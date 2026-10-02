"""Explicit input-resolution transfer must never weaken exact-resume checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from test_recognizer_training_resume_optional import (  # noqa: E402
    _config,
    _samples,
    _TinyRecognizer,
    _TinyRecognizerConfig,
)

import lao_document_ocr.recognizer_model as models  # noqa: E402
from lao_document_ocr.recognizer_training import (  # noqa: E402
    _initialize_model_weights,
    train_recognizer,
)
from lao_document_ocr.vocabulary import CharacterVocabulary  # noqa: E402


@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setattr(models, "RecognizerConfig", _TinyRecognizerConfig)
    monkeypatch.setattr(models, "LaoCrnnRecognizer", _TinyRecognizer)


def test_resize_transfers_exact_weights_then_records_fresh_history(tmp_path, monkeypatch, tiny):
    samples = _samples(tmp_path)
    parent = train_recognizer(samples, tmp_path / "parent", training_config=_config(epochs=1))
    digest = hashlib.sha256(parent["checkpoint"].read_bytes()).hexdigest()
    payload = torch.load(parent["checkpoint"], weights_only=True, map_location="cpu")
    expected = payload["state_dict"]
    forward = _TinyRecognizer.forward
    checked = []

    def inspect(self, images):
        assert tuple(images.shape[-2:]) == (32, 96)
        if not checked:
            assert all(torch.equal(v, expected[k]) for k, v in self.state_dict().items())
            checked.append(True)
        return forward(self, images)

    monkeypatch.setattr(_TinyRecognizer, "forward", inspect)
    config = replace(_config(epochs=1), image_height=32, max_width=96, seed=99)
    result = train_recognizer(
        samples,
        tmp_path / "child",
        training_config=config,
        initialize_from=parent["checkpoint"],
        initialize_resize=True,
    )
    state = torch.load(result["training_state"], map_location="cpu", weights_only=False)
    metadata = json.loads(result["metadata"].read_text())
    initial = state["initialization"]
    assert initial["input_geometry_changed"] is True
    assert initial["source_input_geometry"] == {"image_height": 16, "max_width": 64}
    assert initial["target_input_geometry"] == {"image_height": 32, "max_width": 96}
    assert initial["resize_authorized"] is True
    assert initial["source_sha256"] == digest
    assert initial["optimizer_restored"] is False
    assert metadata["initialization"] == initial
    assert state["completed_epoch"] == 1
    assert [row["epoch"] for row in state["history"]] == [1]
    assert {int(v["step"].item()) for v in state["optimizer_state_dict"]["state"].values()} == {2}
    assert checked == [True]
    resumed = train_recognizer(
        samples,
        tmp_path / "child",
        training_config=replace(config, epochs=2),
        resume_from=result["training_state"],
    )
    continued = torch.load(resumed["training_state"], map_location="cpu", weights_only=False)
    assert continued["initialization"] == initial
    assert continued["history"][:1] == state["history"]
    assert hashlib.sha256(parent["checkpoint"].read_bytes()).hexdigest() == digest
    with pytest.raises(ValueError, match="configuration mismatch"):
        train_recognizer(
            samples,
            tmp_path / "bad-resume",
            training_config=_config(epochs=3),
            resume_from=resumed["training_state"],
        )
    assert not (tmp_path / "bad-resume").exists()


def test_resize_remains_opt_in(tmp_path, tiny):
    samples = _samples(tmp_path)
    parent = train_recognizer(samples, tmp_path / "parent", training_config=_config(epochs=1))
    with pytest.raises(ValueError, match="configuration mismatch"):
        train_recognizer(
            samples,
            tmp_path / "untouched",
            training_config=replace(_config(epochs=1), image_height=32, max_width=96),
            initialize_from=parent["checkpoint"],
        )
    assert not (tmp_path / "untouched").exists()


@pytest.mark.parametrize("value", [1, "true", None])
def test_resize_option_rejects_non_boolean_before_loading(tmp_path, value):
    with pytest.raises(ValueError, match="boolean"):
        train_recognizer([], tmp_path / "untouched", initialize_resize=value)
    assert not (tmp_path / "untouched").exists()


def test_resize_flag_requires_initialization_and_cannot_relax_resume(tmp_path):
    with pytest.raises(ValueError, match="requires --initialize-from"):
        train_recognizer([], tmp_path / "untouched", initialize_resize=True)
    with pytest.raises(ValueError, match="requires --initialize-from"):
        train_recognizer(
            [],
            tmp_path / "untouched",
            initialize_resize=True,
            resume_from=tmp_path / "state.pt",
        )
    assert not (tmp_path / "untouched").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "architecture",
        "direction",
        "unknown_field",
        "source_bool",
        "source_float",
        "source_missing",
        "source_small",
        "target_float",
        "vocabulary",
        "nan",
        "shape",
    ],
)
def test_resize_does_not_allow_other_incompatibilities(tmp_path, tiny, fault):
    samples = _samples(tmp_path)
    parent = train_recognizer(samples, tmp_path / "parent", training_config=_config(epochs=1))
    payload = torch.load(parent["checkpoint"], weights_only=True, map_location="cpu")
    config = replace(_config(epochs=1), image_height=32, max_width=96)
    if fault == "architecture":
        payload["model_config"]["hidden_size"] += 1
    elif fault == "direction":
        payload["model_config"]["bidirectional"] = False
    elif fault == "unknown_field":
        payload["model_config"]["unrecognized"] = 0
    elif fault == "source_bool":
        payload["model_config"]["image_height"] = True
    elif fault == "source_float":
        payload["model_config"]["max_width"] = 64.0
    elif fault == "source_missing":
        payload["model_config"].pop("max_width")
    elif fault == "source_small":
        payload["model_config"]["image_height"] = 8
    elif fault == "target_float":
        config = replace(config, image_height=32.0)
    elif fault == "vocabulary":
        payload["vocabulary"]["characters"] = list(reversed(payload["vocabulary"]["characters"]))
    elif fault == "nan":
        payload["state_dict"]["projection.weight"][0, 0] = float("nan")
    else:
        payload["state_dict"]["projection.weight"] = torch.zeros(100, 100)
    source = tmp_path / "invalid.pt"
    torch.save(payload, source)
    with pytest.raises(ValueError):
        train_recognizer(
            samples,
            tmp_path / "untouched",
            training_config=config,
            initialize_from=source,
            initialize_resize=True,
        )
    assert not (tmp_path / "untouched").exists()


def test_real_crnn_tensor_shapes_transfer_at_new_geometry(tmp_path):
    # Actual CNN/LSTM rather than a toy geometry assumption.
    source_config = models.RecognizerConfig(
        image_height=32,
        max_width=64,
        cnn_channels=16,
        hidden_size=8,
        lstm_layers=1,
    )
    target_config = replace(source_config, image_height=64, max_width=128)
    vocabulary = CharacterVocabulary.from_texts(["ກຂ"])
    source = models.LaoCrnnRecognizer(vocabulary.size, source_config).eval()
    path = tmp_path / "parent.pt"
    torch.save(
        {
            "model": "LaoCrnnRecognizer",
            "model_version": "crnn-ctc-v2",
            "model_config": source_config.to_dict(),
            "vocabulary": vocabulary.to_dict(),
            "vocabulary_checksum": vocabulary.checksum(),
            "state_dict": source.state_dict(),
        },
        path,
    )
    target = models.LaoCrnnRecognizer(vocabulary.size, target_config).eval()
    metadata = _initialize_model_weights(
        target,
        path,
        model_config=target_config,
        model_version="crnn-ctc-v2",
        vocabulary=vocabulary,
        allow_resize=True,
    )
    assert metadata["input_geometry_changed"] is True
    assert all(torch.equal(v, source.state_dict()[k]) for k, v in target.state_dict().items())
    with torch.no_grad():
        output = target(torch.zeros((1, 1, 64, 128)))
    assert tuple(output.shape) == (32, 1, vocabulary.size)
    assert torch.isfinite(output).all()
