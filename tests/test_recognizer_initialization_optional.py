"""Weights-only fine tuning is a new experiment, never a relaxed resume."""

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
from lao_document_ocr.recognizer_training import train_recognizer  # noqa: E402


@pytest.fixture(autouse=True)
def tiny_model(monkeypatch):
    monkeypatch.setattr(models, "RecognizerConfig", _TinyRecognizerConfig)
    monkeypatch.setattr(models, "LaoCrnnRecognizer", _TinyRecognizer)


def test_initialization_allows_new_data_and_resets_optimizer_history(tmp_path):
    samples = _samples(tmp_path)
    parent = train_recognizer(samples, tmp_path / "parent", training_config=_config(epochs=2))
    before = hashlib.sha256(parent["checkpoint"].read_bytes()).hexdigest()
    changed = [replace(sample, id="new-" + sample.id) for sample in samples]
    config = replace(_config(epochs=1), learning_rate=0.0003, seed=42)
    result = train_recognizer(
        changed,
        tmp_path / "child",
        training_config=config,
        initialize_from=parent["checkpoint"],
    )
    state = torch.load(result["training_state"], map_location="cpu", weights_only=False)
    metadata = json.loads(result["metadata"].read_text())
    assert state["completed_epoch"] == 1
    assert [row["epoch"] for row in state["history"]] == [1]
    assert state["initialization"]["mode"] == "weights-only-new-experiment"
    assert state["initialization"]["source_sha256"] == before
    assert state["initialization"]["optimizer_restored"] is False
    assert metadata["initialization"] == state["initialization"]
    assert metadata["resume"] is None
    steps = {int(v["step"].item()) for v in state["optimizer_state_dict"]["state"].values()}
    assert steps == {2}
    assert hashlib.sha256(parent["checkpoint"].read_bytes()).hexdigest() == before
    resumed = train_recognizer(
        changed,
        tmp_path / "child",
        training_config=replace(config, epochs=2),
        resume_from=result["training_state"],
    )
    continued = torch.load(resumed["training_state"], map_location="cpu", weights_only=False)
    assert continued["initialization"] == state["initialization"]
    assert continued["history"][:1] == state["history"]


def test_initialization_loads_exact_weights_before_first_forward(tmp_path, monkeypatch):
    samples = _samples(tmp_path)
    parent = train_recognizer(samples, tmp_path / "parent", training_config=_config(epochs=1))
    checkpoint = torch.load(parent["checkpoint"], map_location="cpu", weights_only=True)
    expected = checkpoint["state_dict"]
    forward = _TinyRecognizer.forward
    checked = []

    def inspect(self, images):
        if not checked:
            assert all(
                torch.equal(value, expected[key]) for key, value in self.state_dict().items()
            )
            checked.append(True)
        return forward(self, images)

    monkeypatch.setattr(_TinyRecognizer, "forward", inspect)
    train_recognizer(
        samples,
        tmp_path / "child",
        training_config=_config(epochs=1),
        initialize_from=parent["checkpoint"],
    )
    assert checked == [True]


@pytest.mark.parametrize("fault", ["model", "version", "config", "vocabulary", "nan", "shape"])
def test_incompatible_initialization_fails_before_creating_output(tmp_path, fault):
    samples = _samples(tmp_path)
    parent = train_recognizer(samples, tmp_path / "parent", training_config=_config(epochs=1))
    payload = torch.load(parent["checkpoint"], map_location="cpu", weights_only=True)
    if fault == "model":
        payload["model"] = "Other"
    elif fault == "version":
        payload["model_version"] = "unknown"
    elif fault == "config":
        payload["model_config"]["max_width"] += 4
    elif fault == "vocabulary":
        payload["vocabulary"]["characters"] = list(reversed(payload["vocabulary"]["characters"]))
    elif fault == "nan":
        payload["state_dict"]["projection.weight"][0, 0] = float("nan")
    else:
        payload["state_dict"]["projection.weight"] = torch.zeros(100, 100)
    source = tmp_path / "invalid.pt"
    torch.save(payload, source)
    output = tmp_path / "untouched"
    with pytest.raises(ValueError, match="Initialization"):
        train_recognizer(samples, output, training_config=_config(epochs=1), initialize_from=source)
    assert not output.exists()


def test_initialization_cannot_overwrite_parent_or_combine_with_resume(tmp_path):
    samples = _samples(tmp_path)
    parent = train_recognizer(samples, tmp_path / "parent", training_config=_config(epochs=1))
    with pytest.raises(ValueError, match="empty"):
        train_recognizer(samples, tmp_path / "parent", initialize_from=parent["checkpoint"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        train_recognizer(
            samples,
            tmp_path / "untouched",
            initialize_from=parent["checkpoint"],
            resume_from=parent["training_state"],
        )
    with pytest.raises(ValueError, match="requires --resume-from"):
        train_recognizer(
            samples,
            tmp_path / "untouched",
            initialize_from=parent["checkpoint"],
            recompute_resume_metrics=True,
        )
    assert not (tmp_path / "untouched").exists()


def test_full_training_state_is_not_implicitly_selected_for_initialization(tmp_path):
    samples = _samples(tmp_path)
    parent = train_recognizer(samples, tmp_path / "parent", training_config=_config(epochs=1))
    with pytest.raises(ValueError, match="Initialization"):
        train_recognizer(
            samples,
            tmp_path / "untouched",
            training_config=_config(epochs=1),
            initialize_from=parent["training_state"],
        )
    assert not (tmp_path / "untouched").exists()
