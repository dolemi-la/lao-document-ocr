from __future__ import annotations

import hashlib
import json

import pytest
from PIL import Image, ImageDraw

torch = pytest.importorskip("torch")

import lao_document_ocr.recognizer_model as recognizer_model_module  # noqa: E402
from lao_document_ocr.recognizer_training import (  # noqa: E402
    DEV_EVALUATION_VERSION,
    TRAINING_PADDING_STRATEGY,
    TrainingConfig,
    _greedy_decode,
    train_recognizer,
)
from lao_document_ocr.training_manifest import TrainingSample  # noqa: E402
from lao_document_ocr.vocabulary import CharacterVocabulary  # noqa: E402


class _TinyRecognizerConfig:
    def __init__(
        self,
        image_height: int = 16,
        max_width: int = 64,
        cnn_channels: int = 1,
        hidden_size: int = 1,
        lstm_layers: int = 1,
        blank_logit_bias: float = -2.0,
        bidirectional: bool = True,
    ) -> None:
        self.image_height = image_height
        self.max_width = max_width
        self.cnn_channels = cnn_channels
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.blank_logit_bias = blank_logit_bias
        self.bidirectional = bidirectional

    def to_dict(self) -> dict[str, int | float | bool]:
        return {
            "image_height": self.image_height,
            "max_width": self.max_width,
            "cnn_channels": self.cnn_channels,
            "hidden_size": self.hidden_size,
            "lstm_layers": self.lstm_layers,
            "blank_logit_bias": self.blank_logit_bias,
            "bidirectional": self.bidirectional,
        }


class _TinyRecognizer(torch.nn.Module):
    width_downsample_factor = 4

    def __init__(self, num_classes: int, config: _TinyRecognizerConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = torch.nn.Dropout(p=0.2)
        self.projection = torch.nn.Linear(1, num_classes)
        with torch.no_grad():
            self.projection.bias[0] = config.blank_logit_bias

    def forward(self, images):
        sequence = images.mean(dim=2)[:, :, :: self.width_downsample_factor]
        sequence = sequence.permute(2, 0, 1)
        sequence = self.dropout(sequence)
        return self.projection(sequence).log_softmax(dim=-1)

    @classmethod
    def output_lengths(cls, input_widths):
        return torch.clamp(input_widths // cls.width_downsample_factor, min=1)


@pytest.fixture(autouse=True)
def _use_tiny_recognizer(monkeypatch):
    monkeypatch.setattr(
        recognizer_model_module,
        "RecognizerConfig",
        _TinyRecognizerConfig,
    )
    monkeypatch.setattr(
        recognizer_model_module,
        "LaoCrnnRecognizer",
        _TinyRecognizer,
    )


def _samples(tmp_path) -> list[TrainingSample]:
    texts = ["ກກ", "ກຂ", "ຂກ", "ຂຂ", "ກຄ", "ຄກ", "ຂຄ", "ຄຂ"]
    samples: list[TrainingSample] = []
    for index, text in enumerate(texts):
        path = tmp_path / f"sample-{index}.png"
        image = Image.new("L", (48, 16), 255)
        draw = ImageDraw.Draw(image)
        draw.rectangle((4 + index, 3, 12 + index, 12), fill=0)
        image.save(path)
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        samples.append(
            TrainingSample(
                id=f"sample-{index}",
                image=path,
                text=text,
                sha256=sha256,
            )
        )
    return samples


def _config(*, epochs: int) -> TrainingConfig:
    return TrainingConfig(
        epochs=epochs,
        batch_size=4,
        learning_rate=1e-3,
        dev_ratio=0.25,
        seed=12345,
        image_height=16,
        max_width=64,
        num_workers=0,
        device="cpu",
    )


def test_resumed_training_matches_uninterrupted_training(tmp_path) -> None:
    samples = _samples(tmp_path)

    full = train_recognizer(
        samples,
        tmp_path / "full",
        training_config=_config(epochs=2),
    )
    partial = train_recognizer(
        samples,
        tmp_path / "partial",
        training_config=_config(epochs=1),
    )
    resumed = train_recognizer(
        samples,
        tmp_path / "resumed",
        training_config=_config(epochs=2),
        resume_from=partial["training_state"],
    )

    full_state = torch.load(full["training_state"], map_location="cpu", weights_only=False)
    resumed_state = torch.load(
        resumed["training_state"],
        map_location="cpu",
        weights_only=False,
    )

    assert resumed_state["history"] == full_state["history"]
    assert resumed_state["best_dev_cer"] == full_state["best_dev_cer"]
    assert (
        resumed_state["training_samples_checksum"]
        == full_state["training_samples_checksum"]
    )
    assert resumed_state["completed_epoch"] == 2
    assert resumed_state["training_padding_strategy"] == TRAINING_PADDING_STRATEGY
    for key, tensor in full_state["latest_state_dict"].items():
        assert torch.equal(resumed_state["latest_state_dict"][key], tensor), key

    metadata = json.loads(resumed["metadata"].read_text(encoding="utf-8"))
    resume = metadata["resume"]
    assert resume["completed_epoch"] == 1
    assert resume["training_samples_checksum_verified"] is True
    assert resume["resolved_device_verified"] is True
    assert resume["training_padding_strategy_verified"] is True
    assert resume["optimizer_state_restored"] is True
    assert resume["data_loader_generator_state_restored"] is True
    assert resume["rng_state_restored"] is True




def test_resume_rejects_missing_padding_strategy(tmp_path) -> None:
    samples = _samples(tmp_path)
    partial = train_recognizer(
        samples,
        tmp_path / "padding-partial",
        training_config=_config(epochs=1),
    )

    state = torch.load(
        partial["training_state"],
        map_location="cpu",
        weights_only=False,
    )
    state.pop("training_padding_strategy")
    legacy = tmp_path / "legacy-padding-training-state.pt"
    torch.save(state, legacy)

    with pytest.raises(ValueError, match="padding strategy mismatch"):
        train_recognizer(
            samples,
            tmp_path / "padding-resume",
            training_config=_config(epochs=2),
            resume_from=legacy,
        )


def test_resume_rejects_different_training_samples(tmp_path) -> None:
    samples = _samples(tmp_path)
    partial = train_recognizer(
        samples,
        tmp_path / "partial",
        training_config=_config(epochs=1),
    )

    changed = list(samples)
    changed[0] = TrainingSample(
        id=changed[0].id,
        image=changed[0].image,
        text=changed[0].text,
        sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="training samples mismatch"):
        train_recognizer(
            changed,
            tmp_path / "changed",
            training_config=_config(epochs=2),
            resume_from=partial["training_state"],
        )


def test_resume_accepts_legacy_v2_config_without_bidirectional_field(tmp_path) -> None:
    samples = _samples(tmp_path)
    partial = train_recognizer(
        samples,
        tmp_path / "legacy-v2-partial",
        training_config=_config(epochs=1),
    )

    state = torch.load(
        partial["training_state"],
        map_location="cpu",
        weights_only=False,
    )
    state["model_config"].pop("bidirectional", None)
    state["training_config"].pop("bidirectional", None)
    legacy = tmp_path / "legacy-v2-training-state.pt"
    torch.save(state, legacy)

    resumed = train_recognizer(
        samples,
        tmp_path / "legacy-v2-resumed",
        training_config=_config(epochs=2),
        resume_from=legacy,
    )

    metadata = json.loads(resumed["metadata"].read_text(encoding="utf-8"))
    assert metadata["history"][-1]["epoch"] == 2
    assert metadata["model_version"] == "crnn-ctc-v2"


def test_resume_rejects_conflicting_v2_bidirectional_config(tmp_path) -> None:
    samples = _samples(tmp_path)
    partial = train_recognizer(
        samples,
        tmp_path / "conflicting-v2-partial",
        training_config=_config(epochs=1),
    )

    state = torch.load(
        partial["training_state"],
        map_location="cpu",
        weights_only=False,
    )
    state["model_config"]["bidirectional"] = False
    conflicting = tmp_path / "conflicting-v2-training-state.pt"
    torch.save(state, conflicting)

    with pytest.raises(ValueError, match="model configuration mismatch"):
        train_recognizer(
            samples,
            tmp_path / "conflicting-v2-resumed",
            training_config=_config(epochs=2),
            resume_from=conflicting,
        )


def test_resume_rejects_incomplete_training_state(tmp_path) -> None:
    samples = _samples(tmp_path)
    partial = train_recognizer(
        samples,
        tmp_path / "partial-incomplete",
        training_config=_config(epochs=1),
    )

    state = torch.load(
        partial["training_state"],
        map_location="cpu",
        weights_only=False,
    )
    state.pop("rng_state")
    broken = tmp_path / "broken-training-state.pt"
    torch.save(state, broken)

    with pytest.raises(ValueError, match="Resume training state is incomplete: RNG"):
        train_recognizer(
            samples,
            tmp_path / "broken-resume",
            training_config=_config(epochs=2),
            resume_from=broken,
        )


def test_greedy_decode_ignores_padded_timesteps() -> None:
    vocabulary = CharacterVocabulary.from_texts(["ab"])
    a_id = vocabulary.encode("a")[0]
    b_id = vocabulary.encode("b")[0]
    classes = vocabulary.size

    log_probs = torch.full((4, 2, classes), -10.0)
    log_probs[:, :, 0] = -5.0

    log_probs[0, 0, a_id] = 5.0
    log_probs[1, 0, 0] = 5.0
    log_probs[2, 0, b_id] = 5.0
    log_probs[3, 0, 0] = 5.0

    log_probs[0, 1, b_id] = 5.0
    log_probs[1, 1, 0] = 5.0
    log_probs[2, 1, b_id] = 5.0
    log_probs[3, 1, 0] = 5.0

    decoded = _greedy_decode(
        log_probs,
        vocabulary,
        input_lengths=torch.tensor([2, 4]),
    )

    assert decoded == ["a", "bb"]


def test_exact_resume_records_dev_evaluation_version(tmp_path) -> None:
    samples = _samples(tmp_path)
    partial = train_recognizer(
        samples,
        tmp_path / "version-partial",
        training_config=_config(epochs=1),
    )
    state = torch.load(
        partial["training_state"],
        map_location="cpu",
        weights_only=False,
    )

    assert state["dev_evaluation_version"] == DEV_EVALUATION_VERSION
    assert all(
        row["dev_evaluation_version"] == DEV_EVALUATION_VERSION
        for row in state["history"]
    )


def test_all_blank_training_warns_and_persists_health(tmp_path, monkeypatch) -> None:
    class BlankRecognizer(_TinyRecognizer):
        def __init__(self, num_classes, config):
            super().__init__(num_classes, config)
            with torch.no_grad():
                self.projection.weight.zero_()
                self.projection.bias.zero_()
                self.projection.bias[0] = 20.0

    monkeypatch.setattr(recognizer_model_module, "LaoCrnnRecognizer", BlankRecognizer)
    with pytest.warns(RuntimeWarning, match="all .* development predictions are empty"):
        output = train_recognizer(
            _samples(tmp_path), tmp_path / "blank", training_config=_config(epochs=1),
        )
    metadata = json.loads(output["metadata"].read_text(encoding="utf-8"))
    state = torch.load(output["training_state"], map_location="cpu", weights_only=False)
    checkpoint = torch.load(output["checkpoint"], map_location="cpu", weights_only=False)
    health = metadata["history"][-1]["dev_prediction_diagnostics"]
    assert health["samples"] == metadata["dev_samples"]
    assert health["empty_predictions"] == health["samples"]
    assert health["empty_prediction_ratio"] == 1.0
    assert health["predicted_characters"] == 0
    assert health["blank_timestep_ratio"] == 1.0
    assert state["history"][-1]["dev_prediction_diagnostics"] == health
    assert checkpoint["history"][-1]["dev_prediction_diagnostics"] == health
    json.dumps(health, allow_nan=False)


def test_resume_does_not_invent_legacy_prediction_health(tmp_path) -> None:
    samples = _samples(tmp_path)
    partial = train_recognizer(
        samples, tmp_path / "health-partial", training_config=_config(epochs=1),
    )
    state = torch.load(partial["training_state"], map_location="cpu", weights_only=False)
    for row in state["history"]:
        row.pop("dev_prediction_diagnostics")
    legacy = tmp_path / "legacy-health-state.pt"
    torch.save(state, legacy)
    resumed = train_recognizer(
        samples, tmp_path / "health-resumed", training_config=_config(epochs=2),
        resume_from=legacy,
    )
    metadata = json.loads(resumed["metadata"].read_text(encoding="utf-8"))
    assert "dev_prediction_diagnostics" not in metadata["history"][0]
    health = metadata["history"][1]["dev_prediction_diagnostics"]
    assert health["samples"] == metadata["dev_samples"]
    assert metadata["resume"]["rng_state_restored"] is True



def test_resize_preflight_persists_in_all_training_artifacts(tmp_path) -> None:
    output = train_recognizer(
        _samples(tmp_path), tmp_path / "resize-preflight", training_config=_config(epochs=1),
    )
    metadata = json.loads(output["metadata"].read_text(encoding="utf-8"))
    state = torch.load(output["training_state"], map_location="cpu", weights_only=False)
    checkpoint = torch.load(output["checkpoint"], map_location="cpu", weights_only=False)
    report = metadata["ctc_preflight"]
    assert report == state["ctc_preflight"] == checkpoint["ctc_preflight"]
    assert report["input_image_height"] == 16
    assert report["input_max_width"] == 64
    assert report["width_capped_samples"] == 0
    assert report["min_resized_height"] == report["max_resized_height"] == 16
    assert report["min_resized_width"] == report["max_resized_width"] == 48



def _legacy_metric_state(tmp_path, samples):
    partial = train_recognizer(
        samples, tmp_path / "metric-partial", training_config=_config(epochs=1),
    )
    state = torch.load(partial["training_state"], map_location="cpu", weights_only=False)
    state["dev_evaluation_version"] = "valid-timestep-v1"
    state.pop("best_epoch", None)
    state.pop("metric_migrations", None)
    for row in state["history"]:
        row["dev_evaluation_version"] = "valid-timestep-v1"
    return state


def test_legacy_metric_state_requires_explicit_migration(tmp_path) -> None:
    samples = _samples(tmp_path)
    state = _legacy_metric_state(tmp_path, samples)
    path = tmp_path / "legacy-metric.pt"
    torch.save(state, path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="--recompute-resume-metrics"):
        train_recognizer(
            samples, tmp_path / "rejected", training_config=_config(epochs=2),
            resume_from=path,
        )
    assert path.read_bytes() == before
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("latest_score,best_score,expected", [
    (0.2, 0.6, "latest"), (0.6, 0.2, "retained-best"), (0.2, 0.2, "retained-best"),
])
def test_metric_migration_rescores_both_retained_states_and_preserves_history(
    tmp_path, monkeypatch, latest_score, best_score, expected,
) -> None:
    import lao_document_ocr.recognizer_training as module

    samples = _samples(tmp_path)
    state = _legacy_metric_state(tmp_path, samples)
    # Deliberately obsolete score: it must not compete with normalized CER.
    state["best_dev_cer"] = 0.001
    state["history"][0]["dev_cer"] = 0.001
    original_history = [dict(row) for row in state["history"]]
    for value in state["best_state_dict"].values():
        value.add_(0.1)
    path = tmp_path / "retained-metric.pt"
    torch.save(state, path)
    before = path.read_bytes()
    real_evaluate = module.evaluate
    scores = iter([latest_score, best_score, 0.9])
    seen = []

    def controlled_evaluate(model, *args):
        seen.append(module._cpu_state_dict(model.state_dict()))
        result = real_evaluate(model, *args)
        result["cer"] = next(scores)
        return result

    monkeypatch.setattr(module, "evaluate", controlled_evaluate)
    output = train_recognizer(
        samples, tmp_path / "migrated", training_config=_config(epochs=2),
        resume_from=path, recompute_resume_metrics=True,
    )
    metadata = json.loads(output["metadata"].read_text())
    assert metadata["history"][:1] == original_history
    assert metadata["history"][1]["dev_evaluation_version"] == DEV_EVALUATION_VERSION
    assert metadata["best_dev_cer"] == min(latest_score, best_score)
    assert metadata["best_epoch"] == 1
    migration = metadata["metric_migrations"][0]
    assert migration["from_version"] == "valid-timestep-v1"
    assert migration["to_version"] == DEV_EVALUATION_VERSION
    assert migration["selected_state"] == expected
    assert migration["selection_scope"] == "retained-states-only"
    assert migration["historical_scores_rewritten"] is False
    assert migration["source_sha256"] == hashlib.sha256(before).hexdigest()
    for i, name in enumerate(("latest_state_dict", "best_state_dict")):
        for key, value in state[name].items():
            assert torch.equal(seen[i][key], value)
    for artifact in (output["training_state"], output["checkpoint"]):
        payload = torch.load(artifact, map_location="cpu", weights_only=False)
        assert payload["metric_migrations"] == metadata["metric_migrations"]
        assert payload["best_epoch"] == metadata["best_epoch"]
    assert path.read_bytes() == before
    assert metadata["resume"]["metric_migration_performed"] is True


def test_migration_preserves_rng_optimizer_and_continuation_weights(tmp_path, monkeypatch):
    import random

    import numpy as np

    import lao_document_ocr.recognizer_training as module

    samples = _samples(tmp_path)
    partial = train_recognizer(
        samples, tmp_path / "rng-partial", training_config=_config(epochs=1),
    )
    direct = train_recognizer(
        samples, tmp_path / "direct", training_config=_config(epochs=2),
        resume_from=partial["training_state"],
    )
    legacy = torch.load(partial["training_state"], map_location="cpu", weights_only=False)
    legacy["dev_evaluation_version"] = "valid-timestep-v1"
    for row in legacy["history"]:
        row["dev_evaluation_version"] = "valid-timestep-v1"
    path = tmp_path / "rng-legacy.pt"
    torch.save(legacy, path)
    real_evaluate = module.evaluate
    calls = 0

    def consume_random_during_migration(*args):
        nonlocal calls
        calls += 1
        if calls <= 2:
            random.random()
            np.random.random(5)
            torch.rand(5)
        return real_evaluate(*args)

    monkeypatch.setattr(module, "evaluate", consume_random_during_migration)
    migrated = train_recognizer(
        samples, tmp_path / "rng-migrated", training_config=_config(epochs=2),
        resume_from=path, recompute_resume_metrics=True,
    )
    a = torch.load(direct["training_state"], map_location="cpu", weights_only=False)
    b = torch.load(migrated["training_state"], map_location="cpu", weights_only=False)
    assert a["history"][-1] == b["history"][-1]
    for key, value in a["latest_state_dict"].items():
        assert torch.equal(value, b["latest_state_dict"][key]), key
    assert torch.equal(a["data_loader_generator_state"], b["data_loader_generator_state"])
    assert torch.equal(a["rng_state"]["torch"], b["rng_state"]["torch"])
    assert a["rng_state"]["python"] == b["rng_state"]["python"]
    np.testing.assert_equal(a["rng_state"]["numpy"], b["rng_state"]["numpy"])
    for param, values in a["optimizer_state_dict"]["state"].items():
        for key, value in values.items():
            other = b["optimizer_state_dict"]["state"][param][key]
            assert torch.equal(value, other) if torch.is_tensor(value) else value == other
    # A later exact resume keeps the migration trail and does not rescore again.
    monkeypatch.setattr(module, "evaluate", real_evaluate)
    onward = train_recognizer(
        samples, tmp_path / "onward", training_config=_config(epochs=3),
        resume_from=migrated["training_state"],
    )
    metadata = json.loads(onward["metadata"].read_text())
    assert metadata["metric_migrations"] == b["metric_migrations"]
    assert not metadata["resume"]["metric_migration_performed"]


@pytest.mark.parametrize("corruption,match", [
    ("unknown-version", "Unsupported.*dev-evaluation"),
    ("missing-version", "dev-evaluation version mismatch"),
    ("padding", "padding strategy mismatch"),
    ("rng", "incomplete: RNG"),
    ("optimizer", "incomplete: optimizer"),
    ("best", "incomplete: retained best weights"),
    ("samples", "training samples mismatch"),
])
def test_metric_migration_does_not_bypass_other_resume_guards(tmp_path, corruption, match):
    samples = _samples(tmp_path)
    state = _legacy_metric_state(tmp_path, samples)
    if corruption == "unknown-version":
        state["dev_evaluation_version"] = "future-metric-v99"
    elif corruption == "missing-version":
        state.pop("dev_evaluation_version")
    elif corruption == "padding":
        state.pop("training_padding_strategy")
    elif corruption == "rng":
        state.pop("rng_state")
    elif corruption == "optimizer":
        state.pop("optimizer_state_dict")
    elif corruption == "best":
        state.pop("best_state_dict")
    elif corruption == "samples":
        state["training_samples_checksum"] = "0" * 64
    path = tmp_path / "invalid-metric.pt"
    torch.save(state, path)
    with pytest.raises(ValueError, match=match):
        train_recognizer(
            samples, tmp_path / "guarded", training_config=_config(epochs=2),
            resume_from=path, recompute_resume_metrics=True,
        )
    assert not (tmp_path / "guarded").exists()


def test_metric_migration_requires_resume_source(tmp_path):
    with pytest.raises(ValueError, match="requires --resume-from"):
        train_recognizer([], tmp_path / "invalid", recompute_resume_metrics=True)


def test_weights_only_resume_uses_recorded_epoch_and_recomputes_old_score(tmp_path):
    samples = _samples(tmp_path)
    partial = train_recognizer(
        samples, tmp_path / "weights-partial", training_config=_config(epochs=2),
    )
    checkpoint = torch.load(partial["checkpoint"], map_location="cpu", weights_only=False)
    checkpoint["dev_evaluation_version"] = "valid-timestep-v1"
    for row in checkpoint["history"]:
        row["dev_evaluation_version"] = "valid-timestep-v1"
    # An explicit best epoch is authoritative even when an old score is obsolete.
    checkpoint["best_dev_cer"] = 0.001
    path = tmp_path / "old-weights.pt"
    torch.save(checkpoint, path)
    resumed = train_recognizer(
        samples, tmp_path / "weights-resume", training_config=_config(epochs=3),
        resume_from=path,
    )
    metadata = json.loads(resumed["metadata"].read_text())
    assert metadata["resume"]["completed_epoch"] == checkpoint["best_epoch"]
    assert metadata["resume"]["metric_migration_performed"]
    assert not metadata["resume"]["optimizer_state_restored"]
    assert metadata["metric_migrations"][0]["selected_state"] == "weights-only"


def test_legacy_weights_only_ties_use_first_retained_epoch(tmp_path):
    samples = _samples(tmp_path)
    partial = train_recognizer(
        samples, tmp_path / "ties-partial", training_config=_config(epochs=1),
    )
    checkpoint = torch.load(partial["checkpoint"], map_location="cpu", weights_only=False)
    checkpoint.pop("best_epoch")
    checkpoint["history"].append({**checkpoint["history"][0], "epoch": 2})
    path = tmp_path / "tied-weights.pt"
    torch.save(checkpoint, path)
    resumed = train_recognizer(
        samples, tmp_path / "ties-resume", training_config=_config(epochs=3), resume_from=path,
    )
    metadata = json.loads(resumed["metadata"].read_text())
    assert metadata["resume"]["completed_epoch"] == 1
