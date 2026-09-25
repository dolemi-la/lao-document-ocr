from __future__ import annotations

import hashlib
import json

import pytest
from PIL import Image, ImageDraw

torch = pytest.importorskip("torch")

import lao_document_ocr.recognizer_model as recognizer_model_module  # noqa: E402
from lao_document_ocr.recognizer_training import (  # noqa: E402
    DEV_EVALUATION_VERSION,
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
    for key, tensor in full_state["latest_state_dict"].items():
        assert torch.equal(resumed_state["latest_state_dict"][key], tensor), key

    metadata = json.loads(resumed["metadata"].read_text(encoding="utf-8"))
    resume = metadata["resume"]
    assert resume["completed_epoch"] == 1
    assert resume["training_samples_checksum_verified"] is True
    assert resume["resolved_device_verified"] is True
    assert resume["optimizer_state_restored"] is True
    assert resume["data_loader_generator_state_restored"] is True
    assert resume["rng_state_restored"] is True


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
