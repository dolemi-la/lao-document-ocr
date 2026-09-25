from __future__ import annotations

import hashlib
import json

import pytest
from PIL import Image, ImageDraw

torch = pytest.importorskip("torch")

from lao_document_ocr.recognizer_training import TrainingConfig, train_recognizer  # noqa: E402
from lao_document_ocr.training_manifest import TrainingSample  # noqa: E402


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
