from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from lao_document_ocr.recognizer_training import (
    MODEL_VERSION,
    TRAINING_PADDING_STRATEGY,
    UNIDIRECTIONAL_MODEL_VERSION,
    TrainingConfig,
    _collate,
    _model_version_for_config,
    _training_samples_checksum,
    ctc_required_timesteps,
    preflight_ctc_capacity,
    prepare_line_image,
)
from lao_document_ocr.training_manifest import TrainingSample


def test_prepare_line_image_normalizes_and_preserves_width(tmp_path) -> None:
    path = tmp_path / "line.png"
    Image.new("L", (200, 40), 255).save(path)

    array, width = prepare_line_image(path, image_height=48, max_width=512)

    assert array.shape == (1, 48, 240)
    assert width == 240
    assert np.all((0 <= array) & (array <= 1))


def test_prepare_line_image_caps_very_wide_input(tmp_path) -> None:
    path = tmp_path / "wide.png"
    Image.new("L", (2000, 40), 255).save(path)

    array, width = prepare_line_image(path, image_height=48, max_width=320)

    assert array.shape == (1, 48, 320)
    assert width == 320


def test_training_config_validation() -> None:
    config = TrainingConfig(epochs=2, batch_size=4)
    payload = json.loads(json.dumps(config.to_dict()))
    assert payload["epochs"] == 2
    assert payload["batch_size"] == 4
    assert payload["max_width"] == 768
    assert payload["device"] == "auto"
    assert payload["bidirectional"] is True


def test_training_config_accepts_supported_devices() -> None:
    for device in ("cpu", "cuda", "mps", "auto"):
        assert TrainingConfig(device=device).device == device


def test_training_config_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="device must be one of"):
        TrainingConfig(device="tpu")


def test_model_version_tracks_recurrent_direction() -> None:
    assert _model_version_for_config(SimpleNamespace(bidirectional=True)) == MODEL_VERSION
    assert (
        _model_version_for_config(SimpleNamespace(bidirectional=False))
        == UNIDIRECTIONAL_MODEL_VERSION
    )


def test_ctc_required_timesteps_counts_adjacent_repeats() -> None:
    assert ctc_required_timesteps([]) == 0
    assert ctc_required_timesteps([1, 2, 3]) == 3
    assert ctc_required_timesteps([1, 1, 2, 2, 2]) == 8




def test_collate_can_pad_to_configured_fixed_width() -> None:
    torch = pytest.importorskip("torch")
    batch = [
        {
            "id": "a",
            "image": torch.ones((1, 16, 20), dtype=torch.float32),
            "width": 20,
            "target": torch.tensor([1, 2], dtype=torch.long),
            "text": "ab",
        },
        {
            "id": "b",
            "image": torch.ones((1, 16, 12), dtype=torch.float32),
            "width": 12,
            "target": torch.tensor([2], dtype=torch.long),
            "text": "b",
        },
    ]

    collated = _collate(batch, padded_width=32)

    assert collated["images"].shape == (2, 1, 16, 32)
    assert collated["widths"].tolist() == [20, 12]
    assert torch.all(collated["images"][0, :, :, :20] == 1)
    assert torch.all(collated["images"][0, :, :, 20:] == 0)
    assert torch.all(collated["images"][1, :, :, :12] == 1)
    assert torch.all(collated["images"][1, :, :, 12:] == 0)


def test_collate_rejects_too_small_fixed_width() -> None:
    torch = pytest.importorskip("torch")
    batch = [
        {
            "id": "a",
            "image": torch.ones((1, 16, 20), dtype=torch.float32),
            "width": 20,
            "target": torch.tensor([1], dtype=torch.long),
            "text": "a",
        }
    ]

    with pytest.raises(ValueError, match="padded_width"):
        _collate(batch, padded_width=19)


def test_training_padding_strategy_is_versioned() -> None:
    assert TRAINING_PADDING_STRATEGY == "fixed-max-width-v1"




def test_ctc_preflight_reports_capacity_margin(tmp_path) -> None:
    image_path = tmp_path / "capacity-ok.png"
    Image.new("L", (32, 16), 255).save(image_path)
    sample = TrainingSample(id="ok", image=image_path, text="aa")
    from lao_document_ocr.vocabulary import CharacterVocabulary

    vocab = CharacterVocabulary.from_texts([sample.text])
    report = preflight_ctc_capacity(
        [sample],
        vocab,
        image_height=16,
        max_width=32,
    )

    assert report["samples"] == 1
    assert report["failures"] == 0
    assert report["min_timestep_margin"] == 5
    assert report["max_required_timesteps"] == 3
    assert report["max_available_timesteps"] == 8


def test_ctc_preflight_rejects_incompatible_sample_before_training(tmp_path) -> None:
    image_path = tmp_path / "capacity-fail.png"
    Image.new("L", (32, 16), 255).save(image_path)
    sample = TrainingSample(id="too-long", image=image_path, text="aaaaaaaaaa")
    from lao_document_ocr.vocabulary import CharacterVocabulary

    vocab = CharacterVocabulary.from_texts([sample.text])
    with pytest.raises(ValueError, match="CTC capacity preflight failed.*too-long"):
        preflight_ctc_capacity(
            [sample],
            vocab,
            image_height=16,
            max_width=32,
        )


def test_training_samples_checksum_binds_order_text_and_image_hash(tmp_path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("L", (8, 8), 255).save(first)
    Image.new("L", (8, 8), 0).save(second)

    samples = [
        TrainingSample(id="a", image=first, text="ກ", sha256="1" * 64),
        TrainingSample(id="b", image=second, text="ຂ", sha256="2" * 64),
    ]

    baseline = _training_samples_checksum(samples)
    assert baseline == _training_samples_checksum(list(samples))
    assert baseline != _training_samples_checksum(list(reversed(samples)))
    assert baseline != _training_samples_checksum(
        [
            TrainingSample(id="a", image=first, text="ກ", sha256="3" * 64),
            samples[1],
        ]
    )
    assert baseline != _training_samples_checksum(
        [
            TrainingSample(id="a", image=first, text="ຄ", sha256="1" * 64),
            samples[1],
        ]
    )
