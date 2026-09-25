from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from lao_document_ocr.recognizer_model import RecognizerConfig
from lao_document_ocr.recognizer_training import (
    MODEL_VERSION,
    UNIDIRECTIONAL_MODEL_VERSION,
    TrainingConfig,
    _model_version_for_config,
    _training_samples_checksum,
    ctc_required_timesteps,
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
    assert _model_version_for_config(RecognizerConfig(bidirectional=True)) == MODEL_VERSION
    assert (
        _model_version_for_config(RecognizerConfig(bidirectional=False))
        == UNIDIRECTIONAL_MODEL_VERSION
    )


def test_ctc_required_timesteps_counts_adjacent_repeats() -> None:
    assert ctc_required_timesteps([]) == 0
    assert ctc_required_timesteps([1, 2, 3]) == 3
    assert ctc_required_timesteps([1, 1, 2, 2, 2]) == 8


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
