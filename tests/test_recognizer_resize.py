from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from lao_document_ocr.recognizer_training import (
    plan_line_resize,
    preflight_ctc_capacity,
    prepare_line_pil_image,
)
from lao_document_ocr.training_manifest import TrainingSample
from lao_document_ocr.vocabulary import CharacterVocabulary


@pytest.mark.parametrize(
    "source_size,image_height,max_width,expected",
    [
        ((200, 40), 48, 512, (240, 48, 0, False)),
        ((2000, 40), 48, 320, (320, 6, 21, True)),
        ((2000, 40), 64, 320, (320, 6, 29, True)),
        ((1000, 100), 48, 480, (480, 48, 0, False)),
        ((1, 100), 48, 512, (1, 48, 0, False)),
        ((100000, 1), 48, 32, (32, 1, 23, True)),
    ],
)
def test_resize_plan_matches_actual_pixels(source_size, image_height, max_width, expected):
    plan = plan_line_resize(*source_size, image_height=image_height, max_width=max_width)
    assert (
        plan.width,
        plan.height,
        plan.offset_y,
        plan.width_capped,
    ) == expected
    image = Image.new("L", source_size, 0)
    prepared, width = prepare_line_pil_image(
        image,
        image_height=image_height,
        max_width=max_width,
    )
    assert prepared.shape == (1, image_height, plan.width)
    assert width == plan.width
    assert np.all(prepared[:, plan.offset_y : plan.offset_y + plan.height, :] == 1)
    assert np.all(prepared[:, : plan.offset_y, :] == 0)
    assert np.all(prepared[:, plan.offset_y + plan.height :, :] == 0)


@pytest.mark.parametrize("field", ["source_width", "source_height", "image_height", "max_width"])
@pytest.mark.parametrize("value", [0, -1, 1.5, float("nan"), True])
def test_resize_plan_rejects_invalid_dimensions(field, value):
    args = dict(source_width=20, source_height=16, image_height=48, max_width=768)
    args[field] = value
    with pytest.raises(ValueError, match="positive integers"):
        plan_line_resize(**args)


def test_preflight_reports_width_cap_without_text_or_paths(tmp_path):
    samples = []
    for index, size in enumerate([(100, 40), (2000, 40)]):
        path = tmp_path / f"private-{index}.png"
        Image.new("L", size, 255).save(path)
        samples.append(TrainingSample(f"private-id-{index}", path, "ab"))
    vocab = CharacterVocabulary.from_texts(["ab"])
    report = preflight_ctc_capacity(samples, vocab, image_height=48, max_width=320)
    assert report["input_image_height"] == 48
    assert report["input_max_width"] == 320
    assert report["width_capped_samples"] == 1
    assert report["min_resized_height"] == 6
    assert report["max_resized_height"] == 48
    assert report["min_resized_width"] == 120
    assert report["max_resized_width"] == 320
    serialized = json.dumps(report, allow_nan=False)
    for secret in ("private-id", "private-0.png", "private-1.png", str(tmp_path), '"ab"'):
        assert secret not in serialized


def test_preflight_uses_exif_corrected_dimensions(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (40, 200), "white").save(path, exif=exif)
    sample = TrainingSample("oriented", path, "ab")
    report = preflight_ctc_capacity(
        [sample],
        CharacterVocabulary.from_texts(["ab"]),
        image_height=48,
        max_width=128,
    )
    assert report["width_capped_samples"] == 1
    assert report["min_resized_width"] == 128
    assert report["min_resized_height"] == 26
    assert report["max_available_timesteps"] == 32


def test_capacity_error_does_not_only_recommend_increasing_width(tmp_path):
    path = tmp_path / "narrow.png"
    Image.new("L", (32, 16), 255).save(path)
    sample = TrainingSample("narrow", path, "aaaaaaaaaa")
    with pytest.raises(ValueError, match="--image-height") as exc:
        preflight_ctc_capacity(
            [sample],
            CharacterVocabulary.from_texts([sample.text]),
            image_height=16,
            max_width=768,
        )
    assert "--max-width" in str(exc.value)
    assert "labels" in str(exc.value)


@pytest.mark.parametrize(
    "size,image_height,max_width",
    [
        ((93, 19), 48, 128),
        ((201, 37), 64, 768),
        ((2053, 77), 48, 768),
        ((2431, 165), 64, 768),
        ((1, 100), 48, 768),
        ((128, 64), 64, 128),
    ],
)
def test_resize_refactor_is_pixel_identical_to_previous_policy(size, image_height, max_width):
    rng = np.random.default_rng(724)
    source = Image.fromarray(rng.integers(0, 256, (size[1], size[0]), dtype=np.uint8))
    scale = image_height / source.height
    width = max(1, int(round(source.width * scale)))
    if width > max_width:
        scale = max_width / source.width
        width = max_width
        height = max(1, int(round(source.height * scale)))
    else:
        height = image_height
    resized = source.resize((width, height), Image.Resampling.LANCZOS)
    previous = Image.new("L", (width, image_height), 255)
    previous.paste(resized, (0, max(0, (image_height - height) // 2)))
    previous_array = 1.0 - np.asarray(previous, dtype=np.float32) / 255.0
    actual, actual_width = prepare_line_pil_image(
        source,
        image_height=image_height,
        max_width=max_width,
    )
    assert actual_width == width
    np.testing.assert_array_equal(actual[0], previous_array)
