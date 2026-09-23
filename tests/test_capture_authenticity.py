from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lao_document_ocr.capture_authenticity import (
    compare_capture_to_digital_source,
)


def _digital_page(path: Path) -> None:
    image = Image.new("RGB", (800, 1100), "white")
    draw = ImageDraw.Draw(image)
    for y in range(120, 900, 70):
        draw.rectangle((90, y, 680, y + 18), fill="black")
    image.save(path)


def test_jpeg_reencode_is_detected_as_likely_digital_copy(tmp_path) -> None:
    source = tmp_path / "source.png"
    capture = tmp_path / "capture.jpg"
    _digital_page(source)
    with Image.open(source) as image:
        image.save(capture, quality=84)

    result = compare_capture_to_digital_source(source, capture)

    assert result.has_visible_content is True
    assert result.likely_digital_copy is True
    assert result.normalized_mae <= 3.0
    assert result.dhash_distance <= 2


def test_realistic_optical_change_is_not_near_identical(tmp_path) -> None:
    source = tmp_path / "source.png"
    capture = tmp_path / "capture.jpg"
    _digital_page(source)

    with Image.open(source) as image:
        page = image.convert("RGB")
    canvas = Image.new("RGB", (1000, 1400), (218, 212, 202))
    page = page.resize((760, 1045), Image.Resampling.BICUBIC)
    canvas.paste(page, (130, 170))
    array = np.asarray(canvas, dtype=np.int16)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 5, size=array.shape)
    array = np.clip(array + noise, 0, 255).astype(np.uint8)
    Image.fromarray(array).save(capture, quality=80)

    result = compare_capture_to_digital_source(source, capture)

    assert result.has_visible_content is True
    assert result.likely_digital_copy is False


def test_blank_capture_is_flagged_as_missing_visible_content(tmp_path) -> None:
    source = tmp_path / "source.png"
    capture = tmp_path / "capture.png"
    _digital_page(source)
    Image.new("RGB", (900, 1200), "white").save(capture)

    result = compare_capture_to_digital_source(source, capture)

    assert result.has_visible_content is False
    assert result.likely_digital_copy is False
