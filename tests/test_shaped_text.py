from pathlib import Path

import pytest
from PIL import Image, ImageFont

from lao_document_ocr.shaped_text import (
    missing_font_codepoints,
    paste_shaped_text,
    shaped_text_image,
    validate_font_coverage,
)


def _font_path() -> Path:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return path

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=20)
        path = getattr(font, "path", None)
        if path and Path(path).is_file():
            return Path(path)
    except OSError:
        pass
    pytest.skip("No portable TrueType font available for shaped-text test")


def test_shaped_text_image_renders_visible_rgba() -> None:
    image = shaped_text_image(_font_path(), 28, "Á mixed text")

    assert image.mode == "RGBA"
    assert image.width > 0
    assert image.height > 0
    assert image.getchannel("A").getbbox() is not None


def test_paste_shaped_text_composites_onto_canvas() -> None:
    canvas = Image.new("RGB", (320, 80), "white")
    paste_shaped_text(
        canvas,
        (12, 12),
        "Shaped text",
        font_path=_font_path(),
        font_size=24,
    )

    assert canvas.getbbox() == (0, 0, 320, 80)
    assert canvas.convert("L").getextrema()[0] < 255


def test_font_coverage_reports_and_rejects_missing_codepoint() -> None:
    font = _font_path()
    missing = missing_font_codepoints(font, ["hello", chr(0x10FFFF)])

    assert 0x10FFFF in missing
    with pytest.raises(ValueError, match="U\\+10FFFF"):
        validate_font_coverage(font, ["hello", chr(0x10FFFF)], label="Test font")
