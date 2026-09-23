from pathlib import Path

import pytest
from PIL import ImageFont

from lao_document_ocr.capture_page_id import decode_page_id_image
from lao_document_ocr.capture_templates import (
    CaptureTemplate,
    render_capture_page,
)


def _font_path() -> Path:
    candidates = [
        Path("/usr/share/fonts/truetype/noto/NotoSansLao-Regular.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
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
    pytest.skip("No TrueType font available for capture template test")


@pytest.mark.parametrize(
    ("template", "required_tag"),
    [
        (CaptureTemplate.PLAIN, "layout:plain"),
        (CaptureTemplate.TWO_COLUMN, "layout:multi-column"),
        (CaptureTemplate.RULED_TABLE, "table:ruled"),
        (CaptureTemplate.BORDERLESS_TABLE, "table:borderless"),
        (CaptureTemplate.RECEIPT, "document:receipt"),
        (CaptureTemplate.FORM, "document:form"),
    ],
)
def test_templates_render_with_expected_tags(template, required_tag) -> None:
    image, truth, tags = render_capture_page(
        page_id=f"test-{template.value}",
        lines=[
            "ສະບາຍດີ ໂລກ",
            "ລາຄາ 20,000 ກີບ",
            "Lao OCR mixed text",
            "ຂອບໃຈ ຫຼາຍ",
        ],
        font_path=_font_path(),
        dpi=96,
        template=template,
    )

    assert image.width > 700
    assert image.height > 1000
    assert "Lao OCR Capture Pack" in truth
    assert f"Page ID: test-{template.value}" in truth
    assert required_tag in tags
    assert "source:capture-pack" in tags
    assert "page-id:qr-v1" in tags
    assert "language:mixed" in tags
    assert decode_page_id_image(image) == f"test-{template.value}"


def test_two_column_truth_is_left_then_right() -> None:
    _, truth, _ = render_capture_page(
        page_id="columns",
        lines=["L1", "L2", "R1", "R2"],
        font_path=_font_path(),
        dpi=96,
        template=CaptureTemplate.TWO_COLUMN,
    )

    assert truth.index("L1") < truth.index("L2")
    assert truth.index("L2") < truth.index("R1")
    assert truth.index("R1") < truth.index("R2")


def test_table_truth_is_tab_separated() -> None:
    _, truth, _ = render_capture_page(
        page_id="table",
        lines=["ກາເຟ", "ຊາ"],
        font_path=_font_path(),
        dpi=96,
        template=CaptureTemplate.RULED_TABLE,
    )

    assert "ລາຍການ ຈຳນວນ (₭)" in truth
    assert "ກາເຟ 5,000 ₭" in truth
    assert "ຊາ 10,000 ₭" in truth
