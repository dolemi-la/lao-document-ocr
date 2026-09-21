import numpy as np
from PIL import Image, ImageDraw

from lao_document_ocr.models import BlockType, BoundingBox
from lao_document_ocr.ocr.base import RecognizedLine
from lao_document_ocr.raster_regions import detect_raster_regions


def _photo(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(42)
    array = rng.integers(20, 235, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(array)


def test_detects_dense_photo_region_after_text_masking() -> None:
    page = Image.new("RGB", (700, 500), "white")
    draw = ImageDraw.Draw(page)
    draw.rectangle((60, 50, 350, 82), fill="black")
    page.paste(_photo(260, 180), (110, 180))

    text_line = RecognizedLine(
        text="Heading",
        bbox=BoundingBox(x=55, y=45, width=305, height=42),
        confidence=0.95,
        block_id=1,
        paragraph_id=1,
        line_id=1,
    )

    blocks = detect_raster_regions(page, [text_line])

    assert len(blocks) == 1
    block = blocks[0]
    assert block.type == BlockType.IMAGE
    assert block.metadata["source"] == "raster-region"
    assert block.bbox.y > 100
    assert block.bbox.width >= 240
    assert block.bbox.height >= 160


def test_text_only_region_is_not_preserved_as_image() -> None:
    page = Image.new("RGB", (700, 500), "white")
    draw = ImageDraw.Draw(page)
    draw.rectangle((60, 50, 500, 82), fill="black")
    line = RecognizedLine(
        text="Text",
        bbox=BoundingBox(x=55, y=45, width=455, height=42),
        confidence=0.95,
        block_id=1,
        paragraph_id=1,
        line_id=1,
    )

    assert detect_raster_regions(page, [line]) == []


def test_exclusion_box_prevents_native_image_duplicate() -> None:
    page = Image.new("RGB", (700, 500), "white")
    page.paste(_photo(260, 180), (110, 180))
    excluded = BoundingBox(x=100, y=170, width=280, height=200)

    assert detect_raster_regions(page, [], exclude_boxes=[excluded]) == []


def test_near_full_page_visual_is_rejected() -> None:
    page = _photo(700, 500)
    assert detect_raster_regions(page, []) == []
