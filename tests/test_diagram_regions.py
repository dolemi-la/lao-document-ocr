from PIL import Image, ImageDraw

from lao_document_ocr.diagram_regions import detect_diagram_regions
from lao_document_ocr.models import BlockType, BoundingBox


def _diagram_page() -> Image.Image:
    image = Image.new("RGB", (700, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 150, 250, 240), outline="black", width=4)
    draw.rectangle((420, 150, 570, 240), outline="black", width=4)
    draw.line((250, 195, 420, 195), fill="black", width=4)
    draw.polygon(
        [(410, 185), (430, 195), (410, 205)],
        fill="black",
    )
    draw.ellipse((260, 310, 410, 410), outline="black", width=4)
    draw.line((335, 240, 335, 310), fill="black", width=4)
    return image


def test_detects_compact_line_art_diagram() -> None:
    blocks = detect_diagram_regions(_diagram_page(), [])

    assert len(blocks) == 1
    block = blocks[0]
    assert block.type == BlockType.IMAGE
    assert block.metadata["source"] == "diagram-region"
    assert block.metadata["detector"] == "edge-line-art-v1"
    assert block.metadata["edge_density"] > 0.01
    assert block.bbox.width > 400
    assert block.bbox.height > 200


def test_exclusion_box_prevents_duplicate_diagram() -> None:
    excluded = BoundingBox(x=70, y=120, width=540, height=330)

    assert detect_diagram_regions(
        _diagram_page(),
        [],
        exclude_boxes=[excluded],
    ) == []


def test_blank_page_has_no_diagrams() -> None:
    image = Image.new("RGB", (700, 500), "white")
    assert detect_diagram_regions(image, []) == []


def test_near_full_page_frame_is_rejected() -> None:
    image = Image.new("RGB", (700, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 690, 490), outline="black", width=5)

    assert detect_diagram_regions(image, []) == []
