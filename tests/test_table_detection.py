from PIL import Image, ImageDraw

from lao_document_ocr.models import BlockType, BoundingBox
from lao_document_ocr.ocr.base import RecognizedLine
from lao_document_ocr.table_detection import (
    build_table_block,
    detect_ruled_tables,
    split_table_lines,
)


def _table_image() -> Image.Image:
    image = Image.new("L", (600, 400), 255)
    draw = ImageDraw.Draw(image)
    for x in (80, 260, 500):
        draw.line((x, 80, x, 300), fill=0, width=3)
    for y in (80, 150, 225, 300):
        draw.line((80, y, 500, y), fill=0, width=3)
    return image.convert("RGB")


def _line(text: str, x: int, y: int, *, line_id: int) -> RecognizedLine:
    return RecognizedLine(
        text=text,
        bbox=BoundingBox(x=x, y=y, width=100, height=25),
        confidence=0.9,
        block_id=1,
        paragraph_id=1,
        line_id=line_id,
    )


def test_detect_ruled_table_grid() -> None:
    tables = detect_ruled_tables(_table_image())

    assert len(tables) == 1
    table = tables[0]
    assert len(table.x_lines) == 3
    assert len(table.y_lines) == 4
    assert table.bbox.width > 400
    assert table.bbox.height > 200


def test_build_table_block_assigns_text_to_cells() -> None:
    table = detect_ruled_tables(_table_image())[0]
    lines = [
        _line("Name", 105, 105, line_id=1),
        _line("Amount", 300, 105, line_id=2),
        _line("Coffee", 105, 175, line_id=3),
        _line("20,000", 300, 175, line_id=4),
        _line("Tea", 105, 250, line_id=5),
        _line("10,000", 300, 250, line_id=6),
    ]

    block = build_table_block(table, lines)

    assert block.type == BlockType.TABLE
    assert block.metadata["rows"] == 3
    assert block.metadata["columns"] == 2
    assert [cell.text for cell in block.cells[:2]] == ["Name", "Amount"]
    assert block.cells[2].text == "Coffee"
    assert block.cells[3].text == "20,000"


def test_split_table_lines_keeps_non_table_text() -> None:
    table = detect_ruled_tables(_table_image())[0]
    inside = _line("Cell", 120, 110, line_id=1)
    outside = _line("Heading", 100, 20, line_id=2)

    remaining, assignments = split_table_lines([inside, outside], [table])

    assert remaining == [outside]
    assert assignments[0][1] == [inside]


def test_no_table_on_blank_page() -> None:
    assert detect_ruled_tables(Image.new("RGB", (600, 400), "white")) == []
