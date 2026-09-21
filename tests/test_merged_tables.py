import zipfile

from PIL import Image, ImageDraw

from lao_document_ocr.exporters import export_docx
from lao_document_ocr.models import BoundingBox, Document, Page
from lao_document_ocr.ocr.base import RecognizedLine
from lao_document_ocr.table_detection import build_table_block, detect_ruled_tables


def _line(text: str, x: int, y: int, width: int = 100) -> RecognizedLine:
    return RecognizedLine(
        text=text,
        bbox=BoundingBox(x=x, y=y, width=width, height=25),
        confidence=0.95,
        block_id=1,
        paragraph_id=1,
        line_id=1,
    )


def _horizontal_merge_image() -> Image.Image:
    image = Image.new("L", (600, 400), 255)
    draw = ImageDraw.Draw(image)
    draw.line((80, 80, 80, 300), fill=0, width=3)
    draw.line((500, 80, 500, 300), fill=0, width=3)
    # Interior vertical border is absent in the first row.
    draw.line((260, 150, 260, 300), fill=0, width=3)
    for y in (80, 150, 225, 300):
        draw.line((80, y, 500, y), fill=0, width=3)
    return image.convert("RGB")


def _vertical_merge_image() -> Image.Image:
    image = Image.new("L", (600, 400), 255)
    draw = ImageDraw.Draw(image)
    for x in (80, 260, 500):
        draw.line((x, 80, x, 300), fill=0, width=3)
    draw.line((80, 80, 500, 80), fill=0, width=3)
    # First interior horizontal border is absent in the first column.
    draw.line((260, 150, 500, 150), fill=0, width=3)
    draw.line((80, 225, 500, 225), fill=0, width=3)
    draw.line((80, 300, 500, 300), fill=0, width=3)
    return image.convert("RGB")


def test_detects_horizontal_column_span() -> None:
    table = detect_ruled_tables(_horizontal_merge_image())[0]
    assert (0, 1) in table.missing_vertical_borders

    block = build_table_block(
        table,
        [
            _line("Summary", 140, 105, width=220),
            _line("Coffee", 105, 175),
            _line("20,000", 300, 175),
            _line("Tea", 105, 250),
            _line("10,000", 300, 250),
        ],
    )

    merged = next(cell for cell in block.cells if cell.row == 0 and cell.column == 0)
    assert merged.text == "Summary"
    assert merged.column_span == 2
    assert merged.row_span == 1
    assert block.metadata["merged_cells"] == 1


def test_detects_vertical_row_span() -> None:
    table = detect_ruled_tables(_vertical_merge_image())[0]
    assert (1, 0) in table.missing_horizontal_borders

    block = build_table_block(
        table,
        [
            _line("Merged", 105, 105),
            _line("Right A", 300, 105),
            _line("Right B", 300, 175),
            _line("Bottom L", 105, 250),
            _line("Bottom R", 300, 250),
        ],
    )

    merged = next(cell for cell in block.cells if cell.row == 0 and cell.column == 0)
    assert merged.text == "Merged"
    assert merged.row_span == 2
    assert merged.column_span == 1


def test_horizontal_merge_exports_as_word_grid_span(tmp_path) -> None:
    table = detect_ruled_tables(_horizontal_merge_image())[0]
    block = build_table_block(
        table,
        [
            _line("Summary", 140, 105, width=220),
            _line("Coffee", 105, 175),
            _line("20,000", 300, 175),
            _line("Tea", 105, 250),
            _line("10,000", 300, 250),
        ],
    )
    document = Document(
        pages=[Page(number=1, width=600, height=400, blocks=[block])]
    )

    path = export_docx(document, tmp_path / "merged.docx")
    with zipfile.ZipFile(path) as package:
        xml = package.read("word/document.xml").decode("utf-8")

    assert "gridSpan" in xml
    assert 'w:val="2"' in xml
    assert "Summary" in xml
