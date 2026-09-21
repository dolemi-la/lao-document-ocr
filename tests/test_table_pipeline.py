import zipfile

from PIL import Image, ImageDraw

from lao_document_ocr.exporters import export_docx
from lao_document_ocr.models import BlockType, BoundingBox
from lao_document_ocr.ocr.base import OcrEngine, RecognizedLine
from lao_document_ocr.pipeline import process_document


class TableEngine(OcrEngine):
    def is_available(self) -> bool:
        return True

    def recognize(self, image: Image.Image) -> list[RecognizedLine]:
        values = [
            ("Name", 105, 105),
            ("Amount", 300, 105),
            ("Coffee", 105, 175),
            ("20,000", 300, 175),
            ("Tea", 105, 250),
            ("10,000", 300, 250),
        ]
        return [
            RecognizedLine(
                text=text,
                bbox=BoundingBox(x=x, y=y, width=100, height=25),
                confidence=0.95,
                block_id=index,
                paragraph_id=index,
                line_id=index,
            )
            for index, (text, x, y) in enumerate(values, start=1)
        ]


def _save_table_page(path) -> None:
    image = Image.new("RGB", (600, 400), "white")
    draw = ImageDraw.Draw(image)
    for x in (80, 260, 500):
        draw.line((x, 80, x, 300), fill="black", width=3)
    for y in (80, 150, 225, 300):
        draw.line((80, y, 500, y), fill="black", width=3)
    image.save(path)


def test_ruled_table_reaches_docx_as_editable_table(tmp_path) -> None:
    source = tmp_path / "table.png"
    _save_table_page(source)

    document = process_document(source, engine=TableEngine())

    assert len(document.pages[0].blocks) == 1
    block = document.pages[0].blocks[0]
    assert block.type == BlockType.TABLE
    assert block.metadata["rows"] == 3
    assert block.metadata["columns"] == 2
    assert block.cells[0].text == "Name"
    assert block.cells[1].text == "Amount"

    docx_path = export_docx(document, tmp_path / "table.docx")
    with zipfile.ZipFile(docx_path) as package:
        xml = package.read("word/document.xml").decode("utf-8")

    assert "<w:tbl>" in xml
    assert "Coffee" in xml
    assert "20,000" in xml
