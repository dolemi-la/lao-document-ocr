import zipfile

from PIL import Image

from lao_document_ocr.exporters import export_docx
from lao_document_ocr.models import BlockType, BoundingBox
from lao_document_ocr.ocr.base import OcrEngine, RecognizedLine
from lao_document_ocr.pipeline import process_document


class LearnedTableEngine(OcrEngine):
    def is_available(self) -> bool:
        return True

    def recognize(self, image: Image.Image) -> list[RecognizedLine]:
        values = [
            ("Item", 60, 80),
            ("Qty", 250, 80),
            ("Amount", 420, 80),
            ("Coffee", 60, 125),
            ("2", 250, 125),
            ("20,000", 420, 125),
            ("Tea", 60, 170),
            ("10,000", 420, 170),
        ]
        return [
            RecognizedLine(
                text=text,
                bbox=BoundingBox(x=x, y=y, width=110, height=24),
                confidence=0.95,
                block_id=5,
                paragraph_id=5,
                line_id=index,
                semantic_type=BlockType.TABLE,
            )
            for index, (text, x, y) in enumerate(values, start=1)
        ]


def test_learned_semantic_table_reaches_editable_docx(tmp_path) -> None:
    source = tmp_path / "learned-table.png"
    Image.new("RGB", (600, 350), "white").save(source)

    document = process_document(source, engine=LearnedTableEngine())

    assert len(document.pages[0].blocks) == 1
    block = document.pages[0].blocks[0]
    assert block.type == BlockType.TABLE
    assert block.metadata["detector"] == "learned-layout-table-v1"
    assert block.metadata["rows"] == 3
    assert block.metadata["columns"] == 3
    assert block.metadata["sparse_cells"] == 1

    path = export_docx(document, tmp_path / "learned-table.docx")
    with zipfile.ZipFile(path) as package:
        xml = package.read("word/document.xml").decode("utf-8")

    assert "<w:tbl>" in xml
    assert "Coffee" in xml
    assert "20,000" in xml
