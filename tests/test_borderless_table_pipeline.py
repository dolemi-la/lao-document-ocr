import zipfile

from PIL import Image

from lao_document_ocr.exporters import export_docx
from lao_document_ocr.models import BlockType, BoundingBox
from lao_document_ocr.ocr.base import OcrEngine, RecognizedLine
from lao_document_ocr.pipeline import process_document


class BorderlessTableEngine(OcrEngine):
    def is_available(self) -> bool:
        return True

    def recognize(self, image: Image.Image) -> list[RecognizedLine]:
        values = [
            ("Item", 70, 80),
            ("Amount", 330, 80),
            ("Coffee", 70, 125),
            ("20,000 ₭", 330, 125),
            ("Tea", 70, 170),
            ("10,000 ₭", 330, 170),
            ("Water", 70, 215),
            ("5,000 ₭", 330, 215),
        ]
        return [
            RecognizedLine(
                text=text,
                bbox=BoundingBox(x=x, y=y, width=110, height=24),
                confidence=0.94,
                block_id=index,
                paragraph_id=index,
                line_id=index,
            )
            for index, (text, x, y) in enumerate(values, start=1)
        ]


def test_borderless_table_reaches_editable_docx_table(tmp_path) -> None:
    source = tmp_path / "borderless.png"
    Image.new("RGB", (600, 350), "white").save(source)

    document = process_document(source, engine=BorderlessTableEngine())

    assert len(document.pages[0].blocks) == 1
    block = document.pages[0].blocks[0]
    assert block.type == BlockType.TABLE
    assert block.metadata["detector"] == "aligned-text-v2"
    assert block.metadata["rows"] == 4
    assert block.metadata["columns"] == 2
    assert block.cells[2].text == "Coffee"
    assert block.cells[3].text == "20,000 ₭"

    path = export_docx(document, tmp_path / "borderless.docx")
    with zipfile.ZipFile(path) as package:
        xml = package.read("word/document.xml").decode("utf-8")

    assert "<w:tbl>" in xml
    assert "Coffee" in xml
    assert "20,000" in xml


class MergedBorderlessTableEngine(OcrEngine):
    def is_available(self) -> bool:
        return True

    def recognize(self, image: Image.Image) -> list[RecognizedLine]:
        values = [
            ("Items", 70, 80, 370),
            ("Coffee", 70, 125, 110),
            ("20,000 ₭", 330, 125, 110),
            ("Tea", 70, 170, 110),
            ("10,000 ₭", 330, 170, 110),
        ]
        return [
            RecognizedLine(
                text=text,
                bbox=BoundingBox(x=x, y=y, width=width, height=24),
                confidence=0.94,
                block_id=index,
                paragraph_id=index,
                line_id=index,
            )
            for index, (text, x, y, width) in enumerate(values, start=1)
        ]


def test_borderless_merged_header_exports_as_word_grid_span(tmp_path) -> None:
    source = tmp_path / "borderless-merged.png"
    Image.new("RGB", (600, 300), "white").save(source)

    document = process_document(source, engine=MergedBorderlessTableEngine())

    assert len(document.pages[0].blocks) == 1
    block = document.pages[0].blocks[0]
    assert block.type == BlockType.TABLE
    assert block.metadata["merged_cells"] == 1
    assert block.cells[0].column_span == 2

    path = export_docx(document, tmp_path / "borderless-merged.docx")
    with zipfile.ZipFile(path) as package:
        xml = package.read("word/document.xml").decode("utf-8")

    assert "gridSpan" in xml
    assert 'w:val="2"' in xml
    assert "Items" in xml


class VerticalMergedBorderlessTableEngine(OcrEngine):
    def is_available(self) -> bool:
        return True

    def recognize(self, image: Image.Image) -> list[RecognizedLine]:
        values = [
            ("Team A", 70, 80, 110, 80),
            ("20,000 ₭", 330, 80, 110, 24),
            ("10,000 ₭", 330, 125, 110, 24),
            ("Team B", 70, 170, 110, 24),
            ("5,000 ₭", 330, 170, 110, 24),
        ]
        return [
            RecognizedLine(
                text=text,
                bbox=BoundingBox(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                ),
                confidence=0.94,
                block_id=index,
                paragraph_id=index,
                line_id=index,
            )
            for index, (text, x, y, width, height) in enumerate(
                values,
                start=1,
            )
        ]


def test_borderless_vertical_merge_exports_as_word_vmerge(tmp_path) -> None:
    source = tmp_path / "borderless-vertical.png"
    Image.new("RGB", (600, 300), "white").save(source)

    document = process_document(
        source,
        engine=VerticalMergedBorderlessTableEngine(),
    )

    assert len(document.pages[0].blocks) == 1
    block = document.pages[0].blocks[0]
    assert block.type == BlockType.TABLE
    merged = next(
        cell
        for cell in block.cells
        if cell.row == 0 and cell.column == 0
    )
    assert merged.row_span == 2
    assert merged.column_span == 1
    assert block.metadata["merge_support"] == "horizontal+vertical"

    path = export_docx(document, tmp_path / "borderless-vertical.docx")
    with zipfile.ZipFile(path) as package:
        xml = package.read("word/document.xml").decode("utf-8")

    assert "vMerge" in xml
    assert "Team A" in xml
