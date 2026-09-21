import zipfile

from lao_document_ocr.exporters import export_docx
from lao_document_ocr.header_footer import mark_repeated_headers_footers
from lao_document_ocr.models import Block, BlockType, BoundingBox, Document, Page


def _block(text: str, y: int) -> Block:
    return Block(
        type=BlockType.PARAGRAPH,
        text=text,
        bbox=BoundingBox(x=50, y=y, width=400, height=24),
    )


def _page(number: int) -> Page:
    return Page(
        number=number,
        width=600,
        height=800,
        blocks=[
            _block("Lao OCR Project", 20),
            _block(f"Body page {number}", 250),
            _block("Open-source document", 750),
        ],
    )


def test_docx_moves_repeated_margin_text_to_header_footer(tmp_path) -> None:
    pages = [_page(1), _page(2), _page(3)]
    mark_repeated_headers_footers(pages)
    document = Document(source_name="multi-page.pdf", pages=pages)

    path = export_docx(document, tmp_path / "result.docx")

    with zipfile.ZipFile(path) as package:
        names = set(package.namelist())
        document_xml = package.read("word/document.xml").decode("utf-8")
        header_name = next(name for name in names if name.startswith("word/header"))
        footer_name = next(name for name in names if name.startswith("word/footer"))
        header_xml = package.read(header_name).decode("utf-8")
        footer_xml = package.read(footer_name).decode("utf-8")

    assert "Lao OCR Project" in header_xml
    assert "Open-source document" in footer_xml
    assert "Lao OCR Project" not in document_xml
    assert "Open-source document" not in document_xml
    assert "Body page 1" in document_xml
    assert "Body page 3" in document_xml
