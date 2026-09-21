import zipfile

from lao_document_ocr.exporters import export_docx, export_json, export_markdown, export_text
from lao_document_ocr.models import Block, BlockType, Document, Page


def sample_document() -> Document:
    return Document(
        source_name="sample.png",
        pages=[
            Page(
                number=1,
                width=1200,
                height=1600,
                blocks=[
                    Block(type=BlockType.HEADING, text="Lao OCR", level=1),
                    Block(type=BlockType.PARAGRAPH, text="Editable document"),
                ],
            )
        ],
    )


def test_text_markdown_json_exports(tmp_path) -> None:
    document = sample_document()
    text_path = export_text(document, tmp_path / "sample.txt")
    markdown_path = export_markdown(document, tmp_path / "sample.md")
    json_path = export_json(document, tmp_path / "sample.json")

    assert "Editable document" in text_path.read_text()
    assert "# Lao OCR" in markdown_path.read_text()
    assert '"source_name": "sample.png"' in json_path.read_text()


def test_docx_export_is_valid_zip_package(tmp_path) -> None:
    path = export_docx(sample_document(), tmp_path / "sample.docx")
    assert path.exists()
    assert zipfile.is_zipfile(path)
