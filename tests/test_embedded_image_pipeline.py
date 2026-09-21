import io
import zipfile

import pymupdf
from PIL import Image

from lao_document_ocr.exporters import export_docx
from lao_document_ocr.models import BlockType
from lao_document_ocr.ocr.base import OcrEngine
from lao_document_ocr.pipeline import process_document


class EmptyEngine(OcrEngine):
    def is_available(self) -> bool:
        return True

    def recognize(self, image: Image.Image):
        return []


def _png_bytes() -> bytes:
    image = Image.new("RGB", (160, 100), (80, 130, 210))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_native_pdf_image_survives_into_docx_media(tmp_path) -> None:
    pdf_path = tmp_path / "figure.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page(width=600, height=800)
    page.insert_image(
        pymupdf.Rect(120, 180, 360, 340),
        stream=_png_bytes(),
        keep_proportion=False,
    )
    pdf.save(pdf_path)
    pdf.close()

    document = process_document(pdf_path, engine=EmptyEngine())

    image_blocks = [
        block
        for block in document.pages[0].blocks
        if block.type == BlockType.IMAGE
    ]
    assert len(image_blocks) == 1
    assert image_blocks[0].metadata["source"] == "pdf-embedded"
    assert image_blocks[0].metadata["media_type"] == "image/png"

    docx_path = export_docx(document, tmp_path / "figure.docx")
    with zipfile.ZipFile(docx_path) as package:
        media = [name for name in package.namelist() if name.startswith("word/media/")]
        document_xml = package.read("word/document.xml").decode("utf-8")

    assert len(media) == 1
    assert "graphicData" in document_xml
