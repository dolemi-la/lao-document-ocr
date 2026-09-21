import zipfile

import numpy as np
from PIL import Image, ImageDraw

from lao_document_ocr.exporters import export_docx
from lao_document_ocr.models import BlockType, BoundingBox
from lao_document_ocr.ocr.base import OcrEngine, RecognizedLine
from lao_document_ocr.pipeline import process_document


class TextEngine(OcrEngine):
    def is_available(self) -> bool:
        return True

    def recognize(self, image: Image.Image) -> list[RecognizedLine]:
        return [
            RecognizedLine(
                text="Document heading",
                bbox=BoundingBox(x=55, y=45, width=305, height=42),
                confidence=0.95,
                block_id=1,
                paragraph_id=1,
                line_id=1,
            )
        ]


def test_image_input_preserves_text_and_photo_in_docx(tmp_path) -> None:
    source = tmp_path / "scan.png"
    page = Image.new("RGB", (700, 500), "white")
    draw = ImageDraw.Draw(page)
    draw.rectangle((60, 50, 350, 82), fill="black")

    rng = np.random.default_rng(7)
    photo_array = rng.integers(20, 235, size=(180, 260, 3), dtype=np.uint8)
    page.paste(Image.fromarray(photo_array), (110, 180))
    page.save(source)

    document = process_document(source, engine=TextEngine())

    text_blocks = [
        block for block in document.pages[0].blocks if block.type != BlockType.IMAGE
    ]
    image_blocks = [
        block for block in document.pages[0].blocks if block.type == BlockType.IMAGE
    ]

    assert any(block.text == "Document heading" for block in text_blocks)
    assert len(image_blocks) == 1
    assert image_blocks[0].metadata["source"] == "raster-region"

    docx_path = export_docx(document, tmp_path / "scan.docx")
    with zipfile.ZipFile(docx_path) as package:
        media = [name for name in package.namelist() if name.startswith("word/media/")]
        xml = package.read("word/document.xml").decode("utf-8")

    assert len(media) == 1
    assert "Document heading" in xml
    assert "graphicData" in xml
