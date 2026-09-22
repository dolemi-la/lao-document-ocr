import zipfile

from PIL import Image, ImageDraw

from lao_document_ocr.exporters import export_docx
from lao_document_ocr.models import BlockType, BoundingBox
from lao_document_ocr.ocr.base import OcrEngine, RecognizedLine
from lao_document_ocr.pipeline import process_document


class HeadingEngine(OcrEngine):
    def is_available(self) -> bool:
        return True

    def recognize(self, image: Image.Image) -> list[RecognizedLine]:
        return [
            RecognizedLine(
                text="Diagram example",
                bbox=BoundingBox(x=50, y=40, width=260, height=35),
                confidence=0.95,
                block_id=1,
                paragraph_id=1,
                line_id=1,
            )
        ]


def _save_diagram_page(path) -> None:
    image = Image.new("RGB", (700, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 45, 310, 78), fill="black")

    draw.rectangle((100, 170, 250, 260), outline="black", width=4)
    draw.rectangle((420, 170, 570, 260), outline="black", width=4)
    draw.line((250, 215, 420, 215), fill="black", width=4)
    draw.polygon(
        [(410, 205), (430, 215), (410, 225)],
        fill="black",
    )
    draw.ellipse((260, 320, 410, 420), outline="black", width=4)
    draw.line((335, 260, 335, 320), fill="black", width=4)
    image.save(path)


def test_diagram_region_survives_into_docx_media(tmp_path) -> None:
    source = tmp_path / "diagram.png"
    _save_diagram_page(source)

    document = process_document(source, engine=HeadingEngine())

    text_blocks = [
        block
        for block in document.pages[0].blocks
        if block.type != BlockType.IMAGE
    ]
    image_blocks = [
        block
        for block in document.pages[0].blocks
        if block.type == BlockType.IMAGE
    ]

    assert any(block.text == "Diagram example" for block in text_blocks)
    assert len(image_blocks) == 1
    assert image_blocks[0].metadata["source"] == "diagram-region"
    assert image_blocks[0].metadata["detector"] == "edge-line-art-v1"

    path = export_docx(document, tmp_path / "diagram.docx")
    with zipfile.ZipFile(path) as package:
        media = [
            name
            for name in package.namelist()
            if name.startswith("word/media/")
        ]
        xml = package.read("word/document.xml").decode("utf-8")

    assert len(media) == 1
    assert "Diagram example" in xml
    assert "graphicData" in xml
