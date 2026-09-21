from PIL import Image

from lao_document_ocr.models import BoundingBox
from lao_document_ocr.ocr.base import OcrEngine, RecognizedLine
from lao_document_ocr.pipeline import process_document


class TwoColumnEngine(OcrEngine):
    def is_available(self) -> bool:
        return True

    def recognize(self, image: Image.Image) -> list[RecognizedLine]:
        specs = [
            ("L1", 60, 100),
            ("R1", 360, 100),
            ("L2", 60, 160),
            ("R2", 360, 160),
            ("L3", 60, 220),
            ("R3", 360, 220),
        ]
        return [
            RecognizedLine(
                text=text,
                bbox=BoundingBox(x=x, y=y, width=180, height=30),
                confidence=0.95,
                block_id=index,
                paragraph_id=index,
                line_id=index,
            )
            for index, (text, x, y) in enumerate(specs, start=1)
        ]


def test_pipeline_uses_column_reading_order(tmp_path) -> None:
    source = tmp_path / "two-columns.png"
    Image.new("RGB", (600, 400), "white").save(source)

    document = process_document(source, engine=TwoColumnEngine())

    assert [block.text for block in document.pages[0].blocks] == [
        "L1",
        "L2",
        "L3",
        "R1",
        "R2",
        "R3",
    ]
    assert document.plain_text == "L1\nL2\nL3\nR1\nR2\nR3"
