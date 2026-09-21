from PIL import Image

from lao_document_ocr.models import BoundingBox
from lao_document_ocr.ocr.base import OcrEngine, RecognizedLine
from lao_document_ocr.pipeline import process_document


class FakeEngine(OcrEngine):
    def is_available(self) -> bool:
        return True

    def recognize(self, image: Image.Image) -> list[RecognizedLine]:
        return [
            RecognizedLine(
                text="Hello OCR",
                bbox=BoundingBox(x=10, y=10, width=120, height=24),
                confidence=0.99,
                block_id=1,
                paragraph_id=1,
                line_id=1,
            )
        ]


def test_process_image_with_fake_engine(tmp_path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (320, 180), "white").save(image_path)

    document = process_document(image_path, engine=FakeEngine())

    assert len(document.pages) == 1
    assert document.pages[0].blocks[0].text == "Hello OCR"
    assert document.metadata["engine"]["name"] == "FakeEngine"
