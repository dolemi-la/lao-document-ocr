from PIL import Image

from lao_document_ocr.conversion import convert_document_to_outputs
from lao_document_ocr.models import BoundingBox
from lao_document_ocr.ocr.base import OcrEngine, RecognizedLine


class FakeEngine(OcrEngine):
    def is_available(self) -> bool:
        return True

    def recognize(self, image: Image.Image) -> list[RecognizedLine]:
        return [
            RecognizedLine(
                text="ສະບາຍດີ OCR",
                bbox=BoundingBox(x=10, y=10, width=180, height=30),
                confidence=0.95,
                block_id=1,
                paragraph_id=1,
                line_id=1,
            )
        ]


def test_convert_document_to_all_outputs(tmp_path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (320, 180), "white").save(source)

    outputs = convert_document_to_outputs(
        source,
        tmp_path / "out",
        engine=FakeEngine(),
    )

    assert outputs.docx.is_file()
    assert outputs.markdown.is_file()
    assert outputs.text.is_file()
    assert outputs.json.is_file()
    assert "ສະບາຍດີ OCR" in outputs.text.read_text(encoding="utf-8")
    assert outputs.to_dict()["docx"].endswith("page.docx")


class OrientationEngine(OcrEngine):
    def is_available(self) -> bool:
        return True

    def recognize(self, image: Image.Image) -> list[RecognizedLine]:
        confidence = 0.95 if image.height > image.width else 0.30
        return [
            RecognizedLine(
                text="orientation export text with enough characters",
                bbox=BoundingBox(x=10, y=10, width=120, height=24),
                confidence=confidence,
                block_id=1,
                paragraph_id=1,
                line_id=1,
            )
        ]


def test_convert_document_can_auto_orient_right_angles(tmp_path) -> None:
    import json

    source = tmp_path / "landscape.png"
    Image.new("RGB", (320, 180), "white").save(source)

    outputs = convert_document_to_outputs(
        source,
        tmp_path / "out-auto",
        engine=OrientationEngine(),
        auto_orient_right_angles=True,
    )

    payload = json.loads(outputs.json.read_text(encoding="utf-8"))
    assert payload["pages"][0]["width"] == 180
    assert payload["pages"][0]["height"] == 320
    assert payload["metadata"]["auto_orientation"]["enabled"] is True
    assert payload["metadata"]["auto_orientation"]["pages"][0][
        "degrees_clockwise"
    ] == 90
