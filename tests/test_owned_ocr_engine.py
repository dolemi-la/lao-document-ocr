from dataclasses import dataclass

from PIL import Image, ImageDraw

from lao_document_ocr.ocr.owned import OwnedRecognizerEngine


@dataclass
class FakeRecognitionResult:
    text: str
    confidence: float
    calibrated_confidence: float | None = None


class FakeLineRecognizer:
    metadata = {"model_version": "fake"}

    def __init__(self) -> None:
        self.calls = 0

    def recognize_image(self, image: Image.Image) -> FakeRecognitionResult:
        self.calls += 1
        return FakeRecognitionResult(
            text=f"line {self.calls}",
            confidence=0.6,
            calibrated_confidence=0.8,
        )


def _page() -> Image.Image:
    image = Image.new("L", (500, 220), 255)
    draw = ImageDraw.Draw(image)
    for x in (40, 110, 200, 300):
        draw.rectangle((x, 45, x + 45, 65), fill=0)
    for x in (55, 140, 250):
        draw.rectangle((x, 135, x + 60, 158), fill=0)
    return image.convert("RGB")


def test_owned_engine_detects_and_recognizes_lines() -> None:
    recognizer = FakeLineRecognizer()
    engine = OwnedRecognizerEngine(recognizer=recognizer)

    lines = engine.recognize(_page())

    assert [line.text for line in lines] == ["line 1", "line 2"]
    assert all(line.confidence == 0.8 for line in lines)
    assert lines[0].bbox.y < lines[1].bbox.y
    assert engine.metadata()["model"]["model_version"] == "fake"


def test_owned_engine_metadata_exposes_region_detector() -> None:
    engine = OwnedRecognizerEngine(recognizer=FakeLineRecognizer())

    metadata = engine.metadata()

    assert metadata["text_region_detector"]["version"] == "morphology-region-v1"


def test_owned_engine_recognizes_two_column_lines_region_by_region() -> None:
    image = Image.new("L", (800, 600), 255)
    draw = ImageDraw.Draw(image)
    for y in (80, 125, 170):
        for x in (50, 105, 165):
            draw.rectangle((x, y, x + 34, y + 16), fill=0)
        for x in (470, 525, 585):
            draw.rectangle((x, y, x + 34, y + 16), fill=0)

    recognizer = FakeLineRecognizer()
    engine = OwnedRecognizerEngine(recognizer=recognizer)
    lines = engine.recognize(image.convert("RGB"))

    assert len(lines) == 6
    assert recognizer.calls == 6
    left = [line for line in lines if line.bbox.x < 300]
    right = [line for line in lines if line.bbox.x > 350]
    assert len(left) == 3
    assert len(right) == 3
