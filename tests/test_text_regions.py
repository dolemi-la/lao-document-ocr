from PIL import Image, ImageDraw

from lao_document_ocr.text_regions import (
    MorphologyTextRegionDetector,
    detect_text_regions,
)


def _draw_words(draw: ImageDraw.ImageDraw, x_values, y: int) -> None:
    for x in x_values:
        draw.rectangle((x, y, x + 34, y + 16), fill=0)


def test_detects_two_separate_column_regions() -> None:
    image = Image.new("L", (800, 600), 255)
    draw = ImageDraw.Draw(image)

    for y in (80, 125, 170):
        _draw_words(draw, (50, 105, 165), y)
        _draw_words(draw, (470, 525, 585), y)

    regions = detect_text_regions(image)

    assert len(regions) == 2
    assert regions[0].bbox.x < 250
    assert regions[1].bbox.x > 350
    assert regions[0].bbox.height > 100
    assert regions[1].bbox.height > 100


def test_full_width_heading_stays_separate_from_columns() -> None:
    image = Image.new("L", (800, 700), 255)
    draw = ImageDraw.Draw(image)

    _draw_words(draw, (70, 145, 225, 305, 385, 465), 40)
    for y in (150, 195, 240):
        _draw_words(draw, (50, 105, 165), y)
        _draw_words(draw, (470, 525, 585), y)

    regions = detect_text_regions(image)

    assert len(regions) == 3
    heading = min(regions, key=lambda region: region.bbox.y)
    assert heading.bbox.width > 400
    assert heading.bbox.y < 80


def test_blank_page_has_no_regions() -> None:
    image = Image.new("L", (500, 400), 255)
    assert detect_text_regions(image) == []


def test_detector_exposes_metadata() -> None:
    metadata = MorphologyTextRegionDetector().metadata()
    assert metadata["name"] == "MorphologyTextRegionDetector"
    assert metadata["version"] == "morphology-region-v1"


def test_fallback_detector_uses_secondary_only_when_primary_empty() -> None:
    from lao_document_ocr.models import BoundingBox
    from lao_document_ocr.text_regions import (
        FallbackTextRegionDetector,
        TextRegion,
    )

    class FakeDetector:
        def __init__(self, name, regions):
            self.name = name
            self.regions = regions
            self.calls = 0

        def metadata(self):
            return {"name": self.name}

        def detect(self, image):
            self.calls += 1
            return self.regions

    fallback_region = TextRegion(
        bbox=BoundingBox(x=10, y=10, width=100, height=40)
    )
    primary = FakeDetector("primary", [])
    fallback = FakeDetector("fallback", [fallback_region])
    detector = FallbackTextRegionDetector(primary, fallback)

    assert detector.detect(Image.new("RGB", (200, 100), "white")) == [
        fallback_region
    ]
    assert primary.calls == 1
    assert fallback.calls == 1

    primary.regions = [fallback_region]
    assert detector.detect(Image.new("RGB", (200, 100), "white")) == [
        fallback_region
    ]
    assert primary.calls == 2
    assert fallback.calls == 1


def test_region_aware_line_records_keep_region_group_and_semantic_type() -> None:
    from lao_document_ocr.line_detection import detect_region_aware_line_records
    from lao_document_ocr.models import BlockType, BoundingBox
    from lao_document_ocr.text_regions import TextRegion

    class FakeDetector:
        def metadata(self):
            return {"name": "fake"}

        def detect(self, image):
            return [
                TextRegion(
                    bbox=BoundingBox(x=20, y=20, width=360, height=130),
                    detector="fake-learned",
                    semantic_type=BlockType.PARAGRAPH,
                )
            ]

    image = Image.new("L", (420, 200), 255)
    draw = ImageDraw.Draw(image)
    _draw_words(draw, (40, 100, 165), 50)
    _draw_words(draw, (40, 100, 165), 105)

    records = detect_region_aware_line_records(
        image.convert("RGB"),
        region_detector=FakeDetector(),
    )

    assert len(records) == 2
    assert [record.region_id for record in records] == [1, 1]
    assert [record.line_id for record in records] == [1, 2]
    assert all(record.semantic_type == BlockType.PARAGRAPH for record in records)
    assert all(record.detector == "fake-learned" for record in records)
