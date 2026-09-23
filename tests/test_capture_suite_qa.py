from pathlib import Path

import pytest
from PIL import Image, ImageFont

from lao_document_ocr.capture_suite import generate_capture_suite
from lao_document_ocr.capture_suite_qa import (
    benchmark_capture_suite,
    build_capture_suite_qa_samples,
)
from lao_document_ocr.capture_templates import CaptureTemplate
from lao_document_ocr.models import BoundingBox
from lao_document_ocr.ocr.base import OcrEngine, RecognizedLine


def _font_path() -> Path:
    candidates = [
        Path("/usr/share/fonts/truetype/noto/NotoSansLao-Regular.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=20)
        path = getattr(font, "path", None)
        if path and Path(path).is_file():
            return Path(path)
    except OSError:
        pass
    pytest.skip("No TrueType font available for capture suite QA test")


class FixedEngine(OcrEngine):
    def is_available(self) -> bool:
        return True

    def recognize(self, image: Image.Image) -> list[RecognizedLine]:
        return [
            RecognizedLine(
                text="digital qa",
                bbox=BoundingBox(x=10, y=10, width=120, height=24),
                confidence=1.0,
                block_id=1,
                paragraph_id=1,
                line_id=1,
            )
        ]


def _suite(tmp_path: Path) -> Path:
    return generate_capture_suite(
        [
            "ສະບາຍດີ ໂລກ",
            "ຂອບໃຈ ຫຼາຍ",
            "Lao OCR 2026",
            "ລາຄາ 20,000 ກີບ",
        ],
        tmp_path / "suite",
        _font_path(),
        suite_id="qa-suite",
        text_license="Apache-2.0",
        text_provenance="QA unit-test corpus",
        templates=[
            CaptureTemplate.PLAIN,
            CaptureTemplate.TWO_COLUMN,
            CaptureTemplate.RULED_TABLE,
            CaptureTemplate.BORDERLESS_TABLE,
            CaptureTemplate.RECEIPT,
            CaptureTemplate.FORM,
        ],
        dpi=96,
        lines_per_page=4,
        max_pages_per_template=1,
    )


def test_builds_dataset_samples_for_every_capture_template(tmp_path) -> None:
    suite = _suite(tmp_path)

    root, suite_id, samples = build_capture_suite_qa_samples(suite)

    assert root == suite.parent
    assert suite_id == "qa-suite"
    assert len(samples) == 6
    assert {sample.subset.value for sample in samples} == {
        "clean-print",
        "multi-column",
        "simple-table",
        "complex-table",
        "receipt",
        "form",
    }
    assert all(
        "source:digital-capture-suite" in sample.tags
        for sample in samples
    )
    assert all(
        "page-id:qr-v1" in sample.tags
        for sample in samples
    )


def test_qa_report_is_explicitly_not_real_benchmark(tmp_path) -> None:
    suite = _suite(tmp_path)

    report = benchmark_capture_suite(suite, FixedEngine())

    assert report["qa"]["kind"] == "digital-capture-suite"
    assert report["qa"]["not_real_benchmark"] is True
    assert report["qa"]["sample_count"] == 6
    assert report["qa"]["templates"] == {
        "borderless-table": 1,
        "form": 1,
        "plain": 1,
        "receipt": 1,
        "ruled-table": 1,
        "two-column": 1,
    }
    assert report["overall"]["samples"] == 6
