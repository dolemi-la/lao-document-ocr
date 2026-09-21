from pathlib import Path

import pytest
from PIL import ImageFont

from lao_document_ocr.capture_campaign import build_capture_campaign_report
from lao_document_ocr.capture_suite import generate_capture_suite
from lao_document_ocr.capture_templates import CaptureTemplate
from lao_document_ocr.dataset import DatasetSample, DatasetSplit, DatasetSubset


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
    pytest.skip("No TrueType font available for capture campaign test")


def _suite(tmp_path) -> Path:
    return generate_capture_suite(
        ["ສະບາຍດີ", "ຂອບໃຈ", "OCR", "20,000 ₭"],
        tmp_path / "suite",
        _font_path(),
        suite_id="campaign",
        text_license="CC0-1.0",
        text_provenance="Unit-test corpus",
        templates=[CaptureTemplate.PLAIN, CaptureTemplate.RECEIPT],
        dpi=96,
        lines_per_page=4,
        max_pages_per_template=1,
    )


def _sample(page_id: str, capture_mode: str, sample_id: str) -> DatasetSample:
    subset = (
        DatasetSubset.PHONE_PHOTO
        if capture_mode == "phone-photo"
        else DatasetSubset.CLEAN_PRINT
    )
    return DatasetSample(
        id=sample_id,
        document_id=page_id,
        split=DatasetSplit.TEST,
        subset=subset,
        source=f"{sample_id}.jpg",
        ground_truth=f"{sample_id}.txt",
        license="CC0-1.0",
        provenance="Campaign unit test",
        tags=[f"capture:{capture_mode}"],
    )


def test_campaign_report_tracks_missing_page_mode_pairs(tmp_path) -> None:
    suite = _suite(tmp_path)
    samples = [
        _sample("campaign-plain-p0001", "flatbed-scan", "plain-flatbed"),
        _sample("campaign-plain-p0001", "phone-photo", "plain-phone"),
        _sample("campaign-receipt-p0001", "phone-photo", "receipt-phone"),
    ]

    report = build_capture_campaign_report(suite, samples)

    assert report["expected_pages"] == 2
    assert report["required_captures"] == 4
    assert report["completed_captures"] == 3
    assert report["completion_ratio"] == 0.75
    assert report["mode_coverage"]["phone-photo"]["completion_ratio"] == 1.0
    assert report["mode_coverage"]["flatbed-scan"]["completion_ratio"] == 0.5
    assert report["template_coverage"]["plain"]["completion_ratio"] == 1.0
    assert report["template_coverage"]["receipt"]["completion_ratio"] == 0.5
    assert report["missing"] == [
        {
            "page_id": "campaign-receipt-p0001",
            "template": "receipt",
            "missing_modes": ["flatbed-scan"],
            "present_modes": ["phone-photo"],
        }
    ]


def test_campaign_report_supports_custom_required_modes(tmp_path) -> None:
    from lao_document_ocr.capture_registration import CaptureMode

    suite = _suite(tmp_path)
    samples = [
        _sample("campaign-plain-p0001", "phone-photo", "plain-phone"),
        _sample("campaign-receipt-p0001", "phone-photo", "receipt-phone"),
    ]

    report = build_capture_campaign_report(
        suite,
        samples,
        required_modes=[CaptureMode.PHONE_PHOTO],
    )

    assert report["required_captures"] == 2
    assert report["completed_captures"] == 2
    assert report["completion_ratio"] == 1.0
    assert report["missing"] == []


def test_campaign_report_rejects_duplicate_required_modes(tmp_path) -> None:
    from lao_document_ocr.capture_registration import CaptureMode

    with pytest.raises(ValueError, match="unique"):
        build_capture_campaign_report(
            _suite(tmp_path),
            [],
            required_modes=[
                CaptureMode.PHONE_PHOTO,
                CaptureMode.PHONE_PHOTO,
            ],
        )
