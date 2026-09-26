import csv
from pathlib import Path

import pymupdf
import pytest
from PIL import ImageFont

from lao_document_ocr.capture_suite import (
    generate_capture_suite,
    load_capture_suite,
)
from lao_document_ocr.capture_templates import CaptureTemplate


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
    pytest.skip("No TrueType font available for capture suite test")




def test_capture_suite_strict_font_coverage_rejects_missing_glyph(tmp_path) -> None:
    with pytest.raises(ValueError, match="missing .*required character"):
        generate_capture_suite(
            ["Lao OCR", "􏿿"],
            tmp_path / "strict-suite",
            _font_path(),
            suite_id="strict",
            text_license="CC0-1.0",
            text_provenance="Unit-test corpus",
            templates=[CaptureTemplate.PLAIN],
            dpi=96,
            lines_per_page=2,
            max_pages_per_template=1,
            require_complete_font=True,
        )


def test_generate_capture_suite_builds_combined_pdf(tmp_path) -> None:
    manifest_path = generate_capture_suite(
        [
            "ສະບາຍດີ ໂລກ",
            "ຂອບໃຈ ຫຼາຍ",
            "Lao OCR 2026",
            "ລາຄາ 20,000 ກີບ",
        ],
        tmp_path / "suite",
        _font_path(),
        suite_id="baseline",
        text_license="CC0-1.0",
        text_provenance="Unit-test corpus",
        templates=[
            CaptureTemplate.PLAIN,
            CaptureTemplate.TWO_COLUMN,
            CaptureTemplate.RULED_TABLE,
        ],
        dpi=96,
        lines_per_page=4,
        max_pages_per_template=1,
    )

    suite = load_capture_suite(manifest_path)
    root = manifest_path.parent

    assert suite.suite_id == "baseline"
    assert [pack.template for pack in suite.packs] == [
        "plain",
        "two-column",
        "ruled-table",
    ]
    assert all(pack.page_count == 1 for pack in suite.packs)
    assert all((root / pack.manifest).is_file() for pack in suite.packs)
    assert all((root / pack.printable_pdf).is_file() for pack in suite.packs)
    assert suite.worksheet == "capture-worksheet.csv"
    worksheet = root / suite.worksheet
    assert worksheet.is_file()
    with worksheet.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["combined_page"] for row in rows] == ["1", "2", "3"]
    assert [row["template"] for row in rows] == [
        "plain",
        "two-column",
        "ruled-table",
    ]
    assert [row["page_id"] for row in rows] == [
        "baseline-plain-p0001",
        "baseline-two-column-p0001",
        "baseline-ruled-table-p0001",
    ]
    assert all(
        row["required_capture_modes"] == "flatbed-scan;phone-photo"
        for row in rows
    )

    combined = root / suite.combined_pdf
    assert combined.is_file()
    pdf = pymupdf.open(combined)
    try:
        assert pdf.page_count == 3
    finally:
        pdf.close()


def test_capture_suite_defaults_to_all_templates(tmp_path) -> None:
    manifest_path = generate_capture_suite(
        ["ສະບາຍດີ", "ຂອບໃຈ", "OCR", "20,000 ₭"],
        tmp_path / "suite",
        _font_path(),
        suite_id="all",
        text_license="CC0-1.0",
        text_provenance="Unit-test corpus",
        dpi=96,
        lines_per_page=4,
        max_pages_per_template=1,
    )

    suite = load_capture_suite(manifest_path)
    assert {pack.template for pack in suite.packs} == {
        template.value for template in CaptureTemplate
    }


def test_capture_suite_rejects_duplicate_templates(tmp_path) -> None:
    with pytest.raises(ValueError, match="unique"):
        generate_capture_suite(
            ["one line"],
            tmp_path / "suite",
            _font_path(),
            suite_id="duplicate",
            text_license="CC0-1.0",
            text_provenance="Unit-test corpus",
            templates=[CaptureTemplate.PLAIN, CaptureTemplate.PLAIN],
            dpi=96,
        )


def test_combined_capture_pdf_is_byte_reproducible(tmp_path) -> None:
    kwargs = dict(
        corpus_lines=["ສະບາຍດີ", "ຂອບໃຈ", "OCR", "20,000 ₭"],
        font_path=_font_path(),
        suite_id="repro",
        text_license="Apache-2.0",
        text_provenance="Reproducibility unit test",
        templates=[CaptureTemplate.PLAIN, CaptureTemplate.RECEIPT],
        dpi=96,
        lines_per_page=4,
        max_pages_per_template=1,
    )

    first_manifest = generate_capture_suite(
        output_dir=tmp_path / "first",
        **kwargs,
    )
    second_manifest = generate_capture_suite(
        output_dir=tmp_path / "second",
        **kwargs,
    )

    first = load_capture_suite(first_manifest)
    second = load_capture_suite(second_manifest)
    first_pdf = first_manifest.parent / first.combined_pdf
    second_pdf = second_manifest.parent / second.combined_pdf

    assert first_pdf.read_bytes() == second_pdf.read_bytes()
    assert b"/ID[" not in first_pdf.read_bytes()



def test_strict_suite_failure_does_not_create_output(tmp_path) -> None:
    output = tmp_path / "rejected-suite"
    with pytest.raises(ValueError, match="missing .*required character"):
        generate_capture_suite(
            ["OCR " + chr(0x10FFFF)],
            output,
            _font_path(),
            suite_id="strict-suite",
            text_license="CC0-1.0",
            text_provenance="Unit-test corpus",
            require_complete_font=True,
        )
    assert not output.exists()


def test_strict_suite_checks_normalized_text_and_propagates_to_packs(
    tmp_path, monkeypatch,
) -> None:
    import lao_document_ocr.capture_pack as pack_module
    import lao_document_ocr.capture_suite as suite_module

    checked_texts = []
    pack_options = []
    real_generate_pack = suite_module.generate_capture_pack

    def check(font_path, texts, *, label):
        checked_texts.append(texts)

    def generate(*args, **kwargs):
        pack_options.append(kwargs)
        return real_generate_pack(*args, **kwargs)

    monkeypatch.setattr(suite_module, "validate_font_coverage", check)
    monkeypatch.setattr(pack_module, "validate_font_coverage", check)
    monkeypatch.setattr(suite_module, "generate_capture_pack", generate)
    generate_capture_suite(
        ["e\u0301 OCR"],
        tmp_path / "suite-propagation",
        _font_path(),
        suite_id="normalized",
        text_license="CC0-1.0",
        text_provenance="Unit-test corpus",
        templates=[CaptureTemplate.PLAIN],
        dpi=96,
        require_complete_font=True,
    )

    assert len(pack_options) == 1
    assert pack_options[0]["require_complete_font"] is True
    assert len(checked_texts) == 2
    assert all("é OCR" in texts for texts in checked_texts)
