import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from PIL import ImageFont

from lao_document_ocr.capture_kit import build_capture_kit
from lao_document_ocr.capture_suite import generate_capture_suite
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
    pytest.skip("No TrueType font available for capture-kit test")


def _suite(tmp_path) -> Path:
    return generate_capture_suite(
        ["ສະບາຍດີ", "ຂອບໃຈ", "OCR", "20,000 ₭"],
        tmp_path / "suite",
        _font_path(),
        suite_id="kit-test",
        text_license="Apache-2.0",
        text_provenance="Project-authored capture-kit test",
        templates=[
            CaptureTemplate.PLAIN,
            CaptureTemplate.RECEIPT,
        ],
        dpi=96,
        lines_per_page=4,
        max_pages_per_template=1,
    )


def test_capture_kit_contains_only_collector_material(tmp_path) -> None:
    suite = _suite(tmp_path)
    output = build_capture_kit(
        suite,
        tmp_path / "kit.zip",
        source_revision="abc123",
    )

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert names == sorted(names)
        assert set(names) == {
            "CAPTURE-INSTRUCTIONS.md",
            "SHA256SUMS",
            "capture-kit.json",
            "capture-worksheet.csv",
            "kit-test.pdf",
        }
        assert not any("ground-truth" in name for name in names)
        assert not any("/pages/" in name for name in names)

        manifest = json.loads(
            archive.read("capture-kit.json").decode("utf-8")
        )
        assert manifest["suite_id"] == "kit-test"
        assert manifest["source_revision"] == "abc123"
        assert manifest["page_count"] == 2
        assert manifest["excludes_ground_truth"] is True
        assert manifest["excludes_digital_page_images"] is True
        assert manifest["excludes_internal_suite_paths"] is True
        assert len(manifest["source_suite_manifest_sha256"]) == 64

        worksheet = archive.read("capture-worksheet.csv").decode("utf-8")
        header = worksheet.splitlines()[0]
        assert header == (
            "combined_page,template,page_id,required_capture_modes,"
            "capture_file,notes"
        )
        assert "ground_truth" not in worksheet
        assert "digital_page" not in worksheet
        assert "pack_manifest" not in worksheet

        instructions = archive.read(
            "CAPTURE-INSTRUCTIONS.md"
        ).decode("utf-8")
        assert "Keep the page-ID QR marker visible" in instructions
        assert "Do not submit screenshots" in instructions


def test_capture_kit_checksums_match_members(tmp_path) -> None:
    output = build_capture_kit(
        _suite(tmp_path),
        tmp_path / "kit.zip",
    )

    with zipfile.ZipFile(output) as archive:
        checksum_lines = archive.read("SHA256SUMS").decode(
            "utf-8"
        ).splitlines()
        for line in checksum_lines:
            digest, name = line.split("  ", 1)
            assert digest == hashlib.sha256(
                archive.read(name)
            ).hexdigest()


def test_capture_kit_is_reproducible(tmp_path) -> None:
    suite = _suite(tmp_path)
    first = build_capture_kit(
        suite,
        tmp_path / "first.zip",
        source_revision="abc123",
    )
    second = build_capture_kit(
        suite,
        tmp_path / "second.zip",
        source_revision="abc123",
    )

    assert first.read_bytes() == second.read_bytes()


def test_capture_kit_rejects_internal_worksheet_without_required_columns(tmp_path) -> None:
    suite_path = _suite(tmp_path)
    worksheet = suite_path.parent / "capture-worksheet.csv"
    worksheet.write_text("page_id\nkit-test-plain-p0001\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        build_capture_kit(suite_path, tmp_path / "bad.zip")
