import json
from pathlib import Path

import pytest
from PIL import ImageFont

from lao_document_ocr.capture_pack import (
    capture_pack_page,
    generate_capture_pack,
    load_capture_pack,
)


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

    pytest.skip("No TrueType font available for capture-pack test")


def test_generate_capture_pack_writes_pages_truth_pdf_and_manifest(tmp_path) -> None:
    manifest_path = generate_capture_pack(
        [
            "ສະບາຍດີ ໂລກ",
            "ຂອບໃຈ ຫຼາຍ",
            "Lao OCR 2026",
            "ລາຄາລວມ 125,000 ກີບ",
        ],
        tmp_path / "pack",
        _font_path(),
        pack_id="baseline",
        text_license="CC0-1.0",
        text_provenance="Unit-test corpus",
        dpi=96,
        lines_per_page=2,
    )

    root, manifest = load_capture_pack(manifest_path)

    assert manifest.pack_id == "baseline"
    assert manifest.text_license == "CC0-1.0"
    assert len(manifest.pages) == 2
    assert (root / "baseline.pdf").is_file()
    assert all((root / page.image).is_file() for page in manifest.pages)
    assert all((root / page.ground_truth).is_file() for page in manifest.pages)
    assert all(len(page.sha256) == 64 for page in manifest.pages)
    assert all("layout:plain" in page.tags for page in manifest.pages)
    assert any("language:mixed" in page.tags for page in manifest.pages)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["pages"][0]["id"] == "baseline-p0001"


def test_capture_pack_truth_contains_page_identifier(tmp_path) -> None:
    manifest_path = generate_capture_pack(
        ["ສະບາຍດີ ໂລກ"],
        tmp_path / "pack",
        _font_path(),
        pack_id="id-test",
        text_license="CC0-1.0",
        text_provenance="Unit-test corpus",
        dpi=96,
    )
    root, manifest = load_capture_pack(manifest_path)
    page = capture_pack_page(manifest, "id-test-p0001")
    truth = (root / page.ground_truth).read_text(encoding="utf-8")

    assert "Page ID: id-test-p0001" in truth
    assert "Capture ID: id-test-p0001" in truth
    assert "ສະບາຍດີ ໂລກ" in truth


def test_missing_page_id_is_rejected(tmp_path) -> None:
    manifest_path = generate_capture_pack(
        ["one line"],
        tmp_path / "pack",
        _font_path(),
        pack_id="missing",
        text_license="CC0-1.0",
        text_provenance="Unit-test corpus",
        dpi=96,
    )
    _, manifest = load_capture_pack(manifest_path)

    with pytest.raises(ValueError, match="not found"):
        capture_pack_page(manifest, "missing-p9999")


def test_capture_pack_rejects_unsafe_pack_id(tmp_path) -> None:
    with pytest.raises(ValueError, match="pack_id"):
        generate_capture_pack(
            ["one line"],
            tmp_path / "pack",
            _font_path(),
            pack_id="../unsafe",
            text_license="CC0-1.0",
            text_provenance="Unit-test corpus",
            dpi=96,
        )


def test_capture_pack_rejects_path_traversal_in_manifest(tmp_path) -> None:
    manifest_path = generate_capture_pack(
        ["one line"],
        tmp_path / "pack",
        _font_path(),
        pack_id="safe",
        text_license="CC0-1.0",
        text_provenance="Unit-test corpus",
        dpi=96,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["pages"][0]["ground_truth"] = "../../outside.txt"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="inside the pack directory"):
        load_capture_pack(manifest_path)


def test_capture_pack_records_structured_template_and_tags(tmp_path) -> None:
    from lao_document_ocr.capture_templates import CaptureTemplate

    manifest_path = generate_capture_pack(
        [
            "ສະບາຍດີ ໂລກ",
            "ຂອບໃຈ ຫຼາຍ",
            "ການອ່ານ ແລະ ຂຽນ",
            "ລາຄາ 20,000 ກີບ",
        ],
        tmp_path / "pack",
        _font_path(),
        pack_id="columns",
        text_license="CC0-1.0",
        text_provenance="Unit-test corpus",
        dpi=96,
        lines_per_page=4,
        template=CaptureTemplate.TWO_COLUMN,
    )

    _, manifest = load_capture_pack(manifest_path)
    page = manifest.pages[0]
    assert page.template == "two-column"
    assert "layout:multi-column" in page.tags
    assert "template:two-column" in page.tags


@pytest.mark.parametrize("existing_output", [False, True])
def test_strict_pack_font_failure_leaves_output_untouched(
    tmp_path, existing_output,
) -> None:
    output = tmp_path / "strict-pack"
    if existing_output:
        output.mkdir()
        (output / "keep.txt").write_bytes(b"existing capture data")

    with pytest.raises(ValueError, match="missing .*required character"):
        generate_capture_pack(
            ["OCR " + chr(0x10FFFF)],
            output,
            _font_path(),
            pack_id="strict",
            text_license="CC0-1.0",
            text_provenance="Unit-test corpus",
            require_complete_font=True,
        )

    if existing_output:
        assert sorted(p.name for p in output.iterdir()) == ["keep.txt"]
        assert (output / "keep.txt").read_bytes() == b"existing capture data"
    else:
        assert not output.exists()


def test_strict_pack_checks_normalized_body_labels_and_identifier(
    tmp_path, monkeypatch,
) -> None:
    import lao_document_ocr.capture_pack as module
    from lao_document_ocr.capture_templates import CAPTURE_TEMPLATE_FONT_PROBE

    calls = []

    def check(font_path, texts, *, label):
        calls.append((font_path, texts, label))

    monkeypatch.setattr(module, "validate_font_coverage", check)
    font = _font_path()
    manifest = generate_capture_pack(
        ["  e\u0301 OCR  "],
        tmp_path / "normalized-pack",
        font,
        pack_id="case-X_9",
        text_license="CC0-1.0",
        text_provenance="Unit-test corpus",
        dpi=96,
        require_complete_font=True,
    )

    assert manifest.is_file()
    assert len(calls) == 1
    checked_font, checked_texts, label = calls[0]
    assert checked_font == font
    assert "é OCR" in checked_texts
    assert "  e\u0301 OCR  " not in checked_texts
    assert CAPTURE_TEMPLATE_FONT_PROBE in checked_texts
    assert "case-X_9" in checked_texts
    assert font.name in label
