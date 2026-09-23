from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from lao_document_ocr.capture_pack import generate_capture_pack
from lao_document_ocr.capture_registration import (
    CaptureMode,
    register_capture,
)
from lao_document_ocr.dataset import DatasetSubset, load_manifest


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

    pytest.skip("No TrueType font available for capture registration test")


def _pack(tmp_path) -> Path:
    return generate_capture_pack(
        ["ສະບາຍດີ ໂລກ", "ລາຄາ 20,000 ກີບ"],
        tmp_path / "pack",
        _font_path(),
        pack_id="capture",
        text_license="CC0-1.0",
        text_provenance="Unit-test corpus",
        dpi=96,
        lines_per_page=2,
    )


def _capture(tmp_path, name: str = "capture.jpg") -> Path:
    path = tmp_path / name
    image = Image.new("RGB", (900, 1200), (225, 220, 210))
    draw = ImageDraw.Draw(image)
    for y in range(180, 900, 80):
        draw.rectangle((130, y, 760, y + 20), fill="black")
    image.save(path, quality=82)
    return path


def test_register_phone_capture_uses_pack_truth_and_phone_subset(tmp_path) -> None:
    pack = _pack(tmp_path)
    dataset_root = tmp_path / "dataset"
    manifest = dataset_root / "manifest.jsonl"

    sample = register_capture(
        capture_pack_manifest=pack,
        page_id="capture-p0001",
        capture_image=_capture(tmp_path),
        capture_id="iphone-a",
        capture_mode=CaptureMode.PHONE_PHOTO,
        contributor="Example Contributor",
        release_license="CC0-1.0",
        dataset_root=dataset_root,
        dataset_manifest=manifest,
        confirm_release=True,
    )

    assert sample.id == "capture-p0001-iphone-a"
    assert sample.document_id == "capture-p0001"
    assert sample.subset == DatasetSubset.PHONE_PHOTO
    assert "Unit-test corpus" in sample.provenance
    assert "text_license=CC0-1.0" in (sample.notes or "")
    assert "capture:phone-photo" in sample.tags
    assert "layout:plain" in sample.tags
    assert "source:real-capture" in sample.tags
    assert "capture:optical-evidence" in sample.tags
    assert (
        "language:lao" in sample.tags
        or "language:mixed" in sample.tags
    )
    assert (dataset_root / sample.ground_truth).read_text(encoding="utf-8")


def test_multiple_captures_of_same_page_share_split(tmp_path) -> None:
    pack = _pack(tmp_path)
    dataset_root = tmp_path / "dataset"
    manifest = dataset_root / "manifest.jsonl"

    for capture_id, mode in (
        ("flatbed-a", CaptureMode.FLATBED_SCAN),
        ("phone-a", CaptureMode.PHONE_PHOTO),
    ):
        register_capture(
            capture_pack_manifest=pack,
            page_id="capture-p0001",
            capture_image=_capture(tmp_path, f"{capture_id}.jpg"),
            capture_id=capture_id,
            capture_mode=mode,
            contributor="Example Contributor",
            release_license="CC-BY-4.0",
            dataset_root=dataset_root,
            dataset_manifest=manifest,
            confirm_release=True,
        )

    samples = load_manifest(manifest)
    assert len(samples) == 2
    assert samples[0].split == samples[1].split
    assert {sample.subset for sample in samples} == {
        DatasetSubset.CLEAN_PRINT,
        DatasetSubset.PHONE_PHOTO,
    }


def test_registration_requires_release_confirmation(tmp_path) -> None:
    pack = _pack(tmp_path)
    with pytest.raises(ValueError, match="explicit confirmation"):
        register_capture(
            capture_pack_manifest=pack,
            page_id="capture-p0001",
            capture_image=_capture(tmp_path),
            capture_id="phone-a",
            capture_mode=CaptureMode.PHONE_PHOTO,
            contributor="Example Contributor",
            release_license="CC0-1.0",
            dataset_root=tmp_path / "dataset",
            dataset_manifest=tmp_path / "dataset" / "manifest.jsonl",
        )


def test_tampered_capture_pack_page_is_rejected(tmp_path) -> None:
    pack = _pack(tmp_path)
    page_path = pack.parent / "pages" / "capture-p0001.png"
    page_path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        register_capture(
            capture_pack_manifest=pack,
            page_id="capture-p0001",
            capture_image=_capture(tmp_path),
            capture_id="phone-a",
            capture_mode=CaptureMode.PHONE_PHOTO,
            contributor="Example Contributor",
            release_license="CC0-1.0",
            dataset_root=tmp_path / "dataset",
            dataset_manifest=tmp_path / "dataset" / "manifest.jsonl",
            confirm_release=True,
        )


def test_capture_id_must_be_filename_safe(tmp_path) -> None:
    pack = _pack(tmp_path)
    with pytest.raises(ValueError, match="capture_id"):
        register_capture(
            capture_pack_manifest=pack,
            page_id="capture-p0001",
            capture_image=_capture(tmp_path),
            capture_id="phone unsafe",
            capture_mode=CaptureMode.PHONE_PHOTO,
            contributor="Example Contributor",
            release_license="CC0-1.0",
            dataset_root=tmp_path / "dataset",
            dataset_manifest=tmp_path / "dataset" / "manifest.jsonl",
            confirm_release=True,
        )


def test_structured_template_tags_flow_into_real_capture(tmp_path) -> None:
    from lao_document_ocr.capture_templates import CaptureTemplate

    pack = generate_capture_pack(
        ["ສະບາຍດີ", "ຂອບໃຈ", "ລາຄາ 20,000 ກີບ", "OCR"],
        tmp_path / "structured-pack",
        _font_path(),
        pack_id="columns",
        text_license="CC0-1.0",
        text_provenance="Unit-test corpus",
        dpi=96,
        lines_per_page=4,
        template=CaptureTemplate.TWO_COLUMN,
    )
    dataset_root = tmp_path / "dataset-structured"
    manifest = dataset_root / "manifest.jsonl"

    sample = register_capture(
        capture_pack_manifest=pack,
        page_id="columns-p0001",
        capture_image=_capture(tmp_path, "structured.jpg"),
        capture_id="phone-a",
        capture_mode=CaptureMode.PHONE_PHOTO,
        contributor="Example Contributor",
        release_license="CC0-1.0",
        dataset_root=dataset_root,
        dataset_manifest=manifest,
        confirm_release=True,
    )

    assert "layout:multi-column" in sample.tags
    assert "template:two-column" in sample.tags
    assert "capture:phone-photo" in sample.tags


def test_digital_reencode_is_rejected_as_non_optical_capture(tmp_path) -> None:
    pack = _pack(tmp_path)
    source = pack.parent / "pages" / "capture-p0001.png"
    copied = tmp_path / "copied.jpg"
    with Image.open(source) as image:
        image.convert("RGB").save(copied, quality=85)

    with pytest.raises(ValueError, match="visually indistinguishable"):
        register_capture(
            capture_pack_manifest=pack,
            page_id="capture-p0001",
            capture_image=copied,
            capture_id="copied",
            capture_mode=CaptureMode.PHONE_PHOTO,
            contributor="Example Contributor",
            release_license="CC0-1.0",
            dataset_root=tmp_path / "dataset-copy",
            dataset_manifest=tmp_path / "dataset-copy" / "manifest.jsonl",
            confirm_release=True,
        )


def test_blank_capture_is_rejected(tmp_path) -> None:
    pack = _pack(tmp_path)
    blank = tmp_path / "blank.jpg"
    Image.new("RGB", (900, 1200), "white").save(blank)

    with pytest.raises(ValueError, match="too little visible page content"):
        register_capture(
            capture_pack_manifest=pack,
            page_id="capture-p0001",
            capture_image=blank,
            capture_id="blank",
            capture_mode=CaptureMode.PHONE_PHOTO,
            contributor="Example Contributor",
            release_license="CC0-1.0",
            dataset_root=tmp_path / "dataset-blank",
            dataset_manifest=tmp_path / "dataset-blank" / "manifest.jsonl",
            confirm_release=True,
        )
