from pathlib import Path

import pytest
from PIL import Image, ImageFont

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
    Image.new("RGB", (900, 1200), "white").save(path)
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
