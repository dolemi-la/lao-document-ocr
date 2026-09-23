from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageFont

from lao_document_ocr.capture_batch import register_capture_directory
from lao_document_ocr.capture_registration import CaptureMode
from lao_document_ocr.capture_suite import generate_capture_suite, load_capture_suite
from lao_document_ocr.capture_templates import CaptureTemplate
from lao_document_ocr.dataset import load_manifest


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
    pytest.skip("No TrueType font available for capture batch test")


def _suite(tmp_path: Path) -> Path:
    return generate_capture_suite(
        [
            "ສະບາຍດີ ໂລກ",
            "ຂອບໃຈ ຫຼາຍ",
            "ລາຄາ 20,000 ກີບ",
            "Lao OCR 2026",
        ],
        tmp_path / "suite",
        _font_path(),
        suite_id="batch",
        text_license="Apache-2.0",
        text_provenance="batch unit test",
        templates=[CaptureTemplate.PLAIN],
        dpi=96,
        lines_per_page=2,
        max_pages_per_template=2,
    )


def _page_sources(suite_manifest: Path) -> list[tuple[str, Path]]:
    suite = load_capture_suite(suite_manifest)
    root = suite_manifest.parent
    pack = suite.packs[0]
    manifest = root / pack.manifest

    from lao_document_ocr.capture_pack import load_capture_pack

    pack_root, pack_data = load_capture_pack(manifest)
    return [
        (page.id, pack_root / page.image)
        for page in pack_data.pages
    ]


def _make_optical_capture(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        page = image.convert("RGB")
    canvas = Image.new(
        "RGB",
        (page.width + 180, page.height + 240),
        (220, 215, 205),
    )
    resized = page.resize(
        (page.width - 40, page.height - 55),
        Image.Resampling.BICUBIC,
    )
    canvas.paste(resized, (110, 140))
    canvas.save(destination, quality=82)


def _captures(tmp_path: Path, suite_manifest: Path, count: int = 2) -> Path:
    directory = tmp_path / "captures"
    directory.mkdir()
    for page_id, source in _page_sources(suite_manifest)[:count]:
        _make_optical_capture(
            source,
            directory / f"{page_id}.jpg",
        )
    return directory


def test_dry_run_plans_batch_without_mutating_dataset(tmp_path) -> None:
    suite = _suite(tmp_path)
    captures = _captures(tmp_path, suite)

    report = register_capture_directory(
        suite_manifest=suite,
        capture_dir=captures,
        capture_id="phone-a",
        capture_mode=CaptureMode.PHONE_PHOTO,
        contributor="Test Contributor",
        release_license="CC0-1.0",
        dataset_root=tmp_path / "dataset",
        dataset_manifest=tmp_path / "dataset" / "manifest.jsonl",
        dry_run=True,
        require_complete=True,
    )

    assert report.dry_run is True
    assert report.expected_pages == 2
    assert report.planned_captures == 2
    assert report.registered_captures == 0
    assert report.missing_page_ids == ()
    assert not (tmp_path / "dataset" / "manifest.jsonl").exists()


def test_registers_complete_capture_directory(tmp_path) -> None:
    suite = _suite(tmp_path)
    captures = _captures(tmp_path, suite)
    dataset_root = tmp_path / "dataset"
    manifest = dataset_root / "manifest.jsonl"

    report = register_capture_directory(
        suite_manifest=suite,
        capture_dir=captures,
        capture_id="phone-a",
        capture_mode=CaptureMode.PHONE_PHOTO,
        contributor="Test Contributor",
        release_license="CC0-1.0",
        dataset_root=dataset_root,
        dataset_manifest=manifest,
        confirm_release=True,
        require_complete=True,
    )

    assert report.registered_captures == 2
    assert report.missing_page_ids == ()
    samples = load_manifest(manifest)
    assert len(samples) == 2
    assert all("capture:optical-evidence" in sample.tags for sample in samples)
    assert all(sample.id.endswith("-phone-a") for sample in samples)


def test_partial_directory_reports_missing_pages(tmp_path) -> None:
    suite = _suite(tmp_path)
    captures = _captures(tmp_path, suite, count=1)

    report = register_capture_directory(
        suite_manifest=suite,
        capture_dir=captures,
        capture_id="flatbed-a",
        capture_mode=CaptureMode.FLATBED_SCAN,
        contributor="Test Contributor",
        release_license="CC0-1.0",
        dataset_root=tmp_path / "dataset",
        dataset_manifest=tmp_path / "dataset" / "manifest.jsonl",
        confirm_release=True,
    )

    assert report.registered_captures == 1
    assert len(report.missing_page_ids) == 1


def test_require_complete_rejects_partial_directory_before_write(tmp_path) -> None:
    suite = _suite(tmp_path)
    captures = _captures(tmp_path, suite, count=1)
    manifest = tmp_path / "dataset" / "manifest.jsonl"

    with pytest.raises(ValueError, match="missing 1 suite pages"):
        register_capture_directory(
            suite_manifest=suite,
            capture_dir=captures,
            capture_id="phone-a",
            capture_mode=CaptureMode.PHONE_PHOTO,
            contributor="Test Contributor",
            release_license="CC0-1.0",
            dataset_root=tmp_path / "dataset",
            dataset_manifest=manifest,
            confirm_release=True,
            require_complete=True,
        )

    assert not manifest.exists()


def test_unknown_page_filename_is_rejected(tmp_path) -> None:
    suite = _suite(tmp_path)
    captures = _captures(tmp_path, suite, count=1)
    _make_optical_capture(
        _page_sources(suite)[0][1],
        captures / "not-a-suite-page.jpg",
    )

    with pytest.raises(ValueError, match="do not match suite page ids"):
        register_capture_directory(
            suite_manifest=suite,
            capture_dir=captures,
            capture_id="phone-a",
            capture_mode=CaptureMode.PHONE_PHOTO,
            contributor="Test Contributor",
            release_license="CC0-1.0",
            dataset_root=tmp_path / "dataset",
            dataset_manifest=tmp_path / "dataset" / "manifest.jsonl",
            confirm_release=True,
        )


def test_batch_rolls_back_if_registration_fails(tmp_path, monkeypatch) -> None:
    import lao_document_ocr.capture_batch as capture_batch

    suite = _suite(tmp_path)
    captures = _captures(tmp_path, suite)
    dataset_root = tmp_path / "dataset"
    manifest = dataset_root / "manifest.jsonl"
    original = capture_batch.register_capture
    calls = {"count": 0}

    def fail_on_second(**kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("synthetic batch failure")
        return original(**kwargs)

    monkeypatch.setattr(capture_batch, "register_capture", fail_on_second)

    with pytest.raises(RuntimeError, match="synthetic batch failure"):
        register_capture_directory(
            suite_manifest=suite,
            capture_dir=captures,
            capture_id="phone-a",
            capture_mode=CaptureMode.PHONE_PHOTO,
            contributor="Test Contributor",
            release_license="CC0-1.0",
            dataset_root=dataset_root,
            dataset_manifest=manifest,
            confirm_release=True,
            require_complete=True,
        )

    assert not manifest.exists()
    assert not list((dataset_root / "data").glob("**/*.*"))
    assert not list((dataset_root / "ground-truth").glob("**/*.*"))
