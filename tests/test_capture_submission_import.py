from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageFont

from lao_document_ocr.capture_submission import build_capture_submission
from lao_document_ocr.capture_submission_import import register_capture_submission
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
    pytest.skip("No TrueType font available for capture submission import test")


def _suite(tmp_path: Path, *, suite_id: str = "submission-import") -> Path:
    return generate_capture_suite(
        [
            "ສະບາຍດີ ໂລກ",
            "ຂອບໃຈ ຫຼາຍ",
            "ລາຄາ 20,000 ກີບ",
            "Lao OCR 2026",
        ],
        tmp_path / suite_id,
        _font_path(),
        suite_id=suite_id,
        text_license="Apache-2.0",
        text_provenance="capture submission import unit test",
        templates=[CaptureTemplate.PLAIN],
        dpi=96,
        lines_per_page=2,
        max_pages_per_template=2,
    )


def _page_sources(suite_manifest: Path) -> list[tuple[str, Path]]:
    from lao_document_ocr.capture_pack import load_capture_pack

    suite = load_capture_suite(suite_manifest)
    root = suite_manifest.parent
    pack = suite.packs[0]
    pack_root, pack_data = load_capture_pack(root / pack.manifest)
    return [(page.id, pack_root / page.image) for page in pack_data.pages]


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


def _submission(
    tmp_path: Path,
    suite_manifest: Path,
    *,
    capture_count: int = 2,
    kit_sha256: str = "a" * 64,
    source_revision: str = "abc123",
) -> Path:
    capture_dir = tmp_path / "collector-session"
    capture_dir.mkdir()
    pages = _page_sources(suite_manifest)
    metadata = {
        "schema_version": "1",
        "suite_id": load_capture_suite(suite_manifest).suite_id,
        "source_revision": source_revision,
        "kit_sha256": kit_sha256,
        "capture_id": "phone-a",
        "mode": "phone-photo",
        "require_qr": True,
        "expected_pages": len(pages),
    }
    (capture_dir / ".collector-session.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    for page_id, source in pages[:capture_count]:
        _make_optical_capture(source, capture_dir / f"{page_id}.jpg")

    output = tmp_path / "submission.zip"
    build_capture_submission(capture_dir, output)
    return output


def test_register_capture_submission_dry_run_uses_verified_metadata(tmp_path) -> None:
    suite = _suite(tmp_path)
    submission = _submission(tmp_path, suite)
    manifest = tmp_path / "dataset" / "manifest.jsonl"

    report = register_capture_submission(
        submission_path=submission,
        suite_manifest=suite,
        contributor="Test Contributor",
        release_license="CC0-1.0",
        dataset_root=tmp_path / "dataset",
        dataset_manifest=manifest,
        require_complete=True,
        dry_run=True,
        expected_kit_sha256="A" * 64,
        expected_source_revision="abc123",
    )

    assert report.suite_id == "submission-import"
    assert report.capture_id == "phone-a"
    assert report.capture_mode == "phone-photo"
    assert report.submission_complete is True
    assert report.batch.planned_captures == 2
    assert report.batch.registered_captures == 0
    assert report.batch.dry_run is True
    assert report.batch.ignored_files == ()
    assert not manifest.exists()


def test_register_capture_submission_writes_dataset_atomically(tmp_path) -> None:
    suite = _suite(tmp_path)
    submission = _submission(tmp_path, suite)
    dataset_root = tmp_path / "dataset"
    manifest = dataset_root / "manifest.jsonl"

    report = register_capture_submission(
        submission_path=submission,
        suite_manifest=suite,
        contributor="Test Contributor",
        release_license="CC0-1.0",
        dataset_root=dataset_root,
        dataset_manifest=manifest,
        confirm_release=True,
        require_complete=True,
    )

    assert report.batch.registered_captures == 2
    samples = load_manifest(manifest)
    assert len(samples) == 2
    assert all(sample.id.endswith("-phone-a") for sample in samples)
    assert all("capture:optical-evidence" in sample.tags for sample in samples)


def test_register_capture_submission_rejects_wrong_suite(tmp_path) -> None:
    suite = _suite(tmp_path)
    submission = _submission(tmp_path, suite)
    wrong_suite = _suite(tmp_path, suite_id="other-suite")
    manifest = tmp_path / "dataset" / "manifest.jsonl"

    with pytest.raises(ValueError, match="suite does not match"):
        register_capture_submission(
            submission_path=submission,
            suite_manifest=wrong_suite,
            contributor="Test Contributor",
            release_license="CC0-1.0",
            dataset_root=tmp_path / "dataset",
            dataset_manifest=manifest,
            dry_run=True,
        )

    assert not manifest.exists()


def test_register_capture_submission_rejects_wrong_kit_hash(tmp_path) -> None:
    suite = _suite(tmp_path)
    submission = _submission(tmp_path, suite)

    with pytest.raises(ValueError, match="kit SHA-256"):
        register_capture_submission(
            submission_path=submission,
            suite_manifest=suite,
            contributor="Test Contributor",
            release_license="CC0-1.0",
            dataset_root=tmp_path / "dataset",
            dataset_manifest=tmp_path / "dataset" / "manifest.jsonl",
            dry_run=True,
            expected_kit_sha256="b" * 64,
        )


def test_register_capture_submission_require_complete_rejects_partial(tmp_path) -> None:
    suite = _suite(tmp_path)
    submission = _submission(tmp_path, suite, capture_count=1)

    with pytest.raises(ValueError, match="incomplete"):
        register_capture_submission(
            submission_path=submission,
            suite_manifest=suite,
            contributor="Test Contributor",
            release_license="CC0-1.0",
            dataset_root=tmp_path / "dataset",
            dataset_manifest=tmp_path / "dataset" / "manifest.jsonl",
            require_complete=True,
            dry_run=True,
        )
