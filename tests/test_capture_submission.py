import hashlib
import json
import zipfile

import pytest
from PIL import Image

from lao_document_ocr.capture_submission import (
    CaptureSubmissionError,
    build_capture_submission,
    extract_capture_submission,
    verify_capture_submission,
)


def _session(tmp_path, *, expected_pages=2):
    root = tmp_path / "captures"
    root.mkdir()
    metadata = {
        "schema_version": "1",
        "suite_id": "suite-v1",
        "source_revision": "abc123",
        "kit_sha256": "a" * 64,
        "capture_id": "phone-a",
        "mode": "phone-photo",
        "require_qr": True,
        "expected_pages": expected_pages,
    }
    path = root / ".collector-session.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return root


def _image(path, color):
    Image.new("RGB", (160, 120), color).save(path)


def test_build_verify_and_extract_capture_submission(tmp_path) -> None:
    capture_dir = _session(tmp_path)
    _image(capture_dir / "suite-v1-p0001.jpg", (120, 160, 200))
    _image(capture_dir / "suite-v1-p0002.png", (180, 120, 100))
    output = tmp_path / "submission.zip"

    build_capture_submission(
        capture_dir,
        output,
        require_complete=True,
    )
    submission = verify_capture_submission(output)

    assert submission.suite_id == "suite-v1"
    assert submission.capture_id == "phone-a"
    assert submission.mode == "phone-photo"
    assert submission.complete is True
    assert submission.captured_pages == 2
    assert [item.page_id for item in submission.files] == [
        "suite-v1-p0001",
        "suite-v1-p0002",
    ]

    extracted = extract_capture_submission(
        output,
        tmp_path / "extracted",
    )
    assert (extracted / "suite-v1-p0001.jpg").is_file()
    assert (extracted / "suite-v1-p0002.png").is_file()
    assert (extracted / "capture-submission.json").is_file()
    assert extracted.stat().st_mode & 0o777 == 0o700
    assert (
        extracted / "suite-v1-p0001.jpg"
    ).stat().st_mode & 0o777 == 0o600


def test_capture_submission_is_byte_reproducible(tmp_path) -> None:
    capture_dir = _session(tmp_path)
    _image(capture_dir / "suite-v1-p0001.jpg", (120, 160, 200))
    _image(capture_dir / "suite-v1-p0002.jpg", (180, 120, 100))
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    build_capture_submission(capture_dir, first)
    build_capture_submission(capture_dir, second)

    assert hashlib.sha256(first.read_bytes()).hexdigest() == hashlib.sha256(
        second.read_bytes()
    ).hexdigest()


def test_require_complete_rejects_partial_session(tmp_path) -> None:
    capture_dir = _session(tmp_path, expected_pages=2)
    _image(capture_dir / "suite-v1-p0001.jpg", (120, 160, 200))

    with pytest.raises(CaptureSubmissionError, match="incomplete"):
        build_capture_submission(
            capture_dir,
            tmp_path / "submission.zip",
            require_complete=True,
        )


def test_verify_detects_zip_member_tamper(tmp_path) -> None:
    capture_dir = _session(tmp_path, expected_pages=1)
    _image(capture_dir / "suite-v1-p0001.jpg", (120, 160, 200))
    output = tmp_path / "submission.zip"
    build_capture_submission(capture_dir, output)

    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(output) as source, zipfile.ZipFile(tampered, "w") as dest:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename.startswith("captures/"):
                data += b"tamper"
            dest.writestr(info.filename, data)

    with pytest.raises(CaptureSubmissionError, match="checksum mismatch"):
        verify_capture_submission(tampered)


def test_verify_rejects_unsafe_zip_member(tmp_path) -> None:
    output = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("../escape.jpg", b"bad")
        archive.writestr("capture-submission.json", b"{}")
        archive.writestr("SHA256SUMS", b"")

    with pytest.raises(CaptureSubmissionError, match="Unsafe"):
        verify_capture_submission(output)
