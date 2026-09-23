import hashlib
import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from lao_document_ocr.capture_collector import (
    CaptureCollectorError,
    create_collector_app,
    load_collector_kit,
)
from lao_document_ocr.capture_page_id import render_page_id_qr
from lao_document_ocr.capture_registration import CaptureMode


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _kit(tmp_path, *, corrupt=False, unsafe=False):
    worksheet = (
        b"combined_page,template,page_id,required_capture_modes,capture_file,notes\n"
        b"1,plain,suite-plain-p0001,flatbed-scan;phone-photo,,\n"
        b"2,form,suite-form-p0001,flatbed-scan,,\n"
    )
    pdf = b"%PDF-1.4\ncollector fixture\n"
    instructions = b"collector instructions\n"
    kit_manifest = {
        "schema_version": "1",
        "suite_id": "suite",
        "source_revision": "abc123",
        "page_count": 2,
        "excludes_ground_truth": True,
        "excludes_digital_page_images": True,
        "excludes_internal_suite_paths": True,
    }
    manifest_bytes = (
        json.dumps(kit_manifest, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")

    payloads = {
        "CAPTURE-INSTRUCTIONS.md": instructions,
        "capture-kit.json": manifest_bytes,
        "capture-worksheet.csv": worksheet,
        "suite.pdf": pdf,
    }
    sums = "\n".join(
        f"{_sha256(data)}  {name}"
        for name, data in sorted(payloads.items())
    ) + "\n"

    path = tmp_path / "collector.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in payloads.items():
            if corrupt and name == "capture-worksheet.csv":
                data = data + b"tampered"
            archive.writestr(name, data)
        archive.writestr("SHA256SUMS", sums)
        if unsafe:
            archive.writestr("../escape.txt", b"bad")
    return path


def _jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (160, 120), (120, 160, 200)).save(
        buffer,
        format="JPEG",
    )
    return buffer.getvalue()


def test_load_collector_kit_verifies_blind_safe_manifest(tmp_path) -> None:
    kit = load_collector_kit(_kit(tmp_path))

    assert kit.suite_id == "suite"
    assert kit.source_revision == "abc123"
    assert kit.pdf_name == "suite.pdf"
    assert len(kit.pages) == 2
    assert kit.pages[0].required_modes == (
        "flatbed-scan",
        "phone-photo",
    )
    assert len(kit.sha256) == 64


def test_load_collector_kit_rejects_checksum_mismatch(tmp_path) -> None:
    try:
        load_collector_kit(_kit(tmp_path, corrupt=True))
    except CaptureCollectorError as exc:
        assert "checksum mismatch" in str(exc).lower()
    else:
        raise AssertionError("corrupt kit should be rejected")


def test_load_collector_kit_rejects_unsafe_zip_member(tmp_path) -> None:
    try:
        load_collector_kit(_kit(tmp_path, unsafe=True))
    except CaptureCollectorError as exc:
        assert "unsafe zip member" in str(exc).lower()
    else:
        raise AssertionError("unsafe kit should be rejected")


def test_collector_phone_session_upload_progress_and_retake(tmp_path) -> None:
    output = tmp_path / "captures"
    app = create_collector_app(
        _kit(tmp_path),
        output,
        capture_id="phone-a",
        mode=CaptureMode.PHONE_PHOTO,
        require_qr=False,
    )
    client = TestClient(app)

    session = client.get("/api/session")
    assert session.status_code == 200
    payload = session.json()
    assert payload["expected"] == 1
    assert payload["completed"] == 0
    assert payload["pages"][0]["page_id"] == "suite-plain-p0001"
    assert "output_dir" not in payload

    index = client.get("/")
    assert index.status_code == 200
    assert 'capture="environment"' in index.text
    assert "suite-plain-p0001" not in index.text
    assert index.headers["cache-control"] == "no-store"
    assert index.headers["x-frame-options"] == "DENY"

    upload = client.post(
        "/api/captures/suite-plain-p0001",
        files={"file": ("camera.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert upload.status_code == 201
    assert upload.json()["capture_file"] == "suite-plain-p0001.jpg"

    saved = output / "suite-plain-p0001.jpg"
    assert saved.is_file()
    assert saved.stat().st_mode & 0o777 == 0o600
    assert output.stat().st_mode & 0o777 == 0o700
    metadata = output / ".collector-session.json"
    assert metadata.is_file()
    assert metadata.stat().st_mode & 0o777 == 0o600

    session = client.get("/api/session").json()
    assert session["completed"] == 1
    assert session["remaining"] == 0
    assert session["pages"][0]["state"] == "complete"

    duplicate = client.post(
        "/api/captures/suite-plain-p0001",
        files={"file": ("camera.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert duplicate.status_code == 409

    deleted = client.delete("/api/captures/suite-plain-p0001")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] == 1
    assert not saved.exists()
    assert client.get("/api/session").json()["completed"] == 0


def test_collector_rejects_invalid_image_without_temp_leak(tmp_path) -> None:
    output = tmp_path / "captures"
    app = create_collector_app(
        _kit(tmp_path),
        output,
        capture_id="phone-a",
        mode=CaptureMode.PHONE_PHOTO,
        require_qr=False,
    )
    client = TestClient(app)

    response = client.post(
        "/api/captures/suite-plain-p0001",
        files={"file": ("camera.jpg", b"not an image", "image/jpeg")},
    )

    assert response.status_code == 422
    assert "invalid capture image" in response.json()["detail"].lower()
    assert not list(output.glob(".upload-*"))
    assert not (output / "suite-plain-p0001.jpg").exists()


def test_collector_filters_pages_by_capture_mode_and_serves_pdf(tmp_path) -> None:
    app = create_collector_app(
        _kit(tmp_path),
        tmp_path / "captures",
        capture_id="flatbed-a",
        mode=CaptureMode.FLATBED_SCAN,
        require_qr=False,
    )
    client = TestClient(app)

    session = client.get("/api/session").json()
    assert session["expected"] == 2
    assert [page["page_id"] for page in session["pages"]] == [
        "suite-plain-p0001",
        "suite-form-p0001",
    ]

    pdf = client.get("/kit.pdf")
    assert pdf.status_code == 200
    assert pdf.content.startswith(b"%PDF-")
    assert pdf.headers["content-type"].startswith("application/pdf")


def test_collector_unknown_page_and_unsupported_extension(tmp_path) -> None:
    app = create_collector_app(
        _kit(tmp_path),
        tmp_path / "captures",
        capture_id="phone-a",
        mode=CaptureMode.PHONE_PHOTO,
        require_qr=False,
    )
    client = TestClient(app)

    unknown = client.post(
        "/api/captures/not-a-page",
        files={"file": ("camera.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert unknown.status_code == 404

    unsupported = client.post(
        "/api/captures/suite-plain-p0001",
        files={"file": ("camera.bmp", b"BM", "image/bmp")},
    )
    assert unsupported.status_code == 422


def test_collector_session_metadata_contains_no_ground_truth(tmp_path) -> None:
    output = tmp_path / "captures"
    create_collector_app(
        _kit(tmp_path),
        output,
        capture_id="phone-a",
        mode=CaptureMode.PHONE_PHOTO,
        require_qr=False,
    )

    metadata = (output / ".collector-session.json").read_text(encoding="utf-8")
    assert "ground_truth" not in metadata
    assert "phone-a" in metadata
    assert "abc123" in metadata


def _qr_png_bytes(page_id: str) -> bytes:
    page = Image.new("RGB", (600, 800), "white")
    qr = render_page_id_qr(page_id, size_px=180)
    page.paste(qr, (390, 20))
    buffer = io.BytesIO()
    page.save(buffer, format="PNG")
    return buffer.getvalue()


def test_collector_requires_matching_qr_by_default(tmp_path) -> None:
    app = create_collector_app(
        _kit(tmp_path),
        tmp_path / "captures-qr",
        capture_id="phone-a",
        mode=CaptureMode.PHONE_PHOTO,
    )
    client = TestClient(app)

    accepted = client.post(
        "/api/captures/suite-plain-p0001",
        files={
            "file": (
                "camera.png",
                _qr_png_bytes("suite-plain-p0001"),
                "image/png",
            )
        },
    )
    assert accepted.status_code == 201


def test_collector_rejects_unreadable_qr_by_default(tmp_path) -> None:
    app = create_collector_app(
        _kit(tmp_path),
        tmp_path / "captures-unreadable",
        capture_id="phone-a",
        mode=CaptureMode.PHONE_PHOTO,
    )
    client = TestClient(app)

    response = client.post(
        "/api/captures/suite-plain-p0001",
        files={"file": ("camera.jpg", _jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 422
    assert "qr marker" in response.json()["detail"].lower()


def test_collector_rejects_qr_for_wrong_page(tmp_path) -> None:
    app = create_collector_app(
        _kit(tmp_path),
        tmp_path / "captures-wrong-qr",
        capture_id="phone-a",
        mode=CaptureMode.PHONE_PHOTO,
    )
    client = TestClient(app)

    response = client.post(
        "/api/captures/suite-plain-p0001",
        files={
            "file": (
                "camera.png",
                _qr_png_bytes("suite-form-p0001"),
                "image/png",
            )
        },
    )

    assert response.status_code == 422
    assert "does not match" in response.json()["detail"].lower()
    assert not list((tmp_path / "captures-wrong-qr").glob("suite-plain-p0001.*"))


def test_collector_can_resume_same_session(tmp_path) -> None:
    output = tmp_path / "resume"
    app = create_collector_app(
        _kit(tmp_path),
        output,
        capture_id="phone-a",
        mode=CaptureMode.PHONE_PHOTO,
        require_qr=False,
    )
    client = TestClient(app)
    saved = client.post(
        "/api/captures/suite-plain-p0001",
        files={"file": ("camera.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert saved.status_code == 201

    resumed = create_collector_app(
        _kit(tmp_path),
        output,
        capture_id="phone-a",
        mode=CaptureMode.PHONE_PHOTO,
        require_qr=False,
    )
    payload = TestClient(resumed).get("/api/session").json()
    assert payload["completed"] == 1


def test_collector_rejects_output_dir_from_different_session(tmp_path) -> None:
    output = tmp_path / "mixed-session"
    create_collector_app(
        _kit(tmp_path),
        output,
        capture_id="phone-a",
        mode=CaptureMode.PHONE_PHOTO,
        require_qr=False,
    )

    try:
        create_collector_app(
            _kit(tmp_path),
            output,
            capture_id="phone-b",
            mode=CaptureMode.PHONE_PHOTO,
            require_qr=False,
        )
    except CaptureCollectorError as exc:
        assert "different collector session" in str(exc).lower()
    else:
        raise AssertionError("mismatched collector session should be rejected")


def test_collector_rejects_existing_capture_dir_without_session_metadata(tmp_path) -> None:
    output = tmp_path / "legacy-dir"
    output.mkdir()
    (output / "suite-plain-p0001.jpg").write_bytes(_jpeg_bytes())

    try:
        create_collector_app(
            _kit(tmp_path),
            output,
            capture_id="phone-a",
            mode=CaptureMode.PHONE_PHOTO,
            require_qr=False,
        )
    except CaptureCollectorError as exc:
        assert "no collector session metadata" in str(exc).lower()
    else:
        raise AssertionError("untracked existing captures should be rejected")


def test_collector_access_token_protects_all_routes(tmp_path) -> None:
    from fastapi.testclient import TestClient

    kit_path = _kit(tmp_path)
    token = "collector-token-1234567890"
    app = create_collector_app(
        kit_path,
        tmp_path / "captures-auth",
        capture_id="phone-auth",
        mode=CaptureMode.PHONE_PHOTO,
        require_qr=False,
        access_token=token,
    )

    client = TestClient(app)
    denied = client.get("/")
    assert denied.status_code == 403
    assert denied.json()["detail"] == "Collector access token required."

    first = client.get(f"/?token={token}")
    assert first.status_code == 200
    cookie = first.headers.get("set-cookie", "")
    assert "lao_ocr_collector" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=strict" in cookie

    # The token query is no longer required after the HttpOnly session cookie is set.
    session = client.get("/api/session")
    assert session.status_code == 200
    pdf = client.get("/kit.pdf")
    assert pdf.status_code == 200


def test_collector_access_token_header_supports_scripted_clients(tmp_path) -> None:
    from fastapi.testclient import TestClient

    kit_path = _kit(tmp_path)
    token = "collector-token-abcdefghijk"
    app = create_collector_app(
        kit_path,
        tmp_path / "captures-header",
        capture_id="phone-header",
        mode=CaptureMode.PHONE_PHOTO,
        require_qr=False,
        access_token=token,
    )
    client = TestClient(app)

    response = client.get(
        "/api/session",
        headers={"X-Collector-Token": token},
    )
    assert response.status_code == 200

    wrong = client.get(
        "/api/session",
        headers={"X-Collector-Token": "collector-token-wrong-1234"},
    )
    assert wrong.status_code == 403


def test_collector_rejects_weak_access_token(tmp_path) -> None:
    kit_path = _kit(tmp_path)
    with pytest.raises(CaptureCollectorError, match="access_token"):
        create_collector_app(
            kit_path,
            tmp_path / "captures-weak-token",
            capture_id="phone-weak",
            mode=CaptureMode.PHONE_PHOTO,
            require_qr=False,
            access_token="short",
        )


def test_collector_exports_verified_capture_submission(tmp_path) -> None:
    from lao_document_ocr.capture_submission import verify_capture_submission

    output = tmp_path / "captures-export"
    app = create_collector_app(
        _kit(tmp_path),
        output,
        capture_id="phone-export",
        mode=CaptureMode.PHONE_PHOTO,
        require_qr=False,
    )
    client = TestClient(app)
    uploaded = client.post(
        "/api/captures/suite-plain-p0001",
        files={"file": ("camera.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert uploaded.status_code == 201

    response = client.get("/api/export")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")

    submission_path = tmp_path / "exported.zip"
    submission_path.write_bytes(response.content)
    submission = verify_capture_submission(submission_path)
    assert submission.suite_id == "suite"
    assert submission.capture_id == "phone-export"
    assert submission.mode == "phone-photo"
    assert submission.captured_pages == 1
