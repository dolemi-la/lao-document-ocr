import pytest
from fastapi.testclient import TestClient

from services.api.app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["engine"] == "tesseract"
    assert payload["psm"] == 3


def test_rejects_unsupported_upload() -> None:
    response = client.post(
        "/v1/parse",
        files={"file": ("document.exe", b"not a document", "application/octet-stream")},
    )
    assert response.status_code == 415


def test_owned_engine_health_requires_model(monkeypatch) -> None:
    import services.api.app.main as api_main

    monkeypatch.setattr(api_main, "OCR_ENGINE", "owned")
    monkeypatch.setattr(api_main, "OCR_MODEL_PATH", None)

    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["engine"] == "owned"
    assert payload["ocr_ready"] is False
    assert "OCR_MODEL_PATH" in payload["error"]


def _png_upload_bytes() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (100, 60), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _poll_job(client, job_id: str, terminal=None, timeout: float = 2.0):
    import time

    terminal = terminal or {"succeeded", "failed", "cancelled"}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in terminal:
            return payload
        time.sleep(0.01)
    raise AssertionError("job did not reach expected state")


def test_async_job_success_status_and_download(tmp_path, monkeypatch) -> None:
    import zipfile

    import services.api.app.main as api_main
    from services.api.app.jobs import ConversionJobManager

    def runner(record, cancel_event):
        path = record.workspace / "sample-ocr.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("sample.txt", "done")
        return path

    manager = ConversionJobManager(
        tmp_path / "jobs",
        runner,
        max_workers=1,
        max_active_jobs=2,
    )
    monkeypatch.setattr(api_main, "JOB_MANAGER", manager)
    try:
        response = client.post(
            "/v1/jobs",
            files={"file": ("sample.png", _png_upload_bytes(), "image/png")},
        )
        assert response.status_code == 202
        job_id = response.json()["id"]

        payload = _poll_job(client, job_id)
        assert payload["status"] == "succeeded"
        assert payload["download_ready"] is True

        download = client.get(f"/v1/jobs/{job_id}/download")
        assert download.status_code == 200
        assert download.headers["content-type"].startswith("application/zip")
        with zipfile.ZipFile(__import__("io").BytesIO(download.content)) as archive:
            assert archive.read("sample.txt") == b"done"
    finally:
        manager.shutdown()


def test_async_job_can_be_cancelled(tmp_path, monkeypatch) -> None:
    import time

    import services.api.app.main as api_main
    from services.api.app.jobs import ConversionJobManager, JobCancelledError

    def runner(record, cancel_event):
        while True:
            if cancel_event.is_set():
                raise JobCancelledError()
            time.sleep(0.01)

    manager = ConversionJobManager(
        tmp_path / "jobs",
        runner,
        max_workers=1,
        max_active_jobs=2,
    )
    monkeypatch.setattr(api_main, "JOB_MANAGER", manager)
    try:
        response = client.post(
            "/v1/jobs",
            files={"file": ("sample.png", _png_upload_bytes(), "image/png")},
        )
        assert response.status_code == 202
        job_id = response.json()["id"]

        _poll_job(client, job_id, terminal={"running"})
        cancelled = client.delete(f"/v1/jobs/{job_id}")
        assert cancelled.status_code == 200
        assert cancelled.json()["cancellation_requested"] is True

        payload = _poll_job(client, job_id)
        assert payload["status"] == "cancelled"
        download = client.get(f"/v1/jobs/{job_id}/download")
        assert download.status_code == 409
    finally:
        manager.shutdown()


def test_async_job_capacity_returns_429(tmp_path, monkeypatch) -> None:
    import time

    import services.api.app.main as api_main
    from services.api.app.jobs import ConversionJobManager, JobCancelledError

    def runner(record, cancel_event):
        while True:
            if cancel_event.is_set():
                raise JobCancelledError()
            time.sleep(0.01)

    manager = ConversionJobManager(
        tmp_path / "jobs",
        runner,
        max_workers=1,
        max_active_jobs=1,
    )
    monkeypatch.setattr(api_main, "JOB_MANAGER", manager)
    try:
        first = client.post(
            "/v1/jobs",
            files={"file": ("first.png", _png_upload_bytes(), "image/png")},
        )
        assert first.status_code == 202
        job_id = first.json()["id"]

        second = client.post(
            "/v1/jobs",
            files={"file": ("second.png", _png_upload_bytes(), "image/png")},
        )
        assert second.status_code == 429
        assert "capacity" in second.json()["detail"].lower()

        client.delete(f"/v1/jobs/{job_id}")
        _poll_job(client, job_id)
    finally:
        manager.shutdown()


def test_unknown_job_returns_404() -> None:
    response = client.get("/v1/jobs/not-a-real-job")
    assert response.status_code == 404


def test_request_id_is_echoed() -> None:
    response = client.get("/health", headers={"X-Request-ID": "req-123"})
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-123"


def test_invalid_request_id_is_replaced() -> None:
    response = client.get("/health", headers={"X-Request-ID": "bad request id"})
    assert response.status_code == 200
    generated = response.headers["x-request-id"]
    assert generated != "bad request id"
    assert len(generated) == 32


def test_metrics_endpoint_normalizes_job_ids(monkeypatch) -> None:
    import services.api.app.main as api_main
    from services.api.app.metrics import ApiMetrics

    monkeypatch.setattr(api_main, "API_METRICS", ApiMetrics())

    response = client.get("/v1/jobs/not-a-real-job")
    assert response.status_code == 404

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.headers["content-type"].startswith("text/plain")
    text = metrics.text
    assert 'route="/v1/jobs/{job_id}"' in text
    assert "not-a-real-job" not in text
    assert 'status="404"' in text


def test_submission_rate_limit_returns_429_and_polling_still_works(
    tmp_path,
    monkeypatch,
) -> None:
    import zipfile

    import services.api.app.main as api_main
    from services.api.app.jobs import ConversionJobManager
    from services.api.app.rate_limit import SlidingWindowRateLimiter

    def runner(record, cancel_event):
        path = record.workspace / "result.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("result.txt", "done")
        return path

    manager = ConversionJobManager(
        tmp_path / "jobs-rate-limit",
        runner,
        max_workers=1,
        max_active_jobs=4,
    )
    limiter = SlidingWindowRateLimiter(requests=1, window_seconds=60)
    monkeypatch.setattr(api_main, "JOB_MANAGER", manager)
    monkeypatch.setattr(api_main, "SUBMISSION_RATE_LIMITER", limiter)
    try:
        first = client.post(
            "/v1/jobs",
            files={"file": ("first.png", _png_upload_bytes(), "image/png")},
        )
        assert first.status_code == 202
        job_id = first.json()["id"]

        second = client.post(
            "/v1/jobs",
            files={"file": ("second.png", _png_upload_bytes(), "image/png")},
        )
        assert second.status_code == 429
        assert second.headers["retry-after"]
        assert "rate limit" in second.json()["detail"].lower()

        status = client.get(f"/v1/jobs/{job_id}")
        assert status.status_code == 200
    finally:
        manager.shutdown()


def test_forwarded_for_is_ignored_unless_explicitly_trusted(
    tmp_path,
    monkeypatch,
) -> None:
    import zipfile

    import services.api.app.main as api_main
    from services.api.app.jobs import ConversionJobManager
    from services.api.app.rate_limit import SlidingWindowRateLimiter

    def runner(record, cancel_event):
        path = record.workspace / "result.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("result.txt", "done")
        return path

    manager = ConversionJobManager(
        tmp_path / "jobs-forwarded",
        runner,
        max_workers=1,
        max_active_jobs=4,
    )
    monkeypatch.setattr(api_main, "JOB_MANAGER", manager)
    monkeypatch.setattr(api_main, "RATE_LIMIT_TRUST_PROXY_HEADERS", False)
    monkeypatch.setattr(
        api_main,
        "SUBMISSION_RATE_LIMITER",
        SlidingWindowRateLimiter(requests=1, window_seconds=60),
    )
    try:
        first = client.post(
            "/v1/jobs",
            headers={"X-Forwarded-For": "203.0.113.10"},
            files={"file": ("first.png", _png_upload_bytes(), "image/png")},
        )
        second = client.post(
            "/v1/jobs",
            headers={"X-Forwarded-For": "203.0.113.11"},
            files={"file": ("second.png", _png_upload_bytes(), "image/png")},
        )
        assert first.status_code == 202
        assert second.status_code == 429
    finally:
        manager.shutdown()


def test_forwarded_for_can_be_trusted_by_explicit_configuration(
    tmp_path,
    monkeypatch,
) -> None:
    import zipfile

    import services.api.app.main as api_main
    from services.api.app.jobs import ConversionJobManager
    from services.api.app.rate_limit import SlidingWindowRateLimiter

    def runner(record, cancel_event):
        path = record.workspace / "result.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("result.txt", "done")
        return path

    manager = ConversionJobManager(
        tmp_path / "jobs-forwarded-trusted",
        runner,
        max_workers=1,
        max_active_jobs=4,
    )
    monkeypatch.setattr(api_main, "JOB_MANAGER", manager)
    monkeypatch.setattr(api_main, "RATE_LIMIT_TRUST_PROXY_HEADERS", True)
    monkeypatch.setattr(
        api_main,
        "SUBMISSION_RATE_LIMITER",
        SlidingWindowRateLimiter(requests=1, window_seconds=60),
    )
    try:
        first = client.post(
            "/v1/jobs",
            headers={"X-Forwarded-For": "203.0.113.10"},
            files={"file": ("first.png", _png_upload_bytes(), "image/png")},
        )
        second = client.post(
            "/v1/jobs",
            headers={"X-Forwarded-For": "203.0.113.11"},
            files={"file": ("second.png", _png_upload_bytes(), "image/png")},
        )
        assert first.status_code == 202
        assert second.status_code == 202
    finally:
        manager.shutdown()


def test_async_batch_submission_creates_multiple_jobs(tmp_path, monkeypatch) -> None:
    import zipfile

    import services.api.app.main as api_main
    from services.api.app.jobs import ConversionJobManager
    from services.api.app.rate_limit import SlidingWindowRateLimiter

    def runner(record, cancel_event):
        path = record.workspace / f"{record.id}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("result.txt", record.filename)
        return path

    manager = ConversionJobManager(
        tmp_path / "jobs-batch-api",
        runner,
        max_workers=2,
        max_active_jobs=4,
    )
    monkeypatch.setattr(api_main, "JOB_MANAGER", manager)
    monkeypatch.setattr(
        api_main,
        "SUBMISSION_RATE_LIMITER",
        SlidingWindowRateLimiter(requests=0, window_seconds=60),
    )
    try:
        response = client.post(
            "/v1/jobs/batch",
            files=[
                ("files", ("a.png", _png_upload_bytes(), "image/png")),
                ("files", ("b.png", _png_upload_bytes(), "image/png")),
            ],
        )
        assert response.status_code == 202
        payload = response.json()
        assert payload["count"] == 2
        assert [job["filename"] for job in payload["jobs"]] == ["a.png", "b.png"]
        assert len({job["id"] for job in payload["jobs"]}) == 2

        terminals = [_poll_job(client, job["id"]) for job in payload["jobs"]]
        assert [item["status"] for item in terminals] == ["succeeded", "succeeded"]
    finally:
        manager.shutdown()


def test_async_batch_capacity_rejection_is_atomic(tmp_path, monkeypatch) -> None:
    import services.api.app.main as api_main
    from services.api.app.jobs import ConversionJobManager
    from services.api.app.rate_limit import SlidingWindowRateLimiter

    def runner(record, cancel_event):
        raise AssertionError("runner should not execute")

    root = tmp_path / "jobs-batch-capacity-api"
    manager = ConversionJobManager(
        root,
        runner,
        max_workers=1,
        max_active_jobs=1,
    )
    monkeypatch.setattr(api_main, "JOB_MANAGER", manager)
    monkeypatch.setattr(
        api_main,
        "SUBMISSION_RATE_LIMITER",
        SlidingWindowRateLimiter(requests=0, window_seconds=60),
    )
    try:
        response = client.post(
            "/v1/jobs/batch",
            files=[
                ("files", ("a.png", _png_upload_bytes(), "image/png")),
                ("files", ("b.png", _png_upload_bytes(), "image/png")),
            ],
        )
        assert response.status_code == 429
        assert manager.snapshot()["active_jobs"] == 0
        assert list(root.iterdir()) == []
    finally:
        manager.shutdown()


def test_async_batch_respects_batch_file_cap(tmp_path, monkeypatch) -> None:
    import services.api.app.main as api_main

    monkeypatch.setattr(api_main, "BATCH_MAX_FILES", 1)

    response = client.post(
        "/v1/jobs/batch",
        files=[
            ("files", ("a.png", _png_upload_bytes(), "image/png")),
            ("files", ("b.png", _png_upload_bytes(), "image/png")),
        ],
    )

    assert response.status_code == 413
    assert "1 file limit" in response.json()["detail"]


def test_async_batch_rate_limit_cost_is_per_document(tmp_path, monkeypatch) -> None:
    import zipfile

    import services.api.app.main as api_main
    from services.api.app.jobs import ConversionJobManager
    from services.api.app.rate_limit import SlidingWindowRateLimiter

    def runner(record, cancel_event):
        path = record.workspace / "result.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("result.txt", "done")
        return path

    manager = ConversionJobManager(
        tmp_path / "jobs-batch-rate-api",
        runner,
        max_workers=2,
        max_active_jobs=4,
    )
    monkeypatch.setattr(api_main, "JOB_MANAGER", manager)
    monkeypatch.setattr(
        api_main,
        "SUBMISSION_RATE_LIMITER",
        SlidingWindowRateLimiter(requests=2, window_seconds=60),
    )
    try:
        batch = client.post(
            "/v1/jobs/batch",
            files=[
                ("files", ("a.png", _png_upload_bytes(), "image/png")),
                ("files", ("b.png", _png_upload_bytes(), "image/png")),
            ],
        )
        assert batch.status_code == 202

        extra = client.post(
            "/v1/jobs",
            files={"file": ("c.png", _png_upload_bytes(), "image/png")},
        )
        assert extra.status_code == 429
    finally:
        manager.shutdown()


def test_async_job_downloads_through_storage_adapter(tmp_path, monkeypatch) -> None:
    import zipfile

    import services.api.app.main as api_main
    from services.api.app.jobs import ConversionJobManager
    from services.api.app.rate_limit import SlidingWindowRateLimiter
    from services.api.app.storage import FilesystemArtifactStorage

    storage = FilesystemArtifactStorage(tmp_path / "results-api")

    def runner(record, cancel_event):
        local = record.workspace / "stored-result.zip"
        with zipfile.ZipFile(local, "w") as archive:
            archive.writestr("result.txt", "stored")
        return storage.put_file(
            local,
            key=f"jobs/{record.id}/stored-result.zip",
            filename="stored-result.zip",
            media_type="application/zip",
        )

    manager = ConversionJobManager(
        tmp_path / "jobs-storage-api",
        runner,
        max_workers=1,
        max_active_jobs=2,
        artifact_exists=storage.exists,
        artifact_cleanup=storage.delete,
    )
    monkeypatch.setattr(api_main, "RESULT_STORAGE", storage)
    monkeypatch.setattr(api_main, "JOB_MANAGER", manager)
    monkeypatch.setattr(
        api_main,
        "SUBMISSION_RATE_LIMITER",
        SlidingWindowRateLimiter(requests=0, window_seconds=60),
    )
    try:
        response = client.post(
            "/v1/jobs",
            files={"file": ("sample.png", _png_upload_bytes(), "image/png")},
        )
        assert response.status_code == 202
        job_id = response.json()["id"]
        payload = _poll_job(client, job_id)
        assert payload["status"] == "succeeded"
        assert payload["download_ready"] is True

        record = manager.get_record(job_id)
        artifact = record.output_artifact
        assert artifact is not None
        assert storage.exists(artifact) is True

        download = client.get(f"/v1/jobs/{job_id}/download")
        assert download.status_code == 200
        assert download.headers["content-type"].startswith("application/zip")
        assert "stored-result.zip" in download.headers["content-disposition"]
        with zipfile.ZipFile(__import__("io").BytesIO(download.content)) as archive:
            assert archive.read("result.txt") == b"stored"

        storage.delete(artifact)
        missing = client.get(f"/v1/jobs/{job_id}/download")
        assert missing.status_code == 410
    finally:
        manager.shutdown()


def test_api_security_headers_are_present() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["permissions-policy"] == "camera=(), microphone=(), geolocation=()"
    assert "default-src 'none'" in response.headers["content-security-policy"]
    assert response.headers["cache-control"] == "no-store"


def test_upload_rejects_extension_content_mismatch(tmp_path, monkeypatch) -> None:
    import services.api.app.main as api_main
    from services.api.app.jobs import ConversionJobManager
    from services.api.app.rate_limit import SlidingWindowRateLimiter

    manager = ConversionJobManager(
        tmp_path / "jobs-mismatch",
        lambda record, cancel_event: record.workspace / "unused.zip",
        max_workers=1,
        max_active_jobs=2,
    )
    monkeypatch.setattr(api_main, "JOB_MANAGER", manager)
    monkeypatch.setattr(
        api_main,
        "SUBMISSION_RATE_LIMITER",
        SlidingWindowRateLimiter(requests=0, window_seconds=60),
    )
    try:
        response = client.post(
            "/v1/jobs",
            files={"file": ("forged.jpg", _png_upload_bytes(), "image/jpeg")},
        )
        assert response.status_code == 422
        assert "does not match" in response.json()["detail"]
        assert manager.snapshot()["active_jobs"] == 0
    finally:
        manager.shutdown()


def test_sync_convert_rejects_fake_pdf_before_ocr() -> None:
    response = client.post(
        "/v1/convert",
        files={"file": ("fake.pdf", b"not a pdf", "application/pdf")},
    )
    assert response.status_code == 422
    assert "does not match" in response.json()["detail"]


def test_api_enforces_max_page_pixels(tmp_path, monkeypatch) -> None:
    import io

    from PIL import Image

    import services.api.app.main as api_main
    from services.api.app.jobs import ConversionJobManager
    from services.api.app.rate_limit import SlidingWindowRateLimiter

    buffer = io.BytesIO()
    Image.new("RGB", (100, 100), "white").save(buffer, format="PNG")
    manager = ConversionJobManager(
        tmp_path / "jobs-pixel-cap",
        lambda record, cancel_event: record.workspace / "unused.zip",
        max_workers=1,
        max_active_jobs=2,
    )
    monkeypatch.setattr(api_main, "JOB_MANAGER", manager)
    monkeypatch.setattr(api_main, "MAX_PAGE_PIXELS", 5_000)
    monkeypatch.setattr(
        api_main,
        "SUBMISSION_RATE_LIMITER",
        SlidingWindowRateLimiter(requests=0, window_seconds=60),
    )
    try:
        response = client.post(
            "/v1/jobs",
            files={"file": ("large.png", buffer.getvalue(), "image/png")},
        )
        assert response.status_code == 422
        assert "pixel limit" in response.json()["detail"]
        assert manager.snapshot()["active_jobs"] == 0
    finally:
        manager.shutdown()


def test_async_upload_filename_is_sanitized_and_private(tmp_path, monkeypatch) -> None:
    import zipfile

    import services.api.app.main as api_main
    from services.api.app.jobs import ConversionJobManager
    from services.api.app.rate_limit import SlidingWindowRateLimiter

    observed = {}

    def runner(record, cancel_event):
        observed["filename"] = record.filename
        observed["mode"] = record.input_path.stat().st_mode & 0o777
        output = record.workspace / "result.zip"
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("result.txt", "done")
        return output

    manager = ConversionJobManager(
        tmp_path / "jobs-private-upload",
        runner,
        max_workers=1,
        max_active_jobs=2,
    )
    monkeypatch.setattr(api_main, "JOB_MANAGER", manager)
    monkeypatch.setattr(
        api_main,
        "SUBMISSION_RATE_LIMITER",
        SlidingWindowRateLimiter(requests=0, window_seconds=60),
    )
    try:
        response = client.post(
            "/v1/jobs",
            files={
                "file": (
                    '../../ສະບາຍດີ?".png',
                    _png_upload_bytes(),
                    "image/png",
                )
            },
        )
        assert response.status_code == 202
        job_id = response.json()["id"]
        terminal = _poll_job(client, job_id)
        assert terminal["status"] == "succeeded"
        sanitized = observed["filename"]
        assert sanitized.startswith("ສະບາຍດີ")
        assert sanitized.endswith(".png")
        assert "/" not in sanitized and "\\" not in sanitized
        assert "?" not in sanitized and '"' not in sanitized
        assert ".." not in sanitized
        assert observed["mode"] == 0o600
    finally:
        manager.shutdown()


def test_download_headers_use_ascii_fallback_and_utf8_filename() -> None:
    import services.api.app.main as api_main

    headers = api_main._download_headers("ສະບາຍດີ.zip")
    disposition = headers["Content-Disposition"]
    assert 'filename="' in disposition
    assert "ສະບາຍດີ" not in disposition.split("filename*=", 1)[0]
    assert "filename*=UTF-8''" in disposition
    assert "%E0%BA" in disposition


def test_docs_csp_keeps_swagger_ui_assets_allowed() -> None:
    response = client.get("/docs")
    assert response.status_code == 200
    csp = response.headers["content-security-policy"]
    assert "https://cdn.jsdelivr.net" in csp
    assert "script-src 'unsafe-inline'" in csp
    assert "connect-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp


def test_build_result_storage_supports_s3(monkeypatch) -> None:
    import services.api.app.main as api_main

    captured = {}

    class FakeS3Storage:
        def __init__(
            self,
            bucket,
            *,
            prefix,
            endpoint_url,
            region_name,
            force_path_style,
        ):
            captured.update(
                {
                    "bucket": bucket,
                    "prefix": prefix,
                    "endpoint_url": endpoint_url,
                    "region_name": region_name,
                    "force_path_style": force_path_style,
                }
            )

    monkeypatch.setattr(api_main, "RESULT_STORAGE_BACKEND", "s3")
    monkeypatch.setattr(api_main, "RESULT_STORAGE_S3_BUCKET", "ocr-results")
    monkeypatch.setattr(api_main, "RESULT_STORAGE_S3_PREFIX", "prod/results")
    monkeypatch.setattr(
        api_main,
        "RESULT_STORAGE_S3_ENDPOINT_URL",
        "https://r2.example.invalid",
    )
    monkeypatch.setattr(api_main, "RESULT_STORAGE_S3_REGION", "auto")
    monkeypatch.setattr(api_main, "RESULT_STORAGE_S3_FORCE_PATH_STYLE", True)
    monkeypatch.setattr(api_main, "S3ArtifactStorage", FakeS3Storage)

    storage = api_main._build_result_storage()

    assert isinstance(storage, FakeS3Storage)
    assert captured == {
        "bucket": "ocr-results",
        "prefix": "prod/results",
        "endpoint_url": "https://r2.example.invalid",
        "region_name": "auto",
        "force_path_style": True,
    }


def test_build_result_storage_requires_s3_bucket(monkeypatch) -> None:
    import services.api.app.main as api_main

    monkeypatch.setattr(api_main, "RESULT_STORAGE_BACKEND", "s3")
    monkeypatch.setattr(api_main, "RESULT_STORAGE_S3_BUCKET", "")

    with pytest.raises(RuntimeError, match="RESULT_STORAGE_S3_BUCKET"):
        api_main._build_result_storage()


def test_owned_engine_is_cached_and_receives_device(monkeypatch) -> None:
    import services.api.app.main as api_main

    calls = []

    class FakeOwnedEngine:
        def __init__(
            self,
            model_path,
            *,
            calibration_path,
            device,
            decoder,
            beam_width,
            language_model_path,
            language_model_weight,
            language_model_token_bonus,
            region_detector_name,
            layout_model_path,
            layout_confidence_threshold,
        ):
            calls.append(
                {
                    "model_path": model_path,
                    "calibration_path": calibration_path,
                    "device": device,
                    "decoder": decoder,
                    "beam_width": beam_width,
                    "language_model_path": language_model_path,
                    "language_model_weight": language_model_weight,
                    "language_model_token_bonus": language_model_token_bonus,
                    "region_detector_name": region_detector_name,
                    "layout_model_path": layout_model_path,
                    "layout_confidence_threshold": layout_confidence_threshold,
                }
            )

    api_main._cached_owned_engine.cache_clear()
    monkeypatch.setattr(api_main, "OCR_ENGINE", "owned")
    monkeypatch.setattr(api_main, "OCR_MODEL_PATH", "/models/recognizer.pt2")
    monkeypatch.setattr(api_main, "OCR_CALIBRATION_PATH", "/models/calibration.json")
    monkeypatch.setattr(api_main, "OCR_DEVICE", "cuda")
    monkeypatch.setattr(api_main, "OCR_DECODER", "beam")
    monkeypatch.setattr(api_main, "OCR_BEAM_WIDTH", 7)
    monkeypatch.setattr(api_main, "OCR_LANGUAGE_MODEL_PATH", "/models/char-lm.json")
    monkeypatch.setattr(api_main, "OCR_LANGUAGE_MODEL_WEIGHT", 0.35)
    monkeypatch.setattr(api_main, "OCR_LANGUAGE_MODEL_TOKEN_BONUS", 0.08)
    monkeypatch.setattr(api_main, "OCR_LAYOUT_DETECTOR", "learned")
    monkeypatch.setattr(api_main, "OCR_LAYOUT_MODEL_PATH", "/models/layout.pt2")
    monkeypatch.setattr(api_main, "OCR_LAYOUT_CONFIDENCE", 0.61)
    monkeypatch.setattr(api_main, "OwnedRecognizerEngine", FakeOwnedEngine)
    try:
        first = api_main._engine()
        second = api_main._engine()
        assert first is second
        assert calls == [
            {
                "model_path": "/models/recognizer.pt2",
                "calibration_path": "/models/calibration.json",
                "device": "cuda",
                "decoder": "beam",
                "beam_width": 7,
                "language_model_path": "/models/char-lm.json",
                "language_model_weight": 0.35,
                "language_model_token_bonus": 0.08,
                "region_detector_name": "learned",
                "layout_model_path": "/models/layout.pt2",
                "layout_confidence_threshold": 0.61,
            }
        ]
    finally:
        api_main._cached_owned_engine.cache_clear()


def test_health_reports_deterministic_reading_order(monkeypatch) -> None:
    import services.api.app.main as api_main

    monkeypatch.setattr(api_main, "OCR_READING_ORDER", "deterministic")
    monkeypatch.setattr(api_main, "OCR_READING_ORDER_MODEL_PATH", None)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["reading_order"]["name"] == "DeterministicReadingOrderResolver"
    assert payload["reading_order"]["version"] == "multi-column-v2"


def test_health_fails_early_when_learned_reading_order_model_is_missing(
    monkeypatch,
) -> None:
    import services.api.app.main as api_main

    monkeypatch.setattr(api_main, "OCR_READING_ORDER", "learned")
    monkeypatch.setattr(api_main, "OCR_READING_ORDER_MODEL_PATH", None)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ocr_ready"] is False
    assert payload["reading_order"] is None
    assert "OCR_READING_ORDER_MODEL_PATH" in payload["error"]


def test_tesseract_engine_uses_configured_tessdata_dir(tmp_path, monkeypatch) -> None:
    import services.api.app.main as api_main
    from lao_document_ocr.ocr.tesseract import TesseractEngine

    (tmp_path / "lao.traineddata").write_bytes(b"lao")
    (tmp_path / "eng.traineddata").write_bytes(b"eng")
    monkeypatch.setattr(api_main, "OCR_ENGINE", "tesseract")
    monkeypatch.setattr(api_main, "OCR_TESSDATA_DIR", str(tmp_path))

    engine = api_main._engine()

    assert isinstance(engine, TesseractEngine)
    assert engine.tessdata_dir == tmp_path.resolve()
    assert set(engine.metadata()["tessdata"]["traineddata_sha256"]) == {
        "lao",
        "eng",
    }
