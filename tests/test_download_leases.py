"""Managed expiry must not delete an already admitted download's input bytes.

Filesystem is real; the S3 adapter uses a local protocol double. No OCR/remote data.
"""

from __future__ import annotations

from datetime import timedelta

import anyio
import pytest
from test_storage import FakeS3Client

from services.api.app.jobs import ConversionJobManager, JobNotFoundError
from services.api.app.storage import FilesystemArtifactStorage, S3ArtifactStorage


@pytest.fixture(params=["filesystem", "s3-protocol-double", "legacy-path"])
def context(tmp_path, monkeypatch, request):
    import services.api.app.main as api

    storage = (
        S3ArtifactStorage("fixture-bucket", prefix="private", client=FakeS3Client())
        if request.param == "s3-protocol-double"
        else FilesystemArtifactStorage(tmp_path / "objects")
    )
    content = b"authored archive fixture bytes" * 7

    def runner(record, cancel_event):
        path = record.workspace / "result.zip"
        path.write_bytes(content)
        if request.param == "legacy-path":
            return path
        return storage.put_file(
            path,
            key=f"jobs/{record.id}/result.zip",
            filename="result.zip",
            media_type="application/zip",
        )

    manager = ConversionJobManager(
        tmp_path / "jobs",
        runner,
        retention_seconds=60,
        artifact_exists=storage.exists,
        artifact_cleanup=storage.delete,
    )
    monkeypatch.setattr(api, "JOB_MANAGER", manager)
    monkeypatch.setattr(api, "RESULT_STORAGE", storage)
    record = manager.reserve("private-source.png", ".png")
    record.input_path.write_bytes(b"authored fixture upload")
    manager.enqueue(record.id)
    record.future.result(timeout=5)
    try:
        yield api, manager, record, storage, content
    finally:
        manager.shutdown()


def consume(response, *, spec="2.4", headers=None, send_override=None):
    async def run():
        messages = []

        async def send(message):
            if send_override is not None:
                await send_override(message)
            messages.append(message)

        async def receive():
            await anyio.sleep_forever()

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/fixture",
            "headers": headers or [],
            "asgi": {"spec_version": spec},
        }
        await response(scope, receive, send)
        return messages

    return anyio.run(run)


def test_admitted_download_survives_expiry_before_first_read(context):
    api, manager, record, storage, expected = context
    artifact = record.output_artifact
    response = api.download_conversion_job(record.id)
    expiry = record.completed_at + timedelta(hours=2)
    assert manager.cleanup_expired(now=expiry) == 1
    # Public expiry must not expose old jobs even though their bytes are still leased.
    with pytest.raises(JobNotFoundError):
        manager.public(record.id)
    assert record.workspace.exists()
    if artifact is not None:
        assert storage.exists(artifact)
    messages = consume(response)
    assert b"".join(item.get("body", b"") for item in messages) == expected
    assert manager.cleanup_expired(now=expiry) == 0
    assert not record.workspace.exists()
    if artifact is not None:
        assert not storage.exists(artifact)


@pytest.mark.parametrize("phase", ["headers", "body"])
def test_send_failure_releases_admission_even_before_first_read(context, phase):
    from starlette.requests import ClientDisconnect

    api, manager, record, storage, _ = context
    response = api.download_conversion_job(record.id)
    expiry = record.completed_at + timedelta(hours=2)
    seen = []

    async def broken_send(message):
        target = "http.response.start" if phase == "headers" else "http.response.body"
        if message["type"] == target:
            assert manager.cleanup_expired(now=expiry) == (1 if not seen else 0)
            seen.append(True)
            assert record.workspace.exists()
            raise OSError("PRIVATE disconnected client")

    with pytest.raises((OSError, ClientDisconnect)):
        consume(response, send_override=broken_send)
    assert seen
    assert manager.snapshot()["active_downloads"] == 0
    manager.cleanup_expired(now=expiry)
    assert not record.workspace.exists()
    if record.output_artifact is not None:
        assert not storage.exists(record.output_artifact)


def test_multiple_leases_need_every_response_to_finish(context):
    from concurrent.futures import ThreadPoolExecutor

    _, manager, record, storage, _ = context
    first = manager.acquire_download(record.id)
    second = manager.acquire_download(record.id)
    expiry = record.completed_at + timedelta(hours=2)
    assert manager.cleanup_expired(now=expiry) == 1
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: first.close(), range(20)))
    assert manager.snapshot()["active_downloads"] == 1
    assert manager.cleanup_expired(now=expiry) == 0
    assert record.workspace.exists()
    second.close()
    second.close()
    assert manager.snapshot()["active_downloads"] == 0
    manager.cleanup_expired(now=expiry)
    assert not record.workspace.exists()
    assert manager.snapshot()["completed_total"] == {"succeeded": 1}
    if record.output_artifact is not None:
        assert not storage.exists(record.output_artifact)


def test_download_limits_reject_without_leaking_or_deleting_other_jobs(context):
    from services.api.app.jobs import (
        MAX_DOWNLOADS_PER_JOB,
        MAX_DOWNLOADS_TOTAL,
        JobDownloadCapacityError,
    )

    _, manager, record, _, _ = context
    leases = []
    try:
        for _ in range(MAX_DOWNLOADS_PER_JOB):
            leases.append(manager.acquire_download(record.id))
        with pytest.raises(JobDownloadCapacityError):
            manager.acquire_download(record.id)
        while len(leases) < MAX_DOWNLOADS_TOTAL:
            other = manager.reserve("other.png", ".png")
            manager.enqueue(other.id)
            other.future.result(timeout=5)
            for _ in range(MAX_DOWNLOADS_PER_JOB):
                leases.append(manager.acquire_download(other.id))
        last = manager.reserve("last.png", ".png")
        manager.enqueue(last.id)
        last.future.result(timeout=5)
        with pytest.raises(JobDownloadCapacityError):
            manager.acquire_download(last.id)
        assert manager.snapshot()["active_downloads"] == MAX_DOWNLOADS_TOTAL
        leases.pop().close()
        leases.append(manager.acquire_download(last.id))
        assert last.workspace.exists() and record.workspace.exists()
    finally:
        for lease in leases:
            lease.close()
    assert manager.snapshot()["active_downloads"] == 0


def test_expired_job_cannot_renew_or_bypass_zero_retention(context):
    from fastapi import HTTPException

    api, manager, record, _, _ = context
    manager.retention_seconds = 0
    with pytest.raises(HTTPException) as rejected:
        api.download_conversion_job(record.id)
    assert rejected.value.status_code == 404
    assert manager.snapshot()["active_downloads"] == 0


def test_admission_rechecks_clock_after_slow_cleanup(context, monkeypatch):
    from datetime import UTC, datetime

    _, manager, record, _, _ = context

    def slow_cleanup():
        # An unrelated callback may consume the rest of this job's retention.
        record.completed_at = datetime.now(UTC) - timedelta(seconds=61)

    monkeypatch.setattr(manager, "cleanup_expired", slow_cleanup)
    with pytest.raises(JobNotFoundError):
        manager.acquire_download(record.id)
    assert not manager._download_leases


def test_pre_response_missing_result_releases_admission(context):
    from fastapi import HTTPException

    api, manager, record, storage, _ = context
    if record.output_artifact is not None:
        storage.delete(record.output_artifact)
    else:
        record.output_path.unlink()
    with pytest.raises(HTTPException) as rejected:
        api.download_conversion_job(record.id)
    assert rejected.value.status_code == 410
    assert manager.snapshot()["active_downloads"] == 0


@pytest.mark.parametrize("status", ["uploading", "queued", "running", "failed", "cancelled"])
def test_unfinished_or_failed_jobs_never_acquire_download_leases(context, status):
    from fastapi import HTTPException

    from services.api.app.jobs import JobStatus

    api, manager, record, _, _ = context
    record.status = JobStatus(status)
    with pytest.raises(HTTPException) as rejected:
        api.download_conversion_job(record.id)
    assert rejected.value.status_code == 409
    assert not manager._download_leases


def test_expired_leased_job_remains_404_and_counts_against_retained_capacity(context):
    from fastapi.testclient import TestClient

    from services.api.app.jobs import JobCapacityError
    from services.api.app.metrics import ApiMetrics

    api, manager, record, _, _ = context
    manager.max_retained_jobs = 1
    lease = manager.acquire_download(record.id)
    try:
        expiry = record.completed_at + timedelta(hours=2)
        assert manager.cleanup_expired(now=expiry) == 1
        client = TestClient(api.app)
        assert client.get(f"/v1/jobs/{record.id}").status_code == 404
        assert client.get(f"/v1/jobs/{record.id}/download").status_code == 404
        assert client.delete(f"/v1/jobs/{record.id}").status_code == 404
        with pytest.raises(JobCapacityError):
            manager.reserve("new.png", ".png")
        stats = manager.snapshot()
        assert stats["active_downloads"] == 1
        assert stats["cleanup_download_blocked_jobs"] == 1
        assert stats["cleanup_attempts_total"] == 0
        rendered = ApiMetrics().render_prometheus(stats)
        assert "lao_ocr_downloads_active 1\n" in rendered
        assert "lao_ocr_cleanup_download_blocked_jobs 1\n" in rendered
        for private in (record.id, "private-source", str(record.workspace)):
            assert private not in rendered
    finally:
        lease.close()
    manager.cleanup_expired(now=expiry)
    assert manager.reserve("new.png", ".png").workspace.is_dir()


def test_download_capacity_http_error_leaves_leases_unchanged(context):
    from fastapi.testclient import TestClient

    from services.api.app.jobs import MAX_DOWNLOADS_PER_JOB

    api, manager, record, _, _ = context
    leases = [manager.acquire_download(record.id) for _ in range(MAX_DOWNLOADS_PER_JOB)]
    try:
        result = TestClient(api.app).get(f"/v1/jobs/{record.id}/download")
        assert result.status_code == 429
        assert result.json()["detail"] == "Too many active downloads. Try again later."
        assert len(leases) == manager.snapshot()["active_downloads"]
    finally:
        for lease in leases:
            lease.close()


def test_discard_cannot_remove_an_admitted_download(context):
    _, manager, record, _, _ = context
    lease = manager.acquire_download(record.id)
    try:
        with pytest.raises(RuntimeError, match="admitted download"):
            manager.discard(record.id)
        assert record.workspace.exists()
    finally:
        lease.close()


def test_actual_http_response_releases_lease_and_preserves_headers(context):
    from fastapi.testclient import TestClient

    api, manager, record, _, expected = context
    response = TestClient(api.app).get(f"/v1/jobs/{record.id}/download")
    assert response.status_code == 200
    assert response.content == expected
    assert response.headers["content-type"].startswith("application/zip")
    assert response.headers["cache-control"] == "no-store"
    assert "result.zip" in response.headers["content-disposition"]
    assert not manager._download_leases
    assert record.workspace.exists()  # Lease release is not a deletion request.


@pytest.mark.parametrize("failure_at", ["exists", "iterator"])
def test_pre_response_storage_failures_release_admission(context, monkeypatch, failure_at):
    api, manager, record, storage, _ = context
    if record.output_artifact is None:
        # Exercise the stored-result route using an owned copy of the legacy file.
        record.output_artifact = storage.put_file(
            record.output_path,
            key=f"jobs/{record.id}/copy.zip",
            filename="copy.zip",
            media_type="application/zip",
        )
        record.output_path = None

    def fail(*args, **kwargs):
        raise OSError("PRIVATE provider detail")

    monkeypatch.setattr(storage, "exists" if failure_at == "exists" else "iter_bytes", fail)
    with pytest.raises(OSError, match="provider detail"):
        api.download_conversion_job(record.id)
    assert not manager._download_leases


def test_admission_limit_is_atomic_across_threads(context):
    from concurrent.futures import ThreadPoolExecutor

    from services.api.app.jobs import MAX_DOWNLOADS_PER_JOB, JobDownloadCapacityError

    _, manager, record, _, _ = context

    def acquire(_):
        try:
            return manager.acquire_download(record.id)
        except JobDownloadCapacityError:
            return None

    with ThreadPoolExecutor(max_workers=12) as pool:
        accepted = [lease for lease in pool.map(acquire, range(40)) if lease is not None]
    try:
        assert len(accepted) == MAX_DOWNLOADS_PER_JOB
        assert manager.snapshot()["active_downloads"] == len(accepted)
    finally:
        for lease in accepted:
            lease.close()


def test_lease_suppression_does_not_spend_cleanup_budget_or_starve_other_jobs(context):
    _, manager, record, _, _ = context
    lease = manager.acquire_download(record.id)
    others = []
    for _ in range(8):
        other = manager.reserve("other.png", ".png")
        manager.enqueue(other.id)
        other.future.result(timeout=5)
        others.append(other)
    expiry = others[-1].completed_at + timedelta(hours=2)
    try:
        assert manager.cleanup_expired(now=expiry) == 9
        assert record.workspace.exists()
        assert all(not item.workspace.exists() for item in others)
        assert manager.snapshot()["cleanup_attempts_total"] == 8
        assert manager.snapshot()["cleanup_failures_total"] == 0
    finally:
        lease.close()
    manager.cleanup_expired(now=expiry)
    assert not record.workspace.exists()



def test_download_metrics_reject_non_integer_or_negative_values():
    from services.api.app.metrics import ApiMetrics

    for key, metric in [
        ("active_downloads", "lao_ocr_downloads_active"),
        ("cleanup_download_blocked_jobs", "lao_ocr_cleanup_download_blocked_jobs"),
    ]:
        for value in (True, -1, None, "PRIVATE", {}, float("nan")):
            rendered = ApiMetrics().render_prometheus({key: value})
            assert metric not in rendered
            assert "PRIVATE" not in rendered
