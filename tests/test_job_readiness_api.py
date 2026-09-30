"""Real HTTP/job routes remain usable while one result-existence call waits.

The archive runner and PNG are authored fixtures, not an OCR accuracy test.
Filesystem storage is real; S3 uses the real adapter with a local client double.
"""

from __future__ import annotations

import io
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_storage import FakeS3Client

import services.api.app.main as api
from services.api.app.jobs import ConversionJobManager
from services.api.app.storage import FilesystemArtifactStorage, S3ArtifactStorage


@pytest.fixture(params=["filesystem", "s3-protocol-double"])
def storage(tmp_path, request):
    if request.param == "filesystem":
        return FilesystemArtifactStorage(tmp_path / "objects")
    return S3ArtifactStorage("fixture-bucket", prefix="fixture", client=FakeS3Client())


def png():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), "white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize("method", ["GET", "DELETE"])
@pytest.mark.parametrize("expire", [False, True])
def test_status_or_terminal_cancel_does_not_block_other_http_work(
    tmp_path,
    monkeypatch,
    storage,
    method,
    expire,
):
    entered, release = threading.Event(), threading.Event()
    target = []
    probes = []

    def runner(record, _):
        archive = record.workspace / "result.zip"
        archive.write_bytes(b"authored fixture archive")
        return storage.put_file(
            archive,
            key=f"jobs/{record.id}/result.zip",
            filename="result.zip",
            media_type="application/zip",
        )

    def exists(artifact):
        available = storage.exists(artifact)
        if target and artifact == target[0]:
            probes.append(artifact)
            entered.set()
            assert release.wait(5), "test did not release the provider"
        return available

    manager = ConversionJobManager(
        tmp_path / "jobs",
        runner,
        artifact_exists=exists,
        artifact_cleanup=storage.delete,
        cleanup_interval_seconds=0,
    )
    monkeypatch.setattr(api, "JOB_MANAGER", manager)
    monkeypatch.setattr(api, "RESULT_STORAGE", storage)
    monkeypatch.setattr(api, "_enforce_submission_rate_limit", lambda *args, **kwargs: None)
    pool = ThreadPoolExecutor(max_workers=3)
    try:
        # Use the actual HTTP upload/validation/admission path for both jobs.
        with TestClient(api.app) as client:
            created = client.post(
                "/v1/jobs",
                files={"file": ("first.png", png(), "image/png")},
                data={"rotate_page": "1:0"},
            )
            assert created.status_code == 202
            record = manager.get_record(created.json()["id"])
            record.future.result(timeout=5)
            target.append(record.output_artifact)
            waiting = pool.submit(client.request, method, f"/v1/jobs/{record.id}")
            assert entered.wait(5)
            submitted = pool.submit(
                client.post,
                "/v1/jobs",
                files={"file": ("second.png", png(), "image/png")},
                data={"rotate_page": "1:0"},
            )
            try:
                second = submitted.result(timeout=2)
                assert second.status_code == 202
                second_id = second.json()["id"]
                other = manager.get_record(second_id)
                other.future.result(timeout=2)
                response = pool.submit(client.get, f"/v1/jobs/{second_id}").result(timeout=2)
                assert response.status_code == 200
                assert response.json()["download_ready"] is True
                assert response.json()["page_rotations"] == [
                    {"page": 1, "degrees_clockwise": 0},
                ]
                assert not waiting.done()
                if expire:
                    # Only the first job becomes due; the second result remains live.
                    with manager._lock:
                        record.completed_at -= timedelta(hours=2)
                    expired = pool.submit(manager.cleanup_expired).result(timeout=2)
                    assert expired == 1
                    assert not storage.exists(target[0])
            finally:
                release.set()
            response = waiting.result(timeout=5)
            if expire:
                assert response.status_code == 404
                assert response.json() == {"detail": "Job not found."}
            else:
                assert response.status_code == 200
                assert response.json()["status"] == "succeeded"
                assert response.json()["download_ready"] is True
                assert response.json()["cancellation_requested"] is False
            assert probes == [target[0]]
            download = client.get(f"/v1/jobs/{second_id}/download")
            assert download.status_code == 200
            assert download.content == b"authored fixture archive"
            assert manager.snapshot()["active_downloads"] == 0
            assert manager.snapshot()["completed_total"] == {"succeeded": 2}
    finally:
        release.set()
        pool.shutdown(wait=True)
        manager.shutdown()
