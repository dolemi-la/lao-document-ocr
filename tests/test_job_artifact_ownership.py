"""Deterministic stored-result ownership checks; no OCR or external storage service."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import pytest

from services.api.app.jobs import ConversionJobManager, JobStatus
from services.api.app.storage import FilesystemArtifactStorage, S3ArtifactStorage


def store_result(storage, record):
    local = record.workspace / "result.zip"
    local.write_bytes(b"authored test archive")
    return storage.put_file(
        local,
        key=f"jobs/{record.id}/result.zip",
        filename="result.zip",
        media_type="application/zip",
    )


def test_cancellation_after_storage_before_handoff_does_not_orphan_result(tmp_path, storage):
    stored, release = threading.Event(), threading.Event()
    artifacts = []
    deleted = []

    def runner(record, cancel_event):
        artifacts.append(store_result(storage, record))
        stored.set()
        assert release.wait(5), "test did not release the runner"
        return artifacts[0]

    def cleanup(artifact):
        deleted.append(artifact)
        storage.delete(artifact)

    manager = ConversionJobManager(
        tmp_path / "jobs",
        runner,
        artifact_exists=storage.exists,
        artifact_cleanup=cleanup,
    )
    try:
        record = manager.reserve("sample.png", ".png")
        manager.enqueue(record.id)
        assert stored.wait(5)
        assert manager.cancel(record.id)["cancellation_requested"]
        release.set()
        record.future.result(timeout=5)
        public = manager.public(record.id)
        assert public["status"] == "cancelled"
        assert public["download_ready"] is False
        assert public["orientation_review"] is None
        assert deleted == artifacts
        assert not storage.exists(artifacts[0])
        assert record.output_artifact is None
        assert record.output_path is None
        assert manager.snapshot()["completed_total"] == {"cancelled": 1}
        assert manager.cleanup_expired(now=record.completed_at + timedelta(hours=2)) == 1
        assert deleted == artifacts  # Expiration must not issue another successful delete.
    finally:
        release.set()
        manager.shutdown()


def test_review_failure_precedes_storage_publish(tmp_path, monkeypatch):
    import services.api.app.main as api
    from lao_document_ocr.models import Document, Page
    from services.api.app.jobs import JobRecord

    storage = FilesystemArtifactStorage(tmp_path / "results")
    calls = []
    document = Document(pages=[Page(number=1, width=10, height=10)])
    monkeypatch.setattr(api, "_engine", lambda: None)
    monkeypatch.setattr(api, "process_document", lambda *args, **kwargs: document)
    monkeypatch.setattr(api, "RESULT_STORAGE", storage)

    def archive(document, workspace, filename):
        local = workspace / "result.zip"
        local.write_bytes(b"authored test archive")
        return local

    def invalid_review(*args, **kwargs):
        raise RuntimeError("review fixture failure")

    original_put = storage.put_file

    def put(*args, **kwargs):
        calls.append("put")
        return original_put(*args, **kwargs)

    monkeypatch.setattr(api, "_write_outputs_archive", archive)
    monkeypatch.setattr(api, "build_orientation_review", invalid_review)
    monkeypatch.setattr(storage, "put_file", put)
    workspace = tmp_path / "work"
    workspace.mkdir()
    record = JobRecord("test-job", "sample.png", workspace, workspace / "input.png")
    with pytest.raises(RuntimeError, match="review fixture failure"):
        api._run_conversion_job(record, record.cancel_event)
    assert calls == []
    assert not list(storage.root.rglob("*.zip"))
    assert record.status == JobStatus.UPLOADING  # Worker does not publish lifecycle states.


class MemoryS3:
    """Exercise the real S3 adapter with a local protocol double, never AWS."""

    def __init__(self):
        self.objects = {}
        self.deleted = []

    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        self.objects[(bucket, key)] = Path(filename).read_bytes()

    def head_object(self, *, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            error = RuntimeError("test missing object")
            error.response = {"Error": {"Code": "NoSuchKey"}}
            raise error
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def delete_object(self, *, Bucket, Key):
        self.deleted.append((Bucket, Key))
        self.objects.pop((Bucket, Key), None)


@pytest.fixture(params=["filesystem", "s3-protocol-double"])
def storage(request, tmp_path):
    if request.param == "filesystem":
        return FilesystemArtifactStorage(tmp_path / "results")
    return S3ArtifactStorage("test-bucket", prefix="test-results", client=MemoryS3())


@pytest.mark.parametrize("retention", [0, 60])
def test_expiry_cannot_race_cancellation_cleanup_or_block_other_jobs(tmp_path, storage, retention):
    entered, release = threading.Event(), threading.Event()
    deleted = []

    def runner(record, cancel_event):
        artifact = store_result(storage, record)
        cancel_event.set()
        return artifact

    def cleanup(artifact):
        deleted.append(artifact)
        entered.set()
        assert release.wait(5), "test did not release storage cleanup"
        storage.delete(artifact)

    manager = ConversionJobManager(
        tmp_path / "jobs",
        runner,
        retention_seconds=retention,
        artifact_exists=storage.exists,
        artifact_cleanup=cleanup,
    )
    observer = ThreadPoolExecutor(max_workers=1)
    try:
        record = manager.reserve("cancelled.png", ".png")
        manager.enqueue(record.id)
        assert entered.wait(5)
        assert record.status == JobStatus.CANCELLED
        assert record.completed_at is not None
        future_time = record.completed_at + timedelta(hours=2)

        def observe():
            for _ in range(3):
                assert manager.cleanup_expired(now=future_time) == 0
            assert manager.public(record.id)["download_ready"] is False
            return manager.reserve("unrelated.png", ".png")

        other = observer.submit(observe).result(timeout=2)
        assert other.workspace.is_dir()
        assert len(deleted) == 1
        release.set()
        record.future.result(timeout=5)
        assert record.output_artifact is None
        assert manager.cleanup_expired(now=future_time) == 1
        assert len(deleted) == 1
        assert other.workspace.is_dir()
        assert manager.public(other.id)["status"] == "uploading"
        assert manager.snapshot()["completed_total"] == {"cancelled": 1}
    finally:
        release.set()
        observer.shutdown()
        manager.shutdown()


def test_cancelled_cleanup_failure_stays_private_and_retries_at_expiry(tmp_path, storage, caplog):
    artifacts, calls = [], []

    def runner(record, cancel_event):
        artifacts.append(store_result(storage, record))
        cancel_event.set()
        return artifacts[0]

    def cleanup(artifact):
        calls.append(artifact)
        if len(calls) == 1:
            raise RuntimeError("PRIVATE-STORAGE-ERROR key/token/document text")
        storage.delete(artifact)

    manager = ConversionJobManager(
        tmp_path / "jobs",
        runner,
        artifact_exists=storage.exists,
        artifact_cleanup=cleanup,
    )
    try:
        record = manager.reserve("sample.png", ".png")
        manager.enqueue(record.id)
        record.future.result(timeout=5)
        assert storage.exists(artifacts[0])
        assert record.output_artifact == artifacts[0]
        assert not record._artifact_cleanup_in_progress
        public = manager.public(record.id)
        assert public["status"] == "cancelled" and public["error"] is None
        assert public["download_ready"] is False and public["orientation_review"] is None
        assert "output_artifact" not in public and "_artifact_cleanup_in_progress" not in public
        assert "retained for expiry" in caplog.text
        assert "PRIVATE-STORAGE-ERROR" not in caplog.text
        assert all(item.exc_info is None for item in caplog.records)
        manager.cancel(record.id)
        assert len(calls) == 1  # Repeated user cancellations do not trigger delete loops.
        assert manager.cleanup_expired(now=record.completed_at + timedelta(hours=2)) == 1
        assert len(calls) == 2 and not storage.exists(artifacts[0])
        assert manager.snapshot()["completed_total"] == {"cancelled": 1}
    finally:
        manager.shutdown()


def test_late_cancel_cannot_delete_a_successful_result(tmp_path, storage):
    calls = []

    def cleanup(artifact):
        calls.append(artifact)
        storage.delete(artifact)

    manager = ConversionJobManager(
        tmp_path / "jobs",
        lambda record, cancel: store_result(storage, record),
        artifact_exists=storage.exists,
        artifact_cleanup=cleanup,
    )
    try:
        record = manager.reserve("sample.png", ".png")
        manager.enqueue(record.id)
        record.future.result(timeout=5)
        artifact = record.output_artifact
        public = manager.cancel(record.id)
        assert public["status"] == "succeeded" and public["download_ready"] is True
        assert public["cancellation_requested"] is False
        assert calls == [] and storage.exists(artifact)
        assert manager.cleanup_expired(now=record.completed_at + timedelta(hours=2)) == 1
        assert calls == [artifact] and not storage.exists(artifact)
        assert manager.snapshot()["completed_total"] == {"succeeded": 1}
    finally:
        manager.shutdown()


def test_absent_cleanup_callback_retains_cancelled_artifact_without_publishing(tmp_path, storage):
    artifacts = []

    def runner(record, cancel_event):
        artifacts.append(store_result(storage, record))
        cancel_event.set()
        return artifacts[0]

    manager = ConversionJobManager(tmp_path / "jobs", runner, artifact_exists=storage.exists)
    try:
        record = manager.reserve("sample.png", ".png")
        manager.enqueue(record.id)
        record.future.result(timeout=5)
        assert record.output_artifact == artifacts[0]
        assert storage.exists(artifacts[0])
        assert not record._artifact_cleanup_in_progress
        assert manager.public(record.id)["download_ready"] is False
        assert manager.public(record.id)["status"] == "cancelled"
    finally:
        manager.shutdown()


def test_legacy_path_cancellation_does_not_unlink_unrelated_files(tmp_path):
    external = tmp_path / "unrelated.zip"
    external.write_bytes(b"not owned by a job workspace")

    def runner(record, cancel_event):
        cancel_event.set()
        return external

    manager = ConversionJobManager(tmp_path / "jobs", runner)
    try:
        record = manager.reserve("sample.png", ".png")
        manager.enqueue(record.id)
        record.future.result(timeout=5)
        assert manager.public(record.id)["status"] == "cancelled"
        assert record.output_artifact is None and record.output_path is None
        assert manager.cleanup_expired(now=record.completed_at + timedelta(hours=2)) == 1
        assert external.read_bytes() == b"not owned by a job workspace"
        assert not record.workspace.exists()
    finally:
        manager.shutdown()


@pytest.mark.parametrize("auto_orient", [False, True])
@pytest.mark.parametrize("pause_at", ["storage", "handoff"])
def test_real_api_worker_cancellation_during_storage_is_owned_and_not_downloadable(
    tmp_path,
    monkeypatch,
    storage,
    auto_orient,
    pause_at,
):
    import io

    from fastapi.testclient import TestClient
    from PIL import Image

    import services.api.app.main as api
    from lao_document_ocr.models import BoundingBox
    from lao_document_ocr.ocr.base import OcrEngine, RecognizedLine

    stored, release = threading.Event(), threading.Event()
    artifacts, deleted, recognized = [], [], []
    source = tmp_path / "unrelated-source.zip"
    source.write_bytes(b"other job artifact")
    unrelated = storage.put_file(
        source,
        key="unrelated/result.zip",
        filename="result.zip",
        media_type="application/zip",
    )
    original_put = storage.put_file

    def paused_put(*args, **kwargs):
        artifacts.append(original_put(*args, **kwargs))
        if pause_at == "storage":
            stored.set()
            assert release.wait(5), "test did not release the stored result"
        return artifacts[0]

    def runner(record, cancel_event):
        artifact = api._run_conversion_job(record, cancel_event)
        if pause_at == "handoff":
            stored.set()
            assert release.wait(5), "test did not release the worker handoff"
        return artifact

    def cleanup(artifact):
        deleted.append(artifact)
        storage.delete(artifact)

    class FixtureEngine(OcrEngine):
        def is_available(self):
            return True

        def recognize(self, image):
            recognized.append(image.size)
            return [
                RecognizedLine(
                    "authored fixture text",
                    BoundingBox(x=5, y=5, width=60, height=10),
                    0.95,
                    1,
                    1,
                    1,
                )
            ]

    monkeypatch.setattr(storage, "put_file", paused_put)
    monkeypatch.setattr(api, "RESULT_STORAGE", storage)
    monkeypatch.setattr(api, "_engine", FixtureEngine)
    monkeypatch.setattr(api, "_enforce_submission_rate_limit", lambda *args, **kwargs: None)
    manager = ConversionJobManager(
        tmp_path / "jobs",
        runner,
        artifact_exists=storage.exists,
        artifact_cleanup=cleanup,
    )
    monkeypatch.setattr(api, "JOB_MANAGER", manager)
    client = TestClient(api.app)
    upload = io.BytesIO()
    Image.new("RGB", (80, 120), "white").save(upload, format="PNG")
    try:
        created = client.post(
            "/v1/jobs",
            files={"file": ("sample.png", upload.getvalue(), "image/png")},
            data={"rotate_page": "1:0", "auto_orient_right_angles": str(auto_orient).lower()},
        )
        assert created.status_code == 202
        job_id = created.json()["id"]
        record = manager.get_record(job_id)
        assert stored.wait(5)
        cancellation = client.delete(f"/v1/jobs/{job_id}")
        assert cancellation.status_code == 200
        assert cancellation.json()["cancellation_requested"] is True
        assert cancellation.json()["status"] == "running"
        release.set()
        record.future.result(timeout=5)
        terminal = client.get(f"/v1/jobs/{job_id}").json()
        assert terminal["status"] == "cancelled" and terminal["error"] is None
        assert terminal["download_ready"] is False and terminal["orientation_review"] is None
        assert terminal["page_rotations"] == [{"page": 1, "degrees_clockwise": 0}]
        assert client.get(f"/v1/jobs/{job_id}/download").status_code == 409
        assert deleted == artifacts and not storage.exists(artifacts[0])
        assert storage.exists(unrelated) and len(recognized) == 1
        assert manager.cleanup_expired(now=record.completed_at + timedelta(hours=2)) == 1
        assert deleted == artifacts and storage.exists(unrelated)
        assert client.get(f"/v1/jobs/{job_id}").status_code == 404
    finally:
        release.set()
        manager.shutdown()
        client.close()


@pytest.mark.parametrize("failure", ["cancel", "export-error"])
def test_worker_export_failure_or_cancellation_never_publishes(tmp_path, monkeypatch, failure):
    import services.api.app.main as api
    from lao_document_ocr.models import Document, Page
    from services.api.app.jobs import JobCancelledError, JobRecord

    storage = FilesystemArtifactStorage(tmp_path / "results")
    calls = []
    document = Document(pages=[Page(number=1, width=10, height=10)])
    monkeypatch.setattr(api, "_engine", lambda: None)
    monkeypatch.setattr(api, "process_document", lambda *args, **kwargs: document)
    monkeypatch.setattr(api, "RESULT_STORAGE", storage)
    workspace = tmp_path / "work"
    workspace.mkdir()
    record = JobRecord("test-job", "sample.png", workspace, workspace / "input.png")

    def archive(document, workspace, filename):
        local = workspace / "result.zip"
        local.write_bytes(b"authored temporary export")
        if failure == "export-error":
            raise RuntimeError("export fixture failure")
        record.cancel_event.set()
        return local

    def unexpected_put(*args, **kwargs):
        calls.append("put")
        raise AssertionError("no publication after export failure or cancellation")

    monkeypatch.setattr(api, "_write_outputs_archive", archive)
    monkeypatch.setattr(storage, "put_file", unexpected_put)
    with pytest.raises(JobCancelledError if failure == "cancel" else RuntimeError):
        api._run_conversion_job(record, record.cancel_event)
    assert calls == []
    assert not list(storage.root.rglob("*.zip"))


def test_cancellation_before_runner_does_not_publish_or_cleanup(tmp_path):
    calls = []

    def unexpected(*args):
        calls.append("unexpected")
        raise AssertionError("neither runner nor cleanup should execute")

    manager = ConversionJobManager(tmp_path / "jobs", unexpected, artifact_cleanup=unexpected)
    try:
        record = manager.reserve("sample.png", ".png")
        manager.cancel(record.id)
        manager.enqueue(record.id)
        assert calls == []
        assert manager.public(record.id)["status"] == "cancelled"
        assert manager.snapshot()["completed_total"] == {"cancelled": 1}
    finally:
        manager.shutdown()
