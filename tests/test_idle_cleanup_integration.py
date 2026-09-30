"""Exercise real cleanup passes without foreground requests driving expiration.

Events bound waits; explicit timestamps determine expiry/backoff eligibility.
Filesystem I/O is real. S3 uses a local protocol double, not a network account.
"""

from __future__ import annotations

import asyncio
import io
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_storage import FakeS3Client

import services.api.app.jobs as jobs
import services.api.app.main as api
from services.api.app.jobs import ConversionJobManager, JobStatus
from services.api.app.storage import FilesystemArtifactStorage, S3ArtifactStorage


@pytest.fixture(params=["filesystem", "s3-protocol-double"])
def storage(tmp_path, request):
    if request.param == "filesystem":
        return FilesystemArtifactStorage(tmp_path / "results")
    return S3ArtifactStorage("fixture", prefix="private", client=FakeS3Client())


def publish(storage, record):
    archive = record.workspace / "result.zip"
    archive.write_bytes(b"authored fixture archive")
    return storage.put_file(
        archive,
        key=f"jobs/{record.id}/result.zip",
        filename="result.zip",
        media_type="application/zip",
    )


def finish(manager):
    record = manager.reserve("fixture.png", ".png")
    manager.enqueue(record.id)
    record.future.result(timeout=5)
    return record


def observe_passes(manager, monkeypatch, callback):
    original = manager.cleanup_expired

    def cleanup():
        result = callback(original)
        return result

    monkeypatch.setattr(manager, "cleanup_expired", cleanup)


def test_idle_worker_expires_stored_result_without_status_or_submit_calls(
    tmp_path, storage, monkeypatch,
):
    manager = ConversionJobManager(
        tmp_path / "jobs", lambda record, _: publish(storage, record),
        artifact_cleanup=storage.delete, cleanup_interval_seconds=0.01,
    )
    done = threading.Event()
    try:
        record = finish(manager)
        artifact = record.output_artifact
        expiry = record.completed_at + timedelta(hours=2)
        calls = []

        def tick(cleanup):
            calls.append(threading.get_ident())
            result = cleanup(now=expiry)
            with manager._lock:
                if record.id not in manager._jobs and not manager._pending_cleanup:
                    done.set()
            return result

        observe_passes(manager, monkeypatch, tick)
        assert not manager._cleanup_worker.is_alive
        manager.start_cleanup_worker()
        # No public(), snapshot(), reserve(), or manual cleanup after starting.
        assert done.wait(5)
        assert all(identity != threading.get_ident() for identity in calls)
        assert not record.workspace.exists()
        assert not storage.exists(artifact)
        assert manager._completed_total == {"succeeded": 1}
    finally:
        manager.shutdown()
    assert not manager._cleanup_worker.is_alive


def test_idle_retry_keeps_backoff_and_stage_ownership(tmp_path, storage, monkeypatch):
    calls = []

    def delete(artifact):
        calls.append(artifact)
        if len(calls) <= 2:
            raise OSError("PRIVATE storage message")
        storage.delete(artifact)

    manager = ConversionJobManager(
        tmp_path / "jobs", lambda record, _: publish(storage, record),
        artifact_cleanup=delete, cleanup_interval_seconds=0.01,
    )
    first, early, second, done = (threading.Event() for _ in range(4))
    clock_lock = threading.Lock()
    try:
        record = finish(manager)
        artifact = record.output_artifact
        expiry = record.completed_at + timedelta(hours=2)
        clock = [expiry]

        def tick(cleanup):
            with clock_lock:
                now = clock[0]
            count = cleanup(now=now)
            with manager._lock:
                pending = manager._pending_cleanup.get(record.id)
                if pending is not None and pending.failures == 1:
                    first.set()
                    if now == expiry + timedelta(seconds=29):
                        early.set()
                if pending is not None and pending.failures == 2:
                    second.set()
                if pending is None and record.id not in manager._jobs:
                    done.set()
            return count

        observe_passes(manager, monkeypatch, tick)
        manager.start_cleanup_worker()
        assert first.wait(5)
        assert not record.workspace.exists()
        assert storage.exists(artifact)
        with clock_lock:
            clock[0] = expiry + timedelta(seconds=29)
        assert early.wait(5)
        assert len(calls) == 1
        with clock_lock:
            clock[0] = expiry + timedelta(seconds=30)
        assert second.wait(5)
        assert len(calls) == 2
        with manager._lock:
            assert manager._pending_cleanup[record.id].retry_at == expiry + timedelta(seconds=90)
        with clock_lock:
            clock[0] = expiry + timedelta(seconds=90)
        assert done.wait(5)
        assert calls == [artifact, artifact, artifact]
        assert not storage.exists(artifact)
        assert manager._cleanup_attempts_total == 3
        assert manager._cleanup_failures_total == 2
    finally:
        manager.shutdown()


def test_idle_and_foreground_passes_cannot_delete_same_artifact_twice(
    tmp_path, storage, monkeypatch,
):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def delete(artifact):
        calls.append(artifact)
        entered.set()
        assert release.wait(5)
        storage.delete(artifact)

    manager = ConversionJobManager(
        tmp_path / "jobs", lambda record, _: publish(storage, record),
        artifact_cleanup=delete, cleanup_interval_seconds=0.01,
    )
    try:
        record = finish(manager)
        expiry = record.completed_at + timedelta(hours=2)
        original = manager.cleanup_expired
        monkeypatch.setattr(manager, "cleanup_expired", lambda: original(now=expiry))
        manager.start_cleanup_worker()
        assert entered.wait(5)
        assert manager._lock.acquire(timeout=1), "idle storage I/O held manager lock"
        manager._lock.release()
        assert original(now=expiry) == 0
        assert len(calls) == 1
        assert manager._pending_cleanup[record.id].in_progress
    finally:
        release.set()
        manager.shutdown()
    assert len(calls) == 1
    assert not manager._pending_cleanup


def test_idle_tick_does_not_expire_running_or_uploading_jobs(tmp_path, monkeypatch):
    entered, release, ticked = threading.Event(), threading.Event(), threading.Event()

    def runner(record, _):
        entered.set()
        assert release.wait(5)
        record.input_path.write_bytes(b"fixture")
        return record.input_path

    manager = ConversionJobManager(tmp_path / "jobs", runner, cleanup_interval_seconds=0.01)
    try:
        uploading = manager.reserve("upload.png", ".png")
        running = manager.reserve("run.png", ".png")
        manager.enqueue(running.id)
        assert entered.wait(5)
        now = datetime.now(UTC) + timedelta(days=1)

        def tick(cleanup):
            result = cleanup(now=now)
            ticked.set()
            return result

        observe_passes(manager, monkeypatch, tick)
        manager.start_cleanup_worker()
        assert ticked.wait(5)
        with manager._lock:
            assert uploading.id in manager._jobs and running.id in manager._jobs
            assert uploading.status == JobStatus.UPLOADING
            assert running.status == JobStatus.RUNNING
    finally:
        release.set()
        manager.shutdown()


def test_shutdown_wait_is_bounded_without_falsely_claiming_callback_stopped(
    tmp_path, monkeypatch, caplog,
):
    entered, release = threading.Event(), threading.Event()
    manager = ConversionJobManager(
        tmp_path / "jobs", lambda *_: None, cleanup_interval_seconds=0.01,
    )

    def blocked():
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(manager, "cleanup_expired", blocked)
    monkeypatch.setattr(jobs, "CLEANUP_SHUTDOWN_TIMEOUT_SECONDS", 0)
    try:
        manager.start_cleanup_worker()
        assert entered.wait(5)
        manager.shutdown()
        assert manager._cleanup_worker.is_alive
        assert manager._executor._shutdown
        assert "still running after shutdown wait" in caplog.text
        assert str(manager.root_dir) not in caplog.text
        with pytest.raises(RuntimeError, match="stopped"):
            manager.start_cleanup_worker()
    finally:
        release.set()
        assert manager._cleanup_worker.stop(timeout=5)
        manager.shutdown()


@pytest.mark.parametrize("interval", [-1, True, "30", float("nan"), float("inf")])
def test_manager_rejects_invalid_interval_before_creating_workspace(tmp_path, interval):
    path = tmp_path / "jobs"
    with pytest.raises(ValueError, match="interval_seconds"):
        ConversionJobManager(path, lambda *_: None, cleanup_interval_seconds=interval)
    assert not path.exists()


def test_api_lifespan_cleans_idle_result_and_joins_owner(tmp_path, storage, monkeypatch):
    storage_done = threading.Event()
    manager = ConversionJobManager(
        tmp_path / "jobs", lambda record, _: publish(storage, record),
        artifact_cleanup=storage.delete, cleanup_interval_seconds=0.01,
    )
    monkeypatch.setattr(api, "JOB_MANAGER", manager)
    monkeypatch.setattr(api, "_enforce_submission_rate_limit", lambda *args, **kw: None)
    original = manager.cleanup_expired
    monitored = []

    def cleanup():
        result = original()
        with manager._lock:
            if monitored and monitored[0] not in manager._jobs and not manager._pending_cleanup:
                storage_done.set()
        return result

    monkeypatch.setattr(manager, "cleanup_expired", cleanup)
    raw = io.BytesIO()
    Image.new("RGB", (5, 5), "white").save(raw, format="PNG")
    assert not manager._cleanup_worker.is_alive
    try:
        with TestClient(api.app) as client:
            assert manager._cleanup_worker.is_alive
            response = client.post(
                "/v1/jobs", files={"file": ("fixture.png", raw.getvalue(), "image/png")},
            )
            assert response.status_code == 202
            job_id = response.json()["id"]
            with manager._lock:
                record = manager._jobs[job_id]
            record.future.result(timeout=5)
            artifact = record.output_artifact
            with manager._lock:
                record.completed_at = datetime.now(UTC) - timedelta(hours=2)
                monitored.append(job_id)
            # No further HTTP/manager activity drives cleanup during this wait.
            assert storage_done.wait(5)
            assert not storage.exists(artifact)
            assert not record.workspace.exists()
        assert not manager._cleanup_worker.is_alive
        assert manager._executor._shutdown
    finally:
        manager.shutdown()


@pytest.mark.parametrize("raise_in_body", [False, True])
def test_lifespan_captures_owner_and_shuts_it_down_off_event_loop(monkeypatch, raise_in_body):
    calls = []
    loop_thread = threading.get_ident()

    class Owner:
        def start_cleanup_worker(self):
            calls.append("start")

        def shutdown(self):
            calls.append("stop")
            assert threading.get_ident() != loop_thread

    original = Owner()
    monkeypatch.setattr(api, "JOB_MANAGER", original)

    async def use_lifespan():
        async with api._lifespan(api.app):
            monkeypatch.setattr(api, "JOB_MANAGER", object())
            if raise_in_body:
                raise ValueError("test body")

    if raise_in_body:
        with pytest.raises(ValueError, match="test body"):
            asyncio.run(use_lifespan())
    else:
        asyncio.run(use_lifespan())
    assert calls == ["start", "stop"]


def test_failed_lifespan_start_still_releases_executor(monkeypatch):
    closed = threading.Event()

    class Owner:
        def start_cleanup_worker(self):
            raise RuntimeError("startup fixture")

        def shutdown(self):
            closed.set()

    monkeypatch.setattr(api, "JOB_MANAGER", Owner())

    async def use_lifespan():
        async with api._lifespan(api.app):
            raise AssertionError("startup failure should not yield")

    with pytest.raises(RuntimeError, match="startup fixture"):
        asyncio.run(use_lifespan())
    assert closed.is_set()


def test_api_config_and_presets_expose_idle_cadence_without_starting_on_import():
    assert api.JOB_MANAGER.cleanup_interval_seconds == api.JOB_CLEANUP_INTERVAL_SECONDS
    assert not api.JOB_MANAGER._cleanup_worker.is_alive
    repo = Path(__file__).resolve().parents[1]
    assert 'JOB_CLEANUP_INTERVAL_SECONDS: "${JOB_CLEANUP_INTERVAL_SECONDS:-30}"' in (
        repo / "docker-compose.yml"
    ).read_text()
    for preset in (repo / "deploy/presets").glob("*.env.example"):
        assert "JOB_CLEANUP_INTERVAL_SECONDS=30" in preset.read_text()


def test_idle_pass_keeps_eight_task_budget_and_finishes_backlog_on_next_tick(
    tmp_path, monkeypatch,
):
    manager = ConversionJobManager(
        tmp_path / "jobs", lambda record, _: record.input_path,
        cleanup_interval_seconds=0.01,
    )
    first, release, done = (threading.Event() for _ in range(3))
    rows = []
    try:
        records = [finish(manager) for _ in range(10)]
        expiry = max(record.completed_at for record in records) + timedelta(hours=2)
        original = manager.cleanup_expired

        def tick():
            before = manager._cleanup_attempts_total
            expired = original(now=expiry)
            with manager._lock:
                rows.append((expired, manager._cleanup_attempts_total - before,
                             len(manager._pending_cleanup)))
            if len(rows) == 1:
                first.set()
                assert release.wait(5)
            elif rows[-1][2] == 0:
                done.set()

        monkeypatch.setattr(manager, "cleanup_expired", tick)
        manager.start_cleanup_worker()
        assert first.wait(5)
        assert rows[0] == (10, 8, 2)
        release.set()
        assert done.wait(5)
        assert rows[1] == (0, 2, 0)
        assert not any(record.workspace.exists() for record in records)
    finally:
        release.set()
        manager.shutdown()


def test_disabled_lifespan_keeps_request_driven_cleanup_available(tmp_path, monkeypatch):
    manager = ConversionJobManager(
        tmp_path / "jobs", lambda record, _: record.input_path, cleanup_interval_seconds=0,
    )
    monkeypatch.setattr(api, "JOB_MANAGER", manager)
    try:
        with TestClient(api.app):
            assert not manager._cleanup_worker.is_alive
            record = finish(manager)
            assert manager.cleanup_expired(
                now=record.completed_at + timedelta(hours=2),
            ) == 1
            assert not record.workspace.exists()
        assert manager._executor._shutdown
    finally:
        manager.shutdown()


def test_health_exposes_configured_cadence_without_claiming_worker_health(monkeypatch):
    class FixtureEngine:
        def is_available(self):
            return True

        def metadata(self):
            return {"name": "fixture"}

    monkeypatch.setattr(api, "_engine", FixtureEngine)
    monkeypatch.setattr(api, "JOB_CLEANUP_INTERVAL_SECONDS", 60)
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    assert response.json()["jobs"]["cleanup_interval_seconds"] == 60
    assert not api.JOB_MANAGER._cleanup_worker.is_alive
    assert "cleanup_worker_healthy" not in response.json()["jobs"]
