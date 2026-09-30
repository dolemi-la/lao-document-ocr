"""A slow result-existence probe must not own the global job-state lock."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from test_storage import FakeS3Client

from services.api.app.jobs import (
    ConversionJobManager,
    JobCancelledError,
    JobManagerClosedError,
    JobNotFoundError,
    JobStatus,
)
from services.api.app.storage import FilesystemArtifactStorage, S3ArtifactStorage


@pytest.fixture(params=["filesystem", "s3-protocol-double"])
def storage(tmp_path, request):
    if request.param == "filesystem":
        return FilesystemArtifactStorage(tmp_path / "objects")
    return S3ArtifactStorage("fixture-bucket", prefix="fixture", client=FakeS3Client())


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
    record = manager.reserve("fixture.png", ".png", page_rotations={1: 0})
    manager.enqueue(record.id)
    record.future.result(timeout=5)
    assert record.status == JobStatus.SUCCEEDED
    return record


@pytest.mark.parametrize("operation", ["public", "cancel"])
def test_slow_readiness_does_not_block_unrelated_reservation(tmp_path, storage, operation):
    entered, release = threading.Event(), threading.Event()

    def exists(artifact):
        entered.set()
        assert release.wait(5), "test did not release the readiness check"
        return storage.exists(artifact)

    manager = ConversionJobManager(
        tmp_path / "jobs",
        lambda record, _: publish(storage, record),
        artifact_exists=exists,
        artifact_cleanup=storage.delete,
    )
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        record = finish(manager)
        query = pool.submit(getattr(manager, operation), record.id)
        assert entered.wait(5)
        other = pool.submit(manager.reserve, "unrelated.png", ".png")
        try:
            # The timeout only detects holding the shared lock; Events establish ordering.
            admitted = other.result(timeout=1)
            assert admitted.status == JobStatus.UPLOADING
            assert not query.done()
        finally:
            release.set()
        result = query.result(timeout=5)
        assert result["status"] == "succeeded"
        assert result["download_ready"] is True
        assert result["page_rotations"] == [{"page": 1, "degrees_clockwise": 0}]
    finally:
        release.set()
        pool.shutdown(wait=True)
        manager.shutdown()


@pytest.mark.parametrize("operation", ["public", "cancel"])
def test_expiry_can_finish_during_probe_and_stale_response_is_rejected(
    tmp_path,
    storage,
    operation,
):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def exists(artifact):
        calls.append(artifact)
        result = storage.exists(artifact)
        entered.set()
        assert release.wait(5)
        return result  # Deliberately stale after concurrent deletion.

    manager = ConversionJobManager(
        tmp_path / "jobs",
        lambda record, _: publish(storage, record),
        artifact_exists=exists,
        artifact_cleanup=storage.delete,
    )
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        record = finish(manager)
        artifact = record.output_artifact
        query = pool.submit(getattr(manager, operation), record.id)
        assert entered.wait(5)
        expiry = pool.submit(
            manager.cleanup_expired,
            now=record.completed_at + timedelta(hours=2),
        )
        try:
            assert expiry.result(timeout=1) == 1
            assert not storage.exists(artifact)
            assert not record.workspace.exists()
            assert not manager._download_leases
        finally:
            release.set()
        with pytest.raises(JobNotFoundError):
            query.result(timeout=5)
        assert calls == [artifact]
        assert manager.snapshot()["completed_total"] == {"succeeded": 1}
    finally:
        release.set()
        pool.shutdown(wait=True)
        manager.shutdown()


def test_probe_does_not_block_worker_completion_cancellation_or_shutdown(tmp_path):
    entered, release, running, finish_running = (threading.Event() for _ in range(4))
    storage = FilesystemArtifactStorage(tmp_path / "objects")

    def runner(record, cancel_event):
        if record.filename == "running.png":
            running.set()
            assert finish_running.wait(5)
            if cancel_event.is_set():
                raise JobCancelledError()
        return publish(storage, record)

    def exists(artifact):
        entered.set()
        assert release.wait(5)
        return storage.exists(artifact)

    manager = ConversionJobManager(tmp_path / "jobs", runner, artifact_exists=exists)
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        completed = finish(manager)
        active = manager.reserve("running.png", ".png")
        manager.enqueue(active.id)
        assert running.wait(5)
        query = pool.submit(manager.public, completed.id)
        assert entered.wait(5)
        cancelled = pool.submit(manager.cancel, active.id).result(timeout=1)
        assert cancelled["cancellation_requested"] is True
        assert cancelled["download_ready"] is False
        finish_running.set()
        active.future.result(timeout=1)
        assert active.status == JobStatus.CANCELLED
        snapshot = pool.submit(manager.snapshot).result(timeout=1)
        assert snapshot["completed_total"] == {"cancelled": 1, "succeeded": 1}
        pool.submit(manager.shutdown, wait=False).result(timeout=1)
        assert not query.done()
        release.set()
        assert query.result(timeout=5)["download_ready"] is True
        with pytest.raises(JobManagerClosedError):
            manager.reserve("closed.png", ".png")
    finally:
        release.set()
        finish_running.set()
        pool.shutdown(wait=True)
        manager.shutdown()


@pytest.mark.parametrize("change", ["result", "provider", "record"])
def test_probe_revalidates_immutable_result_and_owner_without_retrying(tmp_path, change):
    entered, release = threading.Event(), threading.Event()
    storage = FilesystemArtifactStorage(tmp_path / "objects")
    calls = []

    def exists(artifact):
        calls.append(artifact)
        entered.set()
        assert release.wait(5)
        return True

    manager = ConversionJobManager(
        tmp_path / "jobs",
        lambda record, _: publish(storage, record),
        artifact_exists=exists,
    )
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        record = finish(manager)
        artifact = record.output_artifact
        query = pool.submit(manager.public, record.id)
        assert entered.wait(5)
        with manager._lock:
            # Inject changes at the callback boundary to test stale-result rejection.
            if change == "result":
                record.output_artifact = replace(artifact, key="different-result.zip")
            elif change == "provider":
                manager.artifact_exists = lambda _: True
            else:
                manager._jobs[record.id] = replace(record)
        release.set()
        if change == "record":
            with pytest.raises(JobNotFoundError):
                query.result(timeout=5)
        else:
            result = query.result(timeout=5)
            assert result["status"] == "succeeded"
            assert result["download_ready"] is False
        assert calls == [artifact]
    finally:
        release.set()
        pool.shutdown(wait=True)
        manager.shutdown()


def test_local_result_stat_runs_outside_lock_and_permission_error_is_private(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def runner(record, _):
        path = record.workspace / "result.zip"
        path.write_bytes(b"authored fixture archive")
        return path

    manager = ConversionJobManager(tmp_path / "jobs", runner)
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        record = finish(manager)
        original = Path.is_file

        def blocked_stat(path):
            if path == record.output_path:
                entered.set()
                assert release.wait(5)
                raise PermissionError("PRIVATE result path")
            return original(path)

        monkeypatch.setattr(Path, "is_file", blocked_stat)
        query = pool.submit(manager.public, record.id)
        assert entered.wait(5)
        other = pool.submit(manager.reserve, "unrelated.png", ".png").result(timeout=1)
        assert other.status == JobStatus.UPLOADING
        release.set()
        result = query.result(timeout=5)
        assert result["download_ready"] is False
        assert result["error"] is None
        assert "PRIVATE" not in json.dumps(result)
    finally:
        release.set()
        pool.shutdown(wait=True)
        manager.shutdown()


@pytest.mark.parametrize("status", list(JobStatus))
def test_only_successful_status_needs_provider_probe(tmp_path, status, caplog):
    calls = []
    storage = FilesystemArtifactStorage(tmp_path / "objects")

    def broken(artifact):
        calls.append(artifact)
        raise OSError("PRIVATE provider credentials and response body")

    manager = ConversionJobManager(tmp_path / "jobs", lambda *args: None, artifact_exists=broken)
    try:
        record = manager.reserve("fixture.png", ".png")
        record.output_artifact = publish(storage, record)
        record.status = status
        result = manager.public(record.id)
        assert len(calls) == (1 if status == JobStatus.SUCCEEDED else 0)
        assert result["status"] == status.value
        assert result["download_ready"] is False
        assert result["error"] is None
        assert "PRIVATE" not in json.dumps(result)
        assert "PRIVATE" not in caplog.text
    finally:
        manager.shutdown()


def test_active_to_success_transition_returns_current_fields_without_unchecked_readiness(
    tmp_path,
    monkeypatch,
):
    snapshot_taken, release_probe, running, release_runner = (threading.Event() for _ in range(4))
    storage = FilesystemArtifactStorage(tmp_path / "objects")
    calls = []

    def runner(record, _):
        running.set()
        assert release_runner.wait(5)
        return publish(storage, record)

    def exists(artifact):
        calls.append(artifact)
        return storage.exists(artifact)

    manager = ConversionJobManager(tmp_path / "jobs", runner, artifact_exists=exists)
    pool = ThreadPoolExecutor(max_workers=1)
    original = manager._download_ready

    def paused_probe(*args):
        snapshot_taken.set()
        assert release_probe.wait(5)
        return original(*args)

    try:
        record = manager.reserve("fixture.png", ".png")
        manager.enqueue(record.id)
        assert running.wait(5)
        monkeypatch.setattr(manager, "_download_ready", paused_probe)
        query = pool.submit(manager.public, record.id)
        assert snapshot_taken.wait(5)
        release_runner.set()
        record.future.result(timeout=1)
        release_probe.set()
        result = query.result(timeout=5)
        assert result["status"] == "succeeded"
        assert result["completed_at"] is not None
        assert result["download_ready"] is False
        assert calls == []
        monkeypatch.setattr(manager, "_download_ready", original)
        assert manager.public(record.id)["download_ready"] is True
        assert calls == [record.output_artifact]
    finally:
        release_probe.set()
        release_runner.set()
        pool.shutdown(wait=True)
        manager.shutdown()


def test_successful_cancel_keeps_original_outcome_and_detached_public_lists(tmp_path, storage):
    manager = ConversionJobManager(
        tmp_path / "jobs",
        lambda record, _: publish(storage, record),
        artifact_exists=storage.exists,
        artifact_cleanup=storage.delete,
    )
    try:
        record = finish(manager)
        completed_at = record.completed_at
        result = manager.cancel(record.id)
        assert result["status"] == "succeeded"
        assert result["download_ready"] is True
        assert not record.cancel_event.is_set()
        assert not result["cancellation_requested"]
        result["page_rotations"].clear()
        assert manager.public(record.id)["page_rotations"] == [
            {"page": 1, "degrees_clockwise": 0},
        ]
        assert record.completed_at == completed_at
        assert manager.snapshot()["completed_total"] == {"succeeded": 1}
    finally:
        manager.shutdown()
