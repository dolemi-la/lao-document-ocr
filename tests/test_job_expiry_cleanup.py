"""Expiry must hide jobs without forgetting failed storage/workspace cleanup.

Filesystem is real; S3 uses the real adapter with a local protocol double.
Thread events and explicit clocks determine ordering, not sleeps or remote OCR.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest
from test_storage import FakeS3Client

import services.api.app.jobs as jobs
from services.api.app.jobs import (
    ConversionJobManager,
    JobCapacityError,
    JobNotFoundError,
    JobStatus,
)
from services.api.app.metrics import ApiMetrics
from services.api.app.storage import FilesystemArtifactStorage, S3ArtifactStorage


@pytest.fixture(params=["filesystem", "s3-protocol-double"])
def storage(tmp_path, request):
    if request.param == "filesystem":
        return FilesystemArtifactStorage(tmp_path / "objects")
    return S3ArtifactStorage("fixture-bucket", prefix="private-prefix", client=FakeS3Client())


def publish(storage, record):
    archive = record.workspace / "result.zip"
    archive.write_bytes(b"authored fixture archive")
    return storage.put_file(
        archive,
        key=f"jobs/{record.id}/result.zip",
        filename="result.zip",
        media_type="application/zip",
    )


def finish_job(manager):
    record = manager.reserve("private-source.png", ".png")
    record.input_path.write_bytes(b"private fixture input")
    manager.enqueue(record.id)
    record.future.result(timeout=5)
    return record


@pytest.mark.parametrize("retention", [0, 60])
def test_expired_artifact_survives_repeated_delete_failures_without_public_job(
    tmp_path,
    storage,
    retention,
):
    calls = []

    def cleanup(artifact):
        calls.append(artifact)
        if len(calls) <= 2:
            raise OSError("PRIVATE storage failure")
        storage.delete(artifact)

    manager = ConversionJobManager(
        tmp_path / "jobs",
        lambda record, _: publish(storage, record),
        retention_seconds=retention,
        artifact_exists=storage.exists,
        artifact_cleanup=cleanup,
    )
    try:
        record = finish_job(manager)
        artifact = record.output_artifact
        expiry = record.completed_at + timedelta(seconds=retention + 1)
        assert manager.cleanup_expired(now=expiry) == 1
        assert not record.workspace.exists()
        assert storage.exists(artifact)
        for _ in range(3):
            with pytest.raises(JobNotFoundError):
                manager.public(record.id)
            with pytest.raises(JobNotFoundError):
                manager.cancel(record.id)
        assert len(calls) == 1
        assert manager.cleanup_expired(now=expiry + timedelta(seconds=29)) == 0
        assert len(calls) == 1
        assert manager.cleanup_expired(now=expiry + timedelta(seconds=30)) == 0
        assert len(calls) == 2
        assert storage.exists(artifact)
        assert manager.cleanup_expired(now=expiry + timedelta(seconds=89)) == 0
        assert len(calls) == 2
        assert manager.cleanup_expired(now=expiry + timedelta(seconds=90)) == 0
        assert len(calls) == 3
        assert not storage.exists(artifact)
        snapshot = manager.snapshot()
        assert snapshot["completed_total"] == {"succeeded": 1}
        assert snapshot["cleanup_pending_jobs"] == 0
        assert snapshot["cleanup_attempts_total"] == 3
        assert snapshot["cleanup_failures_total"] == 2
    finally:
        manager.shutdown()


def test_expired_workspace_deletion_failure_is_retried_without_redeleting_artifact(
    tmp_path,
    monkeypatch,
    storage,
):
    deleted, removed = [], []
    original = jobs.shutil.rmtree

    def cleanup(artifact):
        deleted.append(artifact)
        storage.delete(artifact)

    def flaky_rmtree(path, *args, **kwargs):
        removed.append(path)
        if len(removed) == 1:
            if kwargs.get("ignore_errors"):
                return None
            raise PermissionError("PRIVATE workspace failure")
        return original(path, *args, **kwargs)

    manager = ConversionJobManager(
        tmp_path / "jobs",
        lambda record, _: publish(storage, record),
        retention_seconds=60,
        artifact_exists=storage.exists,
        artifact_cleanup=cleanup,
    )
    try:
        record = finish_job(manager)
        artifact = record.output_artifact
        monkeypatch.setattr(jobs.shutil, "rmtree", flaky_rmtree)
        expiry = record.completed_at + timedelta(seconds=61)
        assert manager.cleanup_expired(now=expiry) == 1
        assert record.workspace.exists()
        assert not storage.exists(artifact)
        snapshot = manager.snapshot()
        assert snapshot["cleanup_pending_jobs"] == 1
        assert snapshot["cleanup_pending_artifacts"] == 0
        with pytest.raises(JobNotFoundError):
            manager.get_record(record.id)
        assert manager.cleanup_expired(now=expiry + timedelta(seconds=30)) == 0
        assert not record.workspace.exists()
        assert len(removed) == 2
        assert deleted == [artifact]
        assert manager.snapshot()["cleanup_pending_jobs"] == 0
    finally:
        manager.shutdown()


def test_cancel_failure_then_repeated_expiry_failures_keep_one_owner(tmp_path, storage):
    calls = []

    def runner(record, cancel):
        result = publish(storage, record)
        cancel.set()
        return result

    def cleanup(artifact):
        calls.append(artifact)
        if len(calls) < 4:
            raise OSError("PRIVATE failure")
        storage.delete(artifact)

    manager = ConversionJobManager(tmp_path / "jobs", runner, artifact_cleanup=cleanup)
    try:
        record = finish_job(manager)
        artifact = record.output_artifact
        assert record.status == JobStatus.CANCELLED
        assert len(calls) == 1
        expiry = record.completed_at + timedelta(hours=2)
        assert manager.cleanup_expired(now=expiry) == 1
        assert manager.cleanup_expired(now=expiry + timedelta(seconds=30)) == 0
        assert manager.cleanup_expired(now=expiry + timedelta(seconds=90)) == 0
        assert calls == [artifact] * 4
        assert not storage.exists(artifact)
        assert manager.snapshot()["completed_total"] == {"cancelled": 1}
        assert manager.snapshot()["cleanup_attempts_total"] == 3  # Not the immediate attempt.
        assert manager.snapshot()["retained_jobs"] == 0
    finally:
        manager.shutdown()


def test_concurrent_expiry_cannot_double_delete_or_block_status_and_reservation(tmp_path, storage):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def cleanup(artifact):
        calls.append(artifact)
        entered.set()
        assert release.wait(5), "test did not release cleanup"
        storage.delete(artifact)

    manager = ConversionJobManager(
        tmp_path / "jobs",
        lambda record, _: publish(storage, record),
        artifact_cleanup=cleanup,
    )
    try:
        record = finish_job(manager)
        expiry = record.completed_at + timedelta(hours=2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(manager.cleanup_expired, now=expiry)
            try:
                assert entered.wait(5)

                def concurrent_pass():
                    assert manager.cleanup_expired(now=expiry + timedelta(hours=3)) == 0
                    with pytest.raises(JobNotFoundError):
                        manager.get_record(record.id)
                    unrelated = manager.reserve("other.png", ".png")
                    assert manager.public(unrelated.id)["status"] == "uploading"
                    return manager.snapshot()

                snapshot = pool.submit(concurrent_pass).result(timeout=3)
                assert snapshot["cleanup_in_progress"] == 1
                assert snapshot["cleanup_pending_artifacts"] == 1
                assert snapshot["active_jobs"] == 1
                assert len(calls) == 1
            finally:
                release.set()
            assert first.result(timeout=5) == 1
        assert len(calls) == 1
        assert manager.snapshot()["cleanup_in_progress"] == 0
        assert manager.snapshot()["cleanup_pending_jobs"] == 0
    finally:
        release.set()
        manager.shutdown()


def test_retry_backoff_saturates_and_does_not_busy_loop(tmp_path, storage):
    calls = []

    def cleanup(artifact):
        calls.append(artifact)
        raise OSError("PRIVATE retry failure")

    manager = ConversionJobManager(
        tmp_path / "jobs",
        lambda record, _: publish(storage, record),
        artifact_cleanup=cleanup,
    )
    try:
        record = finish_job(manager)
        current = record.completed_at + timedelta(hours=2)
        assert manager.cleanup_expired(now=current) == 1
        for index, delay in enumerate([30, 60, 120, 240, 480, 960, 1920, 3600, 3600], 1):
            assert manager.cleanup_expired(now=current + timedelta(seconds=delay - 1)) == 0
            assert len(calls) == index
            current += timedelta(seconds=delay)
            assert manager.cleanup_expired(now=current) == 0
            assert len(calls) == index + 1
        pending = manager._pending_cleanup[record.id]
        assert not hasattr(pending, "filename")
        assert not hasattr(pending, "orientation_review")
        pending.failures = 10**6
        current += timedelta(seconds=3600)
        manager.cleanup_expired(now=current)
        assert pending.retry_at == current + timedelta(seconds=3600)
    finally:
        manager.shutdown()


def test_each_cleanup_pass_has_fixed_task_budget_and_remaining_work_is_not_dropped(tmp_path):
    manager = ConversionJobManager(tmp_path / "jobs", lambda record, _: record.input_path)
    try:
        records = [finish_job(manager) for _ in range(jobs.CLEANUP_MAX_TASKS_PER_PASS + 3)]
        expiry = max(record.completed_at for record in records) + timedelta(hours=2)
        assert manager.cleanup_expired(now=expiry) == len(records)
        assert len(manager._pending_cleanup) == 3
        assert manager._cleanup_attempts_total == jobs.CLEANUP_MAX_TASKS_PER_PASS
        assert sum(record.workspace.exists() for record in records) == 3
        assert manager.cleanup_expired(now=expiry) == 0
        assert not manager._pending_cleanup
        assert manager._cleanup_attempts_total == len(records)
        assert all(not record.workspace.exists() for record in records)
    finally:
        manager.shutdown()


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5, "2", None, 1])
def test_invalid_retained_capacity_fails_before_creating_root(tmp_path, capacity):
    root = tmp_path / "jobs"
    with pytest.raises(ValueError, match="max_retained_jobs"):
        ConversionJobManager(
            root, lambda *args: None, max_active_jobs=2, max_retained_jobs=capacity
        )
    assert not root.exists()


def test_capacity_includes_terminal_and_failed_cleanup_without_dropping_references(
    tmp_path, storage
):
    failing = True

    def cleanup(artifact):
        if failing:
            raise OSError("PRIVATE capacity failure")
        storage.delete(artifact)

    manager = ConversionJobManager(
        tmp_path / "jobs",
        lambda record, _: publish(storage, record),
        max_workers=1,
        max_active_jobs=1,
        max_retained_jobs=2,
        artifact_cleanup=cleanup,
    )
    try:
        first, second = finish_job(manager), finish_job(manager)
        with pytest.raises(JobCapacityError, match="Retained job capacity"):
            manager.reserve("new.png", ".png")
        expiry = max(first.completed_at, second.completed_at) + timedelta(hours=2)
        assert manager.cleanup_expired(now=expiry) == 2
        assert manager.snapshot()["active_jobs"] == 0
        assert manager.snapshot()["retained_jobs"] == 2
        assert manager.snapshot()["cleanup_pending_jobs"] == 2
        with pytest.raises(JobCapacityError, match="Retained job capacity"):
            manager.reserve("new.png", ".png")
        assert len(manager._pending_cleanup) == 2
        failing = False
        assert manager.cleanup_expired(now=expiry + timedelta(seconds=30)) == 0
        assert manager.reserve("new.png", ".png").status == JobStatus.UPLOADING
        assert manager.snapshot()["retained_jobs"] == 1
    finally:
        manager.shutdown()


def test_batch_admission_is_atomic_at_retained_cap(tmp_path):
    manager = ConversionJobManager(
        tmp_path / "jobs",
        lambda record, _: record.input_path,
        max_workers=1,
        max_active_jobs=3,
        max_retained_jobs=3,
    )
    try:
        record = finish_job(manager)
        before = set(manager.root_dir.iterdir())
        with pytest.raises(JobCapacityError, match="Retained job capacity"):
            manager.reserve_many([("a.png", ".png"), ("b.png", ".png"), ("c.png", ".png")])
        assert set(manager.root_dir.iterdir()) == before
        assert manager.public(record.id)["status"] == "succeeded"
        assert len(manager.reserve_many([("a.png", ".png"), ("b.png", ".png")])) == 2
    finally:
        manager.shutdown()


def test_missing_cleanup_callback_retains_private_artifact_until_callback_is_restored(
    tmp_path, storage
):
    manager = ConversionJobManager(tmp_path / "jobs", lambda record, _: publish(storage, record))
    try:
        record = finish_job(manager)
        artifact = record.output_artifact
        expiry = record.completed_at + timedelta(hours=2)
        assert manager.cleanup_expired(now=expiry) == 1
        assert storage.exists(artifact)
        assert not record.workspace.exists()
        assert manager.snapshot()["cleanup_pending_artifacts"] == 1
        manager.artifact_cleanup = storage.delete
        assert manager.cleanup_expired(now=expiry + timedelta(seconds=30)) == 0
        assert not storage.exists(artifact)
        assert manager.snapshot()["cleanup_pending_jobs"] == 0
    finally:
        manager.shutdown()


def test_already_absent_workspace_is_success_but_missing_child_is_not(tmp_path, monkeypatch):
    original = jobs.shutil.rmtree
    manager = ConversionJobManager(tmp_path / "jobs", lambda record, _: record.input_path)
    try:
        first, second = finish_job(manager), finish_job(manager)
        original(first.workspace)

        def missing_child(path, *args, **kwargs):
            if path == second.workspace:
                raise FileNotFoundError("PRIVATE nested child disappeared")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(jobs.shutil, "rmtree", missing_child)
        expiry = second.completed_at + timedelta(hours=2)
        assert manager.cleanup_expired(now=expiry) == 2
        assert list(manager._pending_cleanup) == [second.id]
        monkeypatch.setattr(jobs.shutil, "rmtree", original)
        manager.cleanup_expired(now=expiry + timedelta(seconds=30))
        assert not manager._pending_cleanup
    finally:
        manager.shutdown()


def test_local_runner_external_path_is_not_deleted(tmp_path):
    external = tmp_path / "unrelated.zip"
    external.write_bytes(b"must remain unchanged")
    manager = ConversionJobManager(tmp_path / "jobs", lambda *_: external)
    try:
        record = finish_job(manager)
        assert manager.cleanup_expired(now=record.completed_at + timedelta(hours=2)) == 1
        assert not record.workspace.exists()
        assert external.read_bytes() == b"must remain unchanged"
    finally:
        manager.shutdown()


def test_cleanup_logs_and_metrics_are_fixed_aggregates_not_storage_error_contents(
    tmp_path,
    storage,
    caplog,
):
    def cleanup(artifact):
        raise RuntimeError(f"PRIVATE {artifact.key} secret credentials /private/path")

    manager = ConversionJobManager(
        tmp_path / "jobs",
        lambda record, _: publish(storage, record),
        artifact_cleanup=cleanup,
    )
    try:
        record = finish_job(manager)
        artifact = record.output_artifact
        manager.cleanup_expired(now=record.completed_at + timedelta(hours=2))
        assert "retained for retry" in caplog.text
        assert "PRIVATE" not in caplog.text
        assert artifact.key not in caplog.text
        assert all(log.exc_info is None for log in caplog.records)
        snapshot = manager.snapshot()
        rendered = ApiMetrics().render_prometheus(snapshot)
        assert "lao_ocr_cleanup_pending_jobs 1\n" in rendered
        assert "lao_ocr_cleanup_pending_artifacts 1\n" in rendered
        assert "lao_ocr_cleanup_failures_total 1\n" in rendered
        assert "lao_ocr_jobs_retained 1\n" in rendered
        assert "lao_ocr_jobs_retained_limit 1024\n" in rendered
        for secret in (
            "PRIVATE",
            record.id,
            artifact.key,
            "private-source.png",
            str(record.workspace),
        ):
            assert secret not in rendered
            assert secret not in json.dumps(snapshot)
    finally:
        manager.shutdown()


def test_no_retry_queue_is_claimed_as_restart_durable(tmp_path, storage):
    def failing(_):
        raise OSError("PRIVATE failure")

    root = tmp_path / "jobs"
    manager = ConversionJobManager(
        root,
        lambda record, _: publish(storage, record),
        artifact_cleanup=failing,
    )
    record = finish_job(manager)
    artifact = record.output_artifact
    manager.cleanup_expired(now=record.completed_at + timedelta(hours=2))
    assert manager.snapshot()["cleanup_pending_jobs"] == 1
    manager.shutdown()
    replacement = ConversionJobManager(root, lambda *_: None, artifact_cleanup=storage.delete)
    try:
        assert replacement.snapshot()["cleanup_pending_jobs"] == 0
        assert storage.exists(artifact)  # Explicit known limit: no discovery of old objects.
    finally:
        replacement.shutdown()
        storage.delete(artifact)


@pytest.mark.parametrize("endpoint", ["/v1/jobs", "/v1/jobs/batch"])
def test_api_expired_result_stays_inaccessible_and_cleanup_backpressure_is_429(
    tmp_path,
    monkeypatch,
    storage,
    endpoint,
):
    import io

    from fastapi.testclient import TestClient
    from PIL import Image

    import services.api.app.main as api

    failing = True
    attempts = []

    def cleanup(artifact):
        attempts.append(artifact)
        if failing:
            raise OSError("PRIVATE adapter exception")
        storage.delete(artifact)

    manager = ConversionJobManager(
        tmp_path / "api-jobs",
        lambda record, _: publish(storage, record),
        max_workers=1,
        max_active_jobs=1,
        max_retained_jobs=1,
        artifact_exists=storage.exists,
        artifact_cleanup=cleanup,
    )
    monkeypatch.setattr(api, "JOB_MANAGER", manager)
    monkeypatch.setattr(api, "RESULT_STORAGE", storage)
    monkeypatch.setattr(api, "_enforce_submission_rate_limit", lambda *a, **kw: None)
    client = TestClient(api.app)
    encoded = io.BytesIO()
    Image.new("RGB", (10, 10), "white").save(encoded, format="PNG")
    upload = ("image.png", encoded.getvalue(), "image/png")
    try:
        created = client.post("/v1/jobs", files={"file": upload})
        assert created.status_code == 202
        record = manager.get_record(created.json()["id"])
        record.future.result(timeout=5)
        artifact = record.output_artifact
        expiry = record.completed_at + timedelta(hours=2)
        assert manager.cleanup_expired(now=expiry) == 1
        assert client.get(f"/v1/jobs/{record.id}").status_code == 404
        assert client.get(f"/v1/jobs/{record.id}/download").status_code == 404
        assert client.delete(f"/v1/jobs/{record.id}").status_code == 404
        assert len(attempts) == 1
        field = "files" if endpoint.endswith("batch") else "file"
        rejected = client.post(endpoint, files={field: upload})
        assert rejected.status_code == 429
        assert "Retained job capacity" in rejected.json()["detail"]
        assert not list(manager.root_dir.iterdir())
        assert storage.exists(artifact)
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "lao_ocr_cleanup_pending_jobs 1" in metrics.text
        assert artifact.key not in metrics.text and record.id not in metrics.text
        failing = False
        assert manager.cleanup_expired(now=expiry + timedelta(seconds=30)) == 0
        assert not storage.exists(artifact)
        accepted = client.post(endpoint, files={field: upload})
        assert accepted.status_code == 202
        payload = accepted.json()
        new_id = payload["jobs"][0]["id"] if field == "files" else payload["id"]
        assert new_id != record.id
        manager.get_record(new_id).future.result(timeout=5)
    finally:
        manager.shutdown()


def test_workspace_symlink_does_not_delete_external_content(tmp_path):
    outside = tmp_path / "unrelated"
    outside.mkdir()
    untouched = outside / "keep.txt"
    untouched.write_text("unrelated fixture")
    manager = ConversionJobManager(tmp_path / "jobs", lambda record, _: record.input_path)
    try:
        record = finish_job(manager)
        jobs.shutil.rmtree(record.workspace)
        record.workspace.symlink_to(outside, target_is_directory=True)
        expiry = record.completed_at + timedelta(hours=2)
        assert manager.cleanup_expired(now=expiry) == 1
        assert manager.snapshot()["cleanup_pending_jobs"] == 1
        assert untouched.read_text() == "unrelated fixture"
        record.workspace.unlink()
        manager.cleanup_expired(now=expiry + timedelta(seconds=30))
        assert manager.snapshot()["cleanup_pending_jobs"] == 0
        assert untouched.read_text() == "unrelated fixture"
    finally:
        manager.shutdown()


def test_storage_commit_then_error_is_retried_idempotently(tmp_path, storage):
    attempts = []

    def cleanup(artifact):
        attempts.append(artifact)
        storage.delete(artifact)
        if len(attempts) == 1:
            raise OSError("PRIVATE response lost after deletion")

    manager = ConversionJobManager(
        tmp_path / "jobs",
        lambda record, _: publish(storage, record),
        artifact_cleanup=cleanup,
    )
    try:
        record = finish_job(manager)
        artifact = record.output_artifact
        expiry = record.completed_at + timedelta(hours=2)
        assert manager.cleanup_expired(now=expiry) == 1
        assert not storage.exists(artifact)
        assert manager.snapshot()["cleanup_pending_artifacts"] == 1
        manager.cleanup_expired(now=expiry + timedelta(seconds=30))
        assert attempts == [artifact, artifact]
        assert manager.snapshot()["cleanup_pending_jobs"] == 0
    finally:
        manager.shutdown()


def test_configuration_and_aggregate_metrics_reject_untrusted_values():
    from pathlib import Path

    from services.api.app import main as api

    assert api.JOB_MANAGER.max_retained_jobs == api.JOB_MAX_RETAINED
    repo = Path(__file__).resolve().parents[1]
    assert (
        'JOB_MAX_RETAINED: "${JOB_MAX_RETAINED:-1024}"' in (repo / "docker-compose.yml").read_text()
    )
    for preset in (repo / "deploy/presets").glob("*.env.example"):
        assert "JOB_MAX_RETAINED=1024" in preset.read_text()
    for invalid in [True, -1, "PRIVATE", float("nan"), {}, None]:
        rendered = ApiMetrics().render_prometheus({"cleanup_pending_jobs": invalid})
        assert "lao_ocr_cleanup_pending_jobs" not in rendered
        assert "PRIVATE" not in rendered


def test_inaccessible_root_after_missing_child_does_not_lose_retry_claim(tmp_path, monkeypatch):
    from pathlib import Path

    manager = ConversionJobManager(tmp_path / "jobs", lambda record, _: record.input_path)
    try:
        record = finish_job(manager)
        original_remove, original_exists = jobs.shutil.rmtree, Path.exists

        def missing_child(*args, **kwargs):
            raise FileNotFoundError("PRIVATE child disappeared")

        def denied(path):
            if path == record.workspace:
                raise PermissionError("PRIVATE root cannot be inspected")
            return original_exists(path)

        monkeypatch.setattr(jobs.shutil, "rmtree", missing_child)
        monkeypatch.setattr(Path, "exists", denied)
        expiry = record.completed_at + timedelta(hours=2)
        assert manager.cleanup_expired(now=expiry) == 1
        assert not manager._pending_cleanup[record.id].in_progress
        assert manager._pending_cleanup[record.id].workspace_pending
        monkeypatch.setattr(jobs.shutil, "rmtree", original_remove)
        monkeypatch.setattr(Path, "exists", original_exists)
        assert manager.cleanup_expired(now=expiry + timedelta(seconds=30)) == 0
        assert not manager._pending_cleanup
        assert not record.workspace.exists()
    finally:
        manager.shutdown()
