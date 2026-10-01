"""Opt-in recovery owns expired cleanup, never old public jobs or new OCR work."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest
from test_storage import FakeS3Client

from services.api.app.jobs import ConversionJobManager, JobCapacityError, JobNotFoundError
from services.api.app.storage import FilesystemArtifactStorage, S3ArtifactStorage


@pytest.fixture(params=["filesystem", "s3-double"])
def storage(tmp_path, request):
    if request.param == "filesystem":
        return FilesystemArtifactStorage(tmp_path / "objects")
    return S3ArtifactStorage(
        "fixture-bucket",
        prefix="fixture",
        endpoint_url="https://storage.invalid",
        client=FakeS3Client(),
    )


def publish(storage, record):
    archive = record.workspace / "result.zip"
    archive.write_bytes(b"authored fixture archive")
    return storage.put_file(
        archive,
        key=f"jobs/{record.id}/result.zip",
        filename="result.zip",
        media_type="application/zip",
    )


def manager_for(root, storage, **kwargs):
    return ConversionJobManager(
        root,
        lambda record, _: publish(storage, record),
        durable_cleanup=True,
        cleanup_namespace=storage.cleanup_namespace(),
        artifact_exists=storage.exists,
        artifact_cleanup=kwargs.pop("artifact_cleanup", storage.delete),
        cleanup_interval_seconds=0,
        **kwargs,
    )


def finish(manager):
    record = manager.reserve("private-source.png", ".png")
    manager.enqueue(record.id)
    record.future.result(timeout=5)
    assert record.status == "succeeded"
    return record


def fail_delete(_):
    raise OSError("PRIVATE provider failure")


def test_failed_expiry_recovers_backoff_and_not_public_access(tmp_path, storage):
    root = tmp_path / "jobs"
    original = manager_for(root, storage, artifact_cleanup=fail_delete)
    try:
        record = finish(original)
        artifact = record.output_artifact
        expiry = record.completed_at + timedelta(hours=2)
        assert original.cleanup_expired(now=expiry) == 1
        assert storage.exists(artifact)
        assert not record.workspace.exists()
    finally:
        original.shutdown()

    calls = []

    def delete(artifact):
        calls.append(artifact.key)
        storage.delete(artifact)

    replacement = manager_for(root, storage, artifact_cleanup=delete)
    try:
        state = replacement.snapshot()
        assert state["retained_jobs"] == state["cleanup_pending_jobs"] == 1
        assert state["completed_total"] == {}
        assert state["cleanup_attempts_total"] == 0
        pending = replacement._pending_cleanup[record.id]
        assert pending.failures == 1
        assert pending.retry_at == expiry + timedelta(seconds=30)
        assert not pending.workspace_pending
        assert not pending.in_progress
        for operation in (replacement.public, replacement.cancel, replacement.acquire_download):
            with pytest.raises(JobNotFoundError):
                operation(record.id)
        assert calls == []
        assert replacement.cleanup_expired(now=expiry + timedelta(seconds=29)) == 0
        assert calls == []
        assert replacement.cleanup_expired(now=expiry + timedelta(seconds=30)) == 0
        assert calls == [artifact.key]
        assert not storage.exists(artifact)
        assert replacement.snapshot()["retained_jobs"] == 0
    finally:
        replacement.shutdown()
    last = manager_for(root, storage)
    try:
        assert last.snapshot()["cleanup_pending_jobs"] == 0
    finally:
        last.shutdown()


def test_persisted_artifact_success_does_not_repeat_after_workspace_failure(
    tmp_path,
    storage,
    monkeypatch,
):
    import services.api.app.jobs as jobs

    root = tmp_path / "jobs"
    original = manager_for(root, storage)
    real_rmtree = jobs.shutil.rmtree
    try:
        record = finish(original)
        artifact = record.output_artifact
        expiry = record.completed_at + timedelta(hours=2)

        def blocked(path, *args, **kwargs):
            if path == record.workspace:
                raise PermissionError("PRIVATE workspace")
            return real_rmtree(path, *args, **kwargs)

        monkeypatch.setattr(jobs.shutil, "rmtree", blocked)
        original.cleanup_expired(now=expiry)
        assert not storage.exists(artifact)
        assert record.workspace.exists()
    finally:
        monkeypatch.setattr(jobs.shutil, "rmtree", real_rmtree)
        original.shutdown()

    calls = []
    replacement = manager_for(
        root, storage, artifact_cleanup=lambda artifact: calls.append(artifact)
    )
    try:
        replacement.cleanup_expired(now=expiry + timedelta(seconds=30))
        assert calls == []
        assert not record.workspace.exists()
        assert replacement.snapshot()["cleanup_pending_jobs"] == 0
    finally:
        replacement.shutdown()


def test_recovered_rows_count_toward_capacity_and_batch_budget(tmp_path, storage):
    root = tmp_path / "jobs"
    original = manager_for(
        root,
        storage,
        artifact_cleanup=fail_delete,
        max_workers=1,
        max_active_jobs=10,
        max_retained_jobs=10,
    )
    try:
        records = [finish(original) for _ in range(10)]
        expiry = records[-1].completed_at + timedelta(hours=2)
        assert original.cleanup_expired(now=expiry) == 10
        assert original.snapshot()["cleanup_pending_jobs"] == 10
    finally:
        original.shutdown()
    replacement = manager_for(
        root,
        storage,
        artifact_cleanup=fail_delete,
        max_workers=1,
        max_active_jobs=10,
        max_retained_jobs=10,
    )
    try:
        with pytest.raises(JobCapacityError):
            replacement.reserve("new.png", ".png")
        calls = []

        def delete(artifact):
            calls.append(artifact.key)
            storage.delete(artifact)

        replacement.artifact_cleanup = delete
        replacement.cleanup_expired(now=expiry + timedelta(hours=2))
        assert len(calls) == 8
        assert len(replacement._pending_cleanup) == 2
        replacement.cleanup_expired(now=expiry + timedelta(hours=2))
        assert len(calls) == 10
        assert not replacement._pending_cleanup
    finally:
        replacement.shutdown()


def test_download_lease_keeps_journal_owned_through_shutdown(tmp_path, storage):
    from services.api.app.cleanup_journal import CleanupJournalError

    root = tmp_path / "jobs"
    original = manager_for(root, storage)
    lease = None
    try:
        record = finish(original)
        artifact = record.output_artifact
        lease = original.acquire_download(record.id)
        expiry = record.completed_at + timedelta(hours=2)
        assert original.cleanup_expired(now=expiry) == 1
        original.shutdown()
        with pytest.raises(CleanupJournalError):
            manager_for(root, storage)
        assert storage.exists(artifact)
        assert record.workspace.exists()
        lease.close()
        lease.close()
        replacement = manager_for(root, storage)
        try:
            replacement.cleanup_expired(now=expiry)
            assert not storage.exists(artifact)
            assert not record.workspace.exists()
        finally:
            replacement.shutdown()
    finally:
        if lease is not None:
            lease.close()
        original.shutdown()


def test_namespace_cannot_change_even_when_ledger_is_empty(tmp_path, storage):
    from services.api.app.cleanup_journal import CleanupJournalError

    root = tmp_path / "jobs"
    original = manager_for(root, storage)
    original.shutdown()
    changed = {**storage.cleanup_namespace(), "different_location": "yes"}
    with pytest.raises(CleanupJournalError, match="namespace"):
        ConversionJobManager(
            root,
            lambda *_: None,
            durable_cleanup=True,
            cleanup_namespace=changed,
        )
    correct = manager_for(root, storage)
    correct.shutdown()


def test_custom_runner_bypassing_tracked_publication_remains_expiry_only(tmp_path, storage):
    # manager_for intentionally calls storage directly, without record.publish_result.
    root = tmp_path / "jobs"
    original = manager_for(root, storage)
    try:
        record = finish(original)
    finally:
        original.shutdown()
    replacement = manager_for(root, storage)
    try:
        assert not replacement._pending_cleanup
        with pytest.raises(JobNotFoundError):
            replacement.public(record.id)
        assert storage.exists(record.output_artifact)
        assert record.workspace.exists()
    finally:
        replacement.shutdown()


def test_slow_cleanup_keeps_exclusive_owner_until_pass_finishes(tmp_path, storage):
    from services.api.app.cleanup_journal import CleanupJournalError

    entered, release = threading.Event(), threading.Event()
    root = tmp_path / "jobs"

    def delete(artifact):
        entered.set()
        assert release.wait(5)
        storage.delete(artifact)

    original = manager_for(root, storage, artifact_cleanup=delete)
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        record = finish(original)
        expiry = record.completed_at + timedelta(hours=2)
        cleanup = pool.submit(original.cleanup_expired, now=expiry)
        assert entered.wait(5)
        original.shutdown()
        with pytest.raises(CleanupJournalError):
            manager_for(root, storage)
        release.set()
        assert cleanup.result(timeout=5) == 1
        replacement = manager_for(root, storage)
        try:
            assert not replacement._pending_cleanup
        finally:
            replacement.shutdown()
    finally:
        release.set()
        pool.shutdown(wait=True)
        original.shutdown()
