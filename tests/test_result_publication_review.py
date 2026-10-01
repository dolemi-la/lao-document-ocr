"""Review regressions for publication owner lifetime and deletion receipts."""

from __future__ import annotations

import threading
from datetime import timedelta

import pytest
from test_durable_cleanup import storage as storage
from test_result_publication import completed, enqueue, tracked_manager

from services.api.app.cleanup_journal import CleanupJournalError
from services.api.app.jobs import ConversionJobManager, JobPublicError, JobStatus
from services.api.app.storage import StoredArtifact


def test_only_the_owning_runner_thread_may_publish(tmp_path, storage):
    entered, release = threading.Event(), threading.Event()
    writes = []

    def runner(record, _):
        entered.set()
        assert release.wait(5)
        archive = record.workspace / "result.zip"
        archive.write_bytes(b"authored thread fixture")
        key = f"jobs/{record.id}/result.zip"
        return record.publish_result(
            key,
            lambda: storage.put_file(
                archive,
                key=key,
                filename="result.zip",
                media_type="application/zip",
            ),
        )

    manager = ConversionJobManager(
        tmp_path / "jobs",
        runner,
        durable_cleanup=True,
        cleanup_namespace=storage.cleanup_namespace(),
        artifact_cleanup=storage.delete,
        cleanup_interval_seconds=0,
    )
    try:
        record = enqueue(manager)
        assert entered.wait(5)
        key = f"jobs/{record.id}/result.zip"

        def off_thread_write():
            writes.append("external writer")
            return StoredArtifact(key, "result.zip", "application/zip", 0)

        with pytest.raises(JobPublicError):
            record.publish_result(key, off_thread_write)
        assert writes == []
        rows = manager._journal.load()
        assert len(rows) == 1
        assert rows[0].job_id == record.id
        assert rows[0].artifact_key is None  # No publication, but the workspace is owned.
        release.set()
        record.future.result(timeout=5)
        assert record.status == JobStatus.SUCCEEDED
    finally:
        release.set()
        manager.shutdown()


@pytest.mark.parametrize("committed", [False, True])
def test_ambiguous_cancellation_receipt_replays_only_unconfirmed_object_delete(
    tmp_path,
    storage,
    monkeypatch,
    committed,
):
    def writer(record, put):
        result = put()
        record.cancel_event.set()
        return result

    root = tmp_path / "jobs"
    original = tracked_manager(root, storage, writer=writer)
    update = original._journal.update

    def uncertain(entry):
        if entry.artifact_key is None:
            if committed:
                update(entry)
            raise CleanupJournalError("PRIVATE cancellation receipt")
        update(entry)

    monkeypatch.setattr(original._journal, "update", uncertain)
    try:
        record = enqueue(original)
        with pytest.raises(CleanupJournalError):
            record.future.result(timeout=5)
        assert not storage.exists(record.output_artifact)
        assert record.workspace.exists()
        assert original._journal_failed
    finally:
        original.shutdown()
    calls = []

    def delete(artifact):
        calls.append(artifact.key)
        storage.delete(artifact)

    replacement = tracked_manager(root, storage, artifact_cleanup=delete)
    try:
        pending = replacement._pending_cleanup[record.id]
        assert (pending.artifact is None) is committed
        replacement.cleanup_expired(now=record.completed_at + timedelta(hours=2))
        assert len(calls) == int(not committed)
        assert not record.workspace.exists()
        assert not replacement._journal.load()
    finally:
        replacement.shutdown()


def test_discard_cannot_drop_tracked_publication(tmp_path, storage):
    manager = tracked_manager(tmp_path / "jobs", storage)
    try:
        record = completed(manager)
        with pytest.raises(RuntimeError, match="durable publication"):
            manager.discard(record.id)
        assert record.id in manager._jobs
        assert record.workspace.exists()
        assert len(manager._journal.load()) == 1
    finally:
        manager.shutdown()


def test_unexpired_download_keeps_exclusive_owner_until_release(tmp_path, storage):
    root = tmp_path / "jobs"
    original = tracked_manager(root, storage)
    lease = None
    try:
        record = completed(original)
        deadline = record.completed_at + timedelta(hours=1)
        lease = original.acquire_download(record.id)
        original.shutdown()
        with pytest.raises(CleanupJournalError):
            tracked_manager(root, storage)
        assert storage.exists(record.output_artifact)
        lease.close()
        replacement = tracked_manager(root, storage)
        try:
            assert replacement._pending_cleanup[record.id].retry_at == deadline
            replacement.cleanup_expired(now=deadline)
            assert not storage.exists(record.output_artifact)
        finally:
            replacement.shutdown()
    finally:
        if lease is not None:
            lease.close()
        original.shutdown()
