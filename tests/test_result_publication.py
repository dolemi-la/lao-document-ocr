"""Tracked result publication owns bytes before provider I/O, not just at expiry."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest
from test_durable_cleanup import storage as storage

from services.api.app.cleanup_journal import CleanupJournalError
from services.api.app.jobs import (
    ConversionJobManager,
    JobNotFoundError,
    JobPublicError,
    JobStatus,
)
from services.api.app.storage import StoredArtifact


def tracked_manager(root, storage, *, writer=None, **kwargs):
    def runner(record, _):
        archive = record.workspace / "result.zip"
        archive.write_bytes(b"authored publication fixture")
        key = f"jobs/{record.id}/result.zip"

        def put():
            return storage.put_file(
                archive,
                key=key,
                filename="result.zip",
                media_type="application/zip",
            )

        return record.publish_result(key, lambda: writer(record, put) if writer else put())

    return ConversionJobManager(
        root,
        runner,
        durable_cleanup=True,
        cleanup_namespace=storage.cleanup_namespace(),
        artifact_exists=storage.exists,
        artifact_cleanup=kwargs.pop("artifact_cleanup", storage.delete),
        cleanup_interval_seconds=0,
        **kwargs,
    )


def enqueue(manager):
    record = manager.reserve("private-source.png", ".png")
    manager.enqueue(record.id)
    return record


def completed(manager):
    record = enqueue(manager)
    record.future.result(timeout=5)
    assert record.status == JobStatus.SUCCEEDED
    return record


def test_unexpired_publication_recovers_cleanup_at_original_retention(tmp_path, storage):
    root = tmp_path / "jobs"
    original = tracked_manager(root, storage)
    try:
        record = completed(original)
        artifact = record.output_artifact
        deadline = record.completed_at + timedelta(seconds=original.retention_seconds)
        rows = original._journal.load()
        assert len(rows) == 1
        assert rows[0].artifact_key == artifact.key
        assert rows[0].retry_at == deadline
        assert original.snapshot()["retained_jobs"] == 1
        assert original.snapshot()["cleanup_pending_jobs"] == 0
        assert original.public(record.id)["download_ready"] is True
    finally:
        original.shutdown()

    # A shorter new retention setting must not shorten the persisted deadline.
    replacement = tracked_manager(root, storage, retention_seconds=0)
    try:
        assert replacement.snapshot()["cleanup_pending_jobs"] == 1
        assert replacement.snapshot()["completed_total"] == {}
        for operation in (replacement.public, replacement.cancel, replacement.acquire_download):
            with pytest.raises(JobNotFoundError):
                operation(record.id)
        assert replacement.cleanup_expired(now=deadline - timedelta(microseconds=1)) == 0
        assert storage.exists(artifact)
        replacement.cleanup_expired(now=deadline)
        assert not storage.exists(artifact)
        assert not record.workspace.exists()
        assert not replacement._journal.load()
    finally:
        replacement.shutdown()


def test_intent_exists_before_write_and_slow_publication_does_not_lock_jobs(tmp_path, storage):
    entered, release = threading.Event(), threading.Event()
    owner = []

    def writer(record, put):
        rows = owner[0]._journal.load()
        assert len(rows) == 1
        assert rows[0].job_id == record.id
        assert rows[0].artifact_key == f"jobs/{record.id}/result.zip"
        entered.set()
        assert release.wait(5)
        return put()

    manager = tracked_manager(tmp_path / "jobs", storage, writer=writer, retention_seconds=0)
    owner.append(manager)
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        record = enqueue(manager)
        assert entered.wait(5)
        other = pool.submit(manager.reserve, "other.png", ".png").result(timeout=1)
        assert other.status == JobStatus.UPLOADING
        assert pool.submit(manager.cleanup_expired).result(timeout=1) == 0
        assert record.workspace.exists()
        assert record.status == JobStatus.RUNNING
        release.set()
        record.future.result(timeout=5)
    finally:
        release.set()
        pool.shutdown(wait=True)
        manager.shutdown()


@pytest.mark.parametrize("committed", [False, True])
def test_uncertain_intent_commit_never_invokes_storage(
    tmp_path, storage, monkeypatch, caplog, committed
):
    writes = []
    root = tmp_path / "jobs"
    manager = tracked_manager(root, storage, writer=lambda record, put: writes.append(record.id))
    add = manager._journal.add

    def uncertain(entries):
        if committed:
            add(entries)
        raise CleanupJournalError("PRIVATE journal path")

    monkeypatch.setattr(manager._journal, "add", uncertain)
    try:
        record = enqueue(manager)
        with pytest.raises(CleanupJournalError):
            record.future.result(timeout=5)
        assert writes == []
        assert manager._journal_failed
        assert "PRIVATE" not in caplog.text
        with pytest.raises(CleanupJournalError):
            manager.public(record.id)
    finally:
        manager.shutdown()
    replacement = tracked_manager(root, storage)
    try:
        assert len(replacement._pending_cleanup) == int(committed)
    finally:
        replacement.shutdown()


def test_lost_storage_response_retains_declared_key_without_private_error(
    tmp_path, storage, caplog
):
    outputs = []

    def writer(record, put):
        outputs.append(put())
        raise OSError("PRIVATE provider credentials and body")

    root = tmp_path / "jobs"
    original = tracked_manager(root, storage, writer=writer)
    try:
        record = enqueue(original)
        record.future.result(timeout=5)
        assert record.status == JobStatus.FAILED
        public = original.public(record.id)
        assert public["download_ready"] is False
        assert "PRIVATE" not in json.dumps(public)
        assert "PRIVATE" not in caplog.text
        assert original._journal.load()[0].artifact_key == outputs[0].key
        deadline = record.completed_at + timedelta(hours=1)
    finally:
        original.shutdown()
    replacement = tracked_manager(root, storage)
    try:
        replacement.cleanup_expired(now=deadline)
        assert not storage.exists(outputs[0])
        assert not record.workspace.exists()
        assert not replacement._journal.load()
    finally:
        replacement.shutdown()


@pytest.mark.parametrize("committed", [False, True])
def test_terminal_commit_failure_preserves_prepublication_owner(
    tmp_path, storage, monkeypatch, caplog, committed
):
    entered, release = threading.Event(), threading.Event()
    written = []

    def writer(record, put):
        result = put()
        written.append(result)
        entered.set()
        assert release.wait(5)
        return result

    root = tmp_path / "jobs"
    original = tracked_manager(root, storage, writer=writer)
    try:
        record = enqueue(original)
        assert entered.wait(5)
        update = original._journal.update

        def uncertain(entry):
            if committed:
                update(entry)
            raise CleanupJournalError("PRIVATE terminal SQL")

        monkeypatch.setattr(original._journal, "update", uncertain)
        release.set()
        with pytest.raises(CleanupJournalError):
            record.future.result(timeout=5)
        with pytest.raises(CleanupJournalError):
            original.public(record.id)
        assert "PRIVATE" not in caplog.text
    finally:
        release.set()
        original.shutdown()
    replacement = tracked_manager(root, storage)
    try:
        assert len(replacement._pending_cleanup) == 1
        replacement.cleanup_expired(now=record.completed_at + timedelta(hours=2))
        assert not storage.exists(written[0])
        assert not record.workspace.exists()
    finally:
        replacement.shutdown()


def test_normal_expiry_transfers_existing_row_without_duplicate_ownership(tmp_path, storage):
    manager = tracked_manager(tmp_path / "jobs", storage)
    try:
        record = completed(manager)
        assert len(manager._journal.load()) == 1
        assert manager.cleanup_expired(now=record.completed_at + timedelta(hours=2)) == 1
        assert not manager._pending_cleanup
        assert not manager._journal.load()
        assert not storage.exists(record.output_artifact)
        assert manager.snapshot()["completed_total"] == {"succeeded": 1}
    finally:
        manager.shutdown()


def test_failed_cancellation_delete_keeps_intent_across_restart(tmp_path, storage):
    entered, release = threading.Event(), threading.Event()
    written = []

    def writer(record, put):
        written.append(put())
        entered.set()
        assert release.wait(5)
        return written[-1]

    def fail_delete(_):
        raise OSError("PRIVATE cleanup failure")

    root = tmp_path / "jobs"
    original = tracked_manager(root, storage, writer=writer, artifact_cleanup=fail_delete)
    try:
        record = enqueue(original)
        assert entered.wait(5)
        original.cancel(record.id)
        release.set()
        record.future.result(timeout=5)
        assert record.status == JobStatus.CANCELLED
        assert original._journal.load()[0].artifact_key == written[0].key
        assert original.public(record.id)["download_ready"] is False
    finally:
        release.set()
        original.shutdown()
    replacement = tracked_manager(root, storage)
    try:
        replacement.cleanup_expired(now=record.completed_at + timedelta(hours=2))
        assert not storage.exists(written[0])
        assert not record.workspace.exists()
    finally:
        replacement.shutdown()


def test_successful_cancellation_delete_retains_workspace_only(tmp_path, storage):
    def writer(record, put):
        result = put()
        record.cancel_event.set()
        return result

    manager = tracked_manager(tmp_path / "jobs", storage, writer=writer)
    try:
        record = enqueue(manager)
        record.future.result(timeout=5)
        assert record.status == JobStatus.CANCELLED
        rows = manager._journal.load()
        assert len(rows) == 1
        assert rows[0].artifact_key is None
        assert rows[0].workspace_pending
        assert record.output_artifact is None
        manager.cleanup_expired(now=record.completed_at + timedelta(hours=2))
        assert not manager._journal.load()
    finally:
        manager.shutdown()


def test_repeated_publisher_is_refused_before_second_write(tmp_path, storage):
    writes = []

    def writer(record, put):
        result = put()
        with pytest.raises(JobPublicError):
            record.publish_result(
                f"jobs/{record.id}/different.zip",
                lambda: writes.append("second"),
            )
        return result

    manager = tracked_manager(tmp_path / "jobs", storage, writer=writer)
    try:
        completed(manager)
        assert writes == []
        assert len(manager._journal.load()) == 1
    finally:
        manager.shutdown()


def test_adapter_return_key_mismatch_fails_without_forgetting_declared_key(tmp_path, storage):
    def writer(record, put):
        put()
        return StoredArtifact("different/key.zip", "result.zip", "application/zip", 1)

    manager = tracked_manager(tmp_path / "jobs", storage, writer=writer)
    try:
        record = enqueue(manager)
        record.future.result(timeout=5)
        assert record.status == JobStatus.FAILED
        rows = manager._journal.load()
        assert rows[0].artifact_key == f"jobs/{record.id}/result.zip"
        manager.cleanup_expired(now=record.completed_at + timedelta(hours=2))
        assert not manager._journal.load()
    finally:
        manager.shutdown()
