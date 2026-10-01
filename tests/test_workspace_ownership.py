"""Durable workspace ownership starts before allocation, not after OCR or storage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from test_durable_cleanup import storage as storage

from services.api.app.cleanup_journal import CleanupJournalError
from services.api.app.jobs import (
    ConversionJobManager,
    JobManagerClosedError,
    JobNotFoundError,
    JobStatus,
)


def manager_for(root, storage, *, runner=None, **kwargs):
    return ConversionJobManager(
        root,
        runner or (lambda record, _: record.input_path),
        durable_cleanup=True,
        cleanup_namespace=storage.cleanup_namespace(),
        artifact_exists=storage.exists,
        artifact_cleanup=storage.delete,
        cleanup_interval_seconds=0,
        **kwargs,
    )


def test_entire_batch_is_journaled_before_any_workspace_is_created(tmp_path, storage, monkeypatch):
    manager = manager_for(tmp_path / "jobs", storage)
    mkdir = Path.mkdir
    observed = []

    def checked_mkdir(path, *args, **kwargs):
        if path.parent == manager.root_dir:
            rows = manager._journal.load()
            assert len(rows) == 3
            row = next(row for row in rows if row.job_id == path.name)
            assert row.artifact_key is None
            assert row.workspace_pending
            assert row.failures == 0
            observed.append(path.name)
        return mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", checked_mkdir)
    try:
        records = manager.reserve_many([(f"private-{i}.png", ".png") for i in range(3)])
        assert observed == [record.id for record in records]
        assert manager._upload_owners == set(observed)
        assert manager.snapshot()["retained_jobs"] == 3
        assert manager.cleanup_expired(now=datetime.now(UTC) + timedelta(hours=2)) == 0
        assert all(record.workspace.exists() for record in records)
        manager.discard_many([record.id for record in records])
        assert not manager._upload_owners
        assert not manager._journal.load()
    finally:
        manager.shutdown()


@pytest.mark.parametrize("committed", [False, True])
def test_uncertain_reservation_commit_creates_no_directory_or_upload_owner(
    tmp_path,
    storage,
    monkeypatch,
    caplog,
    committed,
):
    root = tmp_path / "jobs"
    manager = manager_for(root, storage)
    add = manager._journal.add

    def uncertain(entries):
        if committed:
            add(entries)
        raise CleanupJournalError("PRIVATE SQL and path")

    monkeypatch.setattr(manager._journal, "add", uncertain)
    try:
        with pytest.raises(CleanupJournalError):
            manager.reserve_many([("one.png", ".png"), ("two.png", ".png")])
        assert not manager._jobs
        assert not manager._upload_owners
        assert not [path for path in root.iterdir() if path.is_dir()]
        assert "PRIVATE" not in caplog.text
    finally:
        manager.shutdown()
    recovered = manager_for(root, storage)
    try:
        assert len(recovered._pending_cleanup) == (2 if committed else 0)
        recovered.cleanup_expired(now=datetime.now(UTC) + timedelta(hours=2))
        assert not recovered._journal.load()
    finally:
        recovered.shutdown()


def test_partial_batch_allocation_keeps_all_cleanup_rows_but_no_public_jobs(
    tmp_path,
    storage,
    monkeypatch,
):
    root = tmp_path / "jobs"
    manager = manager_for(root, storage)
    sibling = root / "unrelated.txt"
    sibling.write_bytes(b"must remain")
    mkdir = Path.mkdir
    attempts = []

    def failed_mkdir(path, *args, **kwargs):
        if path.parent == root:
            attempts.append(path)
            if len(attempts) == 2:
                raise OSError("fixture allocation failure")
        return mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", failed_mkdir)
    try:
        with pytest.raises(OSError):
            manager.reserve_many([(f"{i}.png", ".png") for i in range(3)])
        assert not manager._jobs
        assert not manager._upload_owners
        assert len(manager._journal.load()) == 3
        assert len(manager._pending_cleanup) == 3
        assert attempts[0].exists()
    finally:
        monkeypatch.setattr(Path, "mkdir", mkdir)
        manager.shutdown()
    recovered = manager_for(root, storage)
    try:
        recovered.cleanup_expired(now=datetime.now(UTC) + timedelta(hours=2))
        assert not recovered._journal.load()
        assert not attempts[0].exists()
        assert sibling.read_bytes() == b"must remain"
    finally:
        recovered.shutdown()


def test_failed_aborted_upload_cleanup_is_recovered_without_public_access(
    tmp_path,
    storage,
    monkeypatch,
    caplog,
):
    import services.api.app.jobs as jobs

    root = tmp_path / "jobs"
    original = manager_for(root, storage)
    record = original.reserve("private.png", ".png")
    record.input_path.write_bytes(b"partial authored upload")
    rmtree = jobs.shutil.rmtree

    def failed(path, *args, **kwargs):
        if path == record.workspace:
            raise PermissionError("PRIVATE workspace path")
        return rmtree(path, *args, **kwargs)

    monkeypatch.setattr(jobs.shutil, "rmtree", failed)
    try:
        original.discard(record.id)
        assert record.id not in original._jobs
        assert not original._upload_owners
        assert record.workspace.exists()
        rows = original._journal.load()
        assert len(rows) == 1
        assert rows[0].workspace_pending
        assert rows[0].failures == 1
        deadline = rows[0].retry_at
        assert "PRIVATE" not in caplog.text
    finally:
        monkeypatch.setattr(jobs.shutil, "rmtree", rmtree)
        original.shutdown()
    recovered = manager_for(root, storage)
    try:
        with pytest.raises(JobNotFoundError):
            recovered.public(record.id)
        recovered.cleanup_expired(now=deadline)
        assert not record.workspace.exists()
        assert not recovered._journal.load()
    finally:
        recovered.shutdown()


def test_shutdown_retains_exclusive_owner_until_upload_caller_aborts(tmp_path, storage):
    root = tmp_path / "jobs"
    original = manager_for(root, storage)
    record = original.reserve("private.png", ".png")
    try:
        with record.input_path.open("wb") as output:
            output.write(b"first chunk")
            original.shutdown()
            with pytest.raises(CleanupJournalError):
                manager_for(root, storage)
            output.write(b"last chunk")
        with pytest.raises(JobManagerClosedError):
            original.enqueue(record.id)
        original.discard(record.id)
        assert not original._upload_owners
        recovered = manager_for(root, storage)
        try:
            assert record.id in recovered._pending_cleanup
            recovered.cleanup_expired(now=datetime.now(UTC) + timedelta(hours=2))
            assert not record.workspace.exists()
        finally:
            recovered.shutdown()
    finally:
        original.shutdown()


def test_upload_cancellation_waits_for_caller_handoff_and_persists_deadline(tmp_path, storage):
    manager = manager_for(tmp_path / "jobs", storage)
    try:
        record = manager.reserve("private.png", ".png")
        manager.cancel(record.id)
        assert record.status == JobStatus.UPLOADING
        assert record.id in manager._upload_owners
        manager.enqueue(record.id)
        assert record.status == JobStatus.CANCELLED
        assert not manager._upload_owners
        row = manager._journal.load()[0]
        assert row.artifact_key is None
        assert row.retry_at == record.completed_at + timedelta(seconds=manager.retention_seconds)
        manager.cleanup_expired(now=row.retry_at)
        assert not manager._journal.load()
    finally:
        manager.shutdown()
