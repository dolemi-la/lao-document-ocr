"""Regression checks for caller ownership during rejection and batch allocation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from test_workspace_ownership import manager_for
from test_workspace_ownership import storage as storage

from services.api.app.cleanup_journal import CleanupEntry, CleanupJournalError
from services.api.app.jobs import JobCapacityError, JobSchedulingError


def test_rejected_enqueue_cannot_expire_before_upload_caller_finalizes(
    tmp_path, storage, monkeypatch
):
    manager = manager_for(tmp_path / "jobs", storage, retention_seconds=0)
    record = manager.reserve("private.png", ".png")
    record.input_path.write_bytes(b"authored upload")

    def unavailable(*args, **kwargs):
        raise RuntimeError("PRIVATE executor failure")

    monkeypatch.setattr(manager._executor, "submit", unavailable)
    try:
        with pytest.raises(JobSchedulingError):
            manager.enqueue(record.id)
        assert record.id in manager._upload_owners
        assert manager.cleanup_expired(now=datetime.now(UTC) + timedelta(hours=2)) == 0
        assert record.id in manager._jobs
        assert record.workspace.exists()
        manager.discard(record.id)
        assert not manager._upload_owners
        assert not manager._journal.load()
    finally:
        manager.shutdown()


@pytest.mark.parametrize("collision", ["directory", "symlink", "duplicate-id"])
def test_reservations_do_not_adopt_existing_or_duplicate_paths(
    tmp_path, storage, monkeypatch, collision
):
    import services.api.app.jobs as jobs

    manager = manager_for(tmp_path / "jobs", storage)
    job_id = "a" * 32
    outside = tmp_path / "unrelated"
    outside.mkdir()
    sentinel = outside / "private.txt"
    sentinel.write_bytes(b"must not change")
    candidate = manager.root_dir / job_id
    if collision == "directory":
        candidate.mkdir()
        (candidate / "existing.txt").write_bytes(b"existing directory")
    elif collision == "symlink":
        candidate.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(jobs.uuid, "uuid4", lambda: SimpleNamespace(hex=job_id))
    try:
        with pytest.raises(JobCapacityError):
            manager.reserve_many([("one.png", ".png"), ("two.png", ".png")])
        assert not manager._journal.load()
        assert not manager._jobs
        assert not manager._upload_owners
        assert sentinel.read_bytes() == b"must not change"
        if collision == "directory":
            assert (candidate / "existing.txt").read_bytes() == b"existing directory"
        if collision == "symlink":
            assert candidate.is_symlink()
    finally:
        manager.shutdown()


@pytest.mark.parametrize("bad_owner", ["missing", "already-published", "workspace-done"])
def test_publication_promotion_does_not_recreate_or_replace_an_invalid_workspace_row(
    tmp_path,
    storage,
    bad_owner,
):
    manager = manager_for(tmp_path / "jobs", storage)
    try:
        ledger = manager._journal
        job_id = "b" * 32
        deadline = datetime.now(UTC) + timedelta(hours=1)
        if bad_owner == "already-published":
            ledger.add([CleanupEntry(job_id, f"jobs/{job_id}/first.zip", True, deadline)])
        elif bad_owner == "workspace-done":
            ledger.add([CleanupEntry(job_id, None, False, deadline)])
        before = ledger.load()
        with pytest.raises(CleanupJournalError):
            ledger.publish(CleanupEntry(job_id, f"jobs/{job_id}/result.zip", True, deadline))
        assert ledger.load() == before
    finally:
        manager.shutdown()


def test_failed_batch_registration_releases_unreturned_caller_pins(tmp_path, storage, monkeypatch):
    manager = manager_for(tmp_path / "jobs", storage)

    class FailedRegistration(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if len(self) == 2:
                raise RuntimeError("fixture interrupted registration")

    monkeypatch.setattr(manager, "_jobs", FailedRegistration())
    try:
        with pytest.raises(RuntimeError, match="interrupted registration"):
            manager.reserve_many([(f"{i}.png", ".png") for i in range(3)])
        assert not manager._jobs
        assert not manager._upload_owners
        assert len(manager._pending_cleanup) == 3
        assert len(manager._journal.load()) == 3
        manager.cleanup_expired(now=datetime.now(UTC) + timedelta(hours=2))
        assert not manager._journal.load()
    finally:
        manager.shutdown()


def test_shutdown_after_queued_terminal_commit_failure_still_releases_executor(tmp_path, storage):
    import threading

    entered, release = threading.Event(), threading.Event()
    ran = []

    def runner(record, _):
        ran.append(record.id)
        entered.set()
        assert release.wait(10)
        return record.input_path

    root = tmp_path / "jobs"
    manager = manager_for(root, storage, runner=runner, max_workers=1)
    records = []
    try:
        for index in range(3):
            record = manager.reserve(f"{index}.png", ".png")
            record.input_path.write_bytes(b"authored upload")
            records.append(record)
            manager.enqueue(record.id)
            if index == 0:
                assert entered.wait(5)

        def fail(*args, **kwargs):
            raise CleanupJournalError("PRIVATE shutdown journal error")

        manager._journal.update = fail
        manager.shutdown(wait=False)
        assert all(record.future.cancelled() for record in records[1:])
        release.set()
        with pytest.raises(CleanupJournalError):
            records[0].future.result(timeout=5)
        manager.shutdown()
        assert ran == [records[0].id]
        assert not manager._upload_owners
        recovered = manager_for(root, storage)
        try:
            assert len(recovered._journal.load()) == 3
            recovered.cleanup_expired(now=datetime.now(UTC) + timedelta(hours=2))
            assert not recovered._journal.load()
        finally:
            recovered.shutdown()
    finally:
        release.set()
        manager.shutdown()
