"""Review boundaries: descriptor ownership, bounded locks, and journal compatibility."""

from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest
from test_filesystem_staging import artifact, put, stage_for, storage_at
from test_filesystem_staging_recovery import manager_for

from services.api.app.filesystem_writes import LOCK_SLOTS, FilesystemWriteError
from services.api.app.storage import FilesystemArtifactStorage


def test_failed_stage_fstat_closes_descriptor_without_inventing_a_deletion_receipt(
    tmp_path,
    monkeypatch,
):
    storage = storage_at(tmp_path / "objects")
    source = tmp_path / "source.zip"
    source.write_bytes(b"authored bytes")
    opened, closed = [], []
    open_file, close, fstat = os.open, os.close, os.fstat

    def watched_open(path, flags, *args, **kwargs):
        descriptor = open_file(path, flags, *args, **kwargs)
        if flags & os.O_EXCL:
            opened.append(descriptor)
        return descriptor

    def unavailable_stat(descriptor):
        if descriptor in opened:
            raise OSError("PRIVATE descriptor metadata failure")
        return fstat(descriptor)

    def watched_close(descriptor):
        closed.append(descriptor)
        close(descriptor)

    monkeypatch.setattr(os, "open", watched_open)
    monkeypatch.setattr(os, "fstat", unavailable_stat)
    monkeypatch.setattr(os, "close", watched_close)
    try:
        with pytest.raises(FilesystemWriteError) as error:
            put(storage, source)
        assert "PRIVATE" not in str(error.value)
        assert len(opened) == 1
        assert opened[0] in closed
        assert stage_for(storage).exists()  # Unknown metadata remains a cleanup obligation.
    finally:
        # Also clean up the old implementation's leaked descriptor in the red run.
        for descriptor in opened:
            if descriptor not in closed:
                close(descriptor)
    monkeypatch.setattr(os, "fstat", fstat)
    storage.delete(artifact())
    assert not stage_for(storage).exists()


def test_recoverable_mode_refuses_missing_lock_support_without_creating_root(tmp_path, monkeypatch):
    root = tmp_path / "objects"
    monkeypatch.setitem(sys.modules, "fcntl", None)
    with pytest.raises(FilesystemWriteError, match="POSIX"):
        storage_at(root)
    assert not root.exists()
    default = FilesystemArtifactStorage(root)
    assert default.recoverable_writes is False


@pytest.mark.parametrize("invalid", [1, 0, "true", None])
def test_recoverable_option_requires_an_actual_boolean(tmp_path, invalid):
    with pytest.raises(ValueError, match="boolean"):
        FilesystemArtifactStorage(tmp_path / "objects", recoverable_writes=invalid)
    assert not (tmp_path / "objects").exists()


def test_lock_inode_pool_is_bounded_and_stable_after_deletion(tmp_path):
    storage = storage_at(tmp_path / "objects")
    source = tmp_path / "source.zip"
    source.write_bytes(b"authored bytes")
    identities = {}
    for i in range(130):
        key = f"jobs/{i:032x}/result.zip"
        result = put(storage, source, key)
        lock = storage._recoverable.lock_path(storage._path(key))
        info = lock.stat()
        identity = (info.st_dev, info.st_ino)
        if lock in identities:
            assert identities[lock] == identity
        identities[lock] = identity
        storage.delete(result)
        assert lock.exists()
        assert lock.stat().st_size == 0
        assert lock.stat().st_mode & 0o777 == 0o600
    assert 0 < len(identities) <= LOCK_SLOTS
    assert set(storage._recoverable.directory.iterdir()) == set(identities)


def test_same_key_writers_wait_instead_of_reusing_an_active_stage(tmp_path, monkeypatch):
    import shutil

    storage, second = storage_at(tmp_path / "objects"), storage_at(tmp_path / "objects")
    first_source, second_source = tmp_path / "one.zip", tmp_path / "two.zip"
    first_source.write_bytes(b"first bytes")
    second_source.write_bytes(b"second bytes")
    entered, release, waiting = (threading.Event() for _ in range(3))
    writer = []
    copy, flock = shutil.copyfileobj, storage._recoverable._fcntl.flock

    def paused(src, dst, *args, **kwargs):
        if not writer:
            writer.append(threading.get_ident())
            entered.set()
            assert release.wait(5)
        return copy(src, dst, *args, **kwargs)

    def locking(descriptor, operation):
        if writer and threading.get_ident() != writer[0]:
            waiting.set()
        return flock(descriptor, operation)

    monkeypatch.setattr(shutil, "copyfileobj", paused)
    monkeypatch.setattr(storage._recoverable._fcntl, "flock", locking)
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        one = pool.submit(put, storage, first_source)
        assert entered.wait(5)
        two = pool.submit(put, second, second_source)
        assert waiting.wait(5)
        assert not two.done()
        release.set()
        assert one.result(timeout=5).size_bytes == len(b"first bytes")
        assert two.result(timeout=5).size_bytes == len(b"second bytes")
        assert b"".join(storage.iter_bytes(artifact())) == b"second bytes"
        assert not stage_for(storage).exists()
    finally:
        release.set()
        pool.shutdown(wait=True)


def test_existing_v1_journal_remains_bound_and_recoverable_after_storage_upgrade(tmp_path):
    legacy = FilesystemArtifactStorage(tmp_path / "objects")

    def runner(record, _):
        source = record.workspace / "result.zip"
        source.write_bytes(b"legacy final result")
        key = f"jobs/{record.id}/result.zip"
        return record.publish_result(key, lambda: put(legacy, source, key))

    original = manager_for(tmp_path / "jobs", legacy, runner)
    try:
        record = original.reserve("fixture.png", ".png")
        original.enqueue(record.id)
        record.future.result(timeout=5)
        deadline = record.completed_at + timedelta(hours=1)
    finally:
        original.shutdown()
    upgraded = storage_at(legacy.root)
    assert upgraded.cleanup_namespace() == legacy.cleanup_namespace()
    recovered = manager_for(tmp_path / "jobs", upgraded)
    try:
        recovered.cleanup_expired(now=deadline)
        assert not upgraded.exists(record.output_artifact)
        assert not recovered._journal.load()
    finally:
        recovered.shutdown()


def test_manager_download_lease_prevents_new_cleanup_path_from_deleting_live_result(tmp_path):
    storage = storage_at(tmp_path / "objects")

    def runner(record, _):
        source = record.workspace / "result.zip"
        source.write_bytes(b"leased result")
        key = f"jobs/{record.id}/result.zip"
        return record.publish_result(key, lambda: put(storage, source, key))

    owner = manager_for(tmp_path / "jobs", storage, runner)
    lease = None
    try:
        record = owner.reserve("fixture.png", ".png")
        owner.enqueue(record.id)
        record.future.result(timeout=5)
        lease = owner.acquire_download(record.id)
        expiry = record.completed_at + timedelta(hours=2)
        owner.cleanup_expired(now=expiry)
        assert storage.exists(record.output_artifact)
        assert record.workspace.exists()
        lease.close()
        owner.cleanup_expired(now=expiry)
        assert not storage.exists(record.output_artifact)
        assert not owner._journal.load()
    finally:
        if lease is not None:
            lease.close()
        owner.shutdown()
