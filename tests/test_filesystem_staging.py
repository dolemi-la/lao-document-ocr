"""Exact-key staging cleanup, not a glob/age-based orphan collector."""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from services.api.app.storage import FilesystemArtifactStorage, StoredArtifact

KEY = "jobs/" + "a" * 32 + "/result.zip"


def storage_at(root):
    return FilesystemArtifactStorage(root, recoverable_writes=True)


def artifact(key=KEY):
    return StoredArtifact(key, "result.zip", "application/zip", 0)


def put(storage, source, key=KEY):
    return storage.put_file(source, key=key, filename="result.zip", media_type="application/zip")


def stage_for(storage, key=KEY):
    return storage._recoverable.stage_path(storage._path(key))


def test_exact_staging_cleanup_without_published_result(tmp_path):
    storage = storage_at(tmp_path / "objects")
    stage = stage_for(storage)
    stage.parent.mkdir(mode=0o700)
    stage.write_bytes(b"authored interrupted copy")
    other_stage = stage.with_name("unrelated.part")
    other_stage.write_bytes(b"must remain")
    legacy = storage.root / "jobs" / ("a" * 32) / ".result.zip.legacy.tmp"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy ownership unknown")
    storage.delete(artifact())
    assert not stage.exists()
    assert not storage.exists(artifact())
    assert other_stage.read_bytes() == b"must remain"
    assert legacy.read_bytes() == b"legacy ownership unknown"
    storage.delete(artifact())  # Missing object/stage is idempotent.


def test_round_trip_has_private_stage_and_does_not_publish_partial_bytes(tmp_path, monkeypatch):
    storage = storage_at(tmp_path / "objects")
    source = tmp_path / "source.zip"
    source.write_bytes(b"authored binary result\x00" * 100)
    copy = shutil.copyfileobj

    def checked_copy(src, dst, *args, **kwargs):
        stage = stage_for(storage)
        assert stage.exists()
        assert stage.stat().st_mode & 0o777 == 0o600
        assert stage.parent.stat().st_mode & 0o777 == 0o700
        assert not storage.exists(artifact())
        copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfileobj", checked_copy)
    result = put(storage, source)
    assert result.size_bytes == source.stat().st_size
    assert b"".join(storage.iter_bytes(result, chunk_size=7)) == source.read_bytes()
    assert (storage.root / KEY).stat().st_mode & 0o777 == 0o600
    assert not stage_for(storage).exists()
    storage.delete(result)
    assert not storage.exists(result)


def test_active_copy_blocks_cleanup_across_adapter_instances(tmp_path, monkeypatch):
    storage = storage_at(tmp_path / "objects")
    second = storage_at(storage.root)
    source = tmp_path / "source.zip"
    source.write_bytes(b"authored original")
    previous = put(storage, source)
    source.write_bytes(b"authored replacement")
    entered, release = threading.Event(), threading.Event()
    copy = shutil.copyfileobj

    def slow_copy(src, dst, *args, **kwargs):
        dst.write(b"partial")
        dst.flush()
        entered.set()
        assert release.wait(5), "test did not release the writer"
        dst.seek(0)
        dst.truncate()
        copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfileobj", slow_copy)
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        writing = pool.submit(put, storage, source)
        assert entered.wait(5)
        stage = stage_for(storage)
        with pytest.raises(RuntimeError, match="busy"):
            pool.submit(second.delete, previous).result(timeout=1)
        assert not writing.done()
        assert stage.read_bytes() == b"partial"
        assert b"".join(second.iter_bytes(previous)) == b"authored original"
        release.set()
        writing.result(timeout=5)
        assert b"".join(second.iter_bytes(previous)) == b"authored replacement"
        second.delete(previous)
        assert not stage.exists()
        assert not second.exists(previous)
    finally:
        release.set()
        pool.shutdown(wait=True)


def test_different_lock_slot_can_finish_while_another_copy_waits(tmp_path, monkeypatch):
    storage = storage_at(tmp_path / "objects")
    source = tmp_path / "source.zip"
    source.write_bytes(b"authored result")
    entered, release = threading.Event(), threading.Event()
    copy = shutil.copyfileobj
    first_lock = storage._recoverable.lock_path(storage._path(KEY))
    other_key = next(
        f"jobs/{i:032x}/result.zip"
        for i in range(1000)
        if storage._recoverable.lock_path(storage._path(f"jobs/{i:032x}/result.zip")) != first_lock
    )
    first_thread = []

    def gated_copy(src, dst, *args, **kwargs):
        if not first_thread:
            first_thread.append(threading.get_ident())
            entered.set()
            assert release.wait(5)
        copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfileobj", gated_copy)
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        waiting = pool.submit(put, storage, source)
        assert entered.wait(5)
        result = pool.submit(put, storage, source, other_key).result(timeout=1)
        assert storage.exists(result)
        assert not waiting.done()
        pool.submit(storage.delete, result).result(timeout=1)
        release.set()
        waiting.result(timeout=5)
    finally:
        release.set()
        pool.shutdown(wait=True)


def test_copy_error_is_private_and_releases_slot(tmp_path, monkeypatch):
    storage = storage_at(tmp_path / "objects")
    source = tmp_path / "source.zip"
    source.write_bytes(b"authored result")
    copy = shutil.copyfileobj

    def fail(src, dst, *args, **kwargs):
        dst.write(b"partial")
        raise OSError("PRIVATE source path and credentials")

    monkeypatch.setattr(shutil, "copyfileobj", fail)
    with pytest.raises(RuntimeError) as error:
        put(storage, source)
    assert "PRIVATE" not in str(error.value)
    assert not stage_for(storage).exists()
    monkeypatch.setattr(shutil, "copyfileobj", copy)
    assert storage.exists(put(storage, source))


def test_stale_stage_is_not_overwritten_by_new_writer(tmp_path):
    storage = storage_at(tmp_path / "objects")
    source = tmp_path / "source.zip"
    source.write_bytes(b"new result")
    stage = stage_for(storage)
    stage.parent.mkdir(mode=0o700)
    stage.write_bytes(b"unfinished old write")
    with pytest.raises(RuntimeError):
        put(storage, source)
    assert stage.read_bytes() == b"unfinished old write"
    storage.delete(artifact())
    assert storage.exists(put(storage, source))


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory", "fifo"])
@pytest.mark.parametrize("target", ["stage", "lock", "destination"])
def test_unsafe_paths_are_refused_without_touching_outside_data(tmp_path, kind, target):
    storage = storage_at(tmp_path / "objects")
    outside = tmp_path / "outside"
    outside.write_bytes(b"private outside data")
    source = tmp_path / "source.zip"
    source.write_bytes(b"new result")
    selected = {
        "stage": stage_for(storage),
        "lock": storage._recoverable.lock_path(storage._path(KEY)),
        "destination": storage.root / KEY,
    }[target]
    selected.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if kind == "symlink":
        selected.symlink_to(outside)
    elif kind == "hardlink":
        os.link(outside, selected)
    elif kind == "directory":
        selected.mkdir()
    else:
        os.mkfifo(selected)
    for operation in (lambda: put(storage, source), lambda: storage.delete(artifact())):
        with pytest.raises((RuntimeError, ValueError)):
            operation()
        assert outside.read_bytes() == b"private outside data"
    assert selected.exists() or selected.is_symlink()


def test_staging_directory_symlink_is_not_followed(tmp_path):
    storage = storage_at(tmp_path / "objects")
    outside = tmp_path / "outside"
    outside.mkdir()
    stage_for(storage).parent.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError):
        storage.delete(artifact())
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    "key",
    [
        ".lao-ocr-writes-v1/lock-00",
        ".lao-ocr-writes-v1/data.part",
        ".lao-ocr-writes-v1",
    ],
)
def test_internal_namespace_cannot_be_used_as_an_artifact_key(tmp_path, key):
    storage = storage_at(tmp_path / "objects")
    source = tmp_path / "source.zip"
    source.write_bytes(b"authored result")
    for operation in (
        lambda: put(storage, source, key),
        lambda: storage.delete(artifact(key)),
        lambda: storage.exists(artifact(key)),
    ):
        with pytest.raises(ValueError, match="storage key"):
            operation()


def test_staging_identity_uses_canonical_key_not_filename_metadata(tmp_path):
    storage = storage_at(tmp_path / "objects")
    stage = stage_for(storage)
    assert hashlib.sha256(KEY.encode()).hexdigest() in stage.name
    assert stage == stage_for(storage, "jobs//" + "a" * 32 + "/./result.zip")
    assert "result.zip" not in stage.name


def test_default_mode_does_not_create_staging_namespace(tmp_path):
    storage = FilesystemArtifactStorage(tmp_path / "objects")
    source = tmp_path / "source.zip"
    source.write_bytes(b"authored result")
    result = put(storage, source)
    storage.delete(result)
    assert not (storage.root / ".lao-ocr-writes-v1").exists()
