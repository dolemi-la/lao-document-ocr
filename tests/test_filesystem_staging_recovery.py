"""Actual process exits with authored bytes, never OCR or live object storage."""

from __future__ import annotations

import json
import select
import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from test_filesystem_staging import KEY, artifact, put, stage_for, storage_at

from services.api.app.jobs import ConversionJobManager, JobNotFoundError


def manager_for(root, storage, runner=None, **kwargs):
    return ConversionJobManager(
        root,
        runner or (lambda *_: None),
        durable_cleanup=True,
        cleanup_namespace=storage.cleanup_namespace(),
        artifact_exists=storage.exists,
        artifact_cleanup=storage.delete,
        cleanup_interval_seconds=0,
        **kwargs,
    )


CRASH_WRITER = r"""
import os, shutil, sys
from pathlib import Path
from services.api.app.jobs import ConversionJobManager
from services.api.app.storage import FilesystemArtifactStorage
root, phase = Path(sys.argv[1]), sys.argv[2]
storage = FilesystemArtifactStorage(root / "objects", recoverable_writes=True)
copy, replace = shutil.copyfileobj, Path.replace

def interrupted_copy(src, dst, *args, **kwargs):
    if phase == "mid-copy":
        dst.write(b"authored partial")
        dst.flush()
        os._exit(61)
    return copy(src, dst, *args, **kwargs)

def interrupted_replace(path, target):
    if path.suffix == ".part" and phase == "before-rename":
        os._exit(61)
    result = replace(path, target)
    if path.suffix == ".part" and phase == "after-rename":
        os._exit(61)
    return result

shutil.copyfileobj, Path.replace = interrupted_copy, interrupted_replace

def runner(record, _):
    source = record.workspace / "result.zip"
    source.write_bytes(b"authored complete result")
    key = f"jobs/{record.id}/result.zip"
    return record.publish_result(key, lambda: storage.put_file(
        source, key=key, filename="result.zip", media_type="application/zip",
    ))

manager = ConversionJobManager(root / "jobs", runner, durable_cleanup=True,
    cleanup_namespace=storage.cleanup_namespace(), artifact_cleanup=storage.delete,
    cleanup_interval_seconds=0)
record = manager.reserve("fixture.png", ".png")
manager.enqueue(record.id)
record.future.result(timeout=5)
raise AssertionError("crash boundary was not reached")
"""


@pytest.mark.parametrize("phase", ["mid-copy", "before-rename", "after-rename"])
def test_restart_recovers_both_staging_and_final_paths_from_journal_key(tmp_path, phase):
    child = subprocess.run(
        [sys.executable, "-c", CRASH_WRITER, str(tmp_path), phase],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert child.returncode == 61, child.stderr
    storage = storage_at(tmp_path / "objects")
    executions = []
    recovered = manager_for(tmp_path / "jobs", storage, lambda *_: executions.append("OCR"))
    try:
        (row,) = recovered._journal.load()
        stage = stage_for(storage, row.artifact_key)
        final = storage.root / row.artifact_key
        assert stage.exists() is (phase != "after-rename")
        assert final.exists() is (phase == "after-rename")
        if stage.exists():
            assert stage.stat().st_mode & 0o777 == 0o600
        assert executions == []
        assert recovered.snapshot()["cleanup_pending_jobs"] == 1
        assert recovered.cleanup_expired(now=row.retry_at - timedelta(microseconds=1)) == 0
        with pytest.raises(JobNotFoundError):
            recovered.public(row.job_id)
        recovered.cleanup_expired(now=row.retry_at)
        assert not stage.exists()
        assert not final.exists()
        assert not (recovered.root_dir / row.job_id).exists()
        assert not recovered._journal.load()
        assert recovered.snapshot()["completed_total"] == {}
        assert executions == []
    finally:
        recovered.shutdown()


@pytest.mark.parametrize("point", ["stage", "final"])
def test_process_exit_during_cleanup_retains_idempotent_retry_owner(tmp_path, point):
    child = subprocess.run(
        [sys.executable, "-c", CRASH_WRITER, str(tmp_path), "mid-copy"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert child.returncode == 61, child.stderr
    cleanup = r"""
import os, sys
from pathlib import Path
from services.api.app.jobs import ConversionJobManager
from services.api.app.storage import FilesystemArtifactStorage
root, point = Path(sys.argv[1]), sys.argv[2]
storage = FilesystemArtifactStorage(root / "objects", recoverable_writes=True)
manager = ConversionJobManager(root / "jobs", lambda *_: None, durable_cleanup=True,
    cleanup_namespace=storage.cleanup_namespace(), artifact_cleanup=storage.delete,
    cleanup_interval_seconds=0)
row, = manager._journal.load()
stage = storage._recoverable.stage_path(storage._path(row.artifact_key))
final = storage.root / row.artifact_key
unlink = Path.unlink

def interrupted(path, *args, **kwargs):
    result = unlink(path, *args, **kwargs)
    if path == (stage if point == "stage" else final):
        os._exit(62)
    return result

Path.unlink = interrupted
manager.cleanup_expired(now=row.retry_at)
raise AssertionError("cleanup exit boundary not reached")
"""
    result = subprocess.run(
        [sys.executable, "-c", cleanup, str(tmp_path), point],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 62, result.stderr
    storage = storage_at(tmp_path / "objects")
    recovered = manager_for(tmp_path / "jobs", storage)
    try:
        (row,) = recovered._journal.load()
        assert row.artifact_key is not None
        recovered.cleanup_expired(now=row.retry_at)
        assert not recovered._journal.load()
        assert not stage_for(storage, row.artifact_key).exists()
        assert not (storage.root / row.artifact_key).exists()
    finally:
        recovered.shutdown()


def test_different_process_cannot_clean_an_active_copy(tmp_path):
    script = r"""
import shutil, sys
from pathlib import Path
from services.api.app.storage import FilesystemArtifactStorage
root = Path(sys.argv[1])
storage = FilesystemArtifactStorage(root / "objects", recoverable_writes=True)
source = root / "source.zip"
source.write_bytes(b"authored result")
copy = shutil.copyfileobj

def paused(src, dst, *args, **kwargs):
    dst.write(b"partial")
    dst.flush()
    print("ready", flush=True)
    assert sys.stdin.readline().strip() == "release"
    dst.seek(0)
    dst.truncate()
    return copy(src, dst, *args, **kwargs)

shutil.copyfileobj = paused
storage.put_file(source, key=sys.argv[2], filename="result.zip", media_type="application/zip")
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path), KEY],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        readable, _, _ = select.select([child.stdout], [], [], 5)
        assert readable, "writer did not reach the copy boundary"
        assert child.stdout.readline().strip() == "ready"
        storage = storage_at(tmp_path / "objects")
        with pytest.raises(RuntimeError, match="busy"):
            storage.delete(artifact())
        assert stage_for(storage).read_bytes() == b"partial"
        child.stdin.write("release\n")
        child.stdin.flush()
        _, error = child.communicate(timeout=10)
        assert child.returncode == 0, error
        assert not stage_for(storage).exists()
        assert storage.exists(artifact())
        storage.delete(artifact())
        assert not storage.exists(artifact())
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=5)


def test_failed_stage_unlink_stays_journaled_until_later_retry(tmp_path, monkeypatch, caplog):
    storage = storage_at(tmp_path / "objects")
    copy, unlink = shutil.copyfileobj, Path.unlink

    def runner(record, _):
        source = record.workspace / "result.zip"
        source.write_bytes(b"authored result")
        key = f"jobs/{record.id}/result.zip"
        return record.publish_result(key, lambda: put(storage, source, key))

    def failed_copy(src, dst, *args, **kwargs):
        dst.write(b"partial")
        raise OSError("PRIVATE copy details")

    def failed_unlink(path, *args, **kwargs):
        if path.suffix == ".part":
            raise PermissionError("PRIVATE stage path")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfileobj", failed_copy)
    monkeypatch.setattr(Path, "unlink", failed_unlink)
    owner = manager_for(tmp_path / "jobs", storage, runner)
    try:
        record = owner.reserve("fixture.png", ".png")
        owner.enqueue(record.id)
        record.future.result(timeout=5)
        assert record.status == "failed"
        assert "PRIVATE" not in json.dumps(owner.public(record.id))
        (row,) = owner._journal.load()
        stage = stage_for(storage, row.artifact_key)
        assert stage.exists()
        owner.cleanup_expired(now=row.retry_at)
        (pending,) = owner._journal.load()
        assert pending.artifact_key == row.artifact_key
        assert pending.failures == 1
        assert pending.workspace_pending is False
        assert stage.exists()
        assert "PRIVATE" not in caplog.text
    finally:
        owner.shutdown()
        monkeypatch.setattr(shutil, "copyfileobj", copy)
        monkeypatch.setattr(Path, "unlink", unlink)
    recovered = manager_for(tmp_path / "jobs", storage)
    try:
        assert recovered.snapshot()["cleanup_pending_jobs"] == 1
        recovered.cleanup_expired(now=pending.retry_at)
        assert not stage.exists()
        assert not recovered._journal.load()
    finally:
        recovered.shutdown()


@pytest.mark.parametrize("enabled", [False, True])
def test_api_builder_ties_recoverable_writes_to_existing_durable_setting(
    tmp_path,
    monkeypatch,
    enabled,
):
    import services.api.app.main as api

    monkeypatch.setattr(api, "RESULT_STORAGE_BACKEND", "filesystem")
    monkeypatch.setattr(api, "RESULT_STORAGE_ROOT", tmp_path / "objects")
    monkeypatch.setattr(api, "JOB_CLEANUP_DURABLE", enabled)
    storage = api._build_result_storage()
    assert storage.recoverable_writes is enabled
    assert storage.metadata() == {"backend": "filesystem", "root": str(storage.root)}
    assert storage.cleanup_namespace() == storage.metadata()
    assert not (storage.root / ".lao-ocr-writes-v1").exists()  # No scan/thread at construction.
