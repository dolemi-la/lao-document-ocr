"""Abrupt exits before publication recover private workspace cleanup, never OCR."""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pytest

from services.api.app.jobs import ConversionJobManager, JobNotFoundError
from services.api.app.storage import FilesystemArtifactStorage

CHILD = r"""
import os, sys, threading
from pathlib import Path
from services.api.app.jobs import ConversionJobManager, JobPublicError, JobStatus
from services.api.app.storage import FilesystemArtifactStorage
root = Path(sys.argv[1])
phase = sys.argv[2]
storage = FilesystemArtifactStorage(root / "objects")
entered = threading.Event()
release = threading.Event()
def runner(record, _):
    (record.workspace / "intermediate.txt").write_bytes(b"authored intermediate fixture")
    if phase == "running":
        os._exit(47)
    if phase == "failed":
        raise JobPublicError("authored failure")
    entered.set()
    assert release.wait(10)
    return record.input_path
manager = ConversionJobManager(root / "jobs", runner, max_workers=1,
    durable_cleanup=True, cleanup_namespace=storage.cleanup_namespace(),
    artifact_cleanup=storage.delete, cleanup_interval_seconds=0)
if phase == "before-allocation":
    mkdir = Path.mkdir
    def interrupted(path, *args, **kwargs):
        if path.parent == manager.root_dir:
            os._exit(47)
        return mkdir(path, *args, **kwargs)
    Path.mkdir = interrupted
record = manager.reserve("private.png", ".png")
record.input_path.write_bytes(b"authored partial upload")
if phase == "uploading":
    os._exit(47)
if phase == "aborting":
    import services.api.app.jobs as jobs
    jobs.shutil.rmtree = lambda *args, **kwargs: os._exit(47)
    manager.discard(record.id)
if phase == "cancelled":
    manager.cancel(record.id)
    manager.enqueue(record.id)
    assert record.status == JobStatus.CANCELLED
    os._exit(47)
manager.enqueue(record.id)
if phase == "queued":
    assert entered.wait(5)
    other = manager.reserve("second.png", ".png")
    other.input_path.write_bytes(b"second authored upload")
    manager.enqueue(other.id)
    assert other.status == JobStatus.QUEUED
    os._exit(47)
record.future.result(timeout=5)
if phase == "failed":
    assert record.status == JobStatus.FAILED
    os._exit(47)
raise AssertionError("exit point not reached")
"""


@pytest.mark.parametrize(
    "phase",
    [
        "before-allocation",
        "uploading",
        "queued",
        "running",
        "failed",
        "cancelled",
        "aborting",
    ],
)
def test_process_exit_recovers_prepublication_workspaces_without_running_jobs(tmp_path, phase):
    result = subprocess.run(
        [sys.executable, "-c", CHILD, str(tmp_path), phase],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 47, result.stderr
    storage = FilesystemArtifactStorage(tmp_path / "objects")
    runs = []
    manager = ConversionJobManager(
        tmp_path / "jobs",
        lambda *args: runs.append("unexpected OCR replay"),
        durable_cleanup=True,
        cleanup_namespace=storage.cleanup_namespace(),
        artifact_cleanup=storage.delete,
        cleanup_interval_seconds=0,
        retention_seconds=0,
    )
    try:
        rows = manager._journal.load()
        assert len(rows) == (2 if phase == "queued" else 1)
        assert all(row.artifact_key is None and row.workspace_pending for row in rows)
        assert not manager._jobs
        assert not manager._upload_owners
        ids = [row.job_id for row in rows]
        assert not list(storage.root.rglob("*.zip"))
        if phase != "aborting":
            manager.cleanup_expired(now=min(row.retry_at for row in rows) - timedelta(seconds=1))
            assert len(manager._journal.load()) == len(rows)
        for job_id in ids:
            with pytest.raises(JobNotFoundError):
                manager.public(job_id)
        manager.cleanup_expired(now=datetime.now(UTC) + timedelta(hours=2))
        assert not manager._journal.load()
        assert all(not (manager.root_dir / job_id).exists() for job_id in ids)
        assert manager.snapshot()["completed_total"] == {}
        assert runs == []
    finally:
        manager.shutdown()
