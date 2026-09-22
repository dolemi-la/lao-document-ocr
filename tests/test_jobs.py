import time
from datetime import UTC, datetime, timedelta

import pytest

from services.api.app.jobs import (
    ConversionJobManager,
    JobCancelledError,
    JobCapacityError,
    JobStatus,
)


def _wait_for_terminal(manager, job_id: str, timeout: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = manager.public(job_id)
        if payload["status"] in {"succeeded", "failed", "cancelled"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("job did not reach a terminal state")


def test_job_manager_runs_and_exposes_download(tmp_path) -> None:
    def runner(record, cancel_event):
        output = record.workspace / "result.zip"
        output.write_bytes(b"zip")
        return output

    manager = ConversionJobManager(
        tmp_path / "jobs",
        runner,
        max_workers=1,
        max_active_jobs=2,
    )
    try:
        record = manager.reserve("sample.png", ".png")
        record.input_path.write_bytes(b"image")
        manager.enqueue(record.id)

        payload = _wait_for_terminal(manager, record.id)

        assert payload["status"] == "succeeded"
        assert payload["download_ready"] is True
        assert manager.get_record(record.id).output_path.read_bytes() == b"zip"
    finally:
        manager.shutdown()


def test_job_manager_enforces_active_capacity(tmp_path) -> None:
    gate = {"open": False}

    def runner(record, cancel_event):
        while not gate["open"]:
            if cancel_event.is_set():
                raise JobCancelledError()
            time.sleep(0.01)
        output = record.workspace / "result.zip"
        output.write_bytes(b"zip")
        return output

    manager = ConversionJobManager(
        tmp_path / "jobs",
        runner,
        max_workers=1,
        max_active_jobs=1,
    )
    try:
        first = manager.reserve("first.png", ".png")
        first.input_path.write_bytes(b"x")
        manager.enqueue(first.id)

        with pytest.raises(JobCapacityError, match="capacity"):
            manager.reserve("second.png", ".png")

        gate["open"] = True
        assert _wait_for_terminal(manager, first.id)["status"] == "succeeded"
    finally:
        gate["open"] = True
        manager.shutdown()


def test_running_job_can_be_cancelled(tmp_path) -> None:
    def runner(record, cancel_event):
        while True:
            if cancel_event.is_set():
                raise JobCancelledError()
            time.sleep(0.01)

    manager = ConversionJobManager(
        tmp_path / "jobs",
        runner,
        max_workers=1,
        max_active_jobs=2,
    )
    try:
        record = manager.reserve("sample.pdf", ".pdf")
        record.input_path.write_bytes(b"pdf")
        manager.enqueue(record.id)

        deadline = time.monotonic() + 1.0
        while manager.public(record.id)["status"] != "running":
            if time.monotonic() > deadline:
                raise AssertionError("job never started")
            time.sleep(0.01)

        cancel_payload = manager.cancel(record.id)
        assert cancel_payload["cancellation_requested"] is True

        terminal = _wait_for_terminal(manager, record.id)
        assert terminal["status"] == "cancelled"
        assert terminal["download_ready"] is False
    finally:
        manager.shutdown()


def test_cleanup_removes_expired_job_workspace(tmp_path) -> None:
    def runner(record, cancel_event):
        output = record.workspace / "result.zip"
        output.write_bytes(b"zip")
        return output

    manager = ConversionJobManager(
        tmp_path / "jobs",
        runner,
        max_workers=1,
        max_active_jobs=2,
        retention_seconds=60,
    )
    try:
        record = manager.reserve("sample.png", ".png")
        record.input_path.write_bytes(b"x")
        manager.enqueue(record.id)
        _wait_for_terminal(manager, record.id)

        completed = manager.get_record(record.id).completed_at
        removed = manager.cleanup_expired(
            now=completed + timedelta(seconds=61)
        )

        assert removed == 1
        assert not record.workspace.exists()
        with pytest.raises(KeyError):
            manager.get_record(record.id)
    finally:
        manager.shutdown()


def test_cancel_before_enqueue_marks_job_cancelled(tmp_path) -> None:
    def runner(record, cancel_event):
        raise AssertionError("runner should not execute")

    manager = ConversionJobManager(
        tmp_path / "jobs",
        runner,
        max_workers=1,
        max_active_jobs=2,
    )
    try:
        record = manager.reserve("sample.png", ".png")
        manager.cancel(record.id)
        payload = manager.enqueue(record.id)
        assert payload.status == JobStatus.CANCELLED
    finally:
        manager.shutdown()


def test_failed_job_records_error_and_has_no_download(tmp_path) -> None:
    def runner(record, cancel_event):
        raise RuntimeError("synthetic conversion failure")

    manager = ConversionJobManager(
        tmp_path / "jobs",
        runner,
        max_workers=1,
        max_active_jobs=2,
    )
    try:
        record = manager.reserve("sample.png", ".png")
        record.input_path.write_bytes(b"image")
        manager.enqueue(record.id)

        payload = _wait_for_terminal(manager, record.id)

        assert payload["status"] == "failed"
        assert payload["error"] == "synthetic conversion failure"
        assert payload["download_ready"] is False
    finally:
        manager.shutdown()


def test_snapshot_keeps_completed_counters_after_cleanup(tmp_path) -> None:
    def runner(record, cancel_event):
        output = record.workspace / "result.zip"
        output.write_bytes(b"zip")
        return output

    manager = ConversionJobManager(
        tmp_path / "jobs",
        runner,
        max_workers=1,
        max_active_jobs=2,
        retention_seconds=60,
    )
    try:
        record = manager.reserve("sample.png", ".png")
        record.input_path.write_bytes(b"image")
        manager.enqueue(record.id)
        terminal = _wait_for_terminal(manager, record.id)
        assert terminal["status"] == "succeeded"

        snapshot_before = manager.snapshot()
        assert snapshot_before["completed_total"]["succeeded"] == 1
        assert snapshot_before["duration_seconds_sum"]["succeeded"] >= 0

        manager.cleanup_expired(now=datetime.now(UTC) + timedelta(seconds=61))
        snapshot_after = manager.snapshot()
        assert snapshot_after["current_by_status"].get("succeeded", 0) == 0
        assert snapshot_after["completed_total"]["succeeded"] == 1
    finally:
        manager.shutdown()


def test_reserve_many_is_atomic_when_capacity_is_insufficient(tmp_path) -> None:
    def runner(record, cancel_event):
        output = record.workspace / "result.zip"
        output.write_bytes(b"zip")
        return output

    manager = ConversionJobManager(
        tmp_path / "jobs-batch-capacity",
        runner,
        max_workers=1,
        max_active_jobs=2,
    )
    try:
        existing = manager.reserve("existing.png", ".png")
        existing.input_path.write_bytes(b"image")

        with pytest.raises(JobCapacityError, match="slot"):
            manager.reserve_many(
                [
                    ("a.png", ".png"),
                    ("b.png", ".png"),
                ]
            )

        snapshot = manager.snapshot()
        assert snapshot["active_jobs"] == 1
        assert len(list((tmp_path / "jobs-batch-capacity").iterdir())) == 1
    finally:
        manager.shutdown()


def test_reserve_many_creates_all_workspaces(tmp_path) -> None:
    def runner(record, cancel_event):
        output = record.workspace / "result.zip"
        output.write_bytes(b"zip")
        return output

    manager = ConversionJobManager(
        tmp_path / "jobs-batch",
        runner,
        max_workers=1,
        max_active_jobs=4,
    )
    try:
        records = manager.reserve_many(
            [
                ("a.png", ".png"),
                ("b.pdf", ".pdf"),
            ]
        )
        assert len(records) == 2
        assert records[0].input_path.name == "input.png"
        assert records[1].input_path.name == "input.pdf"
        assert all(record.workspace.is_dir() for record in records)
        assert manager.snapshot()["active_jobs"] == 2
    finally:
        manager.shutdown()


def test_stored_artifact_download_readiness_and_cleanup(tmp_path) -> None:
    from services.api.app.storage import FilesystemArtifactStorage

    storage = FilesystemArtifactStorage(tmp_path / "results")

    def runner(record, cancel_event):
        local = record.workspace / "result.zip"
        local.write_bytes(b"stored-result")
        return storage.put_file(
            local,
            key=f"jobs/{record.id}/result.zip",
            filename="result.zip",
            media_type="application/zip",
        )

    manager = ConversionJobManager(
        tmp_path / "jobs-artifact",
        runner,
        max_workers=1,
        max_active_jobs=2,
        retention_seconds=60,
        artifact_exists=storage.exists,
        artifact_cleanup=storage.delete,
    )
    try:
        record = manager.reserve("sample.png", ".png")
        record.input_path.write_bytes(b"image")
        manager.enqueue(record.id)
        terminal = _wait_for_terminal(manager, record.id)

        assert terminal["status"] == "succeeded"
        assert terminal["download_ready"] is True
        stored = manager.get_record(record.id).output_artifact
        assert stored is not None
        assert b"".join(storage.iter_bytes(stored)) == b"stored-result"

        completed = manager.get_record(record.id).completed_at
        removed = manager.cleanup_expired(
            now=completed + timedelta(seconds=61)
        )

        assert removed == 1
        assert storage.exists(stored) is False
    finally:
        manager.shutdown()


def test_missing_stored_artifact_is_not_download_ready(tmp_path) -> None:
    from services.api.app.storage import FilesystemArtifactStorage

    storage = FilesystemArtifactStorage(tmp_path / "results")

    def runner(record, cancel_event):
        local = record.workspace / "result.zip"
        local.write_bytes(b"stored-result")
        return storage.put_file(
            local,
            key=f"jobs/{record.id}/result.zip",
            filename="result.zip",
            media_type="application/zip",
        )

    manager = ConversionJobManager(
        tmp_path / "jobs-artifact-missing",
        runner,
        max_workers=1,
        max_active_jobs=2,
        artifact_exists=storage.exists,
        artifact_cleanup=storage.delete,
    )
    try:
        record = manager.reserve("sample.png", ".png")
        record.input_path.write_bytes(b"image")
        manager.enqueue(record.id)
        _wait_for_terminal(manager, record.id)
        stored = manager.get_record(record.id).output_artifact
        assert stored is not None

        storage.delete(stored)

        assert manager.public(record.id)["download_ready"] is False
    finally:
        manager.shutdown()
