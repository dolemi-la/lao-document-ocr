"""Admission/scheduling lifecycle checks; no OCR or external storage services."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest

from services.api.app.jobs import (
    ConversionJobManager,
    JobManagerClosedError,
    JobSchedulingError,
    JobStatus,
)


def test_shutdown_rejects_reservation_without_allocating_workspace(tmp_path):
    manager = ConversionJobManager(tmp_path / "jobs", lambda *_: None)
    manager.shutdown()
    before = list(manager.root_dir.iterdir())
    with pytest.raises(RuntimeError, match="stopped"):
        manager.reserve("sample.png", ".png")
    assert list(manager.root_dir.iterdir()) == before


def test_shutdown_reconciles_cancelled_future_without_cancelling_running_work(tmp_path):
    entered, release = threading.Event(), threading.Event()
    ran = []

    def runner(record, cancel_event):
        ran.append(record.id)
        entered.set()
        assert release.wait(5), "test did not release running work"
        assert not cancel_event.is_set()
        result = record.workspace / "result.zip"
        result.write_bytes(b"authored fixture archive")
        return result

    manager = ConversionJobManager(tmp_path / "jobs", runner, max_workers=1, max_active_jobs=3)
    try:
        running = manager.reserve("running.png", ".png")
        manager.enqueue(running.id)
        assert entered.wait(5)
        queued = manager.reserve("queued.png", ".png")
        manager.enqueue(queued.id)
        manager.shutdown(wait=False)
        assert queued.future.cancelled()
        assert queued.status == JobStatus.CANCELLED
        assert queued.completed_at is not None
        assert queued.cancellation_requested
        assert manager.snapshot()["completed_total"] == {"cancelled": 1}
        assert running.status == JobStatus.RUNNING
        assert not running.cancel_event.is_set()
        release.set()
        running.future.result(timeout=5)
        assert running.status == JobStatus.SUCCEEDED
        assert ran == [running.id]
    finally:
        release.set()
        manager.shutdown()


def test_executor_rejection_does_not_leave_a_phantom_queued_record(tmp_path, monkeypatch):
    manager = ConversionJobManager(tmp_path / "jobs", lambda *_: None)

    def reject(*args, **kwargs):
        raise RuntimeError("PRIVATE executor detail")

    try:
        record = manager.reserve("sample.png", ".png")
        monkeypatch.setattr(manager._executor, "submit", reject)
        with pytest.raises(RuntimeError):
            manager.enqueue(record.id)
        assert record.status == JobStatus.FAILED
        assert record.completed_at is not None
        assert record.error == "Conversion worker is unavailable."
        assert manager.snapshot()["active_jobs"] == 0
    finally:
        manager.shutdown()


@pytest.mark.parametrize("batch", [False, True])
def test_closed_admission_does_not_even_start_request_side_cleanup(tmp_path, monkeypatch, batch):
    manager = ConversionJobManager(tmp_path / "jobs", lambda *_: None)
    manager.shutdown()
    monkeypatch.setattr(manager, "cleanup_expired", lambda: pytest.fail("cleanup must not run"))
    with pytest.raises(JobManagerClosedError):
        if batch:
            manager.reserve_many([("one.png", ".png"), ("two.png", ".png")])
        else:
            manager.reserve("one.png", ".png")
    assert not list(manager.root_dir.iterdir())


def test_shutdown_does_not_erase_a_still_owned_upload(tmp_path):
    manager = ConversionJobManager(tmp_path / "jobs", lambda *_: None)
    record = manager.reserve("upload.png", ".png", page_rotations={1: 0})
    record.input_path.write_bytes(b"partial upload")
    manager.shutdown()
    assert record.status == JobStatus.UPLOADING
    assert record.future is None
    assert record.input_path.read_bytes() == b"partial upload"
    with pytest.raises(JobManagerClosedError):
        manager.enqueue(record.id)
    assert record.status == JobStatus.UPLOADING
    assert manager.cleanup_expired(now=record.created_at + timedelta(days=1)) == 0
    assert record.input_path.exists()
    # The uploader, not shutdown, decides when writing has stopped.
    manager.discard(record.id)
    assert not record.workspace.exists()


def test_admission_closes_before_waiting_for_idle_worker_shutdown(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    manager = ConversionJobManager(tmp_path / "jobs", lambda *_: None)
    upload = manager.reserve("upload.png", ".png")
    real_stop = manager._cleanup_worker.stop

    def blocked_stop(*, timeout):
        entered.set()
        assert release.wait(5)
        return real_stop(timeout=timeout)

    monkeypatch.setattr(manager._cleanup_worker, "stop", blocked_stop)
    with ThreadPoolExecutor(max_workers=1) as pool:
        shutdown = pool.submit(manager.shutdown)
        try:
            assert entered.wait(5)
            assert not manager._executor._shutdown
            with pytest.raises(JobManagerClosedError):
                manager.reserve("later.png", ".png")
            with pytest.raises(JobManagerClosedError):
                manager.enqueue(upload.id)
            with pytest.raises(JobManagerClosedError):
                manager.start_cleanup_worker()
            assert upload.workspace.exists()
        finally:
            release.set()
            shutdown.result(timeout=5)


@pytest.mark.parametrize("batch", [False, True])
def test_shutdown_during_admission_cleanup_is_rechecked(tmp_path, monkeypatch, batch):
    entered, release = threading.Event(), threading.Event()
    manager = ConversionJobManager(tmp_path / "jobs", lambda *_: None)

    def blocked_cleanup():
        entered.set()
        assert release.wait(5)
        return 0

    monkeypatch.setattr(manager, "cleanup_expired", blocked_cleanup)

    def reserve():
        if batch:
            return manager.reserve_many([("one.png", ".png"), ("two.png", ".png")])
        return manager.reserve("one.png", ".png")

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(reserve)
        try:
            assert entered.wait(5)
            manager.shutdown()
        finally:
            release.set()
        with pytest.raises(JobManagerClosedError):
            pending.result(timeout=5)
    assert not list(manager.root_dir.iterdir())
    assert not manager._jobs


@pytest.mark.parametrize("discard", [False, True])
def test_submit_that_queued_then_raised_never_invokes_runner(tmp_path, monkeypatch, discard):
    ran, submitted = [], []
    manager = ConversionJobManager(tmp_path / "jobs", lambda record, _: ran.append(record.id))
    original = manager._executor.submit

    def uncertain_submit(*args, **kwargs):
        # Model the executor queue insertion before thread-start failure.
        submitted.append(original(*args, **kwargs))
        raise RuntimeError("PRIVATE executor message")

    monkeypatch.setattr(manager._executor, "submit", uncertain_submit)
    try:
        record = manager.reserve("sample.png", ".png", page_rotations={1: 0})
        with pytest.raises(JobSchedulingError, match="^Conversion worker is unavailable.$"):
            manager.enqueue(record.id)
        assert record.status == JobStatus.FAILED
        assert record.future is None
        assert manager.public(record.id)["page_rotations"] == [{"page": 1, "degrees_clockwise": 0}]
        assert manager.public(record.id)["orientation_review"] is None
        if discard:
            manager.discard(record.id)
        submitted[0].result(timeout=5)
        assert ran == []
        assert manager.snapshot()["completed_total"] == {"failed": 1}
        assert manager.snapshot()["active_jobs"] == 0
        manager.shutdown()
        assert manager.snapshot()["completed_total"] == {"failed": 1}
    finally:
        manager.shutdown()


def test_dequeued_but_not_started_future_is_cancelled_at_manager_boundary(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    ran = []
    manager = ConversionJobManager(tmp_path / "jobs", lambda record, _: ran.append(record.id))
    original = manager._run

    def paused_run(job_id):
        entered.set()
        assert release.wait(5)
        original(job_id)

    monkeypatch.setattr(manager, "_run", paused_run)
    try:
        record = manager.reserve("sample.png", ".png")
        manager.enqueue(record.id)
        assert entered.wait(5)
        assert record.future.running()
        assert record.status == JobStatus.QUEUED
        manager.shutdown(wait=False)
        assert not record.future.cancelled()  # Future already dequeued, but OCR not entered.
        assert record.status == JobStatus.CANCELLED
        release.set()
        record.future.result(timeout=5)
        assert ran == []
        assert manager.snapshot()["completed_total"] == {"cancelled": 1}
        assert manager.snapshot()["active_jobs"] == 0
    finally:
        release.set()
        manager.shutdown()


def test_repeated_shutdown_and_user_cancel_count_queued_completion_once(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def blocked(record, event):
        entered.set()
        assert release.wait(5)
        return record.input_path

    manager = ConversionJobManager(tmp_path / "jobs", blocked, max_workers=1, max_active_jobs=2)
    try:
        running, queued = manager.reserve_many([("one.png", ".png"), ("two.png", ".png")])
        manager.enqueue(running.id)
        assert entered.wait(5)
        manager.enqueue(queued.id)
        manager.shutdown(wait=False)
        completed_at = queued.completed_at
        manager.shutdown(wait=False)
        manager.cancel(queued.id)
        assert queued.completed_at == completed_at
        assert manager.snapshot()["completed_total"] == {"cancelled": 1}
        assert manager.public(queued.id)["download_ready"] is False
        assert queued.workspace.exists()
    finally:
        release.set()
        manager.shutdown()


def test_shutdown_future_callbacks_can_acquire_the_manager_lock(tmp_path):
    entered, release = threading.Event(), threading.Event()
    callback_checks = []

    def blocked(record, event):
        entered.set()
        assert release.wait(5)
        return record.input_path

    manager = ConversionJobManager(tmp_path / "jobs", blocked, max_workers=1, max_active_jobs=2)

    def callback(_future):
        acquired = threading.Event()

        def check_lock():
            with manager._lock:
                acquired.set()

        helper = threading.Thread(target=check_lock)
        helper.start()
        callback_checks.append(acquired.wait(2))
        helper.join(timeout=2)

    try:
        first, second = manager.reserve_many([("one.png", ".png"), ("two.png", ".png")])
        manager.enqueue(first.id)
        assert entered.wait(5)
        manager.enqueue(second.id)
        second.future.add_done_callback(callback)
        manager.shutdown(wait=False)
        assert callback_checks == [True]
    finally:
        release.set()
        manager.shutdown()


@pytest.mark.parametrize("discard", [False, True])
def test_actual_thread_start_failure_is_fenced_before_later_executor_work(
    tmp_path,
    monkeypatch,
    discard,
):
    ran = []
    manager = ConversionJobManager(tmp_path / "jobs", lambda record, _: ran.append(record.id))
    original_adjust = manager._executor._adjust_thread_count

    def cannot_start_thread():
        raise RuntimeError("PRIVATE thread-start detail")

    monkeypatch.setattr(manager._executor, "_adjust_thread_count", cannot_start_thread)
    try:
        record = manager.reserve("sample.png", ".png")
        with pytest.raises(JobSchedulingError):
            manager.enqueue(record.id)
        assert record.status == JobStatus.FAILED
        with pytest.raises(JobManagerClosedError):
            manager.reserve("retry.png", ".png")
        if discard:
            manager.discard(record.id)
        # Only a test-level direct executor call starts/drains the previously
        # queued work. Production never reopens this manager after rejection.
        monkeypatch.setattr(manager._executor, "_adjust_thread_count", original_adjust)
        manager._executor.submit(lambda: None).result(timeout=5)
        assert ran == []
        assert manager.snapshot()["completed_total"] == {"failed": 1}
    finally:
        manager.shutdown()


def test_scheduling_failure_closes_new_admission_but_keeps_running_work(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    ran = []

    def runner(record, cancel_event):
        ran.append(record.id)
        entered.set()
        assert release.wait(5)
        assert not cancel_event.is_set()
        return record.input_path

    manager = ConversionJobManager(tmp_path / "jobs", runner, max_workers=1, max_active_jobs=2)

    def unavailable(*args, **kwargs):
        raise RuntimeError("PRIVATE unavailable executor")

    try:
        active, rejected = manager.reserve_many([("one.png", ".png"), ("two.png", ".png")])
        manager.enqueue(active.id)
        assert entered.wait(5)
        monkeypatch.setattr(manager._executor, "submit", unavailable)
        with pytest.raises(JobSchedulingError):
            manager.enqueue(rejected.id)
        with pytest.raises(JobManagerClosedError):
            manager.reserve("later.png", ".png")
        assert not active.cancel_event.is_set()
        release.set()
        active.future.result(timeout=5)
        assert active.status == JobStatus.SUCCEEDED
        assert ran == [active.id]
        assert manager.snapshot()["completed_total"] == {"failed": 1, "succeeded": 1}
    finally:
        release.set()
        manager.shutdown()
