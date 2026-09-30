"""Idle cleanup lifecycle controls: local events/fixtures, never real OCR or S3."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from services.api.app.cleanup_worker import IdleCleanupWorker


@pytest.mark.parametrize("interval", [-1, True, "30", None, float("nan"), float("inf"), 10**100])
def test_invalid_interval_rejected_before_start(interval):
    with pytest.raises(ValueError, match="interval_seconds"):
        IdleCleanupWorker(lambda: None, interval_seconds=interval)


def test_constructing_worker_does_not_start_a_thread():
    called = threading.Event()
    worker = IdleCleanupWorker(called.set, interval_seconds=0.01)
    assert not worker.is_alive
    assert not called.is_set()
    assert worker.stop()


def test_zero_disables_periodic_work():
    called = threading.Event()
    worker = IdleCleanupWorker(called.set, interval_seconds=0)
    assert worker.start() is False
    assert not worker.is_alive
    assert not called.is_set()
    assert worker.stop()


def test_repeated_and_concurrent_start_create_only_one_worker():
    entered, release = threading.Event(), threading.Event()
    calls = []

    def cleanup():
        calls.append(threading.get_ident())
        entered.set()
        assert release.wait(5)

    worker = IdleCleanupWorker(cleanup, interval_seconds=0.01)
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: worker.start(), range(6)))
        assert results.count(True) == 1
        assert results.count(False) == 5
        assert entered.wait(5)
        assert worker.is_alive
        assert worker.stop(timeout=0) is False
        assert calls == [worker._thread.ident]
    finally:
        release.set()
        assert worker.stop()
    assert not worker.is_alive
    with pytest.raises(RuntimeError, match="stopped"):
        worker.start()


def test_stop_interrupts_long_interval_without_running_cleanup():
    called = threading.Event()
    worker = IdleCleanupWorker(called.set, interval_seconds=3600)
    worker.start()
    assert worker.stop(timeout=1)
    assert not called.is_set()
    assert worker.stop(timeout=0)
    with pytest.raises(RuntimeError, match="stopped"):
        worker.start()


def test_blocked_callback_cannot_overlap_restart_or_keep_scheduling():
    entered, release = threading.Event(), threading.Event()
    calls = []

    def cleanup():
        calls.append(1)
        entered.set()
        assert release.wait(5)

    worker = IdleCleanupWorker(cleanup, interval_seconds=0.01)
    try:
        worker.start()
        assert entered.wait(5)
        assert worker.stop(timeout=0.01) is False
        assert worker.is_alive
        assert worker._thread.daemon
        with pytest.raises(RuntimeError, match="stopped"):
            worker.start()
        assert calls == [1]
    finally:
        release.set()
        assert worker.stop()
    assert calls == [1]
    assert not worker.is_alive


def test_unexpected_callback_failure_is_private_and_next_interval_retries(caplog):
    recovered = threading.Event()
    calls = []

    def cleanup():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("PRIVATE document path credentials")
        recovered.set()

    worker = IdleCleanupWorker(cleanup, interval_seconds=0.01)
    try:
        worker.start()
        assert recovered.wait(5)
    finally:
        assert worker.stop()
    assert len(calls) >= 2
    assert "Idle job cleanup pass failed" in caplog.text
    assert "PRIVATE" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.parametrize("timeout", [-1, True, None, float("nan"), float("inf")])
def test_invalid_join_timeout_does_not_stop_worker(timeout):
    worker = IdleCleanupWorker(lambda: None, interval_seconds=3600)
    worker.start()
    try:
        with pytest.raises(ValueError, match="timeout"):
            worker.stop(timeout=timeout)
        assert worker.is_alive
    finally:
        assert worker.stop()


def test_start_failure_can_retry_without_losing_lifecycle_state(monkeypatch):
    original = threading.Thread.start
    worker = IdleCleanupWorker(lambda: None, interval_seconds=3600)

    def fail_start(_):
        raise RuntimeError("thread allocation fixture")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="thread allocation fixture"):
        worker.start()
    assert not worker.is_alive
    monkeypatch.setattr(threading.Thread, "start", original)
    try:
        assert worker.start()
    finally:
        assert worker.stop()


def test_callback_can_request_its_own_stop_without_joining_itself():
    returned = threading.Event()
    results = []

    def cleanup():
        results.append(worker.stop())
        returned.set()

    worker = IdleCleanupWorker(cleanup, interval_seconds=0.01)
    try:
        worker.start()
        assert returned.wait(5)
    finally:
        assert worker.stop()
    assert results == [False]
