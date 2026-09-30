"""ASGI completion, disconnect and storage-reader lifetime tests; no network/OCR."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import anyio
import pytest
from starlette.responses import FileResponse
from test_download_leases import consume
from test_download_leases import context as context

from services.api.app.cleanup_worker import IdleCleanupWorker
from services.api.app.downloads import LeasedDownloadResponse, _ClosingIterator
from services.api.app.jobs import JobDownloadLease, JobNotFoundError


class Source:
    def __init__(self, events, *, fails=False, close_fails=False):
        self.events = events
        self.fails = fails
        self.close_fails = close_fails
        self.reads = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.reads += 1
        if self.reads == 1:
            return b"fixture part"
        if self.fails:
            raise RuntimeError("PRIVATE provider failure")
        raise StopIteration

    def close(self):
        self.events.append("source-closed")
        if self.close_fails:
            raise RuntimeError("PRIVATE provider close failure")


def response_for(source, events):
    lease = JobDownloadLease(None, None, lambda: events.append("lease-released"))
    return LeasedDownloadResponse.streaming(
        source,
        lease,
        media_type="application/zip",
        headers={},
    )


@pytest.mark.parametrize("spec", ["2.0", "2.4"])
@pytest.mark.parametrize("fails", [False, True])
def test_storage_iterator_closes_before_lease_release_on_completion_or_error(spec, fails):
    events = []
    response = response_for(Source(events, fails=fails), events)
    if fails:
        with pytest.raises(RuntimeError, match="provider failure"):
            consume(response, spec=spec)
    else:
        consume(response, spec=spec)
    assert events == ["source-closed", "lease-released"]


@pytest.mark.parametrize("before_read", [False, True])
def test_incoming_disconnect_releases_even_when_body_iterator_did_not_finish(before_read):
    events = []
    response = response_for(Source(events), events)

    async def run():
        sent = anyio.Event()

        async def receive():
            if not before_read:
                await sent.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                sent.set()
                await anyio.sleep_forever()

        await response(
            {"type": "http", "method": "GET", "headers": [], "asgi": {"spec_version": "2.0"}},
            receive,
            send,
        )

    anyio.run(run)
    assert events == ["source-closed", "lease-released"]


def test_cancelled_response_closes_stream_under_shield_before_releasing():
    events = []
    response = response_for(Source(events), events)

    async def run():
        with anyio.CancelScope() as scope:

            async def send(message):
                if message["type"] == "http.response.body":
                    scope.cancel()
                    await anyio.sleep(0)

            async def receive():
                await anyio.sleep_forever()

            await response(
                {"type": "http", "method": "GET", "headers": [], "asgi": {"spec_version": "2.4"}},
                receive,
                send,
            )

    anyio.run(run)
    assert events == ["source-closed", "lease-released"]


def test_close_failure_does_not_expose_provider_message_or_skip_lease_release(caplog):
    events = []
    consume(response_for(Source(events, close_fails=True), events))
    assert events == ["source-closed", "lease-released"]
    assert "Download stream close failed." in caplog.text
    assert "PRIVATE" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_closing_iterator_waits_for_an_active_provider_read():
    entered, release, closing, closed = (threading.Event() for _ in range(4))

    class BlockingSource:
        def __next__(self):
            entered.set()
            assert release.wait(5)
            return b"fixture"

        def close(self):
            closed.set()

    source = _ClosingIterator(BlockingSource())
    with ThreadPoolExecutor(max_workers=2) as pool:
        reading = pool.submit(next, source)
        assert entered.wait(5)

        def close():
            closing.set()
            source.close()

        finishing = pool.submit(close)
        try:
            assert closing.wait(5)
            assert not closed.is_set()
        finally:
            release.set()
        assert reading.result(timeout=5) == b"fixture"
        finishing.result(timeout=5)
    assert closed.is_set()
    source.close()
    with pytest.raises(StopIteration):
        next(source)


@pytest.mark.parametrize(
    "range_value,expected_status,expected_body",
    [
        (None, 200, b"0123456789"),
        (b"bytes=2-5", 206, b"2345"),
        (b"bytes=99-100", 416, b""),
        (b"broken", 400, None),
    ],
)
def test_local_file_headers_and_ranges_survive_without_deferred_pathsend(
    tmp_path,
    range_value,
    expected_status,
    expected_body,
):
    path = tmp_path / "result.zip"
    path.write_bytes(b"0123456789")
    events = []
    lease = JobDownloadLease(path, None, lambda: events.append("released"))
    response = LeasedDownloadResponse(
        FileResponse(path, media_type="application/zip", filename="result.zip"),
        lease,
    )

    async def run():
        messages = []

        async def send(message):
            assert not events  # The lease lasts through the actual send call.
            messages.append(message)

        async def receive():
            await anyio.sleep_forever()

        scope = {
            "type": "http",
            "method": "GET",
            "asgi": {"spec_version": "2.4"},
            "headers": [] if range_value is None else [(b"range", range_value)],
            "extensions": {"http.response.pathsend": {}},
        }
        await response(scope, receive, send)
        assert "http.response.pathsend" in scope["extensions"]  # Not mutated.
        return messages

    messages = anyio.run(run)
    assert messages[0]["status"] == expected_status
    assert all(item["type"] != "http.response.pathsend" for item in messages)
    if expected_body is not None:
        assert b"".join(item.get("body", b"") for item in messages) == expected_body
    assert events == ["released"]


def test_idle_and_concurrent_cleanup_wait_for_active_response(context, monkeypatch):
    api, manager, record, storage, expected = context
    started, release, expired, deleted = (threading.Event() for _ in range(4))
    original = manager.cleanup_expired

    def cleanup():
        result = original()
        with manager._lock:
            if record.id in manager._pending_cleanup:
                expired.set()
            elif expired.is_set():
                deleted.set()
        return result

    monkeypatch.setattr(manager, "cleanup_expired", cleanup)
    manager._cleanup_worker = IdleCleanupWorker(cleanup, interval_seconds=0.01)
    response = api.download_conversion_job(record.id)

    async def paused_send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            started.set()
            assert await anyio.to_thread.run_sync(release.wait, 5)

    with ThreadPoolExecutor(max_workers=3) as pool:
        downloading = pool.submit(consume, response, send_override=paused_send)
        try:
            assert started.wait(5)
            with manager._lock:
                record.completed_at = datetime.now(UTC) - timedelta(seconds=61)
            manager.start_cleanup_worker()
            assert expired.wait(5)  # No intervening status/download/reserve request.
            assert record.workspace.exists()
            if record.output_artifact is not None:
                assert storage.exists(record.output_artifact)
            passes = [pool.submit(original) for _ in range(2)]
            assert [future.result(timeout=5) for future in passes] == [0, 0]
            with pytest.raises(JobNotFoundError):
                manager.acquire_download(record.id)
            assert not deleted.is_set()
        finally:
            release.set()
        messages = downloading.result(timeout=5)
    assert b"".join(item.get("body", b"") for item in messages) == expected
    assert deleted.wait(5)
    assert not record.workspace.exists()
    if record.output_artifact is not None:
        assert not storage.exists(record.output_artifact)
    assert manager.snapshot()["active_downloads"] == 0


@pytest.mark.parametrize("fails", [False, True])
def test_response_background_work_does_not_bypass_release(fails):
    from starlette.background import BackgroundTask

    events = []
    response = response_for(Source(events), events)

    def background():
        events.append("background")
        if fails:
            raise RuntimeError("fixture background failed")

    response.background = BackgroundTask(background)
    if fails:
        with pytest.raises(RuntimeError, match="background failed"):
            consume(response)
    else:
        consume(response)
    assert events == ["background", "source-closed", "lease-released"]


@pytest.mark.parametrize("phase", ["headers", "body"])
def test_actual_api_middleware_send_failure_releases_download(context, phase):
    api, manager, record, storage, _ = context
    reached = []

    async def run():
        request_sent = False

        async def receive():
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": b"", "more_body": False}
            await anyio.sleep_forever()

        async def send(message):
            target = "http.response.start" if phase == "headers" else "http.response.body"
            if message["type"] == target:
                reached.append(True)
                raise OSError("fixture client disconnected")

        path = f"/v1/jobs/{record.id}/download"
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [],
            "server": ("test", 80),
            "client": ("fixture", 1),
            "root_path": "",
        }
        await api.app(scope, receive, send)

    with pytest.raises(OSError, match="fixture client disconnected"):
        anyio.run(run)
    assert reached
    assert manager.snapshot()["active_downloads"] == 0
    if hasattr(storage, "client"):
        body = storage.client.last_body
        if body is not None:
            assert body.closed
