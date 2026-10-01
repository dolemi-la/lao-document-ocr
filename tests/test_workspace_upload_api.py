"""Submission request lifetime, using real upload copying and authored PNGs."""

from __future__ import annotations

import asyncio
import io
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from PIL import Image
from starlette.requests import Request
from test_workspace_ownership import manager_for
from test_workspace_ownership import storage as storage

import services.api.app.main as api
from services.api.app.cleanup_journal import CleanupJournalError


def png():
    output = io.BytesIO()
    Image.new("RGB", (12, 8), "white").save(output, format="PNG")
    return output.getvalue()


def request():
    return Request({"type": "http", "method": "POST", "path": "/v1/jobs", "headers": []})


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("journal_failure", [False, True])
def test_cancelled_request_finishes_every_upload_owner_even_when_abort_commit_fails(
    tmp_path,
    storage,
    monkeypatch,
    batch,
    journal_failure,
):
    root = tmp_path / "jobs"
    manager = manager_for(root, storage)
    monkeypatch.setattr(api, "JOB_MANAGER", manager)
    monkeypatch.setattr(api, "_enforce_submission_rate_limit", lambda *args, **kwargs: None)
    recorded = []
    reserve = manager.reserve_many

    def remember(*args, **kwargs):
        records = reserve(*args, **kwargs)
        recorded.extend(records)
        return records

    monkeypatch.setattr(manager, "reserve_many", remember)
    if journal_failure:

        def unavailable(*args, **kwargs):
            raise CleanupJournalError("PRIVATE abort commit path")

        monkeypatch.setattr(manager._journal, "expire", unavailable)

    async def scenario():
        uploads = [
            UploadFile(io.BytesIO(png()), filename=f"{i}.png") for i in range(3 if batch else 1)
        ]
        paused = asyncio.Event()
        calls = 0

        async def read(size=-1):
            nonlocal calls
            calls += 1
            if calls == 1:
                return png()
            paused.set()
            await asyncio.Future()

        uploads[1 if batch else 0].read = read
        coroutine = (
            api.create_conversion_batch(request(), uploads, False, None)
            if batch
            else api.create_conversion_job(request(), uploads[0], False, None)
        )
        task = asyncio.create_task(coroutine)
        await asyncio.wait_for(paused.wait(), 5)
        assert len(manager._upload_owners) == len(uploads)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        for upload in uploads:
            await upload.close()

    try:
        asyncio.run(scenario())
        assert not manager._upload_owners
        assert all(record.future is None for record in recorded)
        if journal_failure:
            assert len(manager._journal.load()) == len(recorded)
            assert manager._journal_failed
        else:
            assert not manager._jobs
            assert not manager._journal.load()
            assert all(not record.workspace.exists() for record in recorded)
    finally:
        manager.shutdown()
    recovered = manager_for(root, storage)
    try:
        recovered.cleanup_expired(now=datetime.now(UTC) + timedelta(hours=2))
        assert not recovered._journal.load()
        assert all(not record.workspace.exists() for record in recorded)
    finally:
        recovered.shutdown()


@pytest.mark.parametrize("batch", [False, True])
def test_shutdown_during_real_upload_rejects_handoff_without_releasing_a_live_writer(
    tmp_path,
    storage,
    monkeypatch,
    batch,
):
    root = tmp_path / "jobs"
    manager = manager_for(root, storage)
    monkeypatch.setattr(api, "JOB_MANAGER", manager)
    monkeypatch.setattr(api, "_enforce_submission_rate_limit", lambda *args, **kwargs: None)

    async def scenario():
        uploads = [
            UploadFile(io.BytesIO(png()), filename=f"{i}.png") for i in range(3 if batch else 1)
        ]
        paused, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def read(size=-1):
            nonlocal calls
            calls += 1
            if calls == 1:
                return png()
            paused.set()
            await release.wait()
            return b""

        uploads[0].read = read
        task = asyncio.create_task(
            api.create_conversion_batch(request(), uploads, False, None)
            if batch
            else api.create_conversion_job(request(), uploads[0], False, None)
        )
        await asyncio.wait_for(paused.wait(), 5)
        manager.shutdown()
        with pytest.raises(CleanupJournalError):
            manager_for(root, storage)
        release.set()
        with pytest.raises(HTTPException) as error:
            await task
        assert error.value.status_code == 503
        assert not manager._upload_owners
        for upload in uploads:
            await upload.close()

    try:
        asyncio.run(scenario())
    finally:
        manager.shutdown()
    recovered = manager_for(root, storage)
    try:
        assert len(recovered._pending_cleanup) == (3 if batch else 1)
        recovered.cleanup_expired(now=datetime.now(UTC) + timedelta(hours=2))
        assert not recovered._journal.load()
    finally:
        recovered.shutdown()


@pytest.mark.parametrize("batch", [False, True])
def test_http_invalid_upload_is_not_an_untracked_or_active_job(
    tmp_path, storage, monkeypatch, batch
):
    manager = manager_for(tmp_path / "jobs", storage)
    monkeypatch.setattr(api, "JOB_MANAGER", manager)
    monkeypatch.setattr(api, "_enforce_submission_rate_limit", lambda *args, **kwargs: None)
    with TestClient(api.app) as client:
        if batch:
            response = client.post(
                "/v1/jobs/batch",
                files=[
                    ("files", ("one.png", png(), "image/png")),
                    ("files", ("bad.png", b"not an image", "image/png")),
                    ("files", ("three.png", png(), "image/png")),
                ],
            )
        else:
            response = client.post(
                "/v1/jobs",
                files={
                    "file": ("bad.png", b"not an image", "image/png"),
                },
            )
        assert response.status_code == 422
        assert not manager._upload_owners
        assert not manager._jobs
        assert not manager._journal.load()


@pytest.mark.parametrize("batch", [False, True])
def test_http_reservation_commit_failure_never_starts_copying(
    tmp_path, storage, monkeypatch, batch
):
    manager = manager_for(tmp_path / "jobs", storage)
    monkeypatch.setattr(api, "JOB_MANAGER", manager)
    monkeypatch.setattr(api, "_enforce_submission_rate_limit", lambda *args, **kwargs: None)
    copied = []

    async def copy(*args, **kwargs):
        copied.append("unexpected copy")

    def fail(*args, **kwargs):
        raise CleanupJournalError("PRIVATE reservation journal detail")

    monkeypatch.setattr(api, "_save_upload", copy)
    with TestClient(api.app) as client:
        monkeypatch.setattr(manager._journal, "add", fail)
        field = "files" if batch else "file"
        path = "/v1/jobs/batch" if batch else "/v1/jobs"
        response = client.post(path, files={field: ("fixture.png", png(), "image/png")})
        assert response.status_code == 503
        assert response.json() == {"detail": "Conversion service is unavailable. Try again later."}
        assert copied == []
        assert not manager._upload_owners
        assert not manager._jobs


@pytest.mark.parametrize("journal_failure", [False, True])
def test_partial_batch_enqueue_failure_releases_all_uploads_but_keeps_running_workspace(
    tmp_path,
    storage,
    monkeypatch,
    journal_failure,
):
    import threading

    entered, release = threading.Event(), threading.Event()
    records = []

    def runner(record, _):
        entered.set()
        assert release.wait(10)
        assert record.workspace.exists()
        return record.input_path

    root = tmp_path / "jobs"
    manager = manager_for(root, storage, runner=runner, max_workers=1)
    monkeypatch.setattr(api, "JOB_MANAGER", manager)
    monkeypatch.setattr(api, "_enforce_submission_rate_limit", lambda *args, **kwargs: None)
    reserve = manager.reserve_many
    submit = manager._executor.submit
    attempts = 0

    def remember(*args, **kwargs):
        created = reserve(*args, **kwargs)
        records.extend(created)
        return created

    def unavailable(*args, **kwargs):
        raise CleanupJournalError("PRIVATE terminal journal detail")

    def rejected_submit(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return submit(*args, **kwargs)
        if journal_failure:
            monkeypatch.setattr(manager._journal, "update", unavailable)
        raise RuntimeError("PRIVATE executor thread-start detail")

    enqueue = manager.enqueue

    def wait_after_enqueue(job_id):
        result = enqueue(job_id)
        if job_id == records[0].id:
            # The manager lock is released only after enqueue returns.
            assert entered.wait(5)
        return result

    monkeypatch.setattr(manager, "reserve_many", remember)
    monkeypatch.setattr(manager._executor, "submit", rejected_submit)
    monkeypatch.setattr(manager, "enqueue", wait_after_enqueue)
    client = TestClient(api.app)
    try:
        response = client.post(
            "/v1/jobs/batch", files=[("files", (f"{i}.png", png(), "image/png")) for i in range(3)]
        )
        assert response.status_code == 503
        assert "PRIVATE" not in response.text
        assert not manager._upload_owners
        assert len(records) == 3
        assert records[0].workspace.exists()
        assert records[0].cancel_event.is_set()
        if not journal_failure:
            assert all(not record.workspace.exists() for record in records[1:])
    finally:
        release.set()
        for record in records:
            if record.future is not None:
                try:
                    record.future.result(timeout=5)
                except CleanupJournalError:
                    assert journal_failure
        client.close()
        manager.shutdown()
    recovered = manager_for(root, storage)
    try:
        recovered.cleanup_expired(now=datetime.now(UTC) + timedelta(hours=2))
        assert not recovered._journal.load()
        assert all(not record.workspace.exists() for record in records)
    finally:
        recovered.shutdown()
