"""Actual HTTP admission/rollback routes with fixture archives, not OCR evaluation."""

from __future__ import annotations

import io
import json
import threading
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import services.api.app.main as api
from services.api.app.jobs import ConversionJobManager, JobCancelledError, JobStatus

UNAVAILABLE = {"detail": "Conversion service is unavailable. Try again later."}


def image_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (8, 6), "white").save(stream, format="PNG")
    return stream.getvalue()


def fixture_archive(record, cancel_event):
    result = record.workspace / "result.zip"
    with zipfile.ZipFile(result, "w") as archive:
        archive.writestr("result.txt", "authored fixture result")
    return result


@pytest.fixture
def context(tmp_path, monkeypatch):
    manager = ConversionJobManager(
        tmp_path / "jobs",
        fixture_archive,
        max_workers=1,
        max_active_jobs=6,
        cleanup_interval_seconds=0,
    )
    monkeypatch.setattr(api, "JOB_MANAGER", manager)
    monkeypatch.setattr(api, "_enforce_submission_rate_limit", lambda *args, **kwargs: None)
    client = TestClient(api.app)
    try:
        yield client, manager
    finally:
        client.close()
        manager.shutdown()


def submit(client, batch):
    options = {"data": {"rotate_page": "1:0", "auto_orient_right_angles": "true"}}
    if batch:
        options["files"] = [
            ("files", (name, image_bytes(), "image/png"))
            for name in ("one.png", "two.png", "three.png")
        ]
        return client.post("/v1/jobs/batch", **options)
    options["files"] = {"file": ("one.png", image_bytes(), "image/png")}
    return client.post("/v1/jobs", **options)


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("point", ["before-reservation", "during-upload", "executor-rejection"])
def test_unavailable_service_returns_503_and_cleans_only_request_owned_files(
    context,
    monkeypatch,
    batch,
    point,
    caplog,
):
    client, manager = context
    sibling = manager.root_dir / "unrelated.txt"
    sibling.write_bytes(b"must remain")
    records, ran = [], []
    original_reserve = manager.reserve_many

    def record_reservations(*args, **kwargs):
        result = original_reserve(*args, **kwargs)
        records.extend(result)
        return result

    monkeypatch.setattr(manager, "reserve_many", record_reservations)
    monkeypatch.setattr(manager, "runner", lambda *args: ran.append("unexpected"))
    if point == "before-reservation":
        manager.shutdown()
    elif point == "during-upload":
        original_save = api._save_upload

        async def stop_after_copy(upload, destination):
            await original_save(upload, destination)
            manager.shutdown(wait=False)
            # Closing admission must not delete uploads while this owner is copying.
            assert destination.is_file()

        monkeypatch.setattr(api, "_save_upload", stop_after_copy)
    else:

        def reject(*args, **kwargs):
            raise RuntimeError("PRIVATE executor detail /secret/storage")

        monkeypatch.setattr(manager._executor, "submit", reject)

    response = submit(client, batch)
    assert response.status_code == 503
    assert response.json() == UNAVAILABLE
    assert "PRIVATE" not in caplog.text
    assert not ran
    assert list(manager.root_dir.iterdir()) == [sibling]
    assert sibling.read_bytes() == b"must remain"
    assert not manager._jobs
    assert manager.snapshot()["active_jobs"] == 0
    assert all(record.future is None for record in records)


@pytest.mark.parametrize("batch", [False, True])
def test_uncertain_executor_rejection_never_runs_discarded_upload(context, monkeypatch, batch):
    client, manager = context
    submitted, ran = [], []
    original_submit = manager._executor.submit

    def queue_then_raise(*args, **kwargs):
        submitted.append(original_submit(*args, **kwargs))
        raise RuntimeError("PRIVATE thread-start failure")

    monkeypatch.setattr(manager._executor, "submit", queue_then_raise)
    monkeypatch.setattr(manager, "runner", lambda *args: ran.append("unexpected"))
    response = submit(client, batch)
    assert response.status_code == 503
    assert response.json() == UNAVAILABLE
    for future in submitted:
        future.result(timeout=5)
    assert not ran
    assert not manager._jobs
    assert not list(manager.root_dir.iterdir())
    assert manager.snapshot()["completed_total"] == {"failed": 1}


@pytest.mark.parametrize("failure", ["shutdown", "scheduler"])
def test_partial_batch_rejection_releases_unstarted_tail_without_disturbing_unrelated_job(
    context,
    monkeypatch,
    failure,
):
    client, manager = context
    entered, release = threading.Event(), threading.Event()
    ran = []

    def held_runner(record, cancel_event):
        ran.append(record.id)
        entered.set()
        assert release.wait(5)
        assert not cancel_event.is_set()
        return fixture_archive(record, cancel_event)

    monkeypatch.setattr(manager, "runner", held_runner)
    unrelated = manager.reserve("unrelated.png", ".png", page_rotations={1: 270})
    unrelated.input_path.write_bytes(b"unrelated input")
    manager.enqueue(unrelated.id)
    assert entered.wait(5)
    original_enqueue, original_submit = manager.enqueue, manager._executor.submit
    attempts = []

    def reject_submission(*args, **kwargs):
        raise RuntimeError("PRIVATE queue rejection")

    def interrupted_enqueue(job_id):
        attempts.append(manager.get_record(job_id))
        if len(attempts) == 2:
            if failure == "shutdown":
                manager.shutdown(wait=False)
            else:
                monkeypatch.setattr(manager._executor, "submit", reject_submission)
        return original_enqueue(job_id)

    monkeypatch.setattr(manager, "enqueue", interrupted_enqueue)
    try:
        response = submit(client, True)
        assert response.status_code == 503
        assert response.json() == UNAVAILABLE
        assert len(attempts) == 2
        admitted, rejected = attempts
        assert admitted.status == JobStatus.CANCELLED
        assert admitted.future.cancelled()
        assert admitted.workspace.exists()
        assert not rejected.workspace.exists()
        assert set(manager._jobs) == {unrelated.id, admitted.id}
        assert unrelated.status == JobStatus.RUNNING
        assert unrelated.input_path.read_bytes() == b"unrelated input"
        assert not unrelated.cancel_event.is_set()
        assert manager.public(admitted.id)["page_rotations"] == [
            {"page": 1, "degrees_clockwise": 0}
        ]
        assert manager.public(admitted.id)["orientation_review"] is None
        assert manager.snapshot()["active_jobs"] == 1
        release.set()
        unrelated.future.result(timeout=5)
        assert ran == [unrelated.id]
        assert unrelated.status == JobStatus.SUCCEEDED
        assert manager.public(unrelated.id)["download_ready"]
    finally:
        release.set()
        monkeypatch.setattr(manager._executor, "submit", original_submit)


def test_partial_batch_cancels_running_member_without_unlinking_its_workspace(context, monkeypatch):
    client, manager = context
    entered, release = threading.Event(), threading.Event()
    running = []

    def held_runner(record, cancel_event):
        running.append(record)
        entered.set()
        assert release.wait(5)
        assert record.input_path.is_file()
        if cancel_event.is_set():
            raise JobCancelledError()
        return fixture_archive(record, cancel_event)

    original_enqueue = manager.enqueue
    calls = []

    def reject(*args, **kwargs):
        raise RuntimeError("PRIVATE scheduling details")

    def interrupted_enqueue(job_id):
        calls.append(job_id)
        if len(calls) == 2:
            assert entered.wait(5)
            monkeypatch.setattr(manager._executor, "submit", reject)
        return original_enqueue(job_id)

    monkeypatch.setattr(manager, "runner", held_runner)
    monkeypatch.setattr(manager, "enqueue", interrupted_enqueue)
    try:
        response = submit(client, True)
        assert response.status_code == 503
        assert len(running) == 1
        record = running[0]
        assert record.cancel_event.is_set()
        assert record.status == JobStatus.RUNNING  # Cancellation is still cooperative.
        assert record.input_path.is_file()
        assert set(manager._jobs) == {record.id}
        release.set()
        record.future.result(timeout=5)
        assert record.status == JobStatus.CANCELLED
        assert record.workspace.exists()
        assert manager.snapshot()["active_jobs"] == 0
    finally:
        release.set()


@pytest.mark.parametrize("batch", [False, True])
def test_upload_request_does_not_switch_manager_owner_mid_submission(
    context,
    tmp_path,
    monkeypatch,
    batch,
):
    client, original = context
    replacement = ConversionJobManager(
        tmp_path / "replacement",
        fixture_archive,
        cleanup_interval_seconds=0,
    )
    other = replacement.reserve("other.png", ".png")
    other.input_path.write_bytes(b"replacement-owned upload")
    save = api._save_upload

    async def swap_after_copy(upload, destination):
        await save(upload, destination)
        original.shutdown(wait=False)
        monkeypatch.setattr(api, "JOB_MANAGER", replacement)

    monkeypatch.setattr(api, "_save_upload", swap_after_copy)
    try:
        response = submit(client, batch)
        assert response.status_code == 503
        assert response.json() == UNAVAILABLE
        assert not original._jobs
        assert not list(original.root_dir.iterdir())
        assert set(replacement._jobs) == {other.id}
        assert other.input_path.read_bytes() == b"replacement-owned upload"
        assert other.status == JobStatus.UPLOADING
    finally:
        replacement.shutdown()


def test_shutdown_retains_existing_successful_download_and_manual_acknowledgement(context):
    client, manager = context
    record = manager.reserve("original.png", ".png", page_rotations={1: 0})
    manager.enqueue(record.id)
    record.future.result(timeout=5)
    manager.shutdown()
    response = client.get(f"/v1/jobs/{record.id}")
    assert response.status_code == 200
    assert response.json()["status"] == "succeeded"
    assert response.json()["page_rotations"] == [{"page": 1, "degrees_clockwise": 0}]
    download = client.get(f"/v1/jobs/{record.id}/download")
    assert download.status_code == 200
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        assert archive.read("result.txt") == b"authored fixture result"
    assert manager.snapshot()["active_downloads"] == 0
    assert "PRIVATE" not in json.dumps(response.json())


@pytest.mark.parametrize("batch", [False, True])
def test_repeated_http_submissions_cannot_feed_a_failed_executor(context, monkeypatch, batch):
    client, manager = context
    submissions = []

    def reject(*args, **kwargs):
        submissions.append(1)
        raise RuntimeError("PRIVATE executor failure")

    monkeypatch.setattr(manager._executor, "submit", reject)
    for _ in range(3):
        response = submit(client, batch)
        assert response.status_code == 503
        assert response.json() == UNAVAILABLE
    assert submissions == [1]
    assert not list(manager.root_dir.iterdir())
    assert not manager._jobs
    assert manager.snapshot()["completed_total"] == {"failed": 1}


def test_partial_batch_rejection_does_not_erase_an_already_completed_member(context, monkeypatch):
    client, manager = context
    original_enqueue = manager.enqueue
    admitted = []

    def reject(*args, **kwargs):
        raise RuntimeError("PRIVATE executor failure")

    def sequential_enqueue(job_id):
        if admitted:
            monkeypatch.setattr(manager._executor, "submit", reject)
            return original_enqueue(job_id)
        record = original_enqueue(job_id)
        record.future.result(timeout=5)
        admitted.append(record)
        return record

    monkeypatch.setattr(manager, "enqueue", sequential_enqueue)
    response = submit(client, True)
    assert response.status_code == 503
    assert response.json() == UNAVAILABLE
    assert len(admitted) == 1
    completed = admitted[0]
    assert completed.status == JobStatus.SUCCEEDED
    assert completed.output_path.is_file()
    assert set(manager._jobs) == {completed.id}
    assert manager.snapshot()["active_jobs"] == 0
    download = client.get(f"/v1/jobs/{completed.id}/download")
    assert download.status_code == 200
    assert manager.snapshot()["active_downloads"] == 0
