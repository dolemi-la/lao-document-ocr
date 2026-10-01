"""Uncertain journal commits must pause work, not silently discard ownership."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime, timedelta

import pytest
from PIL import Image
from test_durable_cleanup import finish, manager_for
from test_durable_cleanup import storage as storage

from services.api.app.cleanup_journal import CleanupJournalError
from services.api.app.jobs import JobManagerClosedError, JobNotFoundError


@pytest.mark.parametrize("committed", [False, True])
def test_failed_expiration_commit_never_deletes_and_latches_admission(
    tmp_path,
    storage,
    monkeypatch,
    caplog,
    committed,
):
    root = tmp_path / "jobs"
    calls = []
    original = manager_for(root, storage, artifact_cleanup=lambda artifact: calls.append(artifact))
    try:
        record = finish(original)
        expiry = record.completed_at + timedelta(hours=2)
        save = original._journal.expire

        def uncertain(entries, *, owned_ids):
            if committed:
                save(entries, owned_ids=owned_ids)
            raise CleanupJournalError("PRIVATE journal path and SQL")

        monkeypatch.setattr(original._journal, "expire", uncertain)
        with pytest.raises(CleanupJournalError) as error:
            original.cleanup_expired(now=expiry)
        assert "PRIVATE" not in str(error.value)
        assert "PRIVATE" not in caplog.text
        assert calls == []
        assert record.workspace.exists()
        assert storage.exists(record.output_artifact)
        assert record.id in original._jobs
        assert not original._pending_cleanup
        monkeypatch.setattr(original._journal, "expire", save)
        with pytest.raises(CleanupJournalError):
            original.cleanup_expired(now=expiry)
        with pytest.raises(CleanupJournalError):
            original.reserve("another.png", ".png")
        with pytest.raises(CleanupJournalError):
            original.cancel(record.id)
    finally:
        original.shutdown()
    replacement = manager_for(root, storage)
    try:
        # Both outcomes retain ownership: the earlier terminal row exists
        # even if the explicit expiry transfer never committed.
        assert len(replacement._pending_cleanup) == 1
        replacement.cleanup_expired(now=expiry)
        assert not storage.exists(record.output_artifact)
        assert not record.workspace.exists()
    finally:
        replacement.shutdown()


@pytest.mark.parametrize("committed", [False, True])
def test_failed_completion_commit_recovers_idempotently_without_false_receipt(
    tmp_path,
    storage,
    monkeypatch,
    caplog,
    committed,
):
    root = tmp_path / "jobs"
    original = manager_for(root, storage)
    try:
        record = finish(original)
        artifact = record.output_artifact
        expiry = record.completed_at + timedelta(hours=2)
        update = original._journal.update

        def uncertain(entry):
            if committed:
                update(entry)
            raise CleanupJournalError("PRIVATE journal failed after deletion")

        monkeypatch.setattr(original._journal, "update", uncertain)
        with pytest.raises(CleanupJournalError):
            original.cleanup_expired(now=expiry)
        assert not storage.exists(artifact)
        assert not record.workspace.exists()
        assert record.id not in original._jobs
        assert record.id in original._pending_cleanup
        pending = original._pending_cleanup[record.id]
        assert pending.artifact is not None  # No uncommitted receipt in memory.
        assert pending.workspace_pending
        assert not pending.in_progress
        assert "PRIVATE" not in caplog.text
    finally:
        original.shutdown()
    calls = []

    def delete(artifact):
        calls.append(artifact.key)
        storage.delete(artifact)

    replacement = manager_for(root, storage, artifact_cleanup=delete)
    try:
        assert len(replacement._pending_cleanup) == int(not committed)
        replacement.cleanup_expired(now=expiry)
        assert calls == ([] if committed else [artifact.key])
        assert not replacement._pending_cleanup
        with pytest.raises(JobNotFoundError):
            replacement.public(record.id)
    finally:
        replacement.shutdown()


def test_nonwaiting_shutdown_keeps_journal_until_waiting_shutdown(tmp_path, storage):
    root = tmp_path / "jobs"
    original = manager_for(root, storage)
    try:
        record = finish(original)
        original.shutdown(wait=False)
        with pytest.raises(CleanupJournalError):
            manager_for(root, storage)
        with pytest.raises(JobManagerClosedError):
            original.acquire_download(record.id)
        original.shutdown(wait=True)
        replacement = manager_for(root, storage)
        replacement.shutdown()
    finally:
        original.shutdown()


def png():
    output = io.BytesIO()
    Image.new("RGB", (24, 24), "white").save(output, format="PNG")
    return output.getvalue()


def test_http_journal_failure_is_private_and_does_not_admit_new_work(
    tmp_path,
    storage,
    monkeypatch,
    caplog,
):
    from fastapi.testclient import TestClient

    import services.api.app.main as api

    manager = manager_for(tmp_path / "jobs", storage)
    monkeypatch.setattr(api, "JOB_MANAGER", manager)
    monkeypatch.setattr(api, "RESULT_STORAGE", storage)
    monkeypatch.setattr(api, "_enforce_submission_rate_limit", lambda *args, **kwargs: None)
    with TestClient(api.app) as client:
        created = client.post("/v1/jobs", files={"file": ("fixture.png", png(), "image/png")})
        assert created.status_code == 202
        record = manager.get_record(created.json()["id"])
        record.future.result(timeout=5)
        with manager._lock:
            record.completed_at -= timedelta(hours=2)

        def unavailable(*args, **kwargs):
            raise CleanupJournalError("PRIVATE path, SQL, credential")

        monkeypatch.setattr(manager._journal, "expire", unavailable)
        expected = {"detail": "Conversion service is unavailable. Try again later."}
        for method, path in [
            ("GET", f"/v1/jobs/{record.id}"),
            ("DELETE", f"/v1/jobs/{record.id}"),
            ("GET", f"/v1/jobs/{record.id}/download"),
            ("GET", "/metrics"),
        ]:
            response = client.request(method, path)
            assert response.status_code == 503
            assert response.json() == expected
        rejected = client.post("/v1/jobs", files={"file": ("other.png", png(), "image/png")})
        assert rejected.status_code == 503
        assert rejected.json() == expected
        assert len(manager._jobs) == 1
        assert record.workspace.exists()
        assert storage.exists(record.output_artifact)
        assert "PRIVATE" not in caplog.text
        assert not manager._pending_cleanup


def test_http_restart_keeps_expired_job_unavailable_and_accepts_new_jobs(
    tmp_path,
    storage,
    monkeypatch,
):
    from fastapi.testclient import TestClient

    import services.api.app.main as api

    root = tmp_path / "jobs"

    def fail(_):
        raise OSError("PRIVATE provider failure")

    original = manager_for(root, storage, artifact_cleanup=fail)
    monkeypatch.setattr(api, "JOB_MANAGER", original)
    monkeypatch.setattr(api, "RESULT_STORAGE", storage)
    monkeypatch.setattr(api, "_enforce_submission_rate_limit", lambda *args, **kwargs: None)
    with TestClient(api.app) as client:
        created = client.post("/v1/jobs", files={"file": ("first.png", png(), "image/png")})
        assert created.status_code == 202
        record = original.get_record(created.json()["id"])
        record.future.result(timeout=5)
        with original._lock:
            record.completed_at -= timedelta(hours=2)
        assert original.cleanup_expired() == 1
        assert client.get(f"/v1/jobs/{record.id}").status_code == 404
    replacement = manager_for(root, storage)
    monkeypatch.setattr(api, "JOB_MANAGER", replacement)
    with TestClient(api.app) as client:
        for method, path in [
            ("GET", f"/v1/jobs/{record.id}"),
            ("DELETE", f"/v1/jobs/{record.id}"),
            ("GET", f"/v1/jobs/{record.id}/download"),
        ]:
            response = client.request(method, path)
            assert response.status_code == 404
            assert response.json() == {"detail": "Job not found."}
        assert replacement.snapshot()["cleanup_pending_jobs"] == 1
        assert replacement.snapshot()["completed_total"] == {}
        assert "PRIVATE" not in client.get("/metrics").text
        replacement.cleanup_expired(now=datetime.now(UTC) + timedelta(minutes=1))
        assert not storage.exists(record.output_artifact)
        created = client.post("/v1/jobs", files={"file": ("new.png", png(), "image/png")})
        assert created.status_code == 202
        new = replacement.get_record(created.json()["id"])
        new.future.result(timeout=5)
        result = client.get(f"/v1/jobs/{new.id}")
        assert result.status_code == 200
        assert result.json()["download_ready"] is True
        assert "journal" not in json.dumps(result.json())
        assert client.get(f"/v1/jobs/{new.id}/download").content == b"authored fixture archive"
