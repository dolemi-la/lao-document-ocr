"""Crash, HTTP, and transaction-boundary checks for managed result publication."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from test_durable_cleanup import storage as storage
from test_durable_cleanup_failures import png
from test_result_publication import completed, tracked_manager

from services.api.app.cleanup_journal import CleanupEntry, CleanupJournal, CleanupJournalError
from services.api.app.jobs import ConversionJobManager, JobNotFoundError
from services.api.app.storage import FilesystemArtifactStorage


@pytest.mark.parametrize(
    "phase",
    [
        "before-write",
        "stored-no-return",
        "before-terminal-commit",
        "after-terminal-commit",
        "cancellation-before-delete",
        "cancellation-after-delete",
    ],
)
def test_abrupt_process_exit_keeps_preexpiry_ownership(tmp_path, phase):
    script = """
import os, sys
from pathlib import Path
from services.api.app.jobs import ConversionJobManager
from services.api.app.storage import FilesystemArtifactStorage
root, phase = Path(sys.argv[1]), sys.argv[2]
storage = FilesystemArtifactStorage(root / "objects")
def runner(record, _):
    archive = record.workspace / "result.zip"
    archive.write_bytes(b"authored crash fixture")
    key = f"jobs/{record.id}/result.zip"
    def put():
        if phase == "before-write":
            os._exit(71)
        result = storage.put_file(archive, key=key, filename="result.zip",
                                  media_type="application/zip")
        if phase == "stored-no-return":
            os._exit(71)
        if phase.startswith("cancellation-"):
            record.cancel_event.set()
        return result
    return record.publish_result(key, put)
def delete(artifact):
    if phase == "cancellation-before-delete":
        os._exit(71)
    storage.delete(artifact)
    if phase == "cancellation-after-delete":
        os._exit(71)
manager = ConversionJobManager(root / "jobs", runner, durable_cleanup=True,
    cleanup_namespace=storage.cleanup_namespace(), artifact_cleanup=delete,
    cleanup_interval_seconds=0)
if phase == "before-terminal-commit":
    manager._journal.update = lambda _: os._exit(71)
record = manager.reserve("private.png", ".png")
manager.enqueue(record.id)
record.future.result(timeout=5)
if phase == "after-terminal-commit":
    os._exit(71)
raise AssertionError("exit point not reached")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), phase],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 71, result.stderr
    storage = FilesystemArtifactStorage(tmp_path / "objects")
    calls = []
    runs = []

    def delete(artifact):
        calls.append(artifact.key)
        storage.delete(artifact)

    recovered = ConversionJobManager(
        tmp_path / "jobs",
        lambda *args: runs.append("unexpected OCR"),
        durable_cleanup=True,
        cleanup_namespace=storage.cleanup_namespace(),
        artifact_cleanup=delete,
        cleanup_interval_seconds=0,
    )
    try:
        assert calls == []  # Startup must not invoke the provider.
        assert len(recovered._pending_cleanup) == 1
        job_id, pending = next(iter(recovered._pending_cleanup.items()))
        artifact = pending.artifact
        assert artifact.key == f"jobs/{job_id}/result.zip"
        assert pending.workspace_pending
        assert pending.workspace.exists()
        assert storage.exists(artifact) == (
            phase not in {"before-write", "cancellation-after-delete"}
        )
        for operation in (recovered.public, recovered.cancel, recovered.acquire_download):
            with pytest.raises(JobNotFoundError):
                operation(job_id)
        assert not calls
        recovered.cleanup_expired(now=pending.retry_at)
        assert calls == [artifact.key]
        assert not storage.exists(artifact)
        assert not pending.workspace.exists()
        assert not recovered._journal.load()
        assert not runs
        assert recovered.snapshot()["completed_total"] == {}
    finally:
        recovered.shutdown()


@pytest.mark.parametrize("committed", [False, True])
def test_uncertain_expiry_transfer_keeps_prior_publication_row(
    tmp_path,
    storage,
    monkeypatch,
    committed,
):
    root = tmp_path / "jobs"
    original = tracked_manager(root, storage)
    try:
        record = completed(original)
        expiry = record.completed_at + timedelta(hours=2)
        transfer = original._journal.expire

        def uncertain(entries, *, owned_ids):
            if committed:
                transfer(entries, owned_ids=owned_ids)
            raise CleanupJournalError("PRIVATE expiry SQL")

        monkeypatch.setattr(original._journal, "expire", uncertain)
        with pytest.raises(CleanupJournalError):
            original.cleanup_expired(now=expiry)
        assert record.id in original._jobs
        assert not original._pending_cleanup
        assert record.workspace.exists()
        assert storage.exists(record.output_artifact)
    finally:
        original.shutdown()
    replacement = tracked_manager(root, storage)
    try:
        # Either transaction outcome still has a durable owner from publication.
        assert len(replacement._pending_cleanup) == 1
        replacement.cleanup_expired(now=expiry)
        assert not storage.exists(record.output_artifact)
        assert not record.workspace.exists()
    finally:
        replacement.shutdown()


def test_transfer_mixes_old_and_tracked_rows_without_double_counting(tmp_path):
    root = tmp_path / "jobs"
    root.mkdir(mode=0o700)
    ledger = CleanupJournal(root, {"backend": "test"}, max_entries=2)
    first = CleanupEntry("a" * 32, "jobs/" + "a" * 32 + "/result.zip", True, datetime.now(UTC))
    second = replace(first, job_id="b" * 32, artifact_key=None)
    try:
        ledger.add([first])
        ledger.expire([first, second], owned_ids={first.job_id})
        assert len(ledger.load()) == 2
    finally:
        ledger.close()


@pytest.mark.parametrize(
    "fault", ["missing", "wrong-key", "finished-workspace", "unknown-id", "duplicate"]
)
def test_transfer_rejects_invalid_ownership_atomically(tmp_path, fault):
    root = tmp_path / "jobs"
    root.mkdir(mode=0o700)
    ledger = CleanupJournal(root, {"backend": "test"}, max_entries=3)
    first = CleanupEntry("a" * 32, "jobs/" + "a" * 32 + "/result.zip", True, datetime.now(UTC))
    second = replace(first, job_id="b" * 32, artifact_key=None)
    try:
        if fault != "missing":
            ledger.add(
                [
                    replace(first, workspace_pending=False)
                    if fault == "finished-workspace"
                    else first
                ]
            )
        before = ledger.load()
        changed = (
            replace(first, artifact_key="jobs/" + "a" * 32 + "/different.zip")
            if fault == "wrong-key"
            else first
        )
        entries = [second, changed] if fault != "duplicate" else [first, first]
        owned = {first.job_id, "c" * 32} if fault == "unknown-id" else {first.job_id}
        with pytest.raises(CleanupJournalError):
            ledger.expire(entries, owned_ids=owned)
        assert ledger.load() == before
    finally:
        ledger.close()


@pytest.mark.parametrize("lost_reply", [False, True])
def test_http_real_runner_journals_before_storage_and_recovers_after_restart(
    tmp_path,
    storage,
    monkeypatch,
    caplog,
    lost_reply,
):
    from fastapi.testclient import TestClient

    import services.api.app.main as api
    from lao_document_ocr.models import Document, Page

    root = tmp_path / "jobs"
    document = Document(pages=[Page(number=1, width=10, height=10)])
    monkeypatch.setattr(api, "_engine", lambda: None)
    monkeypatch.setattr(api, "process_document", lambda *args, **kwargs: document)
    monkeypatch.setattr(api, "RESULT_STORAGE", storage)
    monkeypatch.setattr(api, "_enforce_submission_rate_limit", lambda *args, **kwargs: None)

    def archive(document, workspace, filename):
        output = workspace / "result.zip"
        output.write_bytes(b"authored HTTP fixture")
        return output

    monkeypatch.setattr(api, "_write_outputs_archive", archive)
    original = ConversionJobManager(
        root,
        api._run_conversion_job,
        durable_cleanup=True,
        cleanup_namespace=storage.cleanup_namespace(),
        artifact_exists=storage.exists,
        artifact_cleanup=storage.delete,
        cleanup_interval_seconds=0,
    )
    monkeypatch.setattr(api, "JOB_MANAGER", original)
    put = storage.put_file
    outputs = []

    def checked_put(*args, **kwargs):
        rows = original._journal.load()
        assert len(rows) == 1
        assert rows[0].artifact_key == kwargs["key"]
        result = put(*args, **kwargs)
        outputs.append(result)
        if lost_reply:
            raise OSError("PRIVATE provider body")
        return result

    monkeypatch.setattr(storage, "put_file", checked_put)
    with TestClient(api.app) as client:
        created = client.post("/v1/jobs", files={"file": ("fixture.png", png(), "image/png")})
        assert created.status_code == 202
        record = original.get_record(created.json()["id"])
        record.future.result(timeout=5)
        response = client.get(f"/v1/jobs/{record.id}")
        assert response.status_code == 200
        public = response.json()
        assert public["status"] == ("failed" if lost_reply else "succeeded")
        assert public["download_ready"] is (not lost_reply)
        assert set(public) == {
            "id",
            "filename",
            "auto_orient_right_angles",
            "page_rotations",
            "status",
            "created_at",
            "started_at",
            "completed_at",
            "cancellation_requested",
            "error",
            "download_ready",
            "orientation_review",
        }
        assert "PRIVATE" not in json.dumps(public)
        assert "PRIVATE" not in caplog.text
        if not lost_reply:
            download = client.get(f"/v1/jobs/{record.id}/download")
            assert download.content == b"authored HTTP fixture"
        assert len(outputs) == 1
        deadline = record.completed_at + timedelta(hours=1)
    replacement = tracked_manager(root, storage)
    monkeypatch.setattr(api, "JOB_MANAGER", replacement)
    with TestClient(api.app) as client:
        assert client.get(f"/v1/jobs/{record.id}").status_code == 404
        assert storage.exists(outputs[0])
        replacement.cleanup_expired(now=deadline)
        assert not storage.exists(outputs[0])
        assert not record.workspace.exists()
        assert len(outputs) == 1  # No replay of the storage publication or OCR.
