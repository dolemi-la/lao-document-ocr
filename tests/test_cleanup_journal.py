"""Journal validation and process-boundary tests with authored temporary data."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from services.api.app.cleanup_journal import (
    JOURNAL_FILENAME,
    CleanupEntry,
    CleanupJournal,
    CleanupJournalError,
)

JOB_ID = "a" * 32
NAMESPACE = {"backend": "filesystem", "root": "private-storage-root"}


def entry():
    return CleanupEntry(JOB_ID, f"jobs/{JOB_ID}/result.zip", True, datetime.now(UTC))


def journal(root, limit=10):
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return CleanupJournal(root, NAMESPACE, max_entries=limit)


def test_minimal_private_rows_and_atomic_duplicate_rejection(tmp_path):
    ledger = journal(tmp_path)
    try:
        first = entry()
        ledger.add([first])
        with pytest.raises(CleanupJournalError):
            ledger.add([replace(first, job_id="b" * 32, artifact_key=None), first])
        assert ledger.load() == [first]
        assert ledger.path.stat().st_mode & 0o777 == 0o600
        assert b"private-storage-root" not in ledger.path.read_bytes()
        ledger.update(replace(first, artifact_key=None, workspace_pending=False))
        assert ledger.load() == []
    finally:
        ledger.close()
        ledger.close()
    with pytest.raises(CleanupJournalError, match="closed"):
        ledger.load()


@pytest.mark.parametrize(
    "change",
    [
        {"job_id": "../outside"},
        {"job_id": "A" * 32},
        {"artifact_key": "../outside"},
        {"artifact_key": "/absolute/path"},
        {"artifact_key": f"jobs/{'b' * 32}/result.zip"},
        {"artifact_key": f"jobs/{JOB_ID}/../result.zip"},
        {"artifact_key": f"jobs/{JOB_ID}/."},
        {"artifact_key": f"jobs/{JOB_ID}/sub/result.zip"},
        {"artifact_key": f"jobs/{JOB_ID}/bad\\name.zip"},
        {"artifact_key": f"jobs/{JOB_ID}/bad\x00name.zip"},
        {"artifact_key": "x" * 1025},
        {"workspace_pending": 1},
        {"failures": -1},
        {"failures": True},
        {"failures": 2**40},
        {"retry_at": datetime(2026, 1, 1)},
    ],
)
def test_invalid_entries_never_enter_journal(tmp_path, change):
    ledger = journal(tmp_path)
    try:
        with pytest.raises(CleanupJournalError):
            ledger.add([replace(entry(), **change)])
        assert ledger.load() == []
    finally:
        ledger.close()


@pytest.mark.parametrize(
    "sql, parameters",
    [
        ("UPDATE pending SET job_id=?", ("../outside",)),
        ("UPDATE pending SET artifact_key=?", (f"jobs/{'b' * 32}/result.zip",)),
        ("UPDATE pending SET workspace_pending=3", ()),
        ("UPDATE pending SET failures=-1", ()),
        ("UPDATE pending SET retry_at=?", ("2026-01-01T00:00:00",)),
        ("UPDATE pending SET retry_at=?", ("PRIVATE bad timestamp",)),
        ("UPDATE pending SET artifact_key=?", (b"PRIVATE blob",)),
        ("UPDATE pending SET artifact_key=?", ("z" * 9000,)),
        ("UPDATE identity SET version=900", ()),
        ("PRAGMA user_version=900", ()),
        ("CREATE TABLE unexpected (value TEXT)", ()),
    ],
)
def test_malformed_database_fails_closed_without_silent_repair(tmp_path, sql, parameters):
    ledger = journal(tmp_path)
    ledger.add([entry()])
    ledger.close()
    with sqlite3.connect(tmp_path / JOURNAL_FILENAME) as db:
        db.execute(sql, parameters)
    with pytest.raises(CleanupJournalError) as error:
        journal(tmp_path)
    assert "PRIVATE" not in str(error.value)
    with sqlite3.connect(tmp_path / JOURNAL_FILENAME) as db:
        assert db.execute("SELECT COUNT(*) FROM pending").fetchone()[0] == 1


def test_lower_capacity_does_not_truncate_recovered_rows(tmp_path):
    ledger = journal(tmp_path)
    first = entry()
    ledger.add([first, replace(first, job_id="b" * 32, artifact_key=None)])
    ledger.close()
    with pytest.raises(CleanupJournalError, match="capacity"):
        journal(tmp_path, limit=1)
    reopened = journal(tmp_path)
    try:
        assert len(reopened.load()) == 2
    finally:
        reopened.close()


def test_database_copied_to_different_job_root_is_not_rebound(tmp_path):
    original = journal(tmp_path / "first")
    original.add([entry()])
    original.close()
    copy_root = tmp_path / "second"
    copy_root.mkdir(mode=0o700)
    (copy_root / JOURNAL_FILENAME).write_bytes((tmp_path / "first" / JOURNAL_FILENAME).read_bytes())
    with pytest.raises(CleanupJournalError, match="namespace"):
        journal(copy_root)


@pytest.mark.parametrize("suffix", ["", "-journal", "-wal", "-shm"])
def test_linked_database_or_sidecar_is_never_opened(tmp_path, suffix):
    outside = tmp_path / "outside"
    outside.write_bytes(b"do not modify")
    root = tmp_path / "jobs"
    root.mkdir(mode=0o700)
    (root / (JOURNAL_FILENAME + suffix)).symlink_to(outside)
    with pytest.raises(CleanupJournalError, match="Unsafe"):
        journal(root)
    assert outside.read_bytes() == b"do not modify"


def test_corrupt_database_is_not_replaced_with_an_empty_queue(tmp_path):
    path = tmp_path / JOURNAL_FILENAME
    path.write_bytes(b"PRIVATE invalid database")
    with pytest.raises(CleanupJournalError) as error:
        journal(tmp_path)
    assert "PRIVATE" not in str(error.value)
    assert path.read_bytes() == b"PRIVATE invalid database"


def test_same_process_open_attempt_does_not_release_cross_process_ownership(tmp_path):
    ledger = journal(tmp_path)
    code = """
import json, sys
from pathlib import Path
from services.api.app.cleanup_journal import CleanupJournal, CleanupJournalError
try:
    ledger = CleanupJournal(Path(sys.argv[1]), json.loads(sys.argv[2]), max_entries=10)
except CleanupJournalError:
    print("blocked")
else:
    ledger.close()
    print("opened")
"""
    args = [sys.executable, "-c", code, str(tmp_path), json.dumps(NAMESPACE)]
    try:
        ledger.add([entry()])
        with pytest.raises(CleanupJournalError):
            journal(tmp_path)
        result = subprocess.run(args, capture_output=True, text=True, timeout=10, check=True)
        assert result.stdout.strip() == "blocked", result.stderr
    finally:
        ledger.close()
    result = subprocess.run(args, capture_output=True, text=True, timeout=10, check=True)
    assert result.stdout.strip() == "opened", result.stderr


def test_inherited_owner_is_rejected_before_a_transaction(tmp_path, monkeypatch):
    ledger = journal(tmp_path)
    owner = os.getpid()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "getpid", lambda: owner + 1)
            with pytest.raises(CleanupJournalError, match="forked"):
                ledger.add([entry()])
            with pytest.raises(CleanupJournalError, match="forked"):
                ledger.load()
        assert ledger.load() == []
    finally:
        ledger.close()


@pytest.mark.parametrize("phase", ["before-delete", "after-delete"])
def test_real_process_exit_recovers_committed_intent(tmp_path, phase):
    from services.api.app.jobs import ConversionJobManager
    from services.api.app.storage import FilesystemArtifactStorage

    code = """
import os, sys
from datetime import timedelta
from pathlib import Path
from services.api.app.jobs import ConversionJobManager
from services.api.app.storage import FilesystemArtifactStorage
root = Path(sys.argv[1])
storage = FilesystemArtifactStorage(root / "objects")
def runner(record, _):
    path = record.workspace / "result.zip"
    path.write_bytes(b"authored archive")
    return storage.put_file(path, key=f"jobs/{record.id}/result.zip",
                            filename="result.zip", media_type="application/zip")
def interrupted_delete(artifact):
    if sys.argv[2] == "after-delete":
        storage.delete(artifact)
    os._exit(47)
manager = ConversionJobManager(root / "jobs", runner, durable_cleanup=True,
    cleanup_namespace=storage.cleanup_namespace(), artifact_cleanup=interrupted_delete,
    cleanup_interval_seconds=0)
record = manager.reserve("private-source.png", ".png")
manager.enqueue(record.id)
record.future.result(timeout=5)
record.completed_at -= timedelta(hours=2)
manager.cleanup_expired()
raise AssertionError("exit point not reached")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), phase],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 47, result.stderr
    storage = FilesystemArtifactStorage(tmp_path / "objects")
    runs = []
    recovered = ConversionJobManager(
        tmp_path / "jobs",
        lambda *args: runs.append(args),
        durable_cleanup=True,
        cleanup_namespace=storage.cleanup_namespace(),
        artifact_cleanup=storage.delete,
        cleanup_interval_seconds=0,
    )
    try:
        assert len(recovered._pending_cleanup) == 1
        pending = next(iter(recovered._pending_cleanup.values()))
        assert pending.workspace.exists()
        assert storage.exists(pending.artifact) == (phase == "before-delete")
        assert b"private-source.png" not in (tmp_path / "jobs" / JOURNAL_FILENAME).read_bytes()
        assert recovered.cleanup_expired() == 0
        assert not recovered._pending_cleanup
        assert not pending.workspace.exists()
        assert runs == []
    finally:
        recovered.shutdown()


def test_recovery_rejects_schema_without_unique_job_ownership(tmp_path):
    ledger = journal(tmp_path)
    ledger.add([entry()])
    ledger.close()
    with sqlite3.connect(tmp_path / JOURNAL_FILENAME) as db:
        db.execute("ALTER TABLE pending RENAME TO saved")
        db.execute("CREATE TABLE pending AS SELECT * FROM saved")
        db.execute("INSERT INTO pending SELECT * FROM saved")
        db.execute("DROP TABLE saved")
    with pytest.raises(CleanupJournalError):
        recovered = journal(tmp_path)
        recovered.close()


def test_actual_sqlite_commit_denial_rolls_back_the_whole_intent(tmp_path):
    ledger = journal(tmp_path)
    try:

        def deny_commit(action, argument, *unused):
            if action == sqlite3.SQLITE_TRANSACTION and argument == "COMMIT":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        ledger._connection.set_authorizer(deny_commit)
        with pytest.raises(CleanupJournalError):
            ledger.add([entry()])
        ledger._connection.set_authorizer(None)
        assert ledger.load() == []
    finally:
        ledger.close()


def test_non_utf8_key_is_rejected_with_fixed_error(tmp_path):
    ledger = journal(tmp_path)
    try:
        key = f"jobs/{JOB_ID}/" + chr(0xD800) + ".zip"
        with pytest.raises(CleanupJournalError):
            ledger.add([replace(entry(), artifact_key=key)])
        assert ledger.load() == []
    finally:
        ledger.close()
