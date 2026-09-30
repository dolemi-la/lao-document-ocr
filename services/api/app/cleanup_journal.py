"""Private, single-owner SQLite journal for already-expired job cleanup.

No directory/object discovery and no public job restoration. The caller must
serialize ownership changes around commits. Storage callbacks never run here.
Keep the root on a trusted local filesystem.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

JOURNAL_FILENAME = ".expired-cleanup.sqlite3"
SCHEMA_VERSION = 1
MAX_DATABASE_BYTES = 64 * 1024 * 1024
MAX_FAILURES = 2**31 - 1
_JOB_ID = re.compile(r"[0-9a-f]{32}\Z")


class CleanupJournalError(RuntimeError):
    """Fixed messages only: never include SQL, provider, or path details."""


@dataclass(frozen=True)
class CleanupEntry:
    job_id: str
    artifact_key: str | None
    workspace_pending: bool
    retry_at: datetime
    failures: int = 0


def _validate(entry: CleanupEntry) -> None:
    if not isinstance(entry.job_id, str) or not _JOB_ID.fullmatch(entry.job_id):
        raise CleanupJournalError("Invalid cleanup journal entry.")
    if type(entry.workspace_pending) is not bool:
        raise CleanupJournalError("Invalid cleanup journal entry.")
    if type(entry.failures) is not int or not 0 <= entry.failures <= MAX_FAILURES:
        raise CleanupJournalError("Invalid cleanup journal entry.")
    if not isinstance(entry.retry_at, datetime) or entry.retry_at.utcoffset() is None:
        raise CleanupJournalError("Invalid cleanup journal entry.")
    try:
        entry.retry_at.astimezone(UTC)
    except (OverflowError, ValueError):
        raise CleanupJournalError("Invalid cleanup journal timestamp.") from None
    key = entry.artifact_key
    if key is not None:
        if not isinstance(key, str) or len(key) > 1024:
            raise CleanupJournalError("Invalid cleanup journal artifact key.")
        try:
            encoded = key.encode("utf-8")
        except UnicodeError:
            raise CleanupJournalError("Invalid cleanup journal artifact key.") from None
        if len(encoded) > 1024:
            raise CleanupJournalError("Invalid cleanup journal artifact key.")
        parts = key.split("/")
        if (
            len(parts) != 3
            or parts[:2] != ["jobs", entry.job_id]
            or parts[2] in {"", ".", ".."}
            or "\\" in key
            or any(ord(char) < 32 or ord(char) == 127 for char in key)
        ):
            raise CleanupJournalError("Invalid cleanup journal artifact key.")


class CleanupJournal:
    def __init__(self, root: Path, namespace: Mapping, *, max_entries: int) -> None:
        if type(max_entries) is not int or max_entries < 1:
            raise CleanupJournalError("Invalid cleanup journal capacity.")
        if not isinstance(namespace, Mapping) or not namespace:
            raise CleanupJournalError("Durable cleanup requires a storage namespace.")
        if root.is_symlink():
            raise CleanupJournalError("Durable cleanup requires a non-symlink job root.")
        self.root = root.resolve()
        self.path = self.root / JOURNAL_FILENAME
        self.max_entries = max_entries
        self._lock = threading.RLock()
        self._owner_pid = os.getpid()
        self._connection: sqlite3.Connection | None = None
        try:
            identity = json.dumps(
                {"job_root": str(self.root), "storage": dict(namespace)},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            if len(identity) > 8192:
                raise ValueError("oversized namespace")
            self._binding = hashlib.sha256(identity).hexdigest()
            self._open()
        except BaseException as exc:
            self.close()
            if isinstance(exc, CleanupJournalError):
                raise
            if isinstance(exc, (sqlite3.Error, OSError, ValueError, TypeError, OverflowError)):
                raise CleanupJournalError("Cleanup journal unavailable; startup refused.") from None
            raise

    def _open(self) -> None:
        # Refuse links before SQLite opens its own sidecars. The enclosing 0700
        # service-owned directory is the trust boundary, not an untrusted import.
        paths = [self.path, *(Path(str(self.path) + s) for s in ("-journal", "-wal", "-shm"))]
        for path in paths:
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise CleanupJournalError("Unsafe cleanup journal file; startup refused.")
            if info.st_size > MAX_DATABASE_BYTES:
                raise CleanupJournalError("Cleanup journal exceeds the size limit.")
        # Do not open/close a separate descriptor here: on POSIX that can
        # release an existing SQLite owner's process-wide advisory file locks.
        # The 0700 parent protects the initial creation before the chmod below.
        connection = sqlite3.connect(
            self.path,
            timeout=0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection = connection
        self.path.chmod(0o600)
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 8192)
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA locking_mode=EXCLUSIVE")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=EXTRA")
        connection.execute("PRAGMA fullfsync=ON")
        page_size = connection.execute("PRAGMA page_size").fetchone()[0]
        connection.execute(f"PRAGMA max_page_count={MAX_DATABASE_BYTES // page_size}")
        # EXCLUSIVE locking mode retains the owner lock across commits.
        with self._transaction() as db:
            objects = set(
                db.execute(
                    "SELECT name, type FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
                ).fetchall()
            )
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if not objects and version == 0:
                db.execute(
                    "CREATE TABLE identity (version INTEGER NOT NULL, binding TEXT NOT NULL)"
                )
                db.execute(
                    "CREATE TABLE pending (job_id TEXT PRIMARY KEY, artifact_key TEXT, "
                    "workspace_pending INTEGER NOT NULL, retry_at TEXT NOT NULL, "
                    "failures INTEGER NOT NULL)"
                )
                db.execute("INSERT INTO identity VALUES (?, ?)", (SCHEMA_VERSION, self._binding))
                db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            else:
                if objects != {("identity", "table"), ("pending", "table")}:
                    raise CleanupJournalError("Unrecognized cleanup journal schema.")
                identities = db.execute("SELECT version, binding FROM identity").fetchmany(2)
                if version != SCHEMA_VERSION or len(identities) != 1:
                    raise CleanupJournalError("Unsupported cleanup journal version.")
                if identities[0][0] != SCHEMA_VERSION:
                    raise CleanupJournalError("Unsupported cleanup journal version.")
                if identities[0][1] != self._binding:
                    raise CleanupJournalError("Cleanup journal namespace does not match storage.")
                if db.execute("PRAGMA quick_check(1)").fetchone() != ("ok",):
                    raise CleanupJournalError("Cleanup journal is corrupt; startup refused.")
            self._validate_schema_unlocked(db)
            self._load_unlocked(db)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self.assert_process_owner()
            connection = self._connection
            if connection is None:
                raise CleanupJournalError("Cleanup journal is closed.")
            try:
                connection.execute("BEGIN EXCLUSIVE")
                yield connection
                connection.execute("COMMIT")
            except BaseException as exc:
                try:
                    if connection.in_transaction:
                        connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                if isinstance(exc, (sqlite3.Error, OSError)):
                    raise CleanupJournalError(
                        "Cleanup journal write failed; cleanup paused."
                    ) from None
                raise

    @staticmethod
    def _validate_schema_unlocked(connection: sqlite3.Connection) -> None:
        expected = {
            "identity": [("version", "INTEGER", 1, 0), ("binding", "TEXT", 1, 0)],
            "pending": [
                ("job_id", "TEXT", 0, 1),
                ("artifact_key", "TEXT", 0, 0),
                ("workspace_pending", "INTEGER", 1, 0),
                ("retry_at", "TEXT", 1, 0),
                ("failures", "INTEGER", 1, 0),
            ],
        }
        for table, columns in expected.items():
            rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
            actual = [(row[1], row[2], row[3], row[5]) for row in rows]
            if actual != columns:
                raise CleanupJournalError("Unrecognized cleanup journal schema.")

    def _load_unlocked(self, connection: sqlite3.Connection) -> list[CleanupEntry]:
        rows = connection.execute(
            "SELECT job_id, artifact_key, workspace_pending, retry_at, failures "
            "FROM pending ORDER BY job_id"
        ).fetchmany(self.max_entries + 1)
        if len(rows) > self.max_entries:
            raise CleanupJournalError("Cleanup journal exceeds retained-job capacity.")
        entries = []
        seen: set[str] = set()
        try:
            for job_id, key, pending, retry, failures in rows:
                if type(pending) is not int or pending not in (0, 1) or not isinstance(retry, str):
                    raise ValueError("invalid row")
                entry = CleanupEntry(
                    job_id, key, bool(pending), datetime.fromisoformat(retry), failures
                )
                _validate(entry)
                if entry.job_id in seen:
                    raise CleanupJournalError("Duplicate cleanup journal ownership.")
                seen.add(entry.job_id)
                entries.append(entry)
        except (ValueError, TypeError, OverflowError):
            raise CleanupJournalError("Invalid cleanup journal entry.") from None
        return entries

    def assert_process_owner(self) -> None:
        if os.getpid() != self._owner_pid:
            raise CleanupJournalError("Cleanup journal cannot be shared across forked workers.")

    def load(self) -> list[CleanupEntry]:
        with self._lock:
            self.assert_process_owner()
            if self._connection is None:
                raise CleanupJournalError("Cleanup journal is closed.")
            try:
                return self._load_unlocked(self._connection)
            except sqlite3.Error:
                raise CleanupJournalError("Cleanup journal read failed.") from None

    @staticmethod
    def _values(entry: CleanupEntry) -> tuple:
        _validate(entry)
        return (
            entry.job_id,
            entry.artifact_key,
            int(entry.workspace_pending),
            entry.retry_at.astimezone(UTC).isoformat(),
            entry.failures,
        )

    def add(self, entries: Sequence[CleanupEntry]) -> None:
        if not entries:
            return
        values = [self._values(entry) for entry in entries]
        with self._transaction() as db:
            count = db.execute("SELECT COUNT(*) FROM pending").fetchone()[0]
            if count + len(values) > self.max_entries:
                raise CleanupJournalError("Cleanup journal exceeds retained-job capacity.")
            db.executemany("INSERT INTO pending VALUES (?, ?, ?, ?, ?)", values)

    def update(self, entry: CleanupEntry) -> None:
        values = self._values(entry)
        with self._transaction() as db:
            if entry.artifact_key is None and not entry.workspace_pending:
                cursor = db.execute("DELETE FROM pending WHERE job_id = ?", (entry.job_id,))
            else:
                cursor = db.execute(
                    "UPDATE pending SET artifact_key=?, workspace_pending=?, retry_at=?, "
                    "failures=? WHERE job_id=?",
                    (*values[1:], values[0]),
                )
            if cursor.rowcount != 1:
                raise CleanupJournalError("Cleanup journal ownership is missing.")

    def close(self) -> None:
        with self._lock:
            self.assert_process_owner()
            if self._connection is not None:
                self._connection.close()
                self._connection = None
