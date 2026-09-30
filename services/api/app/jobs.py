from __future__ import annotations

import logging
import shutil
import threading
import uuid
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

from lao_document_ocr.orientation_review import OrientationReview
from lao_document_ocr.page_rotations import validate_page_rotations
from services.api.app.cleanup_journal import (
    MAX_FAILURES,
    CleanupEntry,
    CleanupJournal,
    CleanupJournalError,
)
from services.api.app.cleanup_worker import (
    CLEANUP_SHUTDOWN_TIMEOUT_SECONDS,
    DEFAULT_CLEANUP_INTERVAL_SECONDS,
    IdleCleanupWorker,
)
from services.api.app.storage import StoredArtifact

logger = logging.getLogger(__name__)


def _invalid_cleanup_workspace() -> None:
    raise CleanupJournalError("Invalid cleanup workspace ownership.")


DEFAULT_MAX_RETAINED_JOBS = 1024
MAX_DOWNLOADS_PER_JOB = 4
MAX_DOWNLOADS_TOTAL = 32
CLEANUP_MAX_TASKS_PER_PASS = 8
CLEANUP_RETRY_INITIAL_SECONDS = 30
CLEANUP_RETRY_MAX_SECONDS = 3600


class JobStatus(StrEnum):
    UPLOADING = "uploading"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TERMINAL_STATUSES = {
    JobStatus.SUCCEEDED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
}


class JobCapacityError(RuntimeError):
    pass


class JobManagerClosedError(RuntimeError):
    pass


class JobSchedulingError(RuntimeError):
    pass


class JobNotFoundError(KeyError):
    pass


class JobDownloadNotReadyError(RuntimeError):
    pass


class JobDownloadCapacityError(RuntimeError):
    pass


@dataclass(frozen=True)
class JobDownloadLease:
    """An admitted response owns a private snapshot until close, not public retention."""

    output_path: Path | None
    output_artifact: StoredArtifact | None
    _release: Callable[[], None] = field(repr=False, compare=False)

    def close(self) -> None:
        """Idempotently release without storage I/O; normal cleanup handles deletion."""
        self._release()


class JobCancelledError(RuntimeError):
    pass


class JobPublicError(RuntimeError):
    pass


@dataclass
class JobRecord:
    id: str
    filename: str
    workspace: Path
    input_path: Path
    auto_orient_right_angles: bool = False
    status: JobStatus = JobStatus.UPLOADING
    output_path: Path | None = None
    output_artifact: StoredArtifact | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    cancellation_requested: bool = False
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    future: Future | None = field(default=None, repr=False)
    terminal_recorded: bool = field(default=False, repr=False)
    orientation_review: OrientationReview | None = None
    page_rotations: Mapping[int, int] = field(default_factory=dict)
    _artifact_cleanup_in_progress: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        # Snapshot each job independently, retaining explicit zero overrides.
        self.page_rotations = MappingProxyType(validate_page_rotations(self.page_rotations))


@dataclass
class _ExpiredJobCleanup:
    """Minimal private expired ownership; optionally backed by a cleanup journal."""

    workspace: Path
    artifact: StoredArtifact | None
    retry_at: datetime
    workspace_pending: bool = True
    failures: int = 0
    in_progress: bool = False


JobRunner = Callable[[JobRecord, threading.Event], Path | StoredArtifact]
ArtifactExists = Callable[[StoredArtifact], bool]
ArtifactCleanup = Callable[[StoredArtifact], None]


class ConversionJobManager:
    def __init__(
        self,
        root_dir: str | Path,
        runner: JobRunner,
        *,
        max_workers: int = 2,
        max_active_jobs: int = 8,
        retention_seconds: int = 3600,
        max_retained_jobs: int = DEFAULT_MAX_RETAINED_JOBS,
        cleanup_interval_seconds: float = DEFAULT_CLEANUP_INTERVAL_SECONDS,
        artifact_exists: ArtifactExists | None = None,
        artifact_cleanup: ArtifactCleanup | None = None,
        durable_cleanup: bool = False,
        cleanup_namespace: Mapping | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if max_active_jobs < max_workers:
            raise ValueError("max_active_jobs must be >= max_workers")
        if retention_seconds < 0:
            raise ValueError("retention_seconds must be non-negative")

        if type(max_retained_jobs) is not int or max_retained_jobs < max_active_jobs:
            raise ValueError("max_retained_jobs must be an integer >= max_active_jobs")

        if type(durable_cleanup) is not bool:
            raise ValueError("durable_cleanup must be a boolean")
        if durable_cleanup and (
            not isinstance(cleanup_namespace, Mapping) or not cleanup_namespace
        ):
            raise ValueError("Durable cleanup requires a storage namespace")
        if durable_cleanup and Path(root_dir).is_symlink():
            raise ValueError("Durable cleanup requires a non-symlink job root")

        # Validate before allocating directories/executor; no thread starts here.
        self._cleanup_worker = IdleCleanupWorker(
            lambda: self.cleanup_expired(), interval_seconds=cleanup_interval_seconds,
        )
        self.cleanup_interval_seconds = cleanup_interval_seconds

        self.root_dir = Path(root_dir).resolve() if durable_cleanup else Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root_dir.chmod(0o700)
        self.runner = runner
        self.max_active_jobs = max_active_jobs
        self.retention_seconds = retention_seconds
        self.max_retained_jobs = max_retained_jobs
        self.artifact_exists = artifact_exists
        self.artifact_cleanup = artifact_cleanup
        self._jobs: dict[str, JobRecord] = {}
        self._pending_cleanup: dict[str, _ExpiredJobCleanup] = {}
        self._download_leases: dict[str, int] = {}
        self._cleanup_attempts_total = 0
        self._cleanup_failures_total = 0
        self._completed_total: Counter[str] = Counter()
        self._duration_seconds_sum: Counter[str] = Counter()
        self._lock = threading.RLock()
        self._accepting_jobs = True
        self.durable_cleanup = durable_cleanup
        self._journal: CleanupJournal | None = None
        self._journal_failed = False
        self._cleanup_passes = 0
        self._cleanup_stopping = False
        self._journal_shutdown_ready = False
        try:
            if durable_cleanup:
                self._journal = CleanupJournal(
                    self.root_dir, cleanup_namespace, max_entries=max_retained_jobs,
                )
                for entry in self._journal.load():
                    artifact = (
                        StoredArtifact(entry.artifact_key, "result.zip", "application/zip", 0)
                        if entry.artifact_key is not None else None
                    )
                    self._pending_cleanup[entry.job_id] = _ExpiredJobCleanup(
                        self.root_dir / entry.job_id, artifact, entry.retry_at,
                        workspace_pending=entry.workspace_pending, failures=entry.failures,
                    )
            self._executor = ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="lao-ocr-job",
            )
        except BaseException:
            if self._journal is not None:
                self._journal.close()
            raise

    def _ensure_journal_usable_unlocked(self) -> None:
        if self._journal is not None:
            self._journal.assert_process_owner()
        if self._journal_failed:
            raise CleanupJournalError("Cleanup journal unavailable; cleanup paused.")

    def _journal_operation_unlocked(self, operation: Callable[[], None]) -> None:
        self._ensure_journal_usable_unlocked()
        try:
            operation()
        except CleanupJournalError:
            # A failed/ambiguous commit is not permission to delete or forget.
            # No automatic fallback to an unjournaled queue; restart must reconcile.
            self._journal_failed = True
            self._accepting_jobs = False
            for item in self._pending_cleanup.values():
                item.in_progress = False
            logger.warning("Cleanup journal unavailable; admission and cleanup paused.")
            raise CleanupJournalError("Cleanup journal unavailable; cleanup paused.") from None

    def _maybe_close_journal_unlocked(self) -> None:
        if (
            self._journal is not None and self._journal_shutdown_ready
            and self._cleanup_passes == 0 and not self._download_leases
        ):
            self._journal.close()

    @staticmethod
    def _cleanup_entry(job_id: str, item: _ExpiredJobCleanup) -> CleanupEntry:
        return CleanupEntry(
            job_id, item.artifact.key if item.artifact is not None else None,
            item.workspace_pending, item.retry_at, item.failures,
        )

    def _ensure_accepting_unlocked(self) -> None:
        self._ensure_journal_usable_unlocked()
        if not self._accepting_jobs:
            raise JobManagerClosedError("Conversion job manager has stopped accepting work.")

    def _active_count(self) -> int:
        return sum(
            record.status not in _TERMINAL_STATUSES
            for record in self._jobs.values()
        )

    def _record_terminal_unlocked(self, record: JobRecord) -> None:
        if record.terminal_recorded or record.status not in _TERMINAL_STATUSES:
            return
        if record.completed_at is None:
            record.completed_at = datetime.now(UTC)
        start = record.started_at or record.created_at
        duration = max(0.0, (record.completed_at - start).total_seconds())
        status = record.status.value
        self._completed_total[status] += 1
        self._duration_seconds_sum[status] += duration
        record.terminal_recorded = True

    def reserve_many(
        self,
        files: list[tuple[str, str]],
        *,
        auto_orient_right_angles: bool = False,
        page_rotations: Mapping[int, int] | None = None,
    ) -> list[JobRecord]:
        if not files:
            raise ValueError("At least one job reservation is required")
        rotations = validate_page_rotations(page_rotations)
        with self._lock:
            self._ensure_accepting_unlocked()
        self.cleanup_expired()
        with self._lock:
            # Cleanup can block on storage while shutdown closes admission.
            self._ensure_accepting_unlocked()
            active = self._active_count()
            if active + len(files) > self.max_active_jobs:
                available = max(0, self.max_active_jobs - active)
                raise JobCapacityError(
                    f"Job capacity reached ({available} slot(s) available, {len(files)} requested)."
                )

            # Bound both live/terminal records and private failed-cleanup ownership.
            # Backpressure rather than dropping references during a storage outage.
            retained = len(self._jobs) + len(self._pending_cleanup)
            if retained + len(files) > self.max_retained_jobs:
                raise JobCapacityError(
                    "Retained job capacity reached; wait for expiry/cleanup "
                    "or increase JOB_MAX_RETAINED."
                )

            records: list[JobRecord] = []
            try:
                for filename, suffix in files:
                    job_id = uuid.uuid4().hex
                    workspace = self.root_dir / job_id
                    workspace.mkdir(parents=True, exist_ok=False, mode=0o700)
                    input_path = workspace / f"input{suffix}"
                    record = JobRecord(
                        id=job_id,
                        filename=filename,
                        workspace=workspace,
                        input_path=input_path,
                        auto_orient_right_angles=auto_orient_right_angles,
                        page_rotations=rotations,
                    )
                    self._jobs[job_id] = record
                    records.append(record)
            except Exception:
                for record in records:
                    self._jobs.pop(record.id, None)
                    shutil.rmtree(record.workspace, ignore_errors=True)
                raise
            return records

    def reserve(
        self,
        filename: str,
        suffix: str,
        *,
        auto_orient_right_angles: bool = False,
        page_rotations: Mapping[int, int] | None = None,
    ) -> JobRecord:
        return self.reserve_many(
            [(filename, suffix)],
            auto_orient_right_angles=auto_orient_right_angles,
            page_rotations=page_rotations,
        )[0]

    def discard(self, job_id: str) -> None:
        with self._lock:
            if self._download_leases.get(job_id, 0):
                raise RuntimeError("Cannot discard a job with an admitted download.")
            record = self._jobs.pop(job_id, None)
        if record is not None:
            shutil.rmtree(record.workspace, ignore_errors=True)

    def enqueue(self, job_id: str) -> JobRecord:
        with self._lock:
            self._ensure_accepting_unlocked()
            record = self._require(job_id)
            if record.status != JobStatus.UPLOADING:
                raise RuntimeError(
                    f"Job {job_id} cannot be enqueued from status {record.status.value}."
                )
            if record.cancel_event.is_set():
                record.status = JobStatus.CANCELLED
                record.cancellation_requested = True
                record.completed_at = datetime.now(UTC)
                self._record_terminal_unlocked(record)
                return record

            record.status = JobStatus.QUEUED
            try:
                record.future = self._executor.submit(self._run, job_id)
            except Exception as exc:
                # Stop repeated admission to a broken executor. Otherwise each
                # failed thread start could leave another unreturned work item.
                self._accepting_jobs = False
                # submit() can queue work before failing to start a thread. The
                # worker cannot pass our lock until this terminal state is set.
                record.status = JobStatus.FAILED
                record.error = "Conversion worker is unavailable."
                record.cancel_event.set()
                record.completed_at = datetime.now(UTC)
                self._record_terminal_unlocked(record)
                raise JobSchedulingError(record.error) from exc
            return record

    def _run(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            # Rejected submissions can still have queued executor work. A
            # discarded/terminal record is never permission to invoke the runner.
            if record is None or record.status != JobStatus.QUEUED:
                return
            if record.cancel_event.is_set():
                record.status = JobStatus.CANCELLED
                record.completed_at = datetime.now(UTC)
                self._record_terminal_unlocked(record)
                return
            record.status = JobStatus.RUNNING
            record.started_at = datetime.now(UTC)

        cancelled_artifact: StoredArtifact | None = None
        artifact_cleanup = self.artifact_cleanup
        try:
            output = self.runner(record, record.cancel_event)
            with self._lock:
                # Take ownership before deciding whether cancellation won the race.
                # A returned object must remain tracked even when never downloadable.
                if isinstance(output, StoredArtifact):
                    record.output_artifact = output
                    record.output_path = None
                else:
                    record.output_path = Path(output)
                    record.output_artifact = None
                if record.cancel_event.is_set():
                    record.status = JobStatus.CANCELLED
                    record.cancellation_requested = True
                    record.output_path = None
                    if record.output_artifact is not None and artifact_cleanup is not None:
                        cancelled_artifact = record.output_artifact
                        # Prevent expiry from deleting twice or dropping this reference
                        # while storage I/O executes outside the manager lock.
                        record._artifact_cleanup_in_progress = True
                else:
                    record.status = JobStatus.SUCCEEDED
        except JobCancelledError:
            with self._lock:
                record.status = JobStatus.CANCELLED
                record.cancellation_requested = True
                record.output_path = None
                record.output_artifact = None
        except JobPublicError as exc:
            with self._lock:
                record.status = JobStatus.FAILED
                record.error = str(exc)
                record.output_path = None
                record.output_artifact = None
        except Exception:
            logger.exception("Unhandled conversion job failure", extra={"job_id": job_id})
            with self._lock:
                record.status = JobStatus.FAILED
                record.error = "Internal conversion error."
                record.output_path = None
                record.output_artifact = None
        finally:
            with self._lock:
                record.completed_at = datetime.now(UTC)
                self._record_terminal_unlocked(record)

        if cancelled_artifact is not None and artifact_cleanup is not None:
            try:
                artifact_cleanup(cancelled_artifact)
            except Exception:
                # Retain the private reference for the existing expiry cleanup pass.
                # Storage exception bodies may contain keys, paths or credentials.
                logger.warning(
                    "Cancelled job artifact cleanup failed; retained for expiry cleanup.",
                    extra={"job_id": job_id},
                )
            else:
                with self._lock:
                    record.output_artifact = None
            finally:
                with self._lock:
                    record._artifact_cleanup_in_progress = False

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            record = self._require(job_id)
            if record.status not in _TERMINAL_STATUSES:
                record.cancellation_requested = True
                record.cancel_event.set()
                if record.future is not None and record.future.cancel():
                    record.status = JobStatus.CANCELLED
                    record.completed_at = datetime.now(UTC)
                    self._record_terminal_unlocked(record)
        return self._public(record)

    def acquire_download(self, job_id: str) -> JobDownloadLease:
        """Pin an available successful result before checking/opening storage.

        Public expiry still removes the job. Its private cleanup waits until all
        admitted responses close. No new lease may bypass the retention boundary.
        """
        self.cleanup_expired()
        with self._lock:
            if self.durable_cleanup and self._cleanup_stopping:
                raise JobManagerClosedError("Conversion job manager has stopped accepting work.")
            record = self._require(job_id)
            # Cleanup may have spent time on unrelated storage before we got the lock.
            cutoff = datetime.now(UTC) - timedelta(seconds=self.retention_seconds)
            if record.completed_at is not None and record.completed_at <= cutoff:
                raise JobNotFoundError(job_id)
            if record.status != JobStatus.SUCCEEDED:
                raise JobDownloadNotReadyError(
                    f"Job is not ready for download (status={record.status.value})."
                )
            count = self._download_leases.get(job_id, 0)
            if (
                count >= MAX_DOWNLOADS_PER_JOB
                or sum(self._download_leases.values()) >= MAX_DOWNLOADS_TOTAL
            ):
                raise JobDownloadCapacityError("Too many active downloads. Try again later.")
            self._download_leases[job_id] = count + 1
            released = False

            def release() -> None:
                nonlocal released
                with self._lock:
                    if released:
                        return
                    released = True
                    remaining = self._download_leases[job_id] - 1
                    if remaining:
                        self._download_leases[job_id] = remaining
                    else:
                        del self._download_leases[job_id]
                    self._maybe_close_journal_unlocked()

            return JobDownloadLease(record.output_path, record.output_artifact, release)

    def get_record(self, job_id: str) -> JobRecord:
        self.cleanup_expired()
        with self._lock:
            return self._require(job_id)

    def public(self, job_id: str) -> dict:
        self.cleanup_expired()
        with self._lock:
            record = self._require(job_id)
        return self._public(record)

    def _public(self, record: JobRecord) -> dict:
        # Snapshot only immutable readiness inputs. A provider HEAD or even a
        # filesystem stat must not hold the lock needed by every job transition.
        with self._lock:
            self._require_current_record_unlocked(record)
            status = record.status
            output_path = record.output_path
            output_artifact = record.output_artifact
            artifact_exists = self.artifact_exists

        ready = self._download_ready(status, output_path, output_artifact, artifact_exists)

        with self._lock:
            # Expiry/discard can proceed while the probe waits. Never resurrect
            # a removed job or attach its result to a replacement record.
            self._require_current_record_unlocked(record)
            same_result = (
                record.status == status
                and record.output_path == output_path
                and record.output_artifact == output_artifact
                and self.artifact_exists is artifact_exists
            )
            return self._public_fields_unlocked(record, download_ready=ready and same_result)

    def _require_current_record_unlocked(self, record: JobRecord) -> None:
        self._ensure_journal_usable_unlocked()
        if self._jobs.get(record.id) is not record:
            raise JobNotFoundError(record.id)

    def _public_fields_unlocked(self, record: JobRecord, *, download_ready: bool) -> dict:
        return {
            "id": record.id,
            "filename": record.filename,
            "auto_orient_right_angles": record.auto_orient_right_angles,
            "page_rotations": [
                {"page": page, "degrees_clockwise": angle}
                for page, angle in record.page_rotations.items()
            ],
            "status": record.status.value,
            "created_at": record.created_at.isoformat(),
            "started_at": (
                record.started_at.isoformat() if record.started_at is not None else None
            ),
            "completed_at": (
                record.completed_at.isoformat() if record.completed_at is not None else None
            ),
            "cancellation_requested": record.cancellation_requested,
            "error": record.error,
            "download_ready": download_ready,
            "orientation_review": (
                record.orientation_review.to_dict()
                if record.status == JobStatus.SUCCEEDED
                and isinstance(record.orientation_review, OrientationReview)
                else None
            ),
        }

    @staticmethod
    def _download_ready(
        status: JobStatus,
        output_path: Path | None,
        output_artifact: StoredArtifact | None,
        artifact_exists: ArtifactExists | None,
    ) -> bool:
        if status != JobStatus.SUCCEEDED:
            return False
        if output_artifact is not None:
            if artifact_exists is None:
                return True
            try:
                return bool(artifact_exists(output_artifact))
            except Exception:
                return False
        try:
            return output_path is not None and output_path.is_file()
        except OSError:
            # Inaccessible local results are unknown, not a public path/error leak.
            return False

    def _require(self, job_id: str) -> JobRecord:
        record = self._jobs.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id)
        return record

    def snapshot(self) -> dict:
        self.cleanup_expired()
        with self._lock:
            current = Counter(record.status.value for record in self._jobs.values())
            return {
                "current_by_status": dict(sorted(current.items())),
                "completed_total": dict(sorted(self._completed_total.items())),
                "duration_seconds_sum": {
                    status: float(total)
                    for status, total in sorted(self._duration_seconds_sum.items())
                },
                "active_jobs": self._active_count(),
                "max_active_jobs": self.max_active_jobs,
                "retained_jobs": len(self._jobs) + len(self._pending_cleanup),
                "active_downloads": sum(self._download_leases.values()),
                "cleanup_download_blocked_jobs": sum(
                    bool(self._download_leases.get(job_id, 0))
                    for job_id in self._pending_cleanup
                ),
                "max_retained_jobs": self.max_retained_jobs,
                "cleanup_pending_jobs": len(self._pending_cleanup),
                "cleanup_pending_artifacts": sum(
                    item.artifact is not None for item in self._pending_cleanup.values()
                ),
                "cleanup_in_progress": sum(
                    item.in_progress for item in self._pending_cleanup.values()
                ),
                "cleanup_attempts_total": self._cleanup_attempts_total,
                "cleanup_failures_total": self._cleanup_failures_total,
            }

    def cleanup_expired(self, *, now: datetime | None = None) -> int:
        """Expire public records and attempt a bounded batch of due private cleanups.

        Returns newly expired jobs, not successful deletions. Retry ownership
        is process-local by default; the opt-in journal persists expired tasks.
        Provider/workspace I/O runs outside the lock; local journal commits do not.
        A task cannot be claimed by two passes.
        """
        with self._lock:
            if self.durable_cleanup and self._cleanup_stopping:
                return 0
            self._ensure_journal_usable_unlocked()
            self._cleanup_passes += 1
        try:
            return self._cleanup_expired(now=now)
        finally:
            with self._lock:
                self._cleanup_passes -= 1
                self._maybe_close_journal_unlocked()

    def _cleanup_expired(self, *, now: datetime | None = None) -> int:
        current_time = now or datetime.now(UTC)
        cutoff = current_time - timedelta(seconds=self.retention_seconds)
        with self._lock:
            expired = {
                job_id: _ExpiredJobCleanup(record.workspace, record.output_artifact, current_time)
                for job_id, record in self._jobs.items()
                if record.status in _TERMINAL_STATUSES
                and not record._artifact_cleanup_in_progress
                and record.completed_at is not None and record.completed_at <= cutoff
            }
            if self._journal is not None and expired:
                # Commit the private deletion intent before dropping public ownership
                # or performing any provider/workspace deletion. Short local journal
                # transactions are synchronous; provider I/O still runs outside lock.
                for job_id, item in expired.items():
                    if item.workspace != self.root_dir / job_id:
                        self._journal_operation_unlocked(
                            _invalid_cleanup_workspace,
                        )
                self._journal_operation_unlocked(lambda: self._journal.add([
                    self._cleanup_entry(job_id, item) for job_id, item in expired.items()
                ]))
            for job_id, item in expired.items():
                self._pending_cleanup[job_id] = item
                del self._jobs[job_id]
            expired_count = len(expired)

            due = sorted(
                (
                    (job_id, item) for job_id, item in self._pending_cleanup.items()
                    if not item.in_progress and item.retry_at <= current_time
                    and not self._download_leases.get(job_id, 0)
                ),
                key=lambda pair: (pair[1].retry_at, pair[0]),
            )[:CLEANUP_MAX_TASKS_PER_PASS]
            for _, item in due:
                item.in_progress = True
            self._cleanup_attempts_total += len(due)

        for job_id, item in due:
            with self._lock:
                self._ensure_journal_usable_unlocked()
            artifact_deleted = item.artifact is None
            workspace_deleted = not item.workspace_pending
            artifact_cleanup = self.artifact_cleanup
            if item.artifact is not None and artifact_cleanup is not None:
                try:
                    artifact_cleanup(item.artifact)
                except Exception:
                    # Never serialize provider exceptions or storage identities.
                    pass
                else:
                    artifact_deleted = True
            if item.workspace_pending:
                try:
                    shutil.rmtree(item.workspace)
                except FileNotFoundError:
                    # A missing child is not proof that the whole workspace is gone.
                    try:
                        workspace_deleted = not (
                            item.workspace.exists() or item.workspace.is_symlink()
                        )
                    except OSError:
                        # An inaccessible root is unknown, never a deletion receipt.
                        workspace_deleted = False
                except Exception:
                    pass
                else:
                    workspace_deleted = True

            complete = artifact_deleted and workspace_deleted
            with self._lock:
                failures = item.failures if complete else min(item.failures + 1, MAX_FAILURES)
                retry_at = item.retry_at
                if not complete:
                    delay = min(
                        CLEANUP_RETRY_INITIAL_SECONDS * 2 ** min(failures - 1, 7),
                        CLEANUP_RETRY_MAX_SECONDS,
                    )
                    retry_at = (now or datetime.now(UTC)) + timedelta(seconds=delay)
                updated = _ExpiredJobCleanup(
                    item.workspace, None if artifact_deleted else item.artifact, retry_at,
                    workspace_pending=not workspace_deleted, failures=failures,
                )
                if self._journal is not None:
                    self._journal_operation_unlocked(
                        lambda job_id=job_id, updated=updated: self._journal.update(
                            self._cleanup_entry(job_id, updated),
                        ),
                    )
                # Forget ownership only after the stage update/removal commits.
                item.artifact = updated.artifact
                item.workspace_pending = updated.workspace_pending
                item.retry_at = updated.retry_at
                item.failures = updated.failures
                item.in_progress = False
                if complete:
                    self._pending_cleanup.pop(job_id)
                else:
                    self._cleanup_failures_total += 1
            if not complete:
                logger.warning(
                    "Expired job cleanup incomplete; retained for retry.",
                    extra={"job_id": job_id},
                )
        return expired_count

    def start_cleanup_worker(self) -> bool:
        """Called by the API lifespan; ordinary manager construction stays inert."""
        with self._lock:
            self._ensure_accepting_unlocked()
            return self._cleanup_worker.start()

    def shutdown(self, *, wait: bool = True) -> None:
        futures_to_cancel: list[Future] = []
        with self._lock:
            # Close both reservation and enqueue admission before a join or I/O.
            # A runner already marked RUNNING keeps the existing graceful policy.
            self._accepting_jobs = False
            self._cleanup_stopping = True
            for record in self._jobs.values():
                if record.status == JobStatus.QUEUED:
                    record.cancel_event.set()
                    record.status = JobStatus.CANCELLED
                    record.cancellation_requested = True
                    record.completed_at = datetime.now(UTC)
                    self._record_terminal_unlocked(record)
                    if record.future is not None:
                        futures_to_cancel.append(record.future)
        # Future callbacks and all joins run outside the manager lock. Even a
        # dequeued future must pass _run's QUEUED check before starting OCR.
        for future in futures_to_cancel:
            future.cancel()
        # An upload still being written retains its original owner/status; its
        # caller must finish/abort copying and discard after enqueue is rejected.
        # Stop scheduling first. Never join while holding the job-manager lock:
        # an in-flight cleanup needs that lock to record completed deletion work.
        stopped = self._cleanup_worker.stop(
            timeout=CLEANUP_SHUTDOWN_TIMEOUT_SECONDS if wait else 0,
        )
        if wait and not stopped:
            logger.warning("Idle job cleanup still running after shutdown wait; stop requested.")
        self._executor.shutdown(wait=wait, cancel_futures=True)
        with self._lock:
            # Nonwaiting shutdown deliberately retains exclusive ownership until
            # a later waiting shutdown or process exit. Active cleanup/downloads
            # release only after their own finalizers have completed.
            if wait:
                self._journal_shutdown_ready = True
            self._maybe_close_journal_unlocked()
