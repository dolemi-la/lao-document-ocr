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
from services.api.app.storage import StoredArtifact

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETAINED_JOBS = 1024
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


class JobNotFoundError(KeyError):
    pass


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
    """Minimal private ownership after the public job has expired; not durable."""

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
        artifact_exists: ArtifactExists | None = None,
        artifact_cleanup: ArtifactCleanup | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if max_active_jobs < max_workers:
            raise ValueError("max_active_jobs must be >= max_workers")
        if retention_seconds < 0:
            raise ValueError("retention_seconds must be non-negative")

        if type(max_retained_jobs) is not int or max_retained_jobs < max_active_jobs:
            raise ValueError("max_retained_jobs must be an integer >= max_active_jobs")

        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root_dir.chmod(0o700)
        self.runner = runner
        self.max_active_jobs = max_active_jobs
        self.retention_seconds = retention_seconds
        self.max_retained_jobs = max_retained_jobs
        self.artifact_exists = artifact_exists
        self.artifact_cleanup = artifact_cleanup
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="lao-ocr-job",
        )
        self._jobs: dict[str, JobRecord] = {}
        self._pending_cleanup: dict[str, _ExpiredJobCleanup] = {}
        self._cleanup_attempts_total = 0
        self._cleanup_failures_total = 0
        self._completed_total: Counter[str] = Counter()
        self._duration_seconds_sum: Counter[str] = Counter()
        self._lock = threading.RLock()

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
        self.cleanup_expired()
        with self._lock:
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
            record = self._jobs.pop(job_id, None)
        if record is not None:
            shutil.rmtree(record.workspace, ignore_errors=True)

    def enqueue(self, job_id: str) -> JobRecord:
        with self._lock:
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
            record.future = self._executor.submit(self._run, job_id)
            return record

    def _run(self, job_id: str) -> None:
        with self._lock:
            record = self._require(job_id)
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
            if record.status in _TERMINAL_STATUSES:
                return self._public(record)

            record.cancellation_requested = True
            record.cancel_event.set()
            if record.future is not None and record.future.cancel():
                record.status = JobStatus.CANCELLED
                record.completed_at = datetime.now(UTC)
                self._record_terminal_unlocked(record)
            return self._public(record)

    def get_record(self, job_id: str) -> JobRecord:
        self.cleanup_expired()
        with self._lock:
            return self._require(job_id)

    def public(self, job_id: str) -> dict:
        self.cleanup_expired()
        with self._lock:
            return self._public(self._require(job_id))

    def _public(self, record: JobRecord) -> dict:
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
            "download_ready": self._download_ready(record),
            "orientation_review": (
                record.orientation_review.to_dict()
                if record.status == JobStatus.SUCCEEDED
                and isinstance(record.orientation_review, OrientationReview)
                else None
            ),
        }

    def _download_ready(self, record: JobRecord) -> bool:
        if record.status != JobStatus.SUCCEEDED:
            return False
        if record.output_artifact is not None:
            if self.artifact_exists is None:
                return True
            try:
                return bool(self.artifact_exists(record.output_artifact))
            except Exception:
                return False
        return record.output_path is not None and record.output_path.is_file()

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
        survives failures in this process only. Callbacks and workspace I/O run
        outside the manager lock; a task cannot be claimed by two passes.
        """
        current_time = now or datetime.now(UTC)
        cutoff = current_time - timedelta(seconds=self.retention_seconds)
        expired_count = 0
        with self._lock:
            for job_id, record in list(self._jobs.items()):
                if (
                    record.status in _TERMINAL_STATUSES
                    and not record._artifact_cleanup_in_progress
                    and record.completed_at is not None
                    and record.completed_at <= cutoff
                ):
                    self._pending_cleanup[job_id] = _ExpiredJobCleanup(
                        record.workspace, record.output_artifact, current_time,
                    )
                    del self._jobs[job_id]
                    expired_count += 1

            due = sorted(
                (
                    (job_id, item) for job_id, item in self._pending_cleanup.items()
                    if not item.in_progress and item.retry_at <= current_time
                ),
                key=lambda pair: (pair[1].retry_at, pair[0]),
            )[:CLEANUP_MAX_TASKS_PER_PASS]
            for _, item in due:
                item.in_progress = True
            self._cleanup_attempts_total += len(due)

        for job_id, item in due:
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
                if artifact_deleted:
                    item.artifact = None
                item.workspace_pending = not workspace_deleted
                item.in_progress = False
                if complete:
                    self._pending_cleanup.pop(job_id)
                else:
                    item.failures += 1
                    # Saturate the exponent; long outages must not allocate huge integers.
                    delay = min(
                        CLEANUP_RETRY_INITIAL_SECONDS * 2 ** min(item.failures - 1, 7),
                        CLEANUP_RETRY_MAX_SECONDS,
                    )
                    item.retry_at = (now or datetime.now(UTC)) + timedelta(seconds=delay)
                    self._cleanup_failures_total += 1
            if not complete:
                logger.warning(
                    "Expired job cleanup incomplete; retained for retry.",
                    extra={"job_id": job_id},
                )
        return expired_count

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)
