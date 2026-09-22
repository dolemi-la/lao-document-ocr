from __future__ import annotations

import shutil
import threading
import uuid
from collections import Counter
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from services.api.app.storage import StoredArtifact


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


@dataclass
class JobRecord:
    id: str
    filename: str
    workspace: Path
    input_path: Path
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
        artifact_exists: ArtifactExists | None = None,
        artifact_cleanup: ArtifactCleanup | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if max_active_jobs < max_workers:
            raise ValueError("max_active_jobs must be >= max_workers")
        if retention_seconds < 0:
            raise ValueError("retention_seconds must be non-negative")

        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self.max_active_jobs = max_active_jobs
        self.retention_seconds = retention_seconds
        self.artifact_exists = artifact_exists
        self.artifact_cleanup = artifact_cleanup
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="lao-ocr-job",
        )
        self._jobs: dict[str, JobRecord] = {}
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
    ) -> list[JobRecord]:
        if not files:
            raise ValueError("At least one job reservation is required")
        self.cleanup_expired()
        with self._lock:
            active = self._active_count()
            if active + len(files) > self.max_active_jobs:
                available = max(0, self.max_active_jobs - active)
                raise JobCapacityError(
                    "Job capacity reached "
                    f"({available} slot(s) available, {len(files)} requested)."
                )

            records: list[JobRecord] = []
            try:
                for filename, suffix in files:
                    job_id = uuid.uuid4().hex
                    workspace = self.root_dir / job_id
                    workspace.mkdir(parents=True, exist_ok=False)
                    input_path = workspace / f"input{suffix}"
                    record = JobRecord(
                        id=job_id,
                        filename=filename,
                        workspace=workspace,
                        input_path=input_path,
                    )
                    self._jobs[job_id] = record
                    records.append(record)
            except Exception:
                for record in records:
                    self._jobs.pop(record.id, None)
                    shutil.rmtree(record.workspace, ignore_errors=True)
                raise
            return records

    def reserve(self, filename: str, suffix: str) -> JobRecord:
        return self.reserve_many([(filename, suffix)])[0]

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

        try:
            output = self.runner(record, record.cancel_event)
            with self._lock:
                if record.cancel_event.is_set():
                    record.status = JobStatus.CANCELLED
                    record.cancellation_requested = True
                    record.output_path = None
                else:
                    if isinstance(output, StoredArtifact):
                        record.output_artifact = output
                        record.output_path = None
                    else:
                        record.output_path = Path(output)
                        record.output_artifact = None
                    record.status = JobStatus.SUCCEEDED
        except JobCancelledError:
            with self._lock:
                record.status = JobStatus.CANCELLED
                record.cancellation_requested = True
                record.output_path = None
                record.output_artifact = None
        except Exception as exc:
            with self._lock:
                record.status = JobStatus.FAILED
                record.error = str(exc)
                record.output_path = None
                record.output_artifact = None
        finally:
            with self._lock:
                record.completed_at = datetime.now(UTC)
                self._record_terminal_unlocked(record)

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
            "status": record.status.value,
            "created_at": record.created_at.isoformat(),
            "started_at": (
                record.started_at.isoformat()
                if record.started_at is not None
                else None
            ),
            "completed_at": (
                record.completed_at.isoformat()
                if record.completed_at is not None
                else None
            ),
            "cancellation_requested": record.cancellation_requested,
            "error": record.error,
            "download_ready": self._download_ready(record),
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
            }

    def cleanup_expired(self, *, now: datetime | None = None) -> int:
        if self.retention_seconds == 0:
            cutoff = now or datetime.now(UTC)
        else:
            cutoff = (now or datetime.now(UTC)) - timedelta(
                seconds=self.retention_seconds
            )

        expired: list[JobRecord] = []
        with self._lock:
            for job_id, record in list(self._jobs.items()):
                if (
                    record.status in _TERMINAL_STATUSES
                    and record.completed_at is not None
                    and record.completed_at <= cutoff
                ):
                    expired.append(record)
                    del self._jobs[job_id]

        for record in expired:
            if record.output_artifact is not None and self.artifact_cleanup is not None:
                try:
                    self.artifact_cleanup(record.output_artifact)
                except Exception:
                    pass
            shutil.rmtree(record.workspace, ignore_errors=True)
        return len(expired)

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)
