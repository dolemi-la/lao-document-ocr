"""Recoverable exact-key staging on a private local POSIX filesystem.

No directory scanning, age heuristics, journal, or background task lives here.
The manager journals the final key; that key also identifies its one stage.
All cooperating writers/cleaners sharing a root must use this protocol.
"""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

STAGING_DIRECTORY = ".lao-ocr-writes-v1"
LOCK_SLOTS = 64
COPY_CHUNK_BYTES = 1024 * 1024


class FilesystemWriteError(RuntimeError):
    """Fixed messages only; do not expose paths or exception bodies."""


class FilesystemWriteBusyError(FilesystemWriteError):
    pass


class RecoverableFilesystemWrites:
    def __init__(self, root: Path) -> None:
        try:
            import fcntl
        except ImportError:
            raise FilesystemWriteError(
                "Recoverable filesystem storage requires local POSIX locking."
            ) from None
        if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
            raise FilesystemWriteError(
                "Recoverable filesystem storage requires local POSIX locking."
            )
        self.root = root
        self.directory = root / STAGING_DIRECTORY
        self._fcntl = fcntl

    def _relative(self, destination: Path) -> Path:
        try:
            relative = destination.relative_to(self.root)
        except ValueError:
            raise FilesystemWriteError("Invalid filesystem result destination.") from None
        if not relative.parts or relative.parts[0] == STAGING_DIRECTORY:
            raise FilesystemWriteError("Invalid filesystem result destination.")
        return relative

    def _digest(self, destination: Path) -> str:
        try:
            return hashlib.sha256(
                self._relative(destination).as_posix().encode("utf-8")
            ).hexdigest()
        except UnicodeError:
            raise FilesystemWriteError("Invalid filesystem result destination.") from None

    def stage_path(self, destination: Path) -> Path:
        return self.directory / f"{self._digest(destination)}.part"

    def lock_path(self, destination: Path) -> Path:
        slot = int(self._digest(destination), 16) % LOCK_SLOTS
        return self.directory / f"lock-{slot:02x}"

    @staticmethod
    def _regular_or_missing(path: Path) -> os.stat_result | None:
        try:
            info = path.lstat()
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise FilesystemWriteError("Unsafe filesystem result file; operation refused.")
        return info

    def _check_destination(self, destination: Path) -> None:
        relative = self._relative(destination)
        current = self.root
        for part in relative.parts[:-1]:
            current = current / part
            try:
                info = current.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(info.st_mode):
                raise FilesystemWriteError("Unsafe filesystem result directory; operation refused.")
        self._regular_or_missing(destination)

    def _ensure_directory(self) -> None:
        self.directory.mkdir(mode=0o700, exist_ok=True)
        info = self.directory.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise FilesystemWriteError("Unsafe filesystem staging directory; operation refused.")
        self.directory.chmod(0o700)

    @staticmethod
    def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
        return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)

    @contextmanager
    def _locked(self, destination: Path, *, wait: bool) -> Iterator[None]:
        self._ensure_directory()
        path = self.lock_path(destination)
        descriptor = os.open(
            path,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
            0o600,
        )
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != 0:
                raise FilesystemWriteError("Unsafe filesystem staging lock; operation refused.")
            current = self._regular_or_missing(path)
            if current is None or not self._same_file(info, current):
                raise FilesystemWriteError("Filesystem staging lock changed; operation refused.")
            os.fchmod(descriptor, 0o600)
            operation = self._fcntl.LOCK_EX
            if not wait:
                operation |= self._fcntl.LOCK_NB
            try:
                self._fcntl.flock(descriptor, operation)
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN}:
                    raise FilesystemWriteBusyError(
                        "Filesystem result write is busy; retry cleanup."
                    ) from None
                raise
            current = self._regular_or_missing(path)
            if current is None or not self._same_file(info, current):
                raise FilesystemWriteError("Filesystem staging lock changed; operation refused.")
            yield
        finally:
            # Closing this open description releases its lock. Never unlink a
            # stable slot: waiters/other processes must keep seeing the same inode.
            os.close(descriptor)

    def _remove_own_stage(self, stage: Path, created: os.stat_result) -> None:
        current = self._regular_or_missing(stage)
        if current is None:
            return  # Atomic rename already moved this inode to the final result.
        if not self._same_file(created, current):
            raise FilesystemWriteError("Filesystem staging owner changed; operation refused.")
        stage.unlink()

    def put(self, source: Path, destination: Path) -> int:
        """Copy privately, then publish atomically. Return size observed while owned."""
        try:
            with self._locked(destination, wait=True):
                self._check_destination(destination)
                stage = self.stage_path(destination)
                # A stale copy is a cleanup obligation, not permission to overwrite it.
                if self._regular_or_missing(stage) is not None:
                    raise FilesystemWriteError(
                        "Unfinished filesystem result write requires cleanup."
                    )
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                destination.parent.chmod(0o700)
                descriptor = os.open(
                    stage,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                    0o600,
                )
                created: os.stat_result | None = None
                try:
                    # Close the descriptor even when its first metadata lookup fails.
                    created = os.fstat(descriptor)
                    os.fchmod(descriptor, 0o600)
                    with os.fdopen(descriptor, "wb") as output:
                        descriptor = -1  # The file object now owns the descriptor.
                        with source.open("rb") as input_file:
                            shutil.copyfileobj(input_file, output, COPY_CHUNK_BYTES)
                        output.flush()
                        os.fsync(output.fileno())
                        size = os.fstat(output.fileno()).st_size
                    self._check_destination(destination)
                    current = self._regular_or_missing(stage)
                    if current is None or not self._same_file(created, current):
                        raise FilesystemWriteError(
                            "Filesystem staging owner changed; write refused."
                        )
                    stage.replace(destination)
                    return size
                finally:
                    if descriptor != -1:
                        os.close(descriptor)
                    if created is not None:
                        self._remove_own_stage(stage, created)
        except OSError:
            raise FilesystemWriteError("Filesystem result write failed.") from None

    def delete(self, destination: Path) -> None:
        """Remove only this key's known stage and final file; busy means retry."""
        try:
            with self._locked(destination, wait=False):
                self._check_destination(destination)
                stage = self.stage_path(destination)
                self._regular_or_missing(stage)
                # Validate both before deleting either. Partial failures still raise:
                # the manager must retain its key until both paths are absent.
                stage.unlink(missing_ok=True)
                destination.unlink(missing_ok=True)
                # Do not prune shared result directories while independent writers
                # may be preparing them. Empty directory scaffolding is not an object.
        except OSError:
            raise FilesystemWriteError("Filesystem result cleanup failed.") from None
