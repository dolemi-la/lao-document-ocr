"""One process-local periodic cleanup loop, explicitly owned by the API lifespan.

This schedules existing cleanup passes; it does not persist ownership, discover
old files, interrupt storage I/O, or turn expiry into an erasure guarantee.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)
DEFAULT_CLEANUP_INTERVAL_SECONDS = 30
CLEANUP_SHUTDOWN_TIMEOUT_SECONDS = 5


def _validate_seconds(value: float, name: str) -> None:
    # Check range before converting: booleans, NaN, infinity and huge integers
    # must not create hot loops or overflow the platform's timed-lock primitive.
    if type(value) not in (int, float) or not 0 <= value <= threading.TIMEOUT_MAX:
        raise ValueError(f"{name} must be finite non-negative seconds within threading limits")


class IdleCleanupWorker:
    def __init__(
        self,
        cleanup: Callable[[], object],
        *,
        interval_seconds: float = DEFAULT_CLEANUP_INTERVAL_SECONDS,
    ) -> None:
        _validate_seconds(interval_seconds, "interval_seconds")
        self.interval_seconds = interval_seconds
        self._cleanup = cleanup
        self._stop = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._closed = False

    @property
    def is_alive(self) -> bool:
        with self._lifecycle_lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Start once; construction is inert and zero disables periodic passes.

        Repeated starts while running are idempotent. Stop is permanent for this
        owner, so a slow old callback cannot overlap a restarted replacement.
        """
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Idle cleanup worker has been stopped")
            if self.interval_seconds == 0 or self._thread is not None:
                return False
            thread = threading.Thread(
                target=self._run,
                name="lao-ocr-idle-cleanup",
                daemon=True,
            )
            self._thread = thread
            try:
                thread.start()
            except BaseException:
                self._thread = None
                raise
            return True

    def _run(self) -> None:
        # Fixed delay after each pass: no overlapping self-invocations, busy loop,
        # or catch-up burst if a pass is slow. The first pass waits one interval.
        while not self._stop.wait(self.interval_seconds):
            try:
                self._cleanup()
            except Exception:
                # Individual deletion failures/backoff belong to the manager.
                # An unexpected pass error must not log document/provider details.
                logger.warning("Idle job cleanup pass failed; retrying after the next interval.")

    def stop(self, *, timeout: float = CLEANUP_SHUTDOWN_TIMEOUT_SECONDS) -> bool:
        """Signal stop and wait at most timeout seconds for this thread.

        False means an in-flight pass is still running, not that deletion or
        termination was confirmed. The daemon finishes that pass cooperatively.
        The caller must not discard/reuse its private owner as a durable queue.
        """
        _validate_seconds(timeout, "timeout")
        with self._lifecycle_lock:
            self._closed = True
            self._stop.set()
            thread = self._thread
        if thread is None:
            return True
        if thread is threading.current_thread():
            return False
        thread.join(timeout=timeout)
        return not thread.is_alive()
