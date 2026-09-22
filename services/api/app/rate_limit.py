from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    remaining: int
    retry_after_seconds: int


class SlidingWindowRateLimiter:
    def __init__(
        self,
        *,
        requests: int,
        window_seconds: int,
        max_clients: int = 10_000,
    ) -> None:
        if requests < 0:
            raise ValueError("requests must be non-negative")
        if window_seconds < 1:
            raise ValueError("window_seconds must be at least 1")
        if max_clients < 1:
            raise ValueError("max_clients must be at least 1")

        self.requests = requests
        self.window_seconds = window_seconds
        self.max_clients = max_clients
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return self.requests > 0

    def _prune_queue(self, queue: deque[float], cutoff: float) -> None:
        while queue and queue[0] <= cutoff:
            queue.popleft()

    def _prune_stale_clients(self, cutoff: float) -> None:
        stale: list[str] = []
        for key, queue in self._events.items():
            self._prune_queue(queue, cutoff)
            if not queue:
                stale.append(key)
        for key in stale:
            self._events.pop(key, None)

    def check(
        self,
        key: str,
        *,
        now: float | None = None,
    ) -> RateLimitDecision:
        if not self.enabled:
            return RateLimitDecision(
                allowed=True,
                remaining=0,
                retry_after_seconds=0,
            )

        timestamp = time.monotonic() if now is None else float(now)
        cutoff = timestamp - self.window_seconds

        with self._lock:
            queue = self._events.get(key)
            if queue is None:
                if len(self._events) >= self.max_clients:
                    self._prune_stale_clients(cutoff)
                if len(self._events) >= self.max_clients:
                    return RateLimitDecision(
                        allowed=False,
                        remaining=0,
                        retry_after_seconds=self.window_seconds,
                    )
                queue = deque()
                self._events[key] = queue

            self._prune_queue(queue, cutoff)

            if len(queue) >= self.requests:
                retry_after = max(
                    1,
                    math.ceil(queue[0] + self.window_seconds - timestamp),
                )
                return RateLimitDecision(
                    allowed=False,
                    remaining=0,
                    retry_after_seconds=retry_after,
                )

            queue.append(timestamp)
            return RateLimitDecision(
                allowed=True,
                remaining=max(0, self.requests - len(queue)),
                retry_after_seconds=0,
            )

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "requests": self.requests,
                "window_seconds": self.window_seconds,
                "max_clients": self.max_clients,
                "tracked_clients": len(self._events),
            }
