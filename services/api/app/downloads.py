"""Bind private download ownership to the actual ASGI response lifetime."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator

import anyio
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse, Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from services.api.app.jobs import JobDownloadLease

logger = logging.getLogger(__name__)


class _ClosingIterator(Iterator[bytes]):
    """Do not close/release storage while a synchronous read is still executing."""

    def __init__(self, source: Iterator[bytes]) -> None:
        self._source = source
        self._lock = threading.Lock()
        self._closed = False

    def __next__(self) -> bytes:
        with self._lock:
            if self._closed:
                raise StopIteration
            return next(self._source)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            close = getattr(self._source, "close", None)
            if callable(close):
                close()


class LeasedDownloadResponse(Response):
    """Release on completion/error/disconnect, not merely endpoint return.

    A background callback alone is insufficient: it may not run after a failed
    ASGI send or response iteration. Admitted ownership stays held until a
    synchronous source read/close finishes, including when cancellation arrives.
    """

    def __init__(
        self,
        response: Response,
        lease: JobDownloadLease,
        *,
        stream: _ClosingIterator | None = None,
    ) -> None:
        self._response = response
        self._lease = lease
        self._stream = stream
        # Delegate actual sending; share header mutations with the inner response.
        self.status_code = response.status_code
        self.media_type = response.media_type
        self.raw_headers = response.raw_headers
        self.background = None

    @classmethod
    def streaming(
        cls,
        source: Iterator[bytes],
        lease: JobDownloadLease,
        *,
        media_type: str,
        headers: dict[str, str],
    ) -> LeasedDownloadResponse:
        stream = _ClosingIterator(source)
        response = StreamingResponse(stream, media_type=media_type, headers=headers)
        return cls(response, lease, stream=stream)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            if isinstance(self._response, FileResponse):
                # Delegated pathsend can outlive this response's file ownership.
                # Preserve FileResponse headers/ranges but read bytes in this call.
                scope = {
                    **scope,
                    "extensions": {
                        name: value for name, value in scope.get("extensions", {}).items()
                        if name != "http.response.pathsend"
                    },
                }
            await self._response(scope, receive, send)
            # FastAPI may attach response-level dependency cleanup/background work.
            # The inner response runs its own callbacks; this handles only ours.
            if self.background is not None:
                await self.background()
        finally:
            try:
                if self._stream is not None:
                    # Disconnect cancellation must not skip releasing the provider body.
                    with anyio.CancelScope(shield=True):
                        await run_in_threadpool(self._stream.close)
            except Exception:
                logger.warning("Download stream close failed.")
            finally:
                self._lease.close()
