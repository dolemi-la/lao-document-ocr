from __future__ import annotations

import re
import threading
import time
from collections import Counter, defaultdict

_JOB_DOWNLOAD = re.compile(r"^/v1/jobs/[^/]+/download$")
_JOB_DETAIL = re.compile(r"^/v1/jobs/[^/]+$")


def normalize_route(path: str) -> str:
    if path == "/v1/jobs/batch":
        return path
    if _JOB_DOWNLOAD.match(path):
        return "/v1/jobs/{job_id}/download"
    if _JOB_DETAIL.match(path):
        return "/v1/jobs/{job_id}"
    return path


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class ApiMetrics:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._requests: Counter[tuple[str, str, int]] = Counter()
        self._duration_sum: dict[tuple[str, str], float] = defaultdict(float)
        self._duration_count: Counter[tuple[str, str]] = Counter()

    def observe_request(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        route = normalize_route(path)
        key = (method.upper(), route, int(status_code))
        duration_key = (method.upper(), route)
        with self._lock:
            self._requests[key] += 1
            self._duration_sum[duration_key] += max(0.0, float(duration_seconds))
            self._duration_count[duration_key] += 1

    def render_prometheus(self, job_snapshot: dict | None = None) -> str:
        with self._lock:
            request_items = sorted(self._requests.items())
            duration_items = sorted(self._duration_sum.items())
            duration_counts = dict(self._duration_count)

        lines = [
            "# HELP lao_ocr_http_requests_total Total HTTP requests.",
            "# TYPE lao_ocr_http_requests_total counter",
        ]
        for (method, route, status), count in request_items:
            labels = (
                f'method="{_escape_label(method)}",'
                f'route="{_escape_label(route)}",'
                f'status="{status}"'
            )
            lines.append(f"lao_ocr_http_requests_total{{{labels}}} {count}")

        lines.extend(
            [
                "# HELP lao_ocr_http_request_duration_seconds_sum Sum of HTTP request durations.",
                "# TYPE lao_ocr_http_request_duration_seconds_sum counter",
            ]
        )
        for (method, route), total in duration_items:
            labels = f'method="{_escape_label(method)}",route="{_escape_label(route)}"'
            lines.append(
                f"lao_ocr_http_request_duration_seconds_sum{{{labels}}} {total:.9f}"
            )
        lines.extend(
            [
                "# HELP lao_ocr_http_request_duration_seconds_count Count of timed HTTP requests.",
                "# TYPE lao_ocr_http_request_duration_seconds_count counter",
            ]
        )
        for (method, route), count in sorted(duration_counts.items()):
            labels = f'method="{_escape_label(method)}",route="{_escape_label(route)}"'
            lines.append(
                f"lao_ocr_http_request_duration_seconds_count{{{labels}}} {count}"
            )

        if job_snapshot:
            lines.extend(
                [
                    "# HELP lao_ocr_jobs_current Current jobs by status.",
                    "# TYPE lao_ocr_jobs_current gauge",
                ]
            )
            for status, count in sorted(job_snapshot.get("current_by_status", {}).items()):
                lines.append(
                    f'lao_ocr_jobs_current{{status="{_escape_label(status)}"}} {count}'
                )

            lines.extend(
                [
                    "# HELP lao_ocr_jobs_completed_total Completed jobs by terminal status.",
                    "# TYPE lao_ocr_jobs_completed_total counter",
                ]
            )
            for status, count in sorted(job_snapshot.get("completed_total", {}).items()):
                lines.append(
                    f'lao_ocr_jobs_completed_total{{status="{_escape_label(status)}"}} {count}'
                )

            lines.extend(
                [
                    "# HELP lao_ocr_job_duration_seconds_sum Sum of completed job durations.",
                    "# TYPE lao_ocr_job_duration_seconds_sum counter",
                ]
            )
            for status, total in sorted(job_snapshot.get("duration_seconds_sum", {}).items()):
                label = _escape_label(status)
                lines.append(
                    f'lao_ocr_job_duration_seconds_sum{{status="{label}"}} '
                    f'{float(total):.9f}'
                )

        return "\n".join(lines) + "\n"


class RequestTimer:
    def __init__(self) -> None:
        self.started = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.started
