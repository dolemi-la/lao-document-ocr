# Observability

The API includes a dependency-free observability baseline suitable for local and small self-hosted deployments.

## Request IDs

Every HTTP response includes:

```text
X-Request-ID: <id>
```

Clients may supply their own `X-Request-ID`.

Accepted request IDs:

- maximum 128 characters
- letters/numbers
- `.`
- `_`
- `-`

Invalid/missing values are replaced with a generated 32-character ID.

This gives operators one stable correlation ID to pass through reverse-proxy/application logs.

## Metrics endpoint

```http
GET /metrics
```

The endpoint uses Prometheus text exposition format and does not require the Prometheus Python client.

Current metrics include:

### HTTP requests

```text
lao_ocr_http_requests_total
lao_ocr_http_request_duration_seconds_sum
lao_ocr_http_request_duration_seconds_count
```

Labels:

- HTTP method
- normalized route
- status code (request counter)

Dynamic job IDs are normalized:

```text
/v1/jobs/abc123
→ /v1/jobs/{job_id}

/v1/jobs/abc123/download
→ /v1/jobs/{job_id}/download
```

This prevents high-cardinality time series.

### Job state

```text
lao_ocr_jobs_current{status="queued|running|..."}
lao_ocr_jobs_completed_total{status="succeeded|failed|cancelled"}
lao_ocr_job_duration_seconds_sum{status="succeeded|failed|cancelled"}
```

Completed counters remain available after old job workspaces are cleaned up.

## Example

```text
# HELP lao_ocr_http_requests_total Total HTTP requests.
# TYPE lao_ocr_http_requests_total counter
lao_ocr_http_requests_total{method="POST",route="/v1/jobs",status="202"} 4

# HELP lao_ocr_jobs_current Current jobs by status.
# TYPE lao_ocr_jobs_current gauge
lao_ocr_jobs_current{status="running"} 1

# HELP lao_ocr_jobs_completed_total Completed jobs by terminal status.
# TYPE lao_ocr_jobs_completed_total counter
lao_ocr_jobs_completed_total{status="succeeded"} 12
```

## Prometheus scrape

Example Prometheus configuration:

```yaml
scrape_configs:
  - job_name: lao-document-ocr
    static_configs:
      - targets:
          - lao-document-ocr-api:8000
```

The scrape path defaults to `/metrics`.

## Reverse proxy guidance

If a reverse proxy already generates request IDs, forward its value to the application:

```text
X-Request-ID
```

Keep the value bounded and ASCII-safe.

## Scope

This baseline intentionally avoids:

- external telemetry SaaS
- mandatory Prometheus/OpenTelemetry SDKs
- document contents in metrics
- filenames in metric labels
- per-job-ID metric labels

A larger deployment can add OpenTelemetry/exporters later without making them mandatory for local users.
