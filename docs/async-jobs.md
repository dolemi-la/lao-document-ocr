# Asynchronous conversion jobs

The API supports bounded local background jobs for larger PDFs while keeping the original synchronous endpoints available.

No Redis, database, account, or cloud service is required.

## Why jobs exist

A 60-page OCR conversion can take much longer than a normal HTTP request.

The job API separates:

- upload
- queueing
- OCR/export work
- status polling
- result download
- cancellation

The default web UI uses this job flow.

## Create a job

```http
POST /v1/jobs
Content-Type: multipart/form-data
```

Upload field: `file`.

Successful response: HTTP 202.

Example:

```json
{
  "id": "f7d...",
  "filename": "scan.pdf",
  "status": "queued",
  "cancellation_requested": false,
  "error": null,
  "download_ready": false
}
```

A very fast job may already be `running` or `succeeded` by the time the upload response is returned.

## Batch submission

Queue multiple documents atomically:

```http
POST /v1/jobs/batch
Content-Type: multipart/form-data
```

Repeat the form field name `files` for every document.

Example response:

```json
{
  "count": 2,
  "jobs": [
    {"id": "...", "filename": "a.pdf", "status": "queued"},
    {"id": "...", "filename": "b.png", "status": "queued"}
  ]
}
```

Batch rules:

- default maximum: 10 files (`BATCH_MAX_FILES`)
- every file type is validated before reservation
- the whole batch must fit current active-job capacity
- if capacity is insufficient, no jobs are reserved
- if any upload fails before enqueue, all reserved batch workspaces are discarded
- submission rate limiting charges one unit per document, not one unit per HTTP request

Each returned job is polled/downloaded/cancelled through the normal per-job endpoints.

## Check status

```http
GET /v1/jobs/{job_id}
```

Statuses:

- `uploading`
- `queued`
- `running`
- `succeeded`
- `failed`
- `cancelled`

The response includes created/started/completed timestamps when available.

## Download

After `status=succeeded`:

```http
GET /v1/jobs/{job_id}/download
```

The response is the same ZIP bundle used by synchronous conversion:

- editable DOCX
- Markdown
- TXT
- structured JSON

Trying to download a queued/running/failed/cancelled job returns HTTP 409.

## Cancel

```http
DELETE /v1/jobs/{job_id}
```

Cancellation is cooperative.

Queued jobs can usually be cancelled before execution. Running OCR checks cancellation at PDF/page processing boundaries, so it does not kill a native OCR call in the middle of one page.

## Capacity

The local worker queue is deliberately bounded.

Environment variables:

```text
JOB_MAX_WORKERS=2
JOB_MAX_ACTIVE=8
JOB_RETENTION_SECONDS=3600
JOB_ROOT=/tmp/lao-document-ocr-jobs
```

When the active-job limit is reached, new jobs return HTTP 429 instead of allowing unbounded memory/disk/CPU pressure.

Existing limits still apply:

```text
MAX_UPLOAD_BYTES=26214400
MAX_PAGES=60
```

## Retention

Terminal job workspaces are retained long enough for result download, then lazily cleaned when jobs are accessed/submitted.

The default is one hour.

A production deployment with shared storage or multiple API replicas should eventually replace this in-memory manager with a persistent queue/storage adapter.

## Process model

The current implementation is intended for one API process.

The queue and job metadata are in memory. Result files live under `JOB_ROOT`.

For multi-process/multi-host deployments, use a future external queue backend rather than running independent in-memory queues behind a load balancer.

## Synchronous compatibility

These endpoints remain available:

- `POST /v1/parse`
- `POST /v1/convert`

They are useful for simple scripts and small local documents.

The web UI uses `/v1/jobs` by default.

## Optional right-angle auto-orientation

Single-job and batch submissions accept an optional multipart boolean:

```text
auto_orient_right_angles=true
```

The default is `false`. For `POST /v1/jobs/batch`, the value applies to every file in the batch. The option is stored on each `JobRecord`, survives queueing, and is exposed by job-status responses as `auto_orient_right_angles`.

The same optional field is available on synchronous `POST /v1/parse` and `POST /v1/convert`.
