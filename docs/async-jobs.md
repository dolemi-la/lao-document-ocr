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

Result-readiness checks run outside the shared manager lock, so a slow storage
check does not block unrelated job-state transitions. The response revalidates
ownership and current result identity afterward. `download_ready` remains advisory,
not a download lease; a concurrent state change can yield false until the next
check. See [job-status readiness](job-status-readiness.md) for expiry races and
remaining synchronous-I/O limits.

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

## Downloads crossing expiry

An archive download admitted before public expiry holds a private lease through
response completion, error or disconnect. Expiry can hide the job while its
existing transfer finishes; new requests cannot renew expired access. Cleanup
waits for the last admitted transfer, so physical deletion can occur later than
public expiry. Download admission is capped at four responses per job and 32 per
manager; excess requests return HTTP 429. Missing results still return 410 and
non-successful jobs return 409. See [download leases](job-download-leases.md)
for storage-reader closure, range behavior, metrics and limits. This is not a
transfer-success or erasure guarantee.

## Cancel

```http
DELETE /v1/jobs/{job_id}
```

Cancellation is cooperative.

Queued jobs can usually be cancelled before execution. Running OCR checks cancellation at PDF/page processing boundaries, so it does not kill a native OCR call in the middle of one page.

If cancellation wins after result storage finishes, the manager keeps ownership
of that artifact and attempts deletion without publishing a download. A failure
retains its private reference for the existing expiry pass. A cancellation that
arrives after confirmed success does not delete the successful result. See
[stored-result ownership](job-artifact-ownership.md) for the concurrency contract
and limits of best-effort cleanup; cancellation is not proof of data erasure.

## Capacity

The local worker queue is deliberately bounded.

Environment variables:

```text
JOB_MAX_WORKERS=2
JOB_MAX_ACTIVE=8
JOB_RETENTION_SECONDS=3600
JOB_CLEANUP_INTERVAL_SECONDS=30
JOB_ROOT=/tmp/lao-document-ocr-jobs
```

When the active-job limit is reached, new jobs return HTTP 429 instead of allowing unbounded memory/disk/CPU pressure.

Existing limits still apply:

```text
MAX_UPLOAD_BYTES=26214400
MAX_PAGES=60
```

## Shutdown and scheduling failures

New reservations and enqueue attempts stop before manager shutdown joins workers.
Queued work becomes cancelled with terminal accounting; already-running work keeps
its existing graceful completion policy. A scheduling exception marks that job
failed and closes further admission to that executor. Replace the manager/restart
the API to restore admission; no automatic resubmission is performed.

The single/batch creation endpoints return a fixed HTTP 503 for these unavailable
states, separately from capacity's HTTP 429. Unstarted batch uploads are discarded
only after copying stops; earlier scheduled members receive cooperative cancellation.
Already-running or successful members cannot be transactionally undone. See
[job admission and shutdown](job-admission-shutdown.md) for owner isolation,
queued-work fencing, and the remaining upload/retention limits.

## Retention

Public expiry and physical deletion are separate. Failed archive or workspace
cleanup remains in a private, bounded-admission retry queue with 30-second
exponential backoff capped at one hour. Each existing expiry pass claims at most
eight due tasks. The [idle cleanup worker](job-idle-cleanup.md) invokes these
passes without incoming requests, using the same backoff. Job IDs remain
unavailable while deletion is pending. See [expiry cleanup](job-expiry-cleanup.md).

`JOB_MAX_RETAINED=1024` limits all live/terminal records and pending-cleanup tasks
combined, separately from `JOB_MAX_ACTIVE`. It must be at least the active limit.
At capacity, single and batch submissions return HTTP 429 before allocating new
job workspaces. Successful normal expiry or retry cleanup releases capacity.
By default, a restart still loses the in-memory retry queue. Opt-in
`JOB_CLEANUP_DURABLE=true` journals expired tasks and
[tracked API result publication](result-publication-ownership.md) on a persistent
`JOB_ROOT`, including lost storage replies and unexpired successful results.
Recovery preserves private cleanup deadlines, not public jobs or download access.
Pre-publication uploads/workspaces and incomplete or late provider writes remain
separate gaps; none of this guarantees erasure.

Terminal job workspaces are retained for result download, then cleaned by
request activity or the API lifespan worker. `JOB_CLEANUP_INTERVAL_SECONDS=30`
sets its default cadence; `0` disables periodic work. Construction/import starts
no thread. Teardown requests stop and joins the idle worker for up to five
seconds off the ASGI loop before shutting down the conversion executor. A stuck
storage call is not forcibly terminated or reported as deleted.

The default is one hour.

A production deployment with shared storage or multiple API replicas should eventually replace this in-memory manager with a persistent queue/storage adapter.

## Process model

The current implementation is intended for one API process.

The active queue and public job metadata remain in memory. Workspaces live under
`JOB_ROOT`; stored archives use the configured result adapter. Optional expired
cleanup journaling does not turn this into a multi-process job queue.

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
