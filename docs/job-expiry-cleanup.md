# Process-local retry ownership for expired job cleanup

## Failure and scope

Starting checkout: `f26edababc896e338ce7824addd28d0b23280062`.

The [cancellation ownership fix](job-artifact-ownership.md) retained an artifact
when its immediate deletion failed. The later expiry pass still removed the
job record before attempting deletion, then ignored storage exceptions. A second
failure therefore lost the only managed reference. Failed local workspace
removal likewise had no retry owner. Regression tests reproduce these paths.

Expiry now separates public availability from private cleanup ownership.
**This is an in-memory retry queue, not restart-safe or guaranteed erasure.**
The initial queue change added no database, persistent ledger, object-store
account, OCR run, training experiment, or document capture. Periodic scheduling
is now supplied by the [idle cleanup follow-up](job-idle-cleanup.md).

## Expiry and ownership

When an eligible terminal job expires, its public record is removed under the
manager lock. A private cleanup task retains only its workspace path, optional
`StoredArtifact`, independent stage-completion state, and retry bookkeeping.
It does not retain the full document, original filename field, page corrections,
review summary, worker future, or public error. The artifact reference can itself
contain a filename/key; these remain private and are not anonymized.

The removed job stays unavailable through status, cancellation, and download
lookups, even if stored bytes still exist. Failed deletion does not resurrect
its successful status or allow the archive to be downloaded. Existing callers
already holding a record or open stream are not revoked by this change.

Storage and workspace deletion are tracked independently:

- Failed stored-object deletion does not prevent attempting local workspace
  removal. The storage reference remains for a later retry.
- Successful object deletion is not repeated merely because local workspace
  removal failed. The remaining task tracks the workspace stage only.
- An already absent workspace counts as complete. A missing child or inability
  to inspect the root is not treated as proof that the whole directory is gone.
- Workspace removal does not follow a root symlink. Local paths returned by a
  legacy/custom runner outside its assigned workspace are not unlinked.
- Without an artifact-cleanup callback, the object remains pending rather than
  being silently forgotten. Production supplies the configured adapter's delete
  callback. Restoring a custom callback permits later due cleanup.

Callbacks must be idempotent. A provider can commit a deletion and then report a
transport error; the safe outcome is to retain the reference and retry deletion.
The filesystem and S3 adapters already allow deletion of a missing object.
This does not guarantee secure erasure, deletion of all object versions, or that
a faulty callback truthfully reports completion.

`cleanup_expired()` still returns the number of newly expired **public jobs**,
not a count of successful object/workspace deletions. A retry-only pass returns
zero even when it finishes previously pending cleanup. Terminal completion
counters are not incremented again by cleanup retries.

## Retry scheduling and concurrency

An expiry pass claims at most eight due tasks. Each claimed task makes at most
one attempt at each unfinished stage. Unclaimed tasks remain queued. The
initial retry delay is 30 seconds and doubles after each incomplete attempt:
30, 60, 120, 240, 480, 960, 1920, then 3600 seconds, capped at one hour. The
exponent is saturated even for very large failure counts. These are fixed
operational defaults, not measured storage-performance targets.

The deadline is based on UTC time after an attempt; explicit `now` supplies a
logical clock for tests/callers. Clock adjustments and process inactivity can
delay retry. No task is retried repeatedly inside one pass. Ordinary repeated
polling before its due time does not create a tight deletion loop.

Tasks are claimed under the existing manager lock, and deletion I/O runs outside
it. Concurrent cleanup passes cannot claim the same task. A slow deletion does
not hold the lock needed to inspect unrelated jobs or reserve capacity. The
existing cancellation-cleanup flag still prevents expiry from racing an
immediate cancellation delete, including when retention is zero.

The API now runs the [lifespan-managed idle worker](job-idle-cleanup.md), by
default every 30 seconds after each completed pass. Existing manager activity
(reservation, status/record lookup, snapshots/metrics) and embedding callers can
still invoke expiry directly. With the periodic worker disabled, a due task
waits for that activity. Storage and filesystem calls remain synchronous and
have no new server-side I/O deadline.
A hung call can hold its claimed batch and delay the invoking request; the web
request deadline neither terminates it nor certifies deletion.

## Admitted downloads

[Download leases](job-download-leases.md) defer physical cleanup while an already
admitted archive response is using the job. Expiry still removes the public
record. Leased tasks are excluded before the eight-task selection budget and
retain their capacity slot; they do not increment cleanup attempts/failures.
After the last response closes, a later pass retries any due unfinished stages.
This changes neither the existing backoff nor restart-durability limitations.

## Bounded retention and admission

A new configurable process-wide admission limit bounds all live and retained
terminal job records plus private pending-cleanup tasks:

```text
JOB_MAX_RETAINED=1024
```

The Python manager option is `max_retained_jobs`. It must be an integer greater
than or equal to `max_active_jobs`; invalid values fail before creating the job
root or executor. The API passes the environment setting to its manager, reports
it in `/health` job limits, and Docker Compose and all deployment example files
expose the same setting.

The limit includes successful results still inside normal retention, not only
failed cleanups. It is a record-count bound, **not a disk-byte quota**. A busy
installation can reach it without any storage outage. Configure it together
with retention and expected volume; increasing it does not repair a failing
storage service or extend job availability.

New single/batch reservations first attempt due cleanup. If admission would
exceed the retained limit, the existing `JobCapacityError`/HTTP 429 path rejects
them before creating new application job workspaces. Framework multipart parsing
may already have accepted/spooled request data. Batch admission is all-or-none;
existing jobs and cleanup references are not discarded to admit replacements.
The active-job limit remains separate. Completing pending cleanup or normal
expiry restores capacity without a process restart.

## Observability and privacy

The manager snapshot and `/metrics` add fixed aggregate series:

| Metric | Meaning |
| --- | --- |
| `lao_ocr_jobs_retained` | Public records plus private cleanup tasks. |
| `lao_ocr_jobs_retained_limit` | Configured retained-record capacity. |
| `lao_ocr_cleanup_pending_jobs` | Pending tasks, including claimed/in-progress tasks. |
| `lao_ocr_cleanup_pending_artifacts` | References still awaiting confirmed expiry deletion. |
| `lao_ocr_cleanup_in_progress` | Currently claimed tasks. |
| `lao_ocr_cleanup_attempts_total` | Claimed expiry task attempts in this process. |
| `lao_ocr_cleanup_failures_total` | Expiry attempts with either stage still incomplete. |

These counters are task counts, not object-store request counts. They exclude
the separate immediate cancellation-delete attempt. They reset on restart.
Scraping metrics uses the existing snapshot path and can trigger a due cleanup
pass; it is not a passive durable queue inspection.

Incomplete attempts log a fixed warning and internal job ID without exception
messages, tracebacks, object keys, local paths, credentials, or document text.
Metrics contain no per-job/key labels and accept only known nonnegative integer
aggregate fields. Public job responses do not expose retry bookkeeping.

## Verification and remaining limits

The new regressions cover repeated failures, capped backoff, zero retention,
partial-stage success, missing callbacks, concurrent expiry, batch budgets,
retained-capacity rejection/recovery, symlink/external-file preservation,
idempotent deletion after a lost success response, log/metric privacy, and
actual HTTP create/status/download/cancel/metrics routes. Storage coverage uses
the real filesystem adapter and the real S3 adapter with an in-memory client
double. HTTP tests use authored fixture uploads and a stored-result test runner,
not a new OCR engine evaluation or live S3 account.

A restart control explicitly demonstrates the remaining boundary: a replacement
manager does **not** reconstruct pending tasks from the old storage objects.
No restart-durability claim is inferred from successful same-process retries.

Process crashes, restarts, or `put_file` committing bytes before failing to
return a reference can still leave objects outside managed cleanup. Persistent
job/cleanup ownership with storage-namespace binding, provider lifecycle
controls, and external deployment security review remain necessary follow-up
work. Never present API expiry or a cancelled status as verified data erasure.

Local test logs and protected input fingerprints remain Git-ignored under
`benchmarks/private/job-expiry-cleanup-f26edab/`. OCR thresholds, Phetsarath
policy, recognizer weights, remote source registry, optical capture kits,
collector sessions, and frozen challenge inputs remain unchanged.
