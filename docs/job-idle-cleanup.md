# Lifecycle-managed cleanup during idle API periods

## Scope

Starting checkout: `7652a05de85ab017059e259fd1e2e5a04e94b664`.

The process-local expiry queue retained failed deletion work, but only processed
it when another request or embedding caller invoked `cleanup_expired()`. An idle
API could therefore keep expired workspaces or a due deletion retry indefinitely.

The API now starts one periodic cleanup thread during its ASGI lifespan. It calls
the existing cleanup pass even without incoming requests. This adds scheduling,
not a persistent queue, directory scan, new storage account, or OCR operation.
Only records already owned by the current manager are eligible for cleanup.

## Configuration

```text
JOB_CLEANUP_INTERVAL_SECONDS=30
```

The default cadence is 30 seconds. Set it to `0` to disable periodic cleanup;
existing request-triggered and explicitly invoked cleanup remain available.
The API accepts integer seconds. Negative values and values exceeding the
platform's threading timeout limit fail during startup. The Python manager also
accepts finite fractional seconds, which the local regression tests use.
Booleans, strings, NaN and infinity are rejected by the manager before creating
its workspace root or executor.

Docker Compose and the local, public, S3 and GPU environment examples expose the
setting. `/health` includes `jobs.cleanup_interval_seconds` as configuration,
not as a guarantee that a deletion succeeded or a worker is healthy.

The worker waits one interval before its first pass, then one full interval
after each completed pass. Slow passes do not produce catch-up bursts or overlap
another invocation from the same worker. Cadence is not an exact deletion SLA:
thread scheduling, process suspension, backlog, UTC clock changes, and storage
latency can delay physical deletion.

`JOB_RETENTION_SECONDS`, `JOB_MAX_RETAINED`, active-job admission limits, and
all existing deletion/retry eligibility rules are unchanged. In particular,
periodic ticks do not bypass the 30-second exponential retry backoff or its
one-hour maximum. Each pass still claims at most eight due tasks. More work can
remain for the next interval or an intervening request-triggered pass.

## Lifecycle and concurrency

Importing the API or constructing `ConversionJobManager` does not start this
thread. Startup calls `start_cleanup_worker()` on the lifespan's captured
manager. Embedding callers can explicitly start it themselves. Repeated or
concurrent starts on that owner are idempotent. A stopped owner is not restarted;
create a new manager for a new application lifecycle.

The worker delegates to the same claim and deletion machinery used by ordinary
manager activity. Task claims remain protected by the manager lock. Foreground
and idle passes cannot claim the same pending task simultaneously, and immediate
cancellation cleanup retains its existing exclusion flag. Storage and workspace
I/O execute outside that lock. Active/uploading jobs do not become eligible
merely because the worker is running.

ASGI teardown shuts down the exact manager captured at startup, even if a global
reference was subsequently replaced. Shutdown runs through a threadpool rather
than joining workers on the ASGI event-loop thread. Startup failure and context
exceptions also invoke teardown for that owner.

Manager shutdown first [closes job admission](job-admission-shutdown.md) and
reconciles queued work as cancelled, without deleting an in-progress upload or
force-cancelling work already running. It then signals the idle worker and waits
up to **five seconds** for it to finish. Waiting for the next interval is interruptible immediately.
An already-running cleanup pass finishes cooperatively; it is not killed midway
through storage I/O. With `shutdown(wait=False)`, no join wait is requested.

If the five-second join does not finish, shutdown logs a fixed warning instead
of claiming the callback stopped. The worker is a daemon, remains stop-requested,
and may finish its current pass later; it does not schedule a subsequent pass.
No replacement worker is started on that owner. The manager then performs its
existing executor shutdown (`cancel_futures=True`). With `wait=True`, running
conversion/native/storage work may still delay that executor shutdown. **The
five-second bound applies to joining the idle thread, not to total API shutdown
or to every storage operation.** Normal teardown tests verify the idle thread
has exited and the executor is closed.

An unexpected exception escaping a cleanup pass emits a fixed warning and
allows a later interval to invoke the pass again. Exception messages, tracebacks,
object keys, paths, credentials, and document text are not included in that
worker warning. Individual failed deletes remain handled by the existing
private queue and backoff rather than by the scheduling layer.

## Observability, availability, and limits

Existing aggregate cleanup attempt/failure and pending-job metrics now change
without API traffic. No new per-document metric label is added. Metrics requests
can still trigger their own cleanup pass; this change does not make every
request-side storage operation asynchronous or deadline-bounded.

Public job expiry remains separate from physical deletion. Expired jobs stay
unavailable while their cleanup tasks are pending. The subsequent
[download-lease protection](job-download-leases.md) keeps an already admitted
response's archive/workspace out of deletion until its response closes. New
requests cannot extend expired public access. Browser request deadlines and
same-job recovery do not prevent server-side expiration or certify erasure.

**Cleanup ownership remains process-local and is lost on restart.** The worker
does not discover old objects/workspaces, reconstruct jobs, persist failed
deletes, or coordinate multiple API processes. A new process cannot infer that
unrecognized files are safe to delete. Storage-namespace-bound persistent
ownership, provider lifecycle management, and external security review remain
open. A hung provider call, abrupt process termination, or publication that
stores bytes before failing to return an artifact can still leave data outside
completed managed cleanup.

## Verification

New tests exercise the lifecycle helper and real manager cleanup passes. They
cover inert construction, disabled scheduling, one owner under concurrent starts,
interruptible waiting, blocked callbacks and bounded joins, unexpected failure
privacy, explicit shutdown, retry after thread-start failure, and callback-local
stop without self-join.

Integration tests use the real filesystem adapter and the real S3 adapter with
an in-memory client double, not an external provider. Thread events bound waits;
explicit timestamps determine expiry and backoff eligibility. They verify idle
expiry with no public/status/reservation calls, repeated failed deletion followed
by recovery, preservation of the eight-task limit, concurrent foreground cleanup,
active-job protection, and configuration validation. ASGI lifespan tests exercise
actual job creation with an authored upload and a fixture archive runner, then
wait for cleanup without further HTTP requests. Teardown and exception-path
checks verify that the correct owner is shut down off the event-loop thread.

These are lifecycle/storage correctness tests, not OCR accuracy or real optical
evidence. No recognizer was trained and no institutional document was downloaded.
Phetsarath policy, recognizer weights, orientation thresholds, source registries,
frozen evaluation inputs, capture kits, and collector sessions are unchanged.
Local logs and integrity fingerprints remain Git-ignored under
`benchmarks/private/idle-cleanup-7652a05/`.
