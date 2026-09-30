# Close job admission safely during shutdown and scheduling failure

## Reproduced defects and scope

Starting checkout: `e2c5f885cd23d567e37fe1597e2a5a5697989c0a`.

Three regression tests failed against that revision. A manager that had shut down
still accepted reservations and created upload directories. Queued executor
futures cancelled by shutdown retained `queued` job status and active-capacity
accounting. An executor scheduling exception also left its job `queued`, without
a returned future that could execute or complete it.

This change adds an admission boundary and reconciles scheduling with the existing
job lifecycle. It does not persist jobs, restart executors automatically, introduce
an external queue, change OCR, or claim that shutdown erases stored data.

## Manager admission boundary

The manager closes reservation and enqueue admission under its lock before
shutdown waits for the idle cleanup thread or conversion executor. Both single
and batch reservations check this state before request-triggered cleanup, and
again before allocating workspaces. The second check covers shutdown occurring
while cleanup is blocked on unrelated storage I/O.

An already closed owner raises `JobManagerClosedError` with fixed wording.
`start_cleanup_worker()` cannot restart that owner. Construct a replacement
manager for a new lifecycle; toggling the flag back on is not a supported recovery
operation. Existing status, cancellation, download leases, and explicit cleanup
remain available on the original manager while its process and records exist.
Normal retention still applies; shutdown does not renew download access.

## Queued versus running work

Shutdown marks jobs still in the manager's `queued` state as `cancelled`, records
a completion timestamp, sets the cancellation flag, and updates terminal counters
once. It then cancels their executor futures outside the manager lock. Repeated
shutdown and cancellation do not change those completion timestamps or increment
the terminal counter again.

A future may already have been dequeued by an executor thread while its job is
still waiting to enter the manager's worker boundary. `_run()` now requires a
present `queued` record before it calls the runner. If shutdown changed that
record to terminal first, the runner is not invoked, even when `Future.cancel()`
cannot cancel the already-dequeued future. The future itself need not have the
`cancelled` state; job cancellation describes suppressed runner execution.

Work already admitted to `running` keeps the existing graceful-shutdown policy.
Shutdown alone does not set its cancellation event, delete its workspace, or
suppress a successful stored-result handoff. Explicit user cancellation and the
batch error path retain their separate cooperative-cancellation behavior.

An `uploading` reservation is not marked terminal or unlinked by shutdown: its
caller may still be writing input bytes. Enqueue is rejected after closing, and
the upload owner must finish or abort copying before discarding its reservation.
The HTTP handlers perform this cleanup once upload copying returns. Embedding
callers remain responsible for abandoned uploads; an upload that never returns
is not made safe to delete merely by closing admission.

Idle-worker joining and executor shutdown remain outside the manager lock.
Future cancellation callbacks also run outside it. The existing five-second idle
thread join bound is unchanged, and it is still not a total shutdown deadline.
Running native OCR/storage work can delay `shutdown(wait=True)`.

## Executor rejection is not proof that nothing was queued

The installed thread executor inserts a work item before it adjusts its worker
thread count. A thread-start failure can therefore raise from `submit()` after
queue insertion but before a future is returned to the manager.

On a scheduling exception, the manager closes further admission and marks the
rejected record `failed` under the same lock that protects entry into `_run()`.
Its public error is only `Conversion worker is unavailable.` The original
exception remains the internal Python cause, but is not forwarded into public
job or HTTP response bodies and is not newly logged by this path.

A queued work item corresponding to that rejected request cannot subsequently
invoke OCR: `_run()` ignores terminal or already-discarded records. Closing
admission also prevents repeated requests from accumulating additional abandoned
executor work items while the executor cannot start threads.

Already-running or successfully queued work is not cancelled merely because a
different submission failed to schedule. A later explicit manager shutdown still
cancels queued work as described above. Replacing the manager or restarting the
API is required to restore admission after scheduling failure; there is no blind
in-process reopening or automatic request resubmission.

## HTTP behavior and request ownership

`POST /v1/jobs` and `POST /v1/jobs/batch` return HTTP 503 for manager closure or
executor rejection, with this fixed body:

```json
{"detail": "Conversion service is unavailable. Try again later."}
```

Malformed uploads and correction maps keep their existing validation responses.
Capacity limits keep HTTP 429. Framework multipart parsing/spooling can still
precede the endpoint; rejecting application admission is not a guarantee that
no network bytes or framework temporary data were accepted.

Each submission captures its manager before reservation. Upload completion,
enqueue, error cleanup, and the initial acknowledgement all use that same owner,
not a replacement global reference encountered partway through the request.
This preserves owner isolation; it is not live migration of jobs to another
manager or a multi-process routing mechanism.

Single-job enqueue failure discards its request-owned upload workspace. In a
batch, all input copying and validation complete before scheduling begins. If
scheduling is interrupted, members with no returned future are discarded after
the runner-start guard has made rejected scheduling safe. Earlier members with
futures receive cooperative cancellation instead of having their workspaces
unlinked while running. Unrelated jobs are untouched.

**A rejected batch is not a transactional rollback of work already begun.** An
earlier member may be running or even successful by the time a later member
fails to schedule. Its ordinary cancellation/result-retention rules apply; a
late cancellation does not erase an already successful archive. The error
response does not acknowledge new job IDs or promise that every member did
nothing. Clients must not interpret HTTP 503 as an exactly-once guarantee or
blindly resubmit a batch on that basis.

Discarded upload directories retain the existing best-effort filesystem cleanup
behavior. This slice does not add durable retry ownership for failed pre-enqueue
discard operations. Job/cleanup state is still process-local. A crash/restart or
an upload owner that never returns remains outside this shutdown contract.

## Verification

`tests/test_job_admission.py` and `tests/test_job_admission_api.py` add 34 cases.
Thread events determine the relevant concurrency boundaries; timeouts only bound
failed tests. Tests cover shutdown before and during admission cleanup, closure
before a blocked idle-worker join, repeated shutdown/cancellation accounting,
queued-versus-running work, pre-enqueue upload ownership, and callbacks acquiring
the manager lock.

Scheduling tests exercise both a queue-then-raise wrapper and failure at the real
executor thread-count adjustment boundary. They verify that later executor work
cannot revive rejected or discarded jobs, that repeated HTTP requests do not
feed a failed executor, and that unaffected running work can still complete.

Actual single/batch HTTP routes are checked for fixed 503 responses, original
manager ownership, request-local cleanup, partial batch cancellation, preservation
of already completed results, manual-zero acknowledgements, and existing leased
downloads after manager shutdown. These tests use authored PNG upload fixtures
and a tiny archive runner, not OCR or external storage services.

Logs, the initial failing tests, the recorded plan, and protected input fingerprints
remain Git-ignored under `benchmarks/private/job-admission-e2c5f88/`. Phetsarath
policy, OCR selection thresholds, recognizer weights, source registries, capture
kits, collector sessions, and frozen evaluation inputs are unchanged. Restart-safe
cleanup and the genuine optical benchmark remain open.
