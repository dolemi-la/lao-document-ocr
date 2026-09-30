# Job-status readiness checks without global storage locking

## Reproduced problem

Starting checkout: `56da66938116133967e0b199a8bb2efc10e41bfa`.

Constructing a public job response called the result-storage existence callback
while holding the job manager's shared lock. For S3 this can involve a provider
HEAD request; for local results it can involve filesystem I/O. A delayed check
therefore blocked unrelated reservations and job-state transitions. The same
path was used when cancelling a job that had already succeeded.

Four event-gated regression cases failed against that checkout: status and
terminal-cancellation reads, each with the real filesystem adapter or the real
S3 adapter backed by an in-memory protocol double. The provider check was held
at a known boundary, and an unrelated reservation could not finish before it
was released. No OCR or external object-store account was involved.

## Snapshot, probe, and revalidate

Public responses now use three short stages:

1. Under the manager lock, verify the record is still owned and snapshot its
   status, immutable output path/artifact, and configured existence callback.
2. Release the lock and perform the existing readiness check. Only a successful
   job needs that check. No extra provider requests or retry loop are added.
3. Reacquire the lock, verify the same record is still registered, and serialize
   its current public fields. Reuse a positive readiness result only if status,
   output identity, and the configured callback are still unchanged.

An expired/discarded record removed during the probe produces the existing
job-not-found result (HTTP 404). A different record occupying the same identifier
cannot inherit its predecessor's result. If output or callback identity changes,
readiness is false rather than retrying indefinitely or applying a stale result
to the new output.

If a running job finishes between the snapshot and serialization, its latest
status and timestamps may be returned with `download_ready: false`: no existence
check for that newly published result has yet completed in this request. A
subsequent poll can check it. This is conservative advisory metadata, not a new
terminal failure or instruction to restart OCR.

Stored-result callback exceptions continue to yield false without exposing the
provider exception in the job response or a new log. An inaccessible legacy
local-file result likewise yields false for filesystem errors instead of
propagating its path-bearing exception. Custom managers without an existence
callback retain their previous stored-reference behavior; no new storage
verification is invented for them.

Cancellation's state mutation still runs under the lock. Response construction
runs after releasing it. A late cancellation of a successful job still leaves
its result, completion timestamp and terminal counters unchanged. Public
correction lists remain detached copies, and orientation-review summaries remain
limited to the successful state.

## Concurrency and limits

While a status check waits, other threads can reserve/enqueue work, complete
running jobs, request cancellation, inspect aggregate state, close admission,
and expire that record. Actual deletion need not wait for a readiness probe:
a status query is **not** an active-download lease.

The [download lease](job-download-leases.md) contract is unchanged. An actual
download still acquires its own admission/retention protection and checks storage
again before serving bytes. `download_ready: true` is an observation, not a
reservation, guaranteed delivery, permanent availability, or integrity check.
External deletion or provider errors can still make a later transfer fail.

This change isolates the shared state lock; it does **not** make the requesting
thread's storage call asynchronous, coalesce concurrent probes, or impose a
provider timeout. A stuck existence check can still consume that request's
thread. Existing request-triggered cleanup can also perform synchronous storage
I/O before public-state lookup. Browser deadlines do not terminate that server
work. Persistent cleanup ownership, server-side I/O limits, and provider lifecycle
management remain separate work.

## Verification

`tests/test_job_readiness.py` covers blocked status/terminal-cancel probes,
concurrent reservation/completion/cancellation/shutdown, expiry during probing,
replacement records/results/providers, local stat failure, current-state
serialization, unchanged late-cancellation behavior, and no provider calls for
non-successful states. Failure messages and arbitrary provider data are not
added to public responses. Events establish ordering; timeouts only bound a
broken test.

`tests/test_job_readiness_api.py` exercises actual HTTP upload, status, cancel,
and leased-download routes with authored PNGs and a tiny archive runner. While
one completed result's check is blocked, a second upload is admitted, completes,
and is downloadable. Expiring the first job before its stale probe returns
produces HTTP 404 without disturbing the second result. The tests use real local
storage and the real S3 adapter with an in-memory client, not live S3 or Tesseract.

These are software lifecycle tests, not OCR accuracy evidence. Phetsarath policy,
model weights, OCR thresholds, source registries, capture kits, collector sessions,
and frozen evaluation inputs are unchanged. Restart-safe cleanup and genuine
optical benchmark validation remain open.

The plan, initial failures, test logs and protected input hashes are retained
locally under the Git-ignored directory
`benchmarks/private/status-readiness-56da669/`.

## Completion validation

Resumed self-review checked every public-response call site, stale ownership and
result rejection, terminal cancellation, download-lease separation, privacy, and
the browser's completed-job path. No blocking issue remained in this slice; the
synchronous-I/O and restart-safety limits above are not marked complete.

Local validation passed 30 focused regression cases, all 1,350 Python tests,
Ruff, web lint, all 145 web tests, the production web build, and `git diff --check`.
All 15 protected-input hashes matched the pre-implementation manifest. The Python
run retained six dependency deprecation warnings, with no test failures.

The resumed gate logs and self-review record are retained beside the original
red-test evidence in the ignored directory above. No live-S3 or optical-OCR
validation is implied by these software tests.
