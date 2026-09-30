# Protect admitted archive downloads from managed expiry cleanup

## Reproduced failure and scope

Starting checkout: `363a41fea823f9ed5c9f89f27f132f84c4b44c50`.

The download endpoint checked the job and storage, then returned a response whose
bytes would be opened/read later. An idle or request-triggered expiry pass could
delete the archive and workspace between these steps. Three regression cases
reproduced deletion before the first response read: filesystem storage, the real
S3 adapter with an in-memory client, and the legacy local-Path result route.

Downloads now acquire a private, process-local ownership lease before storage
validation. The lease lasts through the ASGI response, not just endpoint return.
Cleanup does not delete that job's archive or workspace while any admitted
response still holds ownership. This prevents this manager's expiry deletion
from racing an admitted response; it is not a delivery, availability, accuracy,
restart-durability, or data-erasure guarantee.

## Public expiry is unchanged

`ConversionJobManager.acquire_download(job_id)` checks the available job under
its lock, validates success, rechecks elapsed retention after any preceding
cleanup work, and snapshots the output path/artifact. No storage existence check
or object read is performed under this admission lock.

The job's completion time is not changed. Expiry still removes its public record
and queues its private cleanup task, even while an older admitted transfer is
running. After expiry, new job lookup/download requests return 404. A transfer
admitted before expiry can finish; a new request cannot renew that old access.
Zero retention admits no completed-job download.

Before expiry, non-successful jobs still return 409. Missing stored/local results
still return 410, and a successful job with no result reference returns 409.
Pre-response failures release admission even when storage validation raises.
A custom caller cannot use `discard()` to remove a job with an active lease.

Leases contain the existing output reference, not OCR data. The release closure
is bound to its original manager, is thread-safe and idempotent, and does not
perform storage I/O. It never releases a sibling transfer. The API retains its
existing single-process ownership model.

## Response lifecycle

`LeasedDownloadResponse` delegates the normal response and releases ownership
in `finally`, including failed header/body sends, stream exceptions and ASGI
cancellation/disconnect. Relying only on a background callback would miss some
of these exits. Response-level background callbacks, when attached, are still
executed without duplicating callbacks owned by the inner response.

Stored-result iterators are explicitly closed before releasing their lease.
Closure is shielded from cooperative task cancellation and performed off the
ASGI event loop. A per-iterator lock prevents closure and release while a
synchronous provider read is still executing. The existing S3 adapter can then
close its response body even when iteration was interrupted. A close failure
logs fixed wording without provider exception text or tracebacks; it does not
invent a successful resource-close receipt.

Legacy local `FileResponse` behavior retains its normal filename/content headers
and range handling, including 206, invalid-range and unsatisfiable-range paths.
The optional `http.response.pathsend` extension is removed from a copied scope
for this response so file reads are not delegated beyond its owned lifetime.
This can forgo a server optimization. It does not add range support to the
stored-artifact streaming route, which had no range implementation before.
The source scope is not mutated.

Normal FastAPI requests invoke the returned response. Embedding callers using
`acquire_download()` directly must close the lease in their own `finally` block;
constructing and abandoning a response object is not a completed transfer.
No destructor or garbage-collection timing is used as a deletion signal.

## Cleanup and bounded admission

The existing expiry pass excludes tasks with active leases before taking its
eight-task budget. Blocked tasks do not count as cleanup attempts or failures,
and do not prevent other due tasks from being selected. Concurrent passes and
lease acquisition/release share the manager lock for bookkeeping only.

When the last response releases, the task remains owned and becomes eligible
for the next due idle/request-triggered cleanup pass. Releasing a response does
not invoke cleanup synchronously, bypass retry backoff, or immediately remove
successful results still inside retention. Cancellation cleanup, terminal
counters and partial-stage retry tracking remain unchanged.

Two fixed process-local limits bound admitted responses:

| Limit | Value |
| --- | ---: |
| Simultaneous downloads per job | 4 |
| Simultaneous downloads per manager | 32 |

Further download requests return HTTP 429 with fixed retry wording. Admission is
atomic across threads, and rejection does not acquire a lease. These constants
are not deployment environment variables. The web's existing request recovery
recognizes 429 as retryable; no frontend changes or automatic retry loop are
introduced by this slice.

An expired task waiting for a transfer still counts toward `JOB_MAX_RETAINED`.
The new limits bound response counts, not stored bytes or total transfer time.
A slow/hung transfer can retain its archive and entire workspace beyond the
nominal public-retention period. Provider/reverse-proxy timeouts and lifecycle
policies still matter; no new server-side transfer deadline is provided.

## Aggregate observability

The manager snapshot adds `active_downloads` and
`cleanup_download_blocked_jobs`. Prometheus renders them as
`lao_ocr_downloads_active` and `lao_ocr_cleanup_download_blocked_jobs`.
These series accept only known nonnegative integers and contain no per-job
labels, file names, object keys, coordinates or document strings. They describe
current-process bookkeeping, not client-confirmed receipt or physical erasure.

Snapshots still invoke normal due cleanup. A nonzero blocked count after public
expiry can explain why retained capacity and physical storage are not yet freed.

## Verification and limits

The regression suite exercises real manager admission/expiry, filesystem storage,
the real S3 adapter with a local protocol double, actual HTTP endpoints and
middleware, and direct ASGI response execution. Event-gated idle/concurrent
cleanup checks hold a response mid-transfer, expire public access without more
HTTP traffic, then release it and observe normal cleanup. Explicit timestamps
determine expiry; thread waits only bound a failing test.

Coverage includes multiple concurrent leases, repeated close, zero retention,
retention advancing during earlier cleanup, per-job/global caps and HTTP 429,
missing results and provider preflight exceptions, blocked-cleanup capacity,
non-starvation of other due tasks, failed sends before/after the first body,
source errors, incoming disconnect, cancellation, source-close ordering,
background callback failure, local-file ranges and disabled deferred pathsend.

These are storage/lifecycle tests using authored fixture bytes, not new OCR or
optical-accuracy evidence. No external S3 service or remote document was used.
Local logs and protected identity checks remain Git-ignored under
`benchmarks/private/download-leases-363a41f/`.

Ownership is still lost on restart and does not coordinate multiple processes.
Independent provider lifecycle deletion, external file removal, provider read
failure, network failure, or process termination can still interrupt downloads.
The wrapper cannot prove that a client persisted an entire ZIP. A hung read or
close can delay lease release. Files awaiting a transfer are not certified
physically erased merely because the job became unavailable.

Restart-safe cleanup ownership and genuine optical benchmark/capture review
remain open roadmap items. Phetsarath policy, OCR defaults and thresholds,
model weights, source registries, frozen evaluation inputs, capture kits and
collector sessions are unchanged.
