# Bounded conversion requests in the web app

## Scope

Starting checkout: `118399365b5b1e006cbe9dbe5c85c43ebdee01cd`.

[Same-job recovery](web-job-recovery.md) handled requests that failed, but a
request that never returned headers, JSON, or its complete archive body could
leave the interface busy indefinitely. A cancellation request could also remain
pending and prevent another cancellation attempt. These are browser transport
failures, not reasons to start another OCR job.

This change bounds conversion-related HTTP operations while preserving the
existing recovery session, manual-correction acknowledgement, orientation
warnings, and stale-response guards. It changes no API endpoint, OCR pipeline,
model weights, server-side timeout, job-retention setting, or training dataset.

## Per-request deadlines

`apps/web/src/requestDeadline.ts` defines fixed defaults:

| Operation | Deadline | Boundary |
| --- | ---: | --- |
| Status GET, including recovered status | 30 seconds | Headers, JSON body, and response parsing |
| Cancellation DELETE | 30 seconds | Headers and consumed acknowledgement/error body |
| New-job POST | 120 seconds | Upload, headers, and complete confirmation/error body |
| Archive GET | 120 seconds | Headers and the entire Blob response body |

The best-effort DELETE after a manual-correction acknowledgement mismatch is
also bounded by the cancellation deadline. It does not consume or trust its
response body; unused fetch resources are released when the attempt finishes.

These are total operation budgets, not inactivity timers. Slow transfers can
reach a deadline even while making progress. The values are bounded defaults,
not measured performance guarantees or estimates of OCR duration. A healthy
long-running job can continue through any number of successful status polls;
there is no new overall conversion time limit. These constants can be changed
in a deployment's source build, but there is no new user setting or environment
variable for them.

The initial `/health` request is outside this conversion/recovery slice and is
unchanged. Browser timers depend on event-loop scheduling: suspended tabs,
device sleep, or synchronous work can delay their callbacks. This is not a
hard real-time cancellation guarantee or a server resource limit.

## Deadline and cancellation ownership

`withRequestDeadline(...)` creates a request-local AbortController linked to the
owning UI attempt. The fetch and every required body-read operation execute
inside the deadline callback. Only fully consumed data may be returned, not a
live Response whose body is read after the timer has been cleared.

The helper settles its result independently of fetch reacting to abort. A
never-settling transport or a late response from a wrapper that ignores its
signal cannot keep the caller pending. Both promise handlers remain attached
to late work so a late rejection is handled and a late success is discarded.
Timers and parent abort listeners are removed on every terminal path. Remaining
request resources are aborted without aborting the parent UI attempt or a
sibling request. Pre-cancelled attempts do not start work.

A request-local timeout uses the fixed `RequestTimeoutError` marker. Status and
download helpers translate it to retryable `JobRecoveryError("request-timeout")`.
No URL, document text, raw server response, or underlying network error becomes
part of the timeout message. User cancellation still propagates the parent's
cancellation reason rather than being mislabeled a retryable deadline.

## User-visible behavior

A stalled status request enables **Resume this conversion**. A stalled archive
response enables **Retry download**; partial archive data is not offered as a
completed download. Recovery refreshes the same job and checks the same saved
manual correction map, including explicit zero. Editing the form does not change
that map. No deadline automatically retries a POST, GET, or DELETE.

A submission is not acknowledged merely because successful HTTP headers
arrived. Until a complete, valid creation response is received, a timeout,
malformed success body, or lost response leaves the submission outcome unknown.
The localized message explains that the server may already be processing it.
There is no invented job ID, same-job recovery button, or automatic resubmission.
A deliberate new conversion can still create a duplicate job. This does not
add server-side idempotency or control browser/proxy transport retransmission.

If cancellation times out, the UI says it was **not confirmed**, releases its
cancellation-button lock, and permits a later explicit cancellation attempt.
It does not mark the job cancelled or discard its known recovery identity. An
actual terminal job status supersedes a pending cancellation request: its late
reply or deadline must not overwrite a completed result or trigger a stale
screen update. Choosing another upload and unmounting retain their existing
attempt-abort behavior.

Aborting an HTTP request does not stop an already accepted server job. Neither
creation timeout nor cancellation timeout certifies what happened remotely.
The existing in-memory-only recovery and server retention limits still apply.

## Validation

Dependency-free Node tests exercise never-resolving requests, body stalls after
successful headers, ignored aborts, late resolutions and rejections, sibling
isolation, fresh same-job retry, invalid deadlines, default budgets, and timer/
listener cleanup. Existing identity, expiry, error classification, corrections,
privacy, and localization tests remain. TypeScript permits explicit `.ts`
imports in this no-emit frontend project so the tested helper import is also
usable by Node's TypeScript-stripping runner; no dependency was added.

A headless Chrome check drives the actual React app against mocked local HTTP.
It covers stalled status headers, partial status JSON, incomplete ZIP streams,
unconfirmed submission headers/body, cancellation timeout and retry, pending
cancellation superseded by success, stale-API cleanup, and a simulated late body
that ignores abort. The status-then-download recovery case submits the original
file once and preserves its corrections and page warnings. Lao timeout text
and mobile overflow checks are included. The previous recovery browser suite
also passes independently against the updated frontend.

For the new browser scenarios only, 30,000/120,000 ms timer delays are mapped to
500/800 ms by an injected test-clock wrapper. Production constants are unchanged
and separately asserted by Node tests. These browser observations validate
state transitions, not real-network latency, actual OCR, or optical accuracy.
They do not constitute a full visual or accessibility audit. Initial harness
selector failures were corrected to target the existing retry control and the
status message rather than its separate recovery-help paragraph; no application
behavior was changed to satisfy those selector failures.

Private validation is under `benchmarks/private/request-deadlines-1183993/`:
`plan.json`, `preservation.json`, `browser-check.mjs`, `browser.summary.json`,
`recovery-regression.mjs`, and `recovery-regression.summary.json`. Reports contain
fixed check names, counts, source hashes, and scope statements rather than
uploaded documents. Test images, browser profiles, and local servers are
temporary. No font binary, remote document, screenshot dataset, or OCR text is
added to the public repository. Phetsarath remains canonical; genuine capture
and issue #17 validation remain separate open work.
