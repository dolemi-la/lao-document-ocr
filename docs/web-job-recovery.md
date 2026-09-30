# Recover an existing web conversion without submitting it again

## Scope

The review/correction workflow previously stopped on a failed job-status or ZIP
request. Pressing Convert again was a new upload and a new OCR job, even when
the original conversion had already completed. This change adds explicit
same-job recovery to the existing web app. It does not change any server
endpoint, OCR pipeline, model, page correction, retention setting, or dataset.

Starting checkout: `dab9dd103f874d817dfe30f9bee7b99e06266533`.

## User workflow

After a temporary status/network failure, **Resume this conversion** refreshes
the already acknowledged job and continues polling. After a failed ZIP fetch
or interrupted ZIP response stream, **Retry download** first refreshes that
same job's status and then retrieves its existing archive. Neither recovery
path calls `POST /v1/jobs`, reuploads the file, or starts another OCR operation.
A queued/running job can continue on the server while the browser is paused.
Recovery is a user action, not an unbounded automatic retry loop.

Each recovery snapshot contains the original acknowledged job ID, source
filename for the download fallback, and a copied/frozen manual-correction
list, including explicit `0`. Editing the correction field or automatic option
after a network failure does not alter the submitted job. The help text explains
that applying those edits requires a new conversion. Even invalid new form
input does not block recovery of a valid earlier submission. Choosing a new
file clears the old recovery action, corrections, and result warnings.

The original file remains selected in the existing file state; recovery itself
does not read or send its bytes. No recovery snapshot or document content is
saved in browser storage. This feature does not resume across page reloads,
tab closure, or API restarts, and does not extend the server's result lifetime.

## Identity and acknowledgement checks

Every received status is validated for a bounded endpoint-safe job identifier
and a known lifecycle state. Once a job has been acknowledged, later responses
must carry that same ID. A foreign or malformed response is not followed to
another job or used to retrieve an archive.

The existing exact manual-correction acknowledgement is checked at creation
and on each status response, including the refreshed status before a download
retry. A nonempty correction request still cannot be treated as supported by
an older API that ignores the fields. Missing/mismatched acknowledgements stop
recovery and prevent download. Ordinary submissions with no corrections retain
the existing compatibility with older responses.

Creation-time acknowledgement failure retains the existing best-effort
cancellation of that submitted job. It does not claim cancellation succeeded.
No changed or rejected response is treated as an orientation certificate.

## Failures and cancellation

Network/response-stream failures and HTTP 408, 425, 429, or 5xx responses can
show a same-job retry action. HTTP 404/410 means the job or archive is no longer
available: remove recovery and stale result/review state and explain that a new
conversion would require the original file. Other HTTP errors do not offer a
blind retry. Malformed JSON is distinguished from an interrupted response body.
Recovery errors use fixed localized messages, not raw response/error bodies.

Failed or cancelled jobs do not download or restart. A successful job can keep
its page-review warnings visible through a transient archive failure; after
recovery, the same selected-output review remains. A ZIP download still does
not establish upright pages or accurate OCR.

Status/archive requests use `cache: no-store`, and attempts carry abort signals.
Poll delays release timers/listeners on abort. Selecting another file,
starting a new attempt, or unmounting discards the previous attempt. UI updates
and download dispatch also check attempt ownership, independently of whether
the transport honors abort. Cancellation responses are bound to both the job
snapshot and the attempt that issued them, so an old cancellation cannot
replace a newer upload's state. A confirmed cancellation stops its polling
attempt and suppresses later responses from it.

No acknowledged job ID exists when the creation response is lost. The app
therefore does not retry that upload automatically or fabricate a recoverable
job. Its message says the server may already be processing the submission.
Choosing Convert again is a new job. This is **not** an idempotent-submission or
exactly-once transport guarantee: lower network/proxy/browser layers can behave
differently. The API has no new idempotency-key contract in this slice. Request
deadlines and cross-reload recovery are also outside its scope.

## Regression coverage

Dependency-free Node tests cover immutable snapshots, explicit-zero retention,
known/foreign/invalid identities, GET-only status/download operations, uncached
requests, expired jobs/results, transient and nontransient HTTP failures,
malformed JSON, interrupted response streams, abort before/during/after a
request or delay, filename fallback, and Lao/English messages. Existing
manual-correction and orientation-review tests remain in place. Source-level
App checks were updated for the new download/acknowledgement call sites; those
checks alone are not browser interaction evidence.

A separate local headless Chrome check drives the actual React app against
mocked HTTP at desktop and mobile widths. The main recovery scenario uses
**one submission** despite a failed status request followed by a failed archive
request; later edits to the correction input cannot change its acknowledged
`1:0` choice. Other cases cover expiry, foreign IDs, changed acknowledgements,
failed/cancelled results, new-file reset, delayed cancellation, delayed status
after cancellation, and a creation response lost after server acceptance.
GET/DELETE fetches intentionally ignore AbortSignal in the race fixtures to
verify ownership guards independently. Browser downloads are denied by the
harness; it verifies archive requests and UI behavior, not a file saved by a
person or a real Tesseract service. No OCR runs in these browser controls.

The earlier manual-correction Chrome harness was also copied into the new
private directory and rerun against the changed app. Its historical script
and historical result files were not overwritten.

Local, Git-ignored evidence is under
`benchmarks/private/job-recovery-dab9dd1/`. It contains the recovery browser
harness, the compatibility harness, aggregate summaries, preservation hashes,
and validation logs. Initial fixture debugging is retained separately: a
concurrent status/cancellation check stalled before uncached requests were
added; forcibly destroying the creation socket caused browser-level wire
retries, so the final response-loss fixture deterministically drops the
response after a single mock acceptance. That correction is not evidence of
API idempotency. Temporary Chrome profiles, files, and localhost services are
removed at the end of each run.

These are software recovery/interaction checks, not an OCR accuracy experiment,
optical capture pilot, visual accessibility audit, or external penetration
review. The six-page physical capture pilot, MAF review, missing remote-source
coverage, and issue #17 remain open. Phetsarath policy, model artifacts, frozen
challenge, capture kits and collector sessions are unchanged.
