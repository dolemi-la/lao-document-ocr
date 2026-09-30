# Stored-result ownership when cancellation races completion

## Defects reproduced

Starting checkout: `570725012b0a76624c0b334db54879d611e7ec38`.

A runner could finish storing a ZIP, then observe cancellation before the job
manager published success. The manager marked the job cancelled without saving
its returned `StoredArtifact`. The object stayed in result storage, but expiry
had no reference with which to delete it. A deterministic event-gated regression
reproduced this, without OCR, remote documents, or a real object-store service.

The API worker also computed orientation review after publishing its archive.
A failure in that calculation could leave a stored object without returning its
reference. A second regression reproduced publication before a review failure.
Both tests failed against the starting implementation and passed after the fix.

## Ownership boundary

The worker now completes selected-orientation review and archive generation
before calling result storage. Cancellation is checked again after archive
creation, so an already-cancelled export is not unnecessarily published.

After `put_file` returns normally, the worker immediately returns its artifact.
The job manager takes ownership under its existing lock before choosing success
or cancellation. There is no intervening worker review or cancellation-delete
step that can lose a successfully returned reference.

If cancellation wins that decision, the result is never made downloadable. The
job is marked cancelled, and the manager makes one immediate best-effort call
to its configured artifact-cleanup callback. Production configuration supplies
the result adapter's `delete` method. The filesystem and S3 adapters share this
handoff; no provider-specific object key or credential is added to job status.

If success wins first, a subsequent cancellation returns the existing successful
status. It does not delete the successful result. Normal retention still applies.

## Concurrency and failure behavior

Storage deletion runs outside the job-manager lock. A private, non-public
in-progress flag prevents expiry from removing the job or issuing a competing
delete while cancellation cleanup is running, including with zero retention.
Other job reservations, status checks, and cancellation can acquire the manager
lock during this cleanup. This does not make every existing storage operation
nonblocking; it only keeps this cleanup callback outside that lock.

Successful cleanup clears the stored reference and the in-progress flag. Later
expiry removes the transient workspace without repeating that successful delete.
The input and locally generated exports remain private workspace files until
normal retention cleanup; cancellation is not immediate erasure of all scratch
files. Legacy runners returning a local `Path` keep the existing workspace-based
cleanup behavior. Arbitrary paths returned by custom runners are not unlinked.

If immediate deletion raises an exception, the job stays cancelled, without a
public error, downloadable result, or completed orientation-review summary. Its
private stored-artifact reference is retained for the existing expiry pass.
A fixed warning is logged with the internal job ID, not the exception message,
stack trace, object key, local path, or document text. Polling/repeated user
cancellations do not create a new immediate-delete retry loop.

**Cleanup remains best-effort, not durable deletion.** In particular:

- Expiry is invoked by manager activity and, in the API, the follow-up
  [idle cleanup worker](job-idle-cleanup.md). Periodic passes use the same
  eligibility and retry backoff; they do not provide restart durability.
- The follow-up [expiry cleanup queue](job-expiry-cleanup.md) now hides expired
  public jobs while retaining unfinished artifact/workspace deletion for delayed
  retries in the same process. It does not add restart durability or a persistent ledger.
- A process crash/restart, or a storage call that commits bytes but raises before
  returning a reference, can still leave stored bytes requiring independent
  retention management. Provider versioning/lifecycle semantics are unchanged.
- Custom managers without an artifact-cleanup callback cannot delete stored
  objects. The cancelled reference is private while the record exists; the
  callback must be configured for manager-driven object deletion.
- No storage-I/O deadline or forced interruption of an in-flight upload/native
  OCR call is introduced. A hung cleanup can delay final worker return and that
  record's expiry. Frontend request deadlines do not cancel server execution.

These boundaries are intentional scope limits, not claims that cancelled data
has been securely erased or that object-store lifecycle controls are unnecessary.

## Verification

`tests/test_job_artifact_ownership.py` adds 25 deterministic cases. Threading
Events gate exact handoff/deletion boundaries; timeouts only bound a broken test,
not determine the ordering. Coverage includes:

- cancellation after storage and immediately after the real API worker returns;
- actual create/cancel/status/download routes, worker, pipeline and exporters,
  with authored fixture OCR lines, explicit zero rotations and auto on/off;
- a real filesystem adapter and the real S3 adapter with an in-memory client
  protocol double, never an external S3 account;
- expiry during blocked deletion, zero retention, unrelated reservations and
  unrelated stored-result preservation;
- immediate deletion failure followed by successful expiry cleanup, fixed-log
  privacy, no duplicate successful delete and exactly-once terminal counters;
- successful-result protection against late cancellation, missing callback
  behavior, pre-run cancellation, pre-publication review/export failures and
  cancellation after local export;
- legacy local-path behavior without deleting a file outside the job workspace.

Cancelled downloads return HTTP 409; once the job expires, its status returns
404. Rotation acknowledgements remain unchanged and review summaries remain
hidden for cancelled jobs. Existing successful-job export/review tests continue
to verify the opposite path.

This is a lifecycle correctness change, not a new OCR or optical-data result.
The frontend, OCR selection thresholds, Phetsarath policy, recognizer weights,
remote source registry, frozen challenge and capture kit are unchanged.

Local validation logs and integrity fingerprints are kept in the Git-ignored
`benchmarks/private/job-artifact-ownership-5707250/` directory. No source images,
remote PDFs, transcription ground truth or fonts are published with this slice.
