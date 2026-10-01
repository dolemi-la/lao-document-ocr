# Durable ownership before managed result publication

Starting checkout: `c6049c726341e460246059f3b999759cf17b9ec0`.

The [filesystem staging follow-up](filesystem-staging-cleanup.md) covers new
key-derived temporary copies with active-write protection. Remaining temporary
file gaps below refer to legacy random files and other storage protocols; no
unknown files are scanned or adopted.

## Scope

With `JOB_CLEANUP_DURABLE=true`, the API now commits private cleanup ownership
**before calling result storage**, rather than waiting until the job expires.
This covers a completed object whose storage reply is lost, successful results
still inside retention, and cancellation before expiry. Recovery owns cleanup
only: old job IDs remain unavailable, and neither OCR nor publication is replayed.

The feature uses the existing journal, storage binding, capacity limits, and
persistent-volume configuration in [durable cleanup](durable-expired-cleanup.md).
Persistence remains off by default. No running deployment is enabled by a code
change, and an ephemeral job root still cannot survive volume or disk loss.

This is not complete durable job scheduling. The later
[workspace follow-up](workspace-ownership.md) journals input/workspace ownership
before directory allocation. A custom runner's returned artifact is recorded at
terminal handoff, but a direct untracked write with a lost reply still has no
predeclared object key. Storage temporary files, multipart fragments, version
histories, and provider writes completing after recovery deletion require
separate reconciliation/lifecycle controls. None of this guarantees erasure.

## Publication protocol

The production runner finishes OCR, review construction, and archive generation
before calling `record.publish_result(key, publisher)`. The manager installs that
callback on its own records; the API does not look up a global replacement
manager to acquire ownership for a job already running under another owner.

For a managed durable job, the publisher performs these steps:

1. Verify that the same record is owned, still running, executing on its owning
   runner thread, not previously published, and not already cancelled.
2. Commit the validated job ID, exact intended artifact key, workspace-stage
   flag, provisional cleanup deadline, and zero failures. New durable reservations
   promote their existing workspace-only row; they do not allocate another row.
   No provider call is made until this commit returns successfully.
3. Execute the synchronous storage callback outside the shared manager lock.
   A slow upload does not hold the state lock needed by unrelated jobs.
4. Require the returned `StoredArtifact` to name the predeclared key. The runner
   must also return that same published result, not substitute another output.
5. Commit the terminal cleanup deadline in the same critical section as terminal
   success publication. Public readers cannot observe success before that
   deadline commit completes.

Only one publication is admitted per job. A different thread, second invocation,
removed record, terminal record, or pre-existing cancellation cannot start a
new managed publication. The thread check rejects accidental detached writers;
the callback must still wait for any provider-managed work it starts before
returning. It must not write additional objects outside the declared key.

## Deadlines and restart behavior

The provisional deadline is the intent time plus the configured retention
interval. A terminal job updates that to its completion time plus retention.
While the original manager is alive, the row is not in its private cleanup queue:
the public job still owns it, and running jobs are never cleaned just because
the provisional deadline passes. Normal retention and download leases continue
to govern the live result.

If the process exits before terminal finalization, recovery uses the committed
provisional deadline. If terminal finalization committed, recovery uses that
persisted completion-based deadline. A new process does not recompute either
one using its new retention setting. Recovery does not reconstruct public jobs
or extend old download availability; it retains private deletion ownership
until the stored deadline becomes due.

A provisional deadline is a cleanup scheduling rule, **not** proof that a remote
provider has stopped a request. A provider can still finish a previously accepted
write after the local process exits, including after a later deletion. This
journal alone cannot prevent that late write from recreating an object. Provider
lifecycle/unfinished-upload controls and reconciliation remain necessary.

## Failed publication and cancellation

A storage callback can commit the object and then raise instead of returning a
reference. The job fails with a fixed public error, but its predeclared artifact
key remains privately owned. Provider exception bodies are not copied into the
public response or logged by the managed publication path. Cleanup can delete
the known key after its deadline even when no result reference was returned.

A returned reference with a different key is rejected. Cleanup retains the
intended key, not an arbitrary replacement key. A broken adapter that writes
elsewhere violates the contract; this check cannot discover undeclared objects.

Cancellation during publication remains cooperative. After a valid result is
returned, the manager records the cancelled outcome and may attempt immediate
object deletion. It commits successful object-stage completion before forgetting
the key; the workspace stage stays journaled until normal expiry. A crash or
lost commit receipt after deletion can cause idempotent deletion to be repeated,
but cannot turn an unconfirmed receipt into a forgotten managed key.

## Expiry transfer and capacity

The journal row format and schema version remain unchanged. A row can now be
owned by a live managed publication or by a pending cleanup task. These are not
two capacity slots: live records and private recovered/expired tasks remain
counted once each against `JOB_MAX_RETAINED`.

Normal expiry atomically transfers a known publication row into cleanup and
adds any legacy expiry-only rows in the same selected set. A known row must
exist and match its key and workspace ownership. Missing rows, changed keys,
duplicates, and invalid batches fail closed rather than silently replacing
ownership. An interrupted transfer leaves either the prior publication row or
the committed expiry row, both sufficient for private recovery.

The existing eight-task cleanup pass budget, backoff, independently persisted
object/workspace stages, storage-namespace checks, one-owner journal exclusion,
and download-lease shutdown behavior remain in force. No bucket or directory
scan is introduced. `discard()` refuses to drop a journaled publication; its
normal cancellation/expiry owner must finish cleanup instead.

A failed or ambiguous intent, terminal, expiry, or deletion-receipt commit
pauses cleanup and admission in the manager. The existing fixed HTTP 503 path
applies. There is no automatic unjournaled fallback or provider retry hidden in
this protocol. A failed pre-publication commit starts no storage write; if the
intent did commit before its reply was lost, recovery safely owns an empty or
partially prepared workspace and an object key that may never have existed.

## Custom runner contract

Use the manager-supplied record and call the helper synchronously from its runner:

```python
key = f"jobs/{record.id}/{archive.name}"
return record.publish_result(
    key,
    lambda: storage.put_file(
        archive,
        key=key,
        filename=archive.name,
        media_type="application/zip",
    ),
)
```

The storage adapter must match the manager's bound namespace and clean objects
idempotently by key. A standalone `JobRecord` without a manager retains its old
untracked behavior. Direct `storage.put_file()` calls also remain possible for
legacy/custom runners. Their returned key gains ownership at terminal handoff,
but the earlier store-before-reference gap requires this publisher helper.
Existing untracked files from earlier runs are not automatically adopted. Opting into durable
cleanup is not retroactive recovery of unknown objects.

## Verification

`test_result_publication.py` covers pre-write ownership, original retention
recovery, concurrent job operations, lost storage replies, uncertain intent and
terminal commits, one-publication enforcement, mismatched returned keys, and
cancellation. `test_result_publication_recovery.py` covers actual HTTP/API runner
wiring, mixed-row atomic transfers, malformed transfer rejection, and six real
subprocess exit boundaries: before writing, after writing without a reply,
before/after terminal commit, and before/after cancellation deletion.

`test_result_publication_review.py` covers the off-thread writer regression,
uncertain immediate-cancellation receipts, discard refusal, and an unexpired
download holding exclusive ownership through shutdown. Filesystem storage is
real; S3 tests use the actual adapter with a local protocol double. HTTP tests
stub OCR/archive production with authored fixtures, not recognition evidence.

Plans, initial failures, review failures, protected-input hashes, and gate logs
are retained locally in `benchmarks/private/result-publication-c6049c7/`, which
is Git-ignored. No recognizer, Phetsarath font, source registry, capture kit,
collector session, or frozen evaluation data is changed by this slice.

## Initial publication-slice local gates

All 54 new focused publication cases passed, including six abrupt subprocess
exit boundaries. The full Python suite passed 1,482 tests; Ruff, web lint, all
145 web tests, the production web build, and `git diff --check` passed. Six
existing dependency deprecation warnings remained, with no failing tests.
All 13 protected-input hashes matched the starting manifest.

Self-review reproduced and fixed an off-thread publication admission gap before
acceptance. The gate logs and review record are in the ignored evidence directory
above. No container startup, live S3, or OCR-quality claim follows from these tests.
