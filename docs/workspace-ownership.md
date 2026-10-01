# Durable ownership from workspace reservation

Starting checkout: `88194a64484730ff442fb221a2acf29b4ec3670c`.

## Scope

With `JOB_CLEANUP_DURABLE=true`, the manager commits cleanup ownership before
creating a job workspace. Interrupted input copying, queued or running OCR,
pre-publication failures, and aborted requests can therefore leave recoverable
private cleanup tasks instead of untracked application directories.

The same row continues through [result publication](result-publication-ownership.md),
terminal retention, cancellation, and expiry. Recovery never reconstructs public
jobs, retries uploads, invokes OCR, or restores download access. Persistence is
still disabled by default, using the existing trusted persistent job root and
single-owner configuration in [durable cleanup](durable-expired-cleanup.md).

This is not a complete storage-reconciliation system. Framework multipart spool
files outside the job root, object-store temporary files, incomplete multipart
uploads, versions/backups, and provider writes completing after recovery deletion
are not covered. Unknown directories from older runs are not adopted or scanned.

## Reservation and allocation ordering

Capacity and correction validation run before reservation. Durable reservation
prepares the entire batch of generated job identifiers and rejects existing paths,
symlinks, repeated identifiers, or already-owned identifiers before journaling.
The whole batch of workspace-only rows commits before the first directory is
created. Each row has a null artifact key, workspace-pending flag, provisional
retention deadline, and zero failures. No source text or separate source filename
is added to the journal.

Only after directory allocation and registration succeed does the caller receive
records. A failed or uncertain journal commit starts no directory allocation or
input copy. When allocation or registration fails after the commit, all committed
rows remain privately owned, including directories never created. Partially
registered public records and unreturned caller pins are removed. A later bounded
cleanup pass can handle an absent directory idempotently.

The service-owned private root must not have independent directory writers.
This protocol does not make concurrent operator edits or a compromised account
safe. It does not adopt a pre-existing directory merely because its name resembles
a generated job ID.

## One row through the lifecycle

The journal schema and version are unchanged. Managed publication now promotes
the existing workspace row to its declared result key instead of inserting a
second row. Promotion requires a matching workspace-only row; missing, already
published, or completed-workspace entries are not silently recreated or replaced.

Terminal jobs persist the completion-based retention deadline for workspace-only
failures and cancellations as well as successful results. A custom runner's
returned artifact is recorded at terminal handoff, but a custom direct storage
write that loses its reply before returning still lacks a predeclared object key.
Use `record.publish_result()` to protect that earlier provider-write boundary.

While the original process owns an uploading, queued, or running job, its
provisional deadline is not permission to delete its workspace. After a crash,
recovery uses the last committed deadline without recomputing it from a replacement
manager's retention setting. Recovered rows count against retained capacity and
remain inaccessible through the public job routes.

## Upload caller ownership and shutdown

Every returned durable reservation pins its original manager until the caller
hands it to `enqueue()` successfully or aborts through `discard()` /
`discard_many()`. The caller must stop all workspace writers before either action
and must not write after handoff. Failed enqueue retains the caller's ownership
until abort finalization. Upload-owned records are excluded from expiry, including
a scheduling-rejected terminal record whose retention has already elapsed.

Waiting shutdown stops new admission and waits for conversion workers as before,
but does not pretend an unfinished upload has stopped. The exclusive journal owner
remains held while caller uploads, admitted downloads, or cleanup passes remain
active. A replacement cannot recover and delete files underneath those users.
After the last upload caller aborts, ownership can be released; any queued cleanup
then belongs to the replacement. The shutdown call itself need not wait forever
for that upload. A caller that abandons a reservation without finalizing it can
hold ownership until process exit; no upload deadline or forced writer interruption
is introduced here.

If a terminal journal update fails during shutdown, the original workspace row
remains available for recovery. Teardown still cancels queued futures and shuts
down the executor rather than stopping halfway through the remaining teardown.

## Request aborts and cancellation

The single and batch API handlers finalize request ownership on ordinary errors
and coroutine cancellation. Real file-copy context managers close their destination
before abort cleanup runs. A batch's unsubmitted members transfer together into
private cleanup; members already submitted are cooperatively cancelled, never
unlinked while their workers might still use them.

Durable abort commits the cleanup transfer before removing public records and
releasing caller pins. It then uses the existing bounded cleanup pass. Failed
workspace deletion retains its row and backoff rather than relying on an ignored
`rmtree` error. Large batches or shutdown can leave deletion pending; returning an
error to the client is not a deletion receipt.

An ambiguous abort commit pauses the manager through the existing fixed journal
failure path. All finished request writers still release their caller pins, so
shutdown is not permanently blocked by a failed cleanup receipt. Recovery uses
the actual committed rows. Task cancellation remains cancellation even when its
cleanup also encounters a journal failure. Ordinary HTTP journal failures retain
the fixed 503 response without private SQL, paths, or provider exception bodies.

The non-durable mode keeps its process-local lifecycle. It also benefits from
submission cleanup on coroutine cancellation, but gains no restart persistence.
Standalone records and writes outside the assigned workspace remain outside the
manager's ownership contract.

## Verification

The new workspace suites contain 63 cases across real filesystem storage and the
S3 adapter with a local protocol double. They cover journal-before-directory
ordering, uncertain reservation/abort commits, partial allocation/registration,
existing-path and duplicate-ID rejection, failed deletion retry, original-deadline
recovery, task cancellation during real file copying, HTTP validation failures,
partial batch scheduling rejection, and shutdown handover.

Seven subprocess scenarios exit abruptly before allocation, during uploading,
while queued, while running, after failure, after pre-publication cancellation,
and after abort transfer before deletion. Replacement managers recover cleanup
without invoking old runners. Self-review reproduced and fixed expiry of a
scheduling-rejected record before its upload caller finalized ownership.

Local evidence is retained in the ignored directory
`benchmarks/private/workspace-ownership-88194a6/`. No model, Phetsarath font,
source registry, capture kit, collector session, or frozen evaluation input is
changed. These are lifecycle tests, not OCR accuracy, live-S3, container-runtime,
or deployment-filesystem validation.

## Completed local gates

All 1,545 Python tests passed, including the 63 new workspace cases and seven
abrupt-exit scenarios. Ruff, web lint, all 145 web tests, the production web
build, and local documentation-link checks passed. Six existing dependency
deprecation warnings remained, with no test failures. All 13 protected-input
hashes matched the starting manifest. No container startup or live provider
operation was used as evidence for this change.
