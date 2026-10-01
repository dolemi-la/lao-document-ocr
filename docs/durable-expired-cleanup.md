# Opt-in restart recovery for expired-job cleanup

Starting checkout: `d30d70e25e99cde3b12ba2e427700850bd0b358c`.

## Scope and boundary

`JOB_CLEANUP_DURABLE=true` adds a private SQLite journal for cleanup ownership.
The initial slice persisted already-expired tasks. The
[result-publication follow-up](result-publication-ownership.md) now commits
ownership before managed API storage writes and preserves it through terminal
retention, cancellation, and expiry. A replacement manager retries unfinished
cleanup without restoring public access or invoking OCR again. The default
remains `false` and retains process-local behavior.

This is not a persistent job queue or complete upload/workspace recovery.
Workspaces before publication intent, custom runners bypassing the tracked
publisher, temporary/multipart upload remnants, and provider writes completing
after recovery deletion remain outside the completed contract. The broader
full-lifecycle cleanup roadmap item remains open.

No scan/photo benchmark, recognizer change, font change, or accuracy claim is
part of this work. Filesystem tests use authored fixtures; S3 tests use the real
adapter with a protocol double, not a live storage account.

## Enable deliberately

For a direct single-process API deployment, set:

```text
JOB_CLEANUP_DURABLE=true
JOB_ROOT=/a/persistent/service-owned/job-directory
JOB_CLEANUP_INTERVAL_SECONDS=30
```

Use a dedicated, trusted local filesystem directory writable by the API account.
The journal lives at `JOB_ROOT/.expired-cleanup.sqlite3`. Keep the same canonical
job-root path and storage configuration across restarts. A temporary directory,
ephemeral container filesystem, removed volume, or lost disk does not become
durable merely because the flag is enabled. Do not share this directory among
independent API processes or use it as an untrusted journal-import location.

The optional Compose overlay provides a named job volume and enables the mode:

```bash
docker compose \
  -f docker-compose.yml \
  -f deploy/compose.durable-cleanup.yml \
  up --build
```

Apply it last when combining it with the existing public, S3, or GPU overlays.
It fixes `JOB_ROOT` at `/data/jobs` and mounts `lao-ocr-jobs` at that location.
Both API image definitions prepare that directory for UID/GID 10001 with private
permissions. Rebuild the applicable image before enabling the overlay. An
existing volume must also be writable by that account; this feature does not
recursively repair permissions on arbitrary operator-owned data.

The base Compose configuration and environment examples still default to the
old temporary job root and disabled persistence. No existing deployment is
silently opted in. Do not remove the named volume during a routine restart.
This volume also retains unexpired workspaces across container recreation.
Only workspaces with a committed ownership row are recovered; unknown older or
pre-publication workspaces are not automatically discovered after a crash.

## Commit and recovery ordering

Expiry selects eligible terminal records under the manager lock. In persistent
mode it atomically transfers known publication rows and adds expiry-only rows
before removing public records or invoking deletion. Managed publication first
creates its row before storage I/O, then records terminal retention. A row contains
only the job identifier, optional artifact key, workspace-stage flag, UTC retry
time, and failure count. Workspace paths are derived from the bound job root
and validated identifiers, not loaded as arbitrary paths from disk.

The normal bounded cleanup pass then attempts due unfinished stages. Confirmed
stage completion, retry time, and failure count are committed before the manager
forgets that ownership in memory. A completely finished task is removed from the
journal before its in-memory record is dropped.

Recovery loads only the validated, bounded journal set. It does not scan result
buckets or directories, infer old jobs from filenames, rebuild download URLs,
restore status responses, carry over terminal counters, or schedule old OCR
runners. Recovered identifiers return HTTP 404. Cleanup can run through the
existing idle worker or request-triggered passes; construction itself performs
no storage-provider deletion.

The existing eight-task pass budget and 30-second exponential backoff, capped
at one hour, remain in place. Each unfinished stage is attempted at most once
per claimed task per pass. Recovered tasks count against `JOB_MAX_RETAINED`.
Lowering that setting below the recovered row count refuses startup instead of
loading only a subset or forgetting entries. Aggregate process counters reset
on restart; per-task retry timing and failure counts do not.

## Ambiguous commits and idempotent retries

A provider can commit deletion before its response is lost. The process can
also stop after deleting an object but before committing the completion row.
Recovery may therefore repeat deletion of an already absent object. Callbacks
must remain idempotent; this is not an exactly-once deletion protocol.
Recovered artifacts carry the persisted key and placeholder non-key metadata;
custom cleanup callbacks must identify objects by key, as the supplied filesystem
and S3 adapters do. Do not opt in a custom adapter that deletes by a filename,
size, or media-type field instead of the bound object key.

A failed or ambiguous journal operation pauses cleanup and closes new admission
in the current manager. No automatic in-memory fallback is used. Public API
requests encountering the failure return a fixed HTTP 503 response without SQL,
paths, keys, credentials, or exception bodies. Already-running provider calls
are not forcibly interrupted. The manager retains its uncommitted in-memory
ownership until shutdown, and a replacement reconciles from the actual committed
journal state.

An expiry transfer that never succeeded leaves the original public record in
memory and performs no deletion. Tracked publications retain their earlier
committed row across restart. A legacy/custom record first journaled at expiry
still has no durable owner if that first commit fails. Journal availability and
the remaining full-lifecycle gaps must not be ignored.

## Storage binding and one active owner

The journal identity hashes the canonical job root and storage namespace. For
filesystem storage this includes its canonical result root. For S3 it includes
the bucket, normalized prefix, and effective endpoint and region, including SDK
resolved defaults. Access-key, secret-key, and session-token fields are not
collected. The canonical identity payload is hashed rather than stored as raw
JSON. An unresolved S3 endpoint cannot enable persistence.

A different binding refuses startup even when there are no pending rows. This
prevents silent rebinding when an operator changes storage configuration or
copies the ledger to another root. Restore the matching configuration or plan
an explicit offline migration; deleting the journal to bypass a mismatch loses
cleanup ownership. Storage identity is an operational guard, not a replacement
for account isolation, correct credentials, or provider authorization.

The SQLite connection uses exclusive locking for its lifetime. A second manager
or process cannot replay the same queue concurrently. A forked child cannot use
an inherited journal connection; do not preload an active manager before
forking workers. The current service remains single-process, not horizontally
scalable through this journal.

Waiting shutdown closes admission and stops new durable cleanup passes, but
keeps the journal owned until in-flight cleanup passes and admitted download
leases finish. A slow transfer therefore cannot lose its file to a replacement
manager. Nonwaiting shutdown deliberately keeps ownership until a later waiting
shutdown or process exit. Process exit releases that process's locks; a live,
stalled provider/transfer can still delay handover indefinitely.

## Validation, privacy, and disk limitations

Startup checks schema/version, binding, SQLite integrity, row validity, unique
job ownership, and retained capacity before making recovered work available to
cleanup. Job identifiers must have the generated 32-character hexadecimal form.
Artifact keys must belong to that job's direct `jobs/<id>/<filename>` namespace
and stay within the byte limit. Unsafe path segments, control characters,
non-UTF-8 names, linked journal files/sidecars, or malformed rows fail closed;
there is no silent repair or partial-row import.

The job root is private and the database is mode 0600. Only a hash of the root
and storage identity is stored, but artifact keys themselves are plaintext and
can contain filenames. No separate source-name field, document text, review
summary, or page-correction list is copied. This is not encryption or verified
erasure: SQLite pages, filesystem snapshots, storage versions, backups, and
provider retention policies can retain data after logical deletion.

The database has a 64 MiB page-growth limit and startup rejects oversized known
journal files. This is not a combined disk quota for uploads, results, sidecars,
or the entire deployment. Preserve and protect the database together with its
SQLite sidecar state; do not copy just a live database file as a recovery scheme.

SQLite uses rollback journaling, `synchronous=EXTRA`, and `fullfsync=ON` where
supported. Local journal transactions execute synchronously inside ownership
critical sections; a slow local disk can delay unrelated operations. Provider
and workspace deletion still run outside the manager lock. No provider timeout,
I/O cancellation, disk-hardware guarantee, or remote/shared-filesystem locking
guarantee is introduced.

The implementation follows SQLite's documented [exclusive locking](https://www.sqlite.org/pragma.html#pragma_locking_mode)
and [synchronization modes](https://www.sqlite.org/pragma.html#pragma_synchronous),
with explicit serialization for Python's [shared-thread connection option](https://docs.python.org/3.11/library/sqlite3.html#sqlite3.connect).
These settings do not substitute for testing the actual deployment filesystem.

## Verification and next slice

Regression coverage includes filesystem/S3-double recovery, backoff and partial
stage persistence, unchanged public expiry, capacity/pass budgets, real
subprocess exits before and after deletion, uncertain intent/completion commits,
a real SQLite commit denial, corrupt/mismatched/schema-invalid ledgers,
cross-process exclusion after a same-process open attempt, fork guards, download
and slow-cleanup shutdown races, API 503 privacy, and API restart behavior.
The default non-durable restart control remains unchanged.

Self-review found and fixed malformed-schema duplicate ownership and invalid
UTF-8 error handling before acceptance. Local plans, initial failures, review
regressions, and gate logs are retained in the ignored directory
`benchmarks/private/durable-expiry-d30d70e/`.

The [publication follow-up](result-publication-ownership.md) covers managed
storage writes, lost replies, and normal retention. Remaining work includes
pre-publication upload/workspace ownership and incomplete/late-provider-write
reconciliation, separately from restoring public jobs or re-running OCR.

## Initial expired-cleanup slice gates

All 1,428 Python tests passed, including 78 new journal, manager, HTTP, process,
and configuration cases. Ruff, web lint, all 145 web tests, the production web
build, and `git diff --check` passed. The Python suite retained six dependency
deprecation warnings without failures.

Compose configuration rendering passed for the durable overlay with base,
public, S3, and GPU configurations. This checks configuration composition, not a
built/running container, CUDA runtime, live S3 account, or deployment filesystem.
All 13 protected-input hashes matched the starting manifest, including the
Phetsarath font, recognizer weights, capture kit, and evaluation inputs.
