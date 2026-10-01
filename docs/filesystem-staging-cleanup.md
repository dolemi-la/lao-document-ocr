# Recoverable filesystem staging without deleting active writes

Starting checkout: `800e1b53bdefec15fae721b871040c9a708102f9`.

## Scope and activation

With `JOB_CLEANUP_DURABLE=true`, the production filesystem result adapter now
uses recoverable staging. The manager already journals the final result key
before storage publication. The adapter derives exactly one staging filename
from that same key, so recovery can remove an interrupted copy even when the
final result was never published. No journal schema migration, directory scan,
new scheduled worker, OCR invocation, or storage-account access is required.

Default non-durable writes keep their existing random sibling temporary-file
behavior. No running deployment is enabled by this code change. The internal
`.lao-ocr-writes-v1` namespace is now reserved and cannot be addressed through
filesystem artifact keys, including in default mode.

Custom durable embeddings must construct the adapter with
`FilesystemArtifactStorage(root, recoverable_writes=True)`, supply its existing
`cleanup_namespace()` and deletion callback to the manager, and publish through
`record.publish_result(...)`. Merely enabling the manager's journal does not
change a custom adapter's write protocol. Direct unjournaled writes can use
recoverable staging, but do not gain an owner to schedule cleanup automatically.

## Exact-key ownership

The adapter canonicalizes the relative destination key and uses its UTF-8
SHA-256 digest for this private staging path:

```text
RESULT_STORAGE_ROOT/.lao-ocr-writes-v1/<key-digest>.part
```

The directory is mode 0700. A stage is created exclusively at mode 0600 before
any document bytes are copied. The file is flushed and synchronized before its
atomic rename into the final destination. The returned size is observed while
the writer still owns the operation. Partial stage bytes are not exposed as a
completed artifact or downloadable result.

A pre-existing stage is not overwritten by a new writer. It remains an explicit
cleanup obligation. A failed write attempts removal of its own stage, checking
inode identity rather than deleting a replacement path. If metadata lookup or
removal fails, the descriptor is still closed and the unresolved file remains
recoverable through the recorded key. Private exception bodies are not included
in the helper's fixed errors or the managed publication response.

On deletion, the adapter checks both the exact stage and final path, removes
only those files, and reports success only after both operations succeed.
Missing files are idempotent success. Failure leaves the manager's existing
artifact-stage ownership and retry backoff intact. Removing one file before
failing on the other is safe to retry; it is not an exactly-once protocol.

## Writer and cleanup exclusion

All cooperating recoverable writers and cleaners use a fixed pool of 64
advisory lock slots under the same private namespace. A key digest selects its
slot. Each operation opens the stable slot file separately, validates its inode
and file type, and holds its exclusive lock through copying/renaming or cleanup.
Lock files contain no document bytes, are mode 0600, and are never unlinked by
normal cleanup. Removing a lock file could split ownership between old and new
inodes, so operators must not prune these files while the service is in use.

A writer waits for its slot. Cleanup makes a single nonblocking lock attempt;
a busy slot raises a fixed error before any stage or final-file deletion. The
manager retains the key and uses its existing retry schedule. Different keys
can hash to the same slot, causing conservative serialization or deferred
cleanup; this is not a claim of collision-free or unbounded parallelism. A slow
writer may delay another writer on its slot. No new server-side I/O or lock-wait
timeout is introduced.

The protocol was exercised across threads, separate adapter instances, and
separate processes. It uses Python's documented
[`fcntl.flock`](https://docs.python.org/3/library/fcntl.html#fcntl.flock)
interface and the local [flock open-description ownership contract](https://man7.org/linux/man-pages/man2/flock.2.html).
The recoverable mode requires POSIX locking support; unsupported platforms fail
explicitly rather than falling back to an unprotected cleanup path. Ordinary
non-durable construction does not import or require `fcntl`.

Use a private, service-owned local filesystem. This is not a validated NFS/SMB
locking protocol or an isolation boundary against an actor who can replace
paths in the service's private directories. Do not mix recoverable writers with
older binaries, external directory writers, or uncoordinated/default-mode
adapters against the same result root. Keep matching configuration through
recovery and do not fork while a write is active.

## Restart and compatibility

Process exits mid-copy, before rename, and after rename leave the same journal
key sufficient to identify either the stage, the final result, or both. Recovery
uses the persisted cleanup deadline, not the new process's retention setting.
A process exit between file deletion and the journal receipt can cause deletion
to repeat; missing files remain safe to delete again.

The storage namespace binding remains the canonical filesystem result root.
Existing v1 cleanup rows can therefore be recovered with the upgraded adapter;
this does not silently rebind the ledger to a different root. The journal still
owns cleanup only, not public status restoration, old download access, or OCR
replay. Existing job-manager download leases continue to defer cleanup for an
admitted response.

Both the journal and result root must persist on the intended storage volumes.
The staging directory must be on the same filesystem as final destinations for
atomic rename; nested mount points are not supported by this protocol. File
synchronization does not establish a power-loss, directory-metadata durability,
storage-device, snapshot-erasure, or secure-erasure guarantee.

## Safety boundaries and remaining work

Symlinked result components, linked/nonregular stage or lock files, and unsafe
final-file types are rejected. The new staging namespace cannot be used as an
artifact key. Cleanup does not glob, recurse, estimate age, or adopt arbitrary
files. It deliberately leaves legacy random `.tmp` files and unrelated staging
names untouched because their ownership has not been established.

Empty result-directory scaffolding is not recursively pruned in recoverable
mode: a different writer may be preparing those directories. The bounded lock
pool also remains in place. These are metadata remnants, not a claim that the
entire storage root becomes empty after deleting all known artifacts.

Legacy random temporary files, framework multipart spool files, incomplete S3
multipart uploads, unknown custom writes, object versions/backups, and provider
writes completing after recovery deletion still need separate reconciliation.
No live S3, container startup, optical benchmark, or OCR accuracy validation is
implied by these filesystem lifecycle tests.

## Verification

The three new staging test modules contain 43 cases. They cover exact-key
cleanup with no final result, preserved unrelated/legacy files, private staging
permissions, writer/cleaner exclusion, waiting writers, bounded stable lock
inodes, five abrupt process-exit boundaries, persisted failed-unlink retries,
download leases, old-journal compatibility, API configuration wiring, reserved
namespaces, and unsafe links/file types. A separate live-process test confirms
that another process cannot clean an active copy.

Self-review reproduced and fixed a descriptor leak when the first stage `fstat`
failed. The fix closes the handle without inventing a deletion receipt for a
file whose identity could not be checked. Initial failures, review evidence,
protected-input hashes, and gate logs are stored locally in the Git-ignored
`benchmarks/private/filesystem-staging-800e1b5/` directory.

## Completed local gates

All 43 focused cases and all 1,588 Python tests passed. The Python suite retained
six dependency deprecation warnings without failures. Ruff, web lint, all 145
web tests, and the production web build passed. All 13 protected-input hashes
matched the starting manifest, including Phetsarath, recognizer weights, capture
assets, and frozen evaluation inputs. These are lifecycle checks, not a
container-startup, live-provider, or OCR-quality result.
