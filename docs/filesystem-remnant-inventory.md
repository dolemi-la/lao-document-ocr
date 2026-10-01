# Read-only filesystem remnant inventory

Starting checkout: `1a4ab2634ed8d49b1f3c2616a97a4fbb276c2a64`.

## Purpose and boundary

The standalone inventory observes names and filesystem metadata below one
explicitly selected result root. It provides bounded evidence about legacy
random temporary-file candidates, recoverable staging candidates, coordination
lock candidates, and other entries. It does not decide that any file is
abandoned, owned by this application, or safe to delete.

This is not a cleanup command, journal importer, migration, background task,
API endpoint, OCR run, or storage-provider scan. It never invokes the job manager,
constructs a storage adapter, reads document contents, opens the cleanup journal,
acquires writer locks, or creates/removes/renames/chmods inventory targets.
Normal operating-system access-time accounting may still change when directory
metadata is read; read-only does not mean a frozen or forensic disk image.

The previous [recoverable staging protocol](filesystem-staging-cleanup.md)
continues to clean exactly known stages through journal ownership. This inventory
does not change that protocol or adopt legacy temporary files for deletion.
All runtime cleanup, admission, storage, and OCR settings remain unchanged.

## Run explicitly

From the repository checkout, select the intended existing filesystem result
root, not an entire home directory or unrelated workspace:

```bash
.venv/bin/python -B -m services.api.app.filesystem_inventory \
  --root "${RESULT_STORAGE_ROOT:?Set the intended filesystem result root}" \
  --max-entries 10000 \
  --max-depth 8 \
  --max-items 200 \
  --max-seconds 5
```

The command requires `--root`; it does not infer a target from service defaults
or instantiate the API. A missing/unavailable root is an error, never an empty
report or an instruction to create it. The `-B` flag also prevents Python from
writing import bytecode. This entry point runs from the source/service tree;
it is not a new subcommand of the packaged `lao-ocr` executable.

A JSON report is written to standard output. There is no output-file, delete,
repair, confirm, or cleanup option. To retain a report, redirect stdout to a
private location **outside** the root being scanned. Shell redirection into the
root would itself create/change a file before inventory begins.

By default, report entries identify relative paths using their SHA-256 digest,
not a filename or the absolute root path. Add `--include-paths` only for an
operator-private report that needs relative filenames. JSON escapes control
characters and unsupported filename bytes; no raw terminal-control names are
printed as a free-form listing. Paths exceeding the internal byte bound remain
omitted even in this mode.

Hashes are not anonymization: predictable names can be guessed, identical
relative names correlate across reports, and size/time metadata is sensitive.
No root identity is asserted by a path digest. Record the selected root and
invocation privately when retaining evidence, and do not publish private reports.

## Classification is not ownership

Only singly linked regular files are assigned the following naming categories:

| Category | Observed naming pattern |
| --- | --- |
| `legacy_temporary_candidate` | A basename matching `.<name>.<32 lowercase hexadecimal characters>.tmp`. |
| `recoverable_stage_candidate` | A direct child of `.lao-ocr-writes-v1` matching `<64 lowercase hexadecimal characters>.part`. |
| `coordination_lock_candidate` | A direct child of `.lao-ocr-writes-v1` named `lock-00` through `lock-3f`. |

Other categories are directories, other regular files, hardlinked files,
symlinks, special files, and entries whose metadata could not be read. Near
matches, staging-like names outside the reserved directory, and filenames with
unexpected suffixes are not promoted to genuine staging ownership. A name match
alone cannot authenticate a writer or establish that copying has stopped.

Symlink targets are neither read nor followed. Hardlinked regular files are
reported separately, not treated as independent temporary-file candidates.
Sockets, FIFOs, and device-like entries are inspected through metadata only,
never opened as data streams. Existing writer-lock files are not opened or
probed for availability: the report cannot tell whether a writer is active.

`mtime_ns` is merely observed metadata, not an age threshold or an abandonment
signal. No relationship to public jobs, storage keys, publication success,
source licenses, document contents, or deletion authorization is inferred.

## Bounds and completeness

| Limit | Default | Accepted range |
| --- | --- | --- |
| Entries visited | 10,000 | 1–100,000 |
| Detail records retained | 200 | 0–2,000 |
| Metadata depth | 8 | 1–64 |
| Cooperative elapsed budget | 5 seconds | Greater than 0, at most 60 seconds |

Root contents are depth 1. A directory at the depth boundary is observed but
not entered; even an apparently empty unvisited subtree makes the traversal
partial. Relative paths have an additional fixed 8,192-byte limit. Validation
occurs before opening the target, and invalid limits are rejected rather than
silently clamped.

Traversal uses incremental filesystem enumeration, without loading/sorting a
whole directory first. The entry cap includes directories, symlinks, special
files, and failed metadata lookups, not only candidate files. Hitting the exact
cap conservatively reports a partial traversal without reading one extra entry
just to prove there was nothing more. Directory descriptors and iterators are
closed on failure and on normal unwinding; traversal depth bounds open ancestors.

The elapsed budget is checked cooperatively between operations. It cannot
interrupt a blocked filesystem call or guarantee a hard wall-clock deadline.
A read-only inventory can still consume storage I/O and time on a slow volume.

The JSON schema is `filesystem-remnant-inventory/v1`:

- `traversal_complete` and `status` describe whether the intended traversal
  finished without a limit, inaccessible entry, race, or skipped device boundary.
  `issues` gives fixed reason codes/counts instead of private exception messages.
- `categories` contains counts and observed regular-file logical bytes, including
  observations beyond the detail-record cap. Partial traversal produces partial
  totals; zero candidates in a partial report does not establish an empty root.
- `details_complete` and `items_omitted` separately describe detail sampling.
  Setting `--max-items 0` retains summary traversal but emits no individual items.
  Details are selected in filesystem order, then sorted by digest for display;
  the selected sample is neither stable nor representative across changing runs.

`logical_bytes` is not physical allocation, a disk quota, or reclaimable space.
Hardlinked names can count the same underlying bytes more than once. Sparse
files, clones, snapshots, and unrelated provider accounting are not resolved.
The root directory itself is not included in `entries_seen`.

Exit codes are 0 for complete traversal, 3 for partial traversal, and 2 for a
fixed-code input/platform/root error. Detail truncation alone does not change a
complete traversal's exit code. Command-line parse failures are also JSON and
do not echo untrusted arguments or paths. Help exits successfully without a root.

Every successful/partial report explicitly sets `snapshot`, `ownership_verified`,
and `deletion_authorized` to false. A complete report is not a consistent snapshot:
files can appear, disappear, change, or move during enumeration. Neither complete
status nor a candidate detail is an executable deletion manifest.

## Directory safety and platform scope

The implementation uses the documented Python
[directory-descriptor interfaces](https://docs.python.org/3.11/library/os.html#files-and-directories)
and [descriptor-based scandir](https://docs.python.org/3.11/library/os.html#os.scandir).
It requires POSIX no-follow directory-open support and refuses unsupported
platforms rather than switching to an unsafe recursive fallback.

The selected root itself must not be a symlink, including when supplied with a
trailing slash. Its operator-selected ancestor path uses ordinary OS resolution
(for example, a platform's temporary-directory alias). Below that opened root,
child directories are opened relative to the parent descriptor without following
links, then checked against the previously observed device/inode before descent.
A replaced directory is skipped and makes the report partial.

Directories on a different device are skipped and reported. Same-device bind
mounts are not universally detectable; repeated directory identities are skipped
and depth/entry limits still apply. This is not a mount-topology audit or an
isolation guarantee against an actor able to move the service's private paths.
Use an operator-controlled filesystem root. No locks are taken to make the tree
stationary, and an active-writer test confirms inventory neither waits for nor
releases the writer's file lock.

## Verification and remaining work

The three inventory test modules cover naming/type classification, default and
explicit-path privacy, invalid roots/options, unchanged file bytes/modes/names,
entry/depth/detail/time bounds, metadata/iterator failures, symlink/replacement
races, skipped devices and repeated identities, descriptor closure, active-writer
lock preservation, and standalone command exit codes/import isolation.

Non-UTF-8 filename serialization uses a directory-entry double because some
host filesystems refuse to create those names. Race, permission, elapsed, and
device-boundary cases inject controlled events/failures; they are not claims of
validated production mounts or live failure behavior on every filesystem.

Only authored fixtures were inventoried in this slice. No production result root,
user document collection, object-store account, or old private temporary files
were scanned. The plan, initial expected failure, protected hashes, review notes,
and gate logs are retained under the Git-ignored directory
`benchmarks/private/filesystem-inventory-1a4ab26/`.

Legacy deletion still requires a separately reviewed policy and ownership/writer
coordination. Framework spool files, incomplete S3 multipart uploads, old unknown
writes, and late provider writes remain open reconciliation work. No automatic
cleanup action should be derived solely from this inventory.

## Completed local gates

All 63 focused inventory cases and all 1,651 Python tests passed. Ruff, web
lint, all 145 web tests, the production web build, `git diff --check`, and local
document-link checks passed. The Python suite retained six existing dependency
deprecation warnings with no failures.

All 16 protected-input hashes matched the starting manifest. These include the
13 protected font/model/capture/evaluation and related inputs from the previous
slice, plus the job manager, storage adapter, and recoverable-write implementation.
No production cleanup behavior or OCR-quality result is implied by this inventory.
