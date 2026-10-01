# Result storage adapters

Async conversion jobs separate transient OCR workspaces from downloadable result storage.

The job/download API uses one storage protocol, so local filesystem and S3-compatible object storage share the same behavior.

## Supported backends

- `filesystem` — default, local-first
- `s3` — optional S3-compatible object storage

The default installation and Docker image do not require boto3.

## Filesystem backend

Configure:

```text
RESULT_STORAGE_BACKEND=filesystem
RESULT_STORAGE_ROOT=/data/results
```

Docker Compose mounts `/data/results` as a named volume:

```text
lao-ocr-results
```

so result ZIP files survive API container recreation.

Filesystem storage:

- rejects absolute keys
- rejects `..` traversal
- resolves keys under the configured root
- uses private directory/file permissions
- writes through a temporary file + atomic rename
- streams downloads in chunks
- removes now-empty artifact directories after deletion in default mode

Durable filesystem storage uses [recoverable staging](filesystem-staging-cleanup.md):
key-derived temporary files and final results share writer/cleanup locks. It
does not scan old random files or prune shared directory scaffolding. Stable
lock files remain in place, bounded to 64 slots.

## S3-compatible backend

Install the optional dependency:

```bash
pip install -e ".[s3]"
```

or build the API image with:

```bash
INSTALL_S3=true docker compose build api
```

Configure:

```text
RESULT_STORAGE_BACKEND=s3
RESULT_STORAGE_S3_BUCKET=lao-ocr-results
RESULT_STORAGE_S3_PREFIX=production
RESULT_STORAGE_S3_ENDPOINT_URL=
RESULT_STORAGE_S3_REGION=
RESULT_STORAGE_S3_FORCE_PATH_STYLE=false
```

Credentials use boto3's normal credential chain. Docker Compose passes through:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_SESSION_TOKEN
```

Do not commit credentials into this repository.

### Amazon S3 example

```bash
export INSTALL_S3=true
export RESULT_STORAGE_BACKEND=s3
export RESULT_STORAGE_S3_BUCKET=my-lao-ocr-results
export RESULT_STORAGE_S3_PREFIX=prod
export RESULT_STORAGE_S3_REGION=ap-southeast-1
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...

docker compose up --build
```

Leave `RESULT_STORAGE_S3_ENDPOINT_URL` empty for normal AWS S3.

### Cloudflare R2 example

R2 exposes an S3-compatible API.

```bash
export INSTALL_S3=true
export RESULT_STORAGE_BACKEND=s3
export RESULT_STORAGE_S3_BUCKET=my-bucket
export RESULT_STORAGE_S3_PREFIX=prod
export RESULT_STORAGE_S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
export RESULT_STORAGE_S3_REGION=auto
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...

docker compose up --build
```

Path-style addressing normally remains disabled unless your provider specifically requires it.

### MinIO example

```bash
export INSTALL_S3=true
export RESULT_STORAGE_BACKEND=s3
export RESULT_STORAGE_S3_BUCKET=lao-ocr
export RESULT_STORAGE_S3_PREFIX=results
export RESULT_STORAGE_S3_ENDPOINT_URL=http://minio:9000
export RESULT_STORAGE_S3_REGION=us-east-1
export RESULT_STORAGE_S3_FORCE_PATH_STYLE=true
export AWS_ACCESS_KEY_ID=minio-user
export AWS_SECRET_ACCESS_KEY=...

docker compose up --build
```

## Object keys

A typical key is:

```text
jobs/<job-id>/<document>-ocr.zip
```

When a prefix is configured:

```text
production/jobs/<job-id>/<document>-ocr.zip
```

The adapter rejects unsafe absolute/traversal keys before making an object-store request.

## Upload/download behavior

For an async job, the service:

1. processes input inside the transient job workspace
2. creates DOCX/Markdown/TXT/JSON
3. builds the result ZIP
4. uploads the ZIP to the configured result-storage backend
5. hands the returned artifact to the job manager before success/cancellation is decided
6. streams later downloads through the storage adapter
7. requests best-effort artifact deletion at cancellation or retention expiry

The S3 adapter uses boto3's managed file upload and chunked object reads.

## Retention cleanup

When a terminal job expires according to:

```text
JOB_RETENTION_SECONDS
```

the job manager expires public access and queues the stored artifact and transient
workspace for cleanup. Failed stages remain privately tracked for delayed,
bounded [process-local retries](job-expiry-cleanup.md). A successful object
deletion is not repeated solely because workspace removal failed. Optional
[durable expired cleanup](durable-expired-cleanup.md) journals those expired tasks;
expired status remains separate from physical deletion.
The [API idle worker](job-idle-cleanup.md) invokes the same bounded cleanup
pass on a configurable cadence, so an idle process no longer requires another
request to start due deletion work.

`JOB_MAX_RETAINED` (default 1024) bounds live/terminal records plus pending-cleanup
tasks. Reaching the limit rejects new reservations instead of discarding deletion
references. It is a record-count limit, not a storage-byte quota.

Cancellation also requests immediate deletion of a successfully returned stored
result when cancellation wins before job success. Expiry cannot race that
in-progress cancellation deletion. A failed immediate delete retains the private
reference for the existing expiry pass. See [job artifact ownership](job-artifact-ownership.md).

The default mode is process-local and loses unfinished cleanup on restart.
With `JOB_CLEANUP_DURABLE=true`, expired tasks and tracked API publication keys
survive through the bound journal. [Publication ownership](result-publication-ownership.md)
is committed before storage is invoked, covering lost replies for that declared
key. [Workspace ownership](workspace-ownership.md) begins before allocation and
continues through publication and expiry. A custom runner's returned key is
recorded at terminal handoff, but direct custom writes that lose their reply,
legacy temporary/multipart remnants, and provider writes completing after recovery
deletion still need independent reconciliation and retention controls.
Local OCR workspaces are still retained until normal expiry, not erased as soon
as cancellation is requested.

## Active archive transfers

[Process-local download leases](job-download-leases.md) now protect admitted
archive responses from this manager's expiry deletion. Public expiry still
hides the job; private artifact/workspace cleanup waits until all admitted
responses close. Stored-response iterators are explicitly closed before their
lease is released, including interrupted S3-adapter responses. External/provider
lifecycle deletion and process crashes are not prevented. A stalled transfer can
hold data beyond public retention; this is not proof of erasure or receipt.

## Status readiness

Public job responses check result existence outside the shared job-manager lock
and revalidate the record/result afterward. A delayed provider check therefore
does not serialize unrelated job transitions. This is advisory readiness, not
a lease or a provider timeout; actual downloads still acquire their own lease
and check storage. See [job-status readiness](job-status-readiness.md).

## Optional durable expired cleanup

The opt-in journal binds cleanup to the canonical job root and result-storage
namespace. S3 binding uses the effective endpoint/region plus bucket and prefix;
filesystem binding uses the result root. Changed namespaces, malformed ledgers,
and a second live owner refuse startup rather than redirecting deletions. Only
keys in a generated job's direct object namespace can be replayed. No bucket
scan, public-job reconstruction, or automatic storage migration is performed.

See [durable expired cleanup](durable-expired-cleanup.md) for persistent-volume
setup, failed-commit behavior, and shutdown/download leases. See
[result publication](result-publication-ownership.md) for the pre-write helper,
retained deadlines, custom-runner contract, and remaining lifecycle gaps.

## Health/privacy

The public health response reports only the storage backend name.

It does not expose:

- access keys
- secret keys
- bucket credentials
- filesystem storage path
- individual object keys

The adapter's internal metadata may contain the configured bucket/endpoint for diagnostics, but credentials are never part of adapter metadata.

## Important current limitation

Job metadata is still stored in the API process memory.

Object storage preserves ZIP bytes across API restarts, but a restarted API process does not yet reconstruct old job IDs from stored objects.

Persistent job metadata belongs with a future external queue/database adapter.

## Security recommendations

For S3-compatible deployments:

- use a dedicated bucket
- grant only required object permissions
- scope credentials to the result prefix where possible
- enable provider-side encryption at rest
- use HTTPS endpoints outside trusted local networks
- apply lifecycle deletion as a second retention layer
- do not make the bucket public
- rotate credentials normally

The open-source default stays filesystem/local-first; object storage is optional.
