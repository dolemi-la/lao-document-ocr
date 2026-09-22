# Result storage adapters

Async conversion jobs separate transient OCR workspaces from downloadable result storage.

The storage abstraction makes it possible to move ZIP results to object storage later without rewriting the job API.

## Current backend

The currently supported production backend is:

```text
filesystem
```

Configure it with:

```text
RESULT_STORAGE_BACKEND=filesystem
RESULT_STORAGE_ROOT=/data/results
```

Docker Compose mounts `/data/results` as a named volume:

```text
lao-ocr-results
```

so result ZIP files survive API container recreation.

## What is stored

For an async job, the OCR worker:

1. processes the input inside the transient job workspace
2. builds the ZIP containing DOCX/Markdown/TXT/JSON
3. atomically copies the ZIP into result storage
4. returns a storage artifact reference to the job manager
5. streams later downloads through the storage adapter

A typical filesystem key is:

```text
jobs/<job-id>/<document>-ocr.zip
```

## Retention cleanup

When a terminal job expires according to:

```text
JOB_RETENTION_SECONDS
```

the job manager asks the storage adapter to delete its stored artifact, then removes the transient workspace.

Result storage therefore follows the same retention window as job metadata.

## Safety

Filesystem storage:

- rejects absolute keys
- rejects `..` traversal
- resolves keys under the configured root
- writes through a temporary file and atomic rename
- streams downloads in chunks
- removes now-empty artifact directories after deletion

The public health response reports only the backend name, not the filesystem path.

## Important current limitation

Job metadata is still stored in the API process memory.

A named volume preserves the ZIP bytes across container recreation, but a restarted API process does not yet reconstruct old job IDs from stored artifacts.

Persistent job metadata belongs with a future external queue/database adapter.

## Future object storage

The storage protocol is designed around:

- put file
- existence check
- chunked read
- delete
- metadata

A future S3-compatible implementation can use the same job/download API.

Potential targets:

- Amazon S3
- Cloudflare R2
- MinIO
- DigitalOcean Spaces

The open-source default stays filesystem/local-first; object storage should remain optional.
