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
- removes now-empty artifact directories after deletion

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

For an async job, the OCR worker:

1. processes input inside the transient job workspace
2. creates DOCX/Markdown/TXT/JSON
3. builds the result ZIP
4. uploads the ZIP to the configured result-storage backend
5. stores only the artifact reference in the job record
6. streams later downloads through the storage adapter
7. deletes the artifact when job retention expires

The S3 adapter uses boto3's managed file upload and chunked object reads.

## Retention cleanup

When a terminal job expires according to:

```text
JOB_RETENTION_SECONDS
```

the job manager asks the storage adapter to delete its stored artifact, then removes the transient workspace.

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
