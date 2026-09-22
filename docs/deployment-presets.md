# Deployment presets

The project stays local-first, but the repository includes deployment examples for common modes.

These presets are intentionally simple Compose overlays and environment examples. They do not contain real credentials.

## Files

```text
deploy/
├── README.md
├── compose.public.yml
├── compose.s3.yml
└── presets/
    ├── local.env.example
    ├── public-single-node.env.example
    └── s3.env.example
```

## Local preset

The normal local setup remains:

```bash
docker compose   --env-file deploy/presets/local.env.example   up --build
```

Defaults:

- API: `0.0.0.0:8000`
- web: `0.0.0.0:5173`
- filesystem result storage
- no submission rate limit
- two OCR workers
- eight active jobs

This is intended for development or trusted local networks.

## Hardened single-node public preset

Use the public environment preset plus the public Compose override:

```bash
docker compose   --env-file deploy/presets/public-single-node.env.example   -f docker-compose.yml   -f deploy/compose.public.yml   up --build -d
```

The example:

- binds API/web only to `127.0.0.1`
- expects a reverse proxy in front
- enables a modest submission rate limit
- makes the API root filesystem read-only
- gives API temporary processing a bounded `/tmp` tmpfs
- drops API Linux capabilities
- enables `no-new-privileges`
- applies CPU/memory/PID limits
- uses restart policies

Replace the example domains:

```text
https://ocr.example.com
https://api.ocr.example.com
```

before use.

## S3-compatible preset

Use the S3 environment plus both overrides:

```bash
docker compose   --env-file deploy/presets/s3.env.example   -f docker-compose.yml   -f deploy/compose.public.yml   -f deploy/compose.s3.yml   up --build -d
```

The S3 override:

- builds the API with `INSTALL_S3=true`
- installs the optional `boto3` dependency
- selects `RESULT_STORAGE_BACKEND=s3`

Configure:

```text
RESULT_STORAGE_S3_BUCKET
RESULT_STORAGE_S3_PREFIX
RESULT_STORAGE_S3_ENDPOINT_URL
RESULT_STORAGE_S3_REGION
RESULT_STORAGE_S3_FORCE_PATH_STYLE
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_SESSION_TOKEN
```

See [storage-adapters.md](storage-adapters.md) for AWS S3, Cloudflare R2, and MinIO examples.

## Bind addresses

The base Compose supports:

```text
API_BIND_ADDRESS
API_PORT
WEB_BIND_ADDRESS
WEB_PORT
VITE_API_URL
```

This lets the same base Compose serve local use or sit behind a reverse proxy without editing YAML.

For public deployments, prefer loopback bindings and terminate TLS at a reverse proxy or ingress layer.

## Secrets

Do not commit a copied preset containing real credentials.

Recommended approaches:

- host environment variables
- Docker/Compose secrets
- your deployment platform's secret manager
- cloud workload identity / instance role where supported

The checked-in `.env.example` files are documentation only.

## Reverse proxy

A public reverse proxy should provide:

- TLS
- request/body size limits
- proxy read/write timeouts
- request IDs
- optional external rate limiting
- access logs
- HSTS after HTTPS is confirmed

When enabling:

```text
RATE_LIMIT_TRUST_PROXY_HEADERS=true
```

make sure the proxy overwrites untrusted `X-Forwarded-For`.

## Resource tuning

The public Compose overlay is only a conservative starting point.

Tune:

- `JOB_MAX_WORKERS`
- `JOB_MAX_ACTIVE`
- `MAX_UPLOAD_BYTES`
- `MAX_PAGES`
- `MAX_PAGE_PIXELS`
- container CPU/memory/PID limits

according to host size and OCR model choice.

The project-owned PyTorch recognizer requires a different resource profile from Tesseract.

## Multi-node warning

The current job metadata and rate limiter are process-local.

Do not horizontally scale the API behind a load balancer without replacing:

- in-memory job metadata
- in-process submission limiter

with shared equivalents.

S3 result storage alone does not make the job queue distributed.

## GPU owned-recognizer preset

For NVIDIA inference with the project-owned recognizer:

```bash
docker compose \
  --env-file deploy/presets/gpu.env.example \
  -f docker-compose.yml \
  -f deploy/compose.public.yml \
  -f deploy/compose.gpu.yml \
  up --build -d
```

The preset uses `Dockerfile.owned-api`, mounts `./models` read-only, requests Docker GPU access, and defaults to one OCR worker. See [gpu-worker.md](gpu-worker.md).
