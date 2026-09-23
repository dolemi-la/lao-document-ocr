# External security review package

The project includes automated hardening and a repeatable container security smoke, but maintainers should **not** mark an external penetration/security review complete based on self-review alone.

This document defines a ready-to-hand-off review scope.

## Reproducible baseline

Run:

```bash
./scripts/security_smoke.sh
```

The script creates an isolated Compose project with dedicated ports and volumes, combines:

- `docker-compose.yml`
- `deploy/compose.public.yml`

and verifies:

- API runs as uid 10001, not root
- writable API directories are mode 0700
- API response security headers
- request ID propagation
- Swagger/ReDoc-compatible CSP
- static web security headers
- content/extension mismatch rejection
- submission rate limiting + `Retry-After`
- polling remains exempt from submission limits
- Prometheus route normalization (no job-ID cardinality)
- read-only API root filesystem
- no-new-privileges
- PID limit

It uses a project-specific Compose name and removes only its own temporary volume/resources.

## Suggested external review target

Review the public single-node deployment:

```bash
docker compose   -f docker-compose.yml   -f deploy/compose.public.yml   up -d
```

Optional variants to include:

- S3 result storage: `deploy/compose.s3.yml`
- project-owned/GPU runtime: `deploy/compose.gpu.yml`

## In-scope HTTP surfaces

Public API:

```text
GET    /health
GET    /metrics
POST   /v1/parse
POST   /v1/convert
POST   /v1/jobs
POST   /v1/jobs/batch
GET    /v1/jobs/{job_id}
DELETE /v1/jobs/{job_id}
GET    /v1/jobs/{job_id}/download
GET    /docs
GET    /redoc
GET    /openapi.json
```

Static web UI:

```text
/
static Vite assets
```

## Priority review themes

### File upload/parser abuse

Test extension/content mismatch, malformed PDFs/images, encrypted PDFs, page-count limits, decompression/raster bombs, oversized dimensions, embedded PDF images, corrupt image metadata, Unicode/control-character filenames, archive/result filename handling, and parser error leakage.

Expected application limits include:

```text
MAX_UPLOAD_BYTES
MAX_PAGES
MAX_PAGE_PIXELS
BATCH_MAX_FILES
```

### Job/queue abuse

Test queue saturation, batch atomicity, cancellation races, repeated polling, retention cleanup, result deletion/races, job ID guessing/enumeration, worker-thread errors, and resource exhaustion across many documents.

### Rate-limit/proxy trust

Test client isolation, `Retry-After`, spoofed `X-Forwarded-For`, trusted-proxy mode, bounded key tracking, and status/download exemption.

### Result storage

Filesystem review should cover path traversal, symlink/rename edge cases, file permissions, and cleanup.

S3-compatible review should cover key traversal/prefix handling, endpoint/TLS configuration, credential scoping, deletion/retention, and bucket policy.

### Container/runtime

Verify non-root user, dropped capabilities, read-only root filesystem, writable mount boundaries, no-new-privileges, CPU/memory/PID constraints, read-only model mounts, and that secrets are not baked into images.

### Browser/web

Review CSP, framing, MIME sniffing, referrer policy, CORS, API URL configuration, malicious filenames/status errors reflected in UI, and dependency/static-asset exposure.

### Privacy

No OCR document text, filenames, job IDs, or user data should appear in Prometheus labels.

Review whether deployment logs contain more document metadata than intended.

## Out of scope / deployment-dependent

The application does not itself terminate TLS.

The deploying organization should separately review:

- TLS/HSTS
- reverse-proxy body/time limits
- WAF/CDN policy
- DNS
- host/kernel hardening
- cloud IAM
- S3 bucket/IAM policies
- backup/snapshot retention
- vulnerability management
- incident response

## Review deliverable

A useful external report should include:

- tested commit SHA
- tested deployment preset
- environment/tooling
- findings with severity and reproduction steps
- affected endpoint/component
- remediation recommendation
- retest status
- explicit limitations

Do not replace findings with one aggregate “security score.”

## Project status

The roadmap item **external deployment penetration/security review** stays open until an independent reviewer has tested a specific release/deployment and findings have been triaged/retested.
