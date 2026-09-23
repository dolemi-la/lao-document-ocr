# Security hardening

This document records the baseline security controls built into Lao Document OCR.

The project is local-first and can be run without authentication, billing, or a cloud service. Public deployments still need normal infrastructure controls around the application.

## Upload validation

The API does not trust a filename extension by itself. After upload, before OCR is queued or executed, the server validates the actual content:

- PDF files must start with the PDF signature and open successfully with PyMuPDF
- encrypted PDFs are rejected
- PDF page count is checked before OCR
- PDF rendered page pixel count is checked before rasterization
- image files must decode as the expected PNG/JPEG/TIFF/WebP format
- image width × height is checked before full conversion/OCR
- empty uploads are rejected
- mismatched extension/content returns HTTP 422

Relevant limits:

```text
MAX_UPLOAD_BYTES=26214400
MAX_PAGES=60
MAX_PAGE_PIXELS=40000000
```

The pixel cap is intentionally independent from the compressed upload byte limit. A small compressed image can still expand into a very large raster.

## Embedded PDF images

Native PDF images are inspected before extraction. Images whose declared source dimensions exceed the page pixel cap are skipped instead of being decoded and re-embedded.

## Filenames

Client filenames are treated as display metadata only. The sanitizer removes path components, handles both slash styles, removes control characters, replaces unsafe punctuation, preserves Lao Unicode letters/combining marks, bounds stem length, and normalizes the extension.

Server-side input files use generated workspace paths rather than client paths. Download responses use an ASCII fallback filename plus UTF-8 `filename*=` encoding.

## File permissions

Async job workspaces are created with mode `0700`. Uploaded files and stored result artifacts use mode `0600`. Filesystem result storage writes through a temporary file followed by atomic rename. Temporary synchronous work uses Python private temporary directories.

## Error exposure

Expected document validation/processing failures may be returned to clients with deliberately safe messages. Unexpected job exceptions are logged server-side and exposed to API clients only as:

```text
Internal conversion error.
```

Parser errors are normalized so temporary/local filesystem paths are not included in normal validation responses.

## API response headers

Every FastAPI response includes:

```text
X-Content-Type-Options: nosniff
Referrer-Policy: no-referrer
X-Frame-Options: DENY
Permissions-Policy: camera=(), microphone=(), geolocation=()
Content-Security-Policy: default-src 'none'; frame-ancestors 'none'; base-uri 'none'
Cache-Control: no-store
X-Request-ID: ...
```

`/docs` and `/redoc` use a narrowly scoped documentation CSP that allows only the specific CDN/font/image origins required by FastAPI's built-in Swagger/ReDoc pages, while retaining `frame-ancestors 'none'`, `base-uri 'none'`, and same-origin API connections. Normal API responses keep the stricter `default-src 'none'` policy.

The web Nginx server independently emits appropriate anti-sniff, referrer, frame, permissions, and CSP headers for the static UI.

## Container privilege

The API Docker image runs as an unprivileged user:

```text
uid=10001
gid=10001
```

Only explicitly prepared result/temp directories are writable by that user. Tesseract and application packages are installed while building the image, before dropping privileges.

## Queue/resource controls

Public deployments can combine upload size limits, page count limits, page pixel limits, bounded async job capacity, bounded worker count, bounded batch size, per-client submission rate limiting, result retention cleanup, and bounded rate-limit client-key memory.

These are application-level safeguards, not a substitute for host/container resource limits.

## Proxy trust

`X-Forwarded-For` is ignored by the built-in rate limiter unless:

```text
RATE_LIMIT_TRUST_PROXY_HEADERS=true
```

Only enable that setting behind a trusted reverse proxy which overwrites untrusted forwarded headers.

## Web Content Security Policy

The static web UI CSP is designed for the current Vite bundle: scripts/styles from self, API connections over HTTP/HTTPS, images from self/blob/data, no plugins/objects, no framing, and self-only base URI.

When adding new third-party assets/services, update the policy deliberately rather than weakening it globally.

## Known limits / future review

This baseline does not replace an external security assessment. Before exposing a high-volume public service, also consider reverse-proxy request/body/time limits, container CPU/memory/pid limits, network egress restrictions, TLS/HSTS at the edge, dependency/SBOM scanning, malware/content scanning if arbitrary public uploads are retained, authentication/authorization for private per-user documents, external queue/object-storage permissions, penetration testing, and abuse monitoring.

The open-source local mode intentionally remains usable without account infrastructure.

## Automated dependency scanning

CI includes a dedicated dependency-security job.

Python runtime plus optional S3 dependencies are extracted directly from `pyproject.toml` and audited with `pip-audit --strict`. The local project package itself is not treated as a PyPI dependency.

Web production dependencies are audited with:

```bash
pnpm audit --prod --audit-level high
```

GitHub Dependabot is configured weekly for:

- Python / `pyproject.toml`
- pnpm / npm ecosystem under `apps/web`
- Docker dependencies
- GitHub Actions

A dependency-audit failure should be fixed or explicitly reviewed before merge rather than silently disabled.
