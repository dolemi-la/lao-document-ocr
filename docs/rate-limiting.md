# Submission rate limiting

The API includes an optional in-process submission rate limiter for public deployments.

It is **disabled by default** so local/self-hosted users are not rate limited.

## Scope

The limiter applies only to OCR submission endpoints:

- `POST /v1/parse`
- `POST /v1/convert`
- `POST /v1/jobs`

It does not rate-limit:

- job status polling
- result downloads
- health checks
- metrics scraping

This avoids penalizing clients for polling a job they already submitted.

## Enable it

Example:

```bash
RATE_LIMIT_REQUESTS=10
RATE_LIMIT_WINDOW_SECONDS=60
docker compose up
```

This allows up to 10 OCR submissions per client in a rolling 60-second window.

Configuration:

```text
RATE_LIMIT_REQUESTS=0
RATE_LIMIT_WINDOW_SECONDS=60
RATE_LIMIT_MAX_CLIENTS=10000
RATE_LIMIT_TRUST_PROXY_HEADERS=false
```

`RATE_LIMIT_REQUESTS=0` disables the limiter.

## Responses

When the limit is exceeded:

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 17
```

The body contains:

```json
{
  "detail": "OCR submission rate limit exceeded."
}
```

The normal observability middleware records the resulting HTTP 429.

## Client identity

By default, the limiter uses the socket peer address from the ASGI server.

It deliberately ignores `X-Forwarded-For` unless:

```text
RATE_LIMIT_TRUST_PROXY_HEADERS=true
```

Do **not** enable this flag unless a trusted reverse proxy overwrites/removes untrusted forwarded headers.

Otherwise a public client could rotate/spoof `X-Forwarded-For` to bypass the limiter.

## Bounded memory

The limiter tracks a bounded number of client keys.

```text
RATE_LIMIT_MAX_CLIENTS=10000
```

When the key limit is reached:

1. expired/stale client windows are pruned
2. if capacity is still full, a new client is temporarily rejected

This avoids unbounded in-process key growth.

## Multi-process / multi-host deployments

The built-in limiter is process-local.

For:

- multiple Uvicorn workers
- multiple containers
- multiple hosts
- public high-volume deployments

prefer a shared limiter at the reverse proxy/API gateway or a future shared backend.

Examples include:

- Nginx/HAProxy rate limiting
- Cloudflare rate limiting
- an API gateway
- Redis-backed distributed limiter

The in-process limiter remains useful as a simple defensive baseline and for single-process deployments.
