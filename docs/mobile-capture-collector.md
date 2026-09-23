# Mobile capture collector

`serve-capture-kit` turns a collector-safe capture-kit ZIP into a small local web app for phone/scanner collection.

It uses only the blind collector kit. Ground truth and internal dataset paths are never served.

## Start locally

```bash
lao-ocr serve-capture-kit \
  --kit benchmarks/capture-packs/project-authored-lao-v1.collector.zip \
  --output-dir /path/to/captures/phone-a \
  --capture-id phone-a \
  --mode phone-photo
```

Default bind:

```text
127.0.0.1:8090
```

The CLI prints a random collector access token and a one-click URL such as:

```text
Collector access token: <random-token>
Local collector: http://127.0.0.1:8090/?token=<random-token>
```

Open the printed token URL. After the first successful request, the server stores the token in an HttpOnly, SameSite=Strict session cookie and the browser removes the token query from the visible URL.

## Use a phone on the same trusted LAN

Bind to all local interfaces:

```bash
lao-ocr serve-capture-kit \
  --kit benchmarks/capture-packs/project-authored-lao-v1.collector.zip \
  --output-dir /path/to/captures/phone-a \
  --capture-id phone-a \
  --mode phone-photo \
  --host 0.0.0.0 \
  --port 8090
```

The CLI still prints `0.0.0.0` because that is the bind address. Replace it with the laptop's LAN address while keeping the generated token, for example:

```text
http://192.168.1.25:8090/?token=<random-token>
```

The session token blocks unauthenticated page/API/PDF access from other LAN clients. The collector still uses plain HTTP by default, so use it only on a trusted local network and do not expose it directly to the public Internet. For stronger network security, put it behind a trusted TLS reverse proxy/VPN.

## Collector access token

`serve-capture-kit` generates a URL-safe random token by default. You may provide your own 16-256 character URL-safe token:

```bash
--access-token collector-session-2026-secret
```

Accepted token transports:

- initial `?token=...` query for browser bootstrap
- HttpOnly `lao_ocr_collector` session cookie after bootstrap
- `X-Collector-Token` header for scripted clients

Requests without the configured token receive HTTP 403. The collector does not include ground truth in the served kit regardless of authentication.

## Collector behavior

The mobile page provides:

- collection progress
- page number/template/page ID selection
- phone camera/file input using `capture="environment"`
- upload/save
- delete + retake
- printable PDF access

Saved filenames are exact page IDs, for example:

```text
project-authored-lao-v1-two-column-p0001.png
```

That makes the output directory directly compatible with `register-capture-directory`.

## Integrity checks

Before serving, the collector verifies:

- safe ZIP member paths
- bounded ZIP size/member count
- complete `SHA256SUMS` coverage
- SHA-256 of every collector-kit member
- capture-kit schema
- blind-collection exclusion flags
- worksheet schema/page IDs/capture modes
- worksheet page count against kit metadata

Before saving an upload, it verifies:

- page ID belongs to the selected collection mode
- supported image extension
- image content matches the extension
- upload byte limit
- decoded pixel limit
- valid image structure
- printed page-ID QR marker
- QR page ID matches the page selected in the UI

If another upload already exists for the same page, the new upload is rejected until the existing capture is deleted/retaken.

## QR behavior

QR matching is required by default for modern capture suites.

For a legacy kit whose pages do not contain the printed QR marker:

```bash
--allow-unreadable-qr
```

This only permits a capture where no QR can be decoded. If a QR is decoded and points to a different page ID, the capture is always rejected.

## Private local files

The session output directory is forced to mode `0700` and saved captures/session metadata to `0600` on supported POSIX systems.

The server also writes:

```text
.collector-session.json
```

with non-answer provenance only:

- suite ID
- kit SHA-256
- kit source revision
- capture ID
- capture mode
- whether QR is required
- expected page count

No ground-truth text is written into this session file.

## After collection

Dry-run the existing bulk importer:

```bash
lao-ocr register-capture-directory \
  --suite-manifest benchmarks/capture-packs/project-authored-lao-v1/capture-suite.json \
  --capture-dir /path/to/captures/phone-a \
  --capture-id phone-a \
  --mode phone-photo \
  --contributor "Contributor Alias" \
  --release-license CC0-1.0 \
  --dataset-root benchmarks/public \
  --dataset-manifest benchmarks/public/manifest.jsonl \
  --require-complete \
  --dry-run \
  --report benchmarks/results/phone-a-import-plan.json
```

Then continue with optical-evidence checks, registration, manual review, readiness checks, and benchmark freeze.
