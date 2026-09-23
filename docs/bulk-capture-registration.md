# Bulk capture registration

Use this workflow after scanning or photographing multiple pages from a capture suite.

It avoids running `register-capture` once per page.

## Page identification

Current capture suites print a `page-id:qr-v1` QR marker in the top-right header. Bulk registration reads that marker directly, so raw camera/scanner filenames are fine:

```text
captures/phone-a/
├── IMG_1842.jpg
├── IMG_1843.jpg
├── ...
└── IMG_1901.jpg
```

If you rename files to exact page IDs, that is also supported. When both a page-ID filename and QR are present, they must agree; a mismatch is rejected.

Legacy capture suites without QR markers still require exact page-ID filenames.

Supported image extensions:

- `.png`
- `.jpg`
- `.jpeg`
- `.tif`
- `.tiff`
- `.webp`

Non-image files such as `.DS_Store` are ignored and reported.

Images whose QR cannot be decoded and whose filename is not a known legacy page ID are rejected instead of being silently skipped. QR-enabled suites require a readable QR marker during bulk import. Hard/degraded captures whose marker is unreadable should be reviewed and registered individually rather than guessed.

## Dry-run first

For a complete 60-page phone run:

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

The dry-run performs preflight checks but does not modify the dataset.

It checks:

- suite page IDs decoded from QR/legacy filenames
- duplicate page mappings
- filename ↔ QR agreement when both identify a page
- missing pages when `--require-complete` is used
- existing sample-ID collisions
- destination collisions
- visible page content
- near-identical digital-copy rejection

## Register the batch

After the dry-run is clean:

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
  --confirm-release \
  --report benchmarks/results/phone-a-import.json
```

For a flatbed run, change:

```text
--capture-id flatbed-a
--mode flatbed-scan
```

## Partial capture sessions

If only some pages were captured, omit `--require-complete`.

The report records the missing page IDs. Later sessions can use another capture ID or fill the remaining pages after checking for existing sample IDs.

## Atomicity

All files are preflighted before mutation.

If a later registration unexpectedly fails, the batch importer restores the original dataset manifest and removes files written by that batch.

Original phone/scan files in the capture directory are never modified.

## Capture IDs

Use one stable capture ID for one device/session, for example:

```text
phone-a
phone-b
flatbed-a
office-scanner-01
```

The resulting sample ID is:

```text
<page-id>-<capture-id>
```

Example:

```text
project-authored-lao-v1-two-column-p0001-phone-a
```

This allows the same printed page to have multiple optical captures while the document-level split rules keep them together.
