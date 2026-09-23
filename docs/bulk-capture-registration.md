# Bulk capture registration

Use this workflow after scanning or photographing multiple pages from a capture suite.

It avoids running `register-capture` once per page.

## Filename convention

Put one capture mode/device run in one directory.

Each image filename stem must exactly match the capture-suite page ID:

```text
captures/phone-a/
├── project-authored-lao-v1-plain-p0001.jpg
├── project-authored-lao-v1-plain-p0002.jpg
├── ...
├── project-authored-lao-v1-form-p0009.jpg
└── project-authored-lao-v1-form-p0010.jpg
```

Supported image extensions:

- `.png`
- `.jpg`
- `.jpeg`
- `.tif`
- `.tiff`
- `.webp`

Non-image files such as `.DS_Store` are ignored and reported.

Unknown image stems are rejected instead of being silently skipped.

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

- suite page IDs
- duplicate filenames
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
