# Collector-safe capture kit

A capture suite contains maintainer-only material such as digital source-page paths and ground-truth references. Contributors collecting real scans/photos do not need that internal material.

`build-capture-kit` creates a deterministic ZIP containing only collector-facing files.

## Build

```bash
lao-ocr build-capture-kit \
  --suite-manifest benchmarks/capture-packs/baseline-suite-v1/capture-suite.json \
  --output benchmarks/capture-packs/baseline-suite-v1.collector.zip \
  --revision "$(git rev-parse HEAD)"
```

## ZIP contents

```text
CAPTURE-INSTRUCTIONS.md
SHA256SUMS
capture-kit.json
capture-worksheet.csv
<site-id>.pdf
```

The kit intentionally excludes:

- ground-truth text files
- digital page PNGs
- internal pack-manifest paths
- internal ground-truth paths
- internal digital-page paths

This keeps data collection blind to benchmark answers while still giving collectors stable page IDs and requested capture modes.

## Sanitized worksheet

The packaged worksheet contains only:

```text
combined_page
template
page_id
required_capture_modes
capture_file
notes
```

Collectors can fill `capture_file` and `notes` while working through the printed pages.

## Integrity

`capture-kit.json` records:

- suite ID
- optional source Git revision
- SHA-256 of the source suite manifest
- text license/provenance
- font and DPI
- template/page counts
- SHA-256 and byte size of every packaged collector file
- explicit flags confirming that ground truth, digital pages, and internal suite paths are excluded

`SHA256SUMS` covers every other member in the ZIP.

ZIP entry timestamps are fixed, and the combined printable PDF is serialized without a random PDF document ID. Rebuilding the campaign and kit from the same source revision therefore produces identical combined-PDF/manifest/worksheet/ZIP bytes.

## Collection guidance

The included instructions ask contributors to:

- print at approximately 100% scale
- keep page-ID QR markers visible
- submit real scanner/phone captures, not screenshots or PDF re-exports
- avoid OCR, sharpening, beautification, or AI enhancement before submission
- avoid unrelated private/confidential material in the frame
- keep original capture files

After collection, maintainers run optical-evidence checks, registration, manual review, readiness checks, and benchmark freeze before publishing accuracy results.

## GitHub Actions artifact

Maintainers can build the canonical project-authored collector kit without committing generated files:

1. open **Actions** in the repository
2. select **Build Collector Capture Kit**
3. choose **Run workflow** on the desired commit/branch
4. download the generated artifact when the workflow succeeds

The artifact contains only:

```text
project-authored-lao-v1.collector.zip
project-authored-lao-v1.collector.zip.sha256
```

The workflow installs the Noto Lao font, regenerates all 60 pages from the project-authored corpus, verifies the suite/collector contents and checksums, binds the kit to `github.sha`, and uploads it with 30-day artifact retention. Repository permissions are read-only.

Workflow source: `.github/workflows/capture-kit.yml`.

## Mobile/browser collection

Instead of manually copying camera files, maintainers can serve a verified collector kit on a trusted local network and capture directly from a phone browser with `lao-ocr serve-capture-kit`. The CLI protects the session with a random access token by default. See [mobile-capture-collector.md](mobile-capture-collector.md).
