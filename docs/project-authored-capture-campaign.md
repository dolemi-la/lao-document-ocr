# Project-authored Lao capture campaign v1

This is the first rights-clear printable campaign intended to turn the project’s benchmark tooling into real optical data.

The source text is written specifically for this repository:

```text
resources/corpora/project-authored-lao-v1.txt
resources/corpora/project-authored-lao-v1.meta.json
```

It is distributed under the repository Apache-2.0 license and contains no real personal records.

## Generate the printable suite

Build the CPU training image once if needed:

```bash
docker build -f Dockerfile.train -t lao-document-ocr-train:latest .
```

Then run either:

```bash
make capture-suite
```

or directly:

```bash
./scripts/generate_public_capture_suite.sh
```

The generated files are intentionally ignored by Git:

```text
benchmarks/capture-packs/
├── project-authored-lao-v1/
│   ├── project-authored-lao-v1.pdf
│   ├── capture-suite.json
│   ├── capture-worksheet.csv
│   ├── plain/
│   ├── two-column/
│   ├── ruled-table/
│   ├── borderless-table/
│   ├── receipt/
│   └── form/
└── project-authored-lao-v1.collector.zip
```

The generator verifies all 60 suite pages and builds the collector-safe ZIP automatically. It prints the collector ZIP SHA-256 for handoff/integrity checks. The combined PDF and collector ZIP are byte-reproducible when regenerated from the same source revision.

The v1 campaign contains:

- 10 plain pages
- 10 two-column pages
- 10 ruled-table pages
- 10 borderless-table pages
- 10 receipt pages
- 10 form pages
- 60 pages total in the combined PDF

The digital PDF/PNGs are source material only. They are **not real OCR benchmark evidence**.

Share this ZIP with collectors instead of the internal suite directory:

```text
benchmarks/capture-packs/project-authored-lao-v1.collector.zip
```

It excludes ground truth, digital page PNGs, and internal suite paths.

## Print

Print:

```text
benchmarks/capture-packs/project-authored-lao-v1/project-authored-lao-v1.pdf
```

Recommended baseline:

- A4 paper
- approximately 100% scale
- normal office printer
- no deliberate image enhancement
- keep the page ID/footer visible

If the printer driver automatically scales slightly for hardware margins, that is acceptable; real optical variation is the point.

### Page-ID QR marker

Every v1 page includes a `page-id:qr-v1` marker in the top-right header. It encodes the exact capture-suite page ID and is outside the textual ground-truth content.

Bulk import uses the marker to prevent page/ground-truth mismatches. Keep the top-right page header visible in scans/photos.

## Capture each page

The initial target is two optical captures per printed page:

1. one flatbed scan
2. one phone photo

That means a complete initial campaign contains:

```text
60 pages × 2 capture modes = 120 real captures
```

A later degraded-scan pass can add photocopy/low-DPI/faded examples.

### Flatbed scan

Suggested baseline:

- 300 DPI
- full page visible
- normal scanner defaults
- PNG/TIFF or high-quality JPEG

### Phone photo

Use a normal handheld capture:

- full page visible
- ordinary indoor/daylight lighting
- moderate natural perspective is fine
- avoid intentionally extreme blur or unsafe setups
- do not crop away the page ID/footer

The registration command rejects blank captures and near-identical digital re-encodes of the source page.

## Register captures

Each template has its own capture-pack manifest.

Example phone photo:

```bash
lao-ocr register-capture \
  --pack-manifest benchmarks/capture-packs/project-authored-lao-v1/two-column/capture-pack.json \
  --page-id project-authored-lao-v1-two-column-p0001 \
  --capture-image /path/to/phone-p0001.jpg \
  --capture-id phone-a \
  --mode phone-photo \
  --contributor "Contributor Alias" \
  --release-license CC0-1.0 \
  --dataset-root benchmarks/public \
  --dataset-manifest benchmarks/public/manifest.jsonl \
  --confirm-release
```

Example flatbed scan:

```bash
lao-ocr register-capture \
  --pack-manifest benchmarks/capture-packs/project-authored-lao-v1/two-column/capture-pack.json \
  --page-id project-authored-lao-v1-two-column-p0001 \
  --capture-image /path/to/scan-p0001.png \
  --capture-id flatbed-a \
  --mode flatbed-scan \
  --contributor "Contributor Alias" \
  --release-license CC0-1.0 \
  --dataset-root benchmarks/public \
  --dataset-manifest benchmarks/public/manifest.jsonl \
  --confirm-release
```

Registered real captures receive:

```text
source:real-capture
capture:optical-evidence
capture:<mode>
template:<template>
<layout/content/language tags>
```

## Bulk-register a full device run

For a 60-page scan/photo session, use `register-capture-directory` instead of 60 individual commands. The current pages carry a top-right page-ID QR marker, so normal camera filenames such as `IMG_1842.jpg` can be used without renaming.

Dry-run first:

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
  --dry-run
```

Then rerun without `--dry-run` and add `--confirm-release`. See [bulk-capture-registration.md](bulk-capture-registration.md) for the complete workflow and rollback behavior.

## Track campaign completion

```bash
lao-ocr capture-campaign-report \
  --suite-manifest benchmarks/capture-packs/project-authored-lao-v1/capture-suite.json \
  --dataset-manifest benchmarks/public/manifest.jsonl \
  --output benchmarks/results/project-authored-lao-v1.campaign.json
```

The default completion target is one flatbed scan plus one phone photo for every page.

## Before publishing a baseline

Do not publish Tesseract or owned-model “real benchmark” numbers just because the printable suite exists.

First:

1. collect and register the real captures
2. review capture images and ground truth
3. run `validate-dataset`
4. run `dataset-report`
5. run `capture-campaign-report`
6. run `benchmark-readiness`
7. freeze the reviewed test split
8. verify the freeze lock
9. run the Tesseract baseline with `--freeze-lock`
10. run the owned model on the same frozen set
11. compare with `compare-benchmarks`

The Phase 1 real-data roadmap items stay open until those optical captures actually exist.
