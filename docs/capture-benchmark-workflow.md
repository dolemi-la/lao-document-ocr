# Real scan and phone-photo benchmark workflow

This workflow creates real optical benchmark images without scraping copyrighted documents.

## 1. Prepare reviewed Lao text

For HPLT or another approved source, first normalize/filter the corpus:

```bash
lao-ocr prepare-corpus \
  --input lao-source.jsonl \
  --format jsonl \
  --field text \
  --output training/data/lao-lines.txt \
  --min-lao-ratio 0.5
```

The source must already be approved in `benchmarks/source-registry.json`.

## 2. Generate printable capture pages

```bash
lao-ocr generate-capture-pack \
  --corpus training/data/lao-lines.txt \
  --output benchmarks/capture-packs/baseline-v1 \
  --font /path/to/NotoSansLao-Regular.ttf \
  --pack-id baseline-v1 \
  --text-license CC0-1.0 \
  --text-provenance "HPLT v2 Lao lao-Laoo cleaned corpus" \
  --dpi 150 \
  --lines-per-page 10 \
  --max-pages 100
```

The output contains:

```text
baseline-v1/
├── baseline-v1.pdf
├── capture-pack.json
├── pages/
│   ├── baseline-v1-p0001.png
│   └── ...
└── ground-truth/
    ├── baseline-v1-p0001.txt
    └── ...
```

The printable PDF and digital page PNGs are reproducible source pages.

## 3. Capture real optical versions

Print at approximately 100% scale.

Recommended capture classes:

### Flatbed scan

Use for `flatbed-scan`.

Examples:

- normal office scanner
- clean 150–600 DPI scan
- standard paper

These enter the `clean-print` benchmark subset.

### Degraded scan

Use for `degraded-scan`.

Examples:

- low DPI
- photocopy then scan
- mild compression
- faded print
- slightly skewed feed

These enter the `noisy-scan` subset.

### Phone photo

Use for `phone-photo`.

Useful natural variation:

- different phones
- indoor daylight
- desk/table backgrounds
- mild perspective
- reasonable handheld framing

Do not deliberately damage devices or perform unsafe capture setups.

## 4. Register a capture

```bash
lao-ocr register-capture \
  --pack-manifest benchmarks/capture-packs/baseline-v1/capture-pack.json \
  --page-id baseline-v1-p0001 \
  --capture-image /path/to/phone-photo.jpg \
  --capture-id iphone-a \
  --mode phone-photo \
  --contributor "Contributor Alias" \
  --release-license CC0-1.0 \
  --dataset-root benchmarks/public \
  --dataset-manifest benchmarks/public/manifest.jsonl \
  --confirm-release
```

The command:

- verifies the original capture-pack page SHA-256
- reuses the exact pack ground truth
- records source-text license/provenance
- records capture contributor/release license
- maps capture mode to benchmark subset
- keeps every capture of the same printed page in the same train/dev/test split
- prevents duplicate sample IDs

## Contributor release meaning

By passing `--confirm-release`, the contributor asserts that:

- they created or otherwise control the submitted capture image
- they have authority to release that capture under `--release-license`
- the capture does not intentionally contain unrelated private/confidential material

The project does not infer ownership from file possession alone.

## Multiple devices per page

This is encouraged.

For example:

```text
baseline-v1-p0001-flatbed-a
baseline-v1-p0001-phone-a
baseline-v1-p0001-phone-b
```

All share document ID `baseline-v1-p0001`, so split leakage protection keeps the same printed content out of multiple train/dev/test splits.

## Before publishing results

Run:

```bash
lao-ocr validate-dataset \
  --manifest benchmarks/public/manifest.jsonl \
  --dataset-root benchmarks/public
```

Then manually review:

- image orientation
- correct page ID
- exact ground truth
- subset classification
- contributor/release metadata
- no unrelated personal/private material

Freeze the final test document IDs before comparing OCR models.

## Multi-axis benchmark tags

Registered capture-pack samples automatically receive tags such as:

- `capture:flatbed-scan`
- `capture:degraded-scan`
- `capture:phone-photo`
- `layout:plain`
- `language:lao` or `language:mixed`
- `source:capture-pack`
- `source:real-capture`

This matters because one real page can simultaneously be a phone photo, mixed Lao/English, multi-column, and table-heavy. The primary `subset` field is retained for compatibility, but CER/WER reports also aggregate by every tag.
