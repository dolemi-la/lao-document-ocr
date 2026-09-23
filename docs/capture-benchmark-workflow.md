# Real scan and phone-photo benchmark workflow

This workflow creates real optical benchmark images without scraping copyrighted documents.

## 1. Prepare independently redistributable Lao text

Public capture-pack ground truth must come from text whose **underlying content rights** permit redistribution. Preferred sources include project-authored text explicitly released for the benchmark, source-specific CC0/public-domain material, or contributor text under a compatible license.

For a reviewed local source, normalize/filter it with `prepare-corpus`:

```bash
lao-ocr prepare-corpus \
  --input benchmark-source.txt \
  --output training/data/capture-lines.txt \
  --min-lao-ratio 0.5
```

The source must already be reviewed in `benchmarks/source-registry.json` or have equivalent provenance attached to the campaign.

HPLT's bounded sampler is useful for model-development/training text, but HPLT's packaging CC0 does not establish redistribution rights for every underlying extracted line. Do not use sampled HPLT lines in a **public** capture pack unless those underlying source rights are separately cleared. See [hplt-sampling.md](hplt-sampling.md).

## 2. Generate printable capture pages

```bash
lao-ocr generate-capture-pack \
  --corpus training/data/capture-lines.txt \
  --output benchmarks/capture-packs/baseline-v1 \
  --font /path/to/NotoSansLao-Regular.ttf \
  --pack-id baseline-v1 \
  --text-license Apache-2.0 \
  --text-provenance "Project-authored Lao benchmark text released Apache-2.0" \
  --dpi 150 \
  --lines-per-page 10 \
  --max-pages 100
```

## Structured capture templates

`generate-capture-pack` supports six deterministic layouts:

- `plain` — normal single-column text
- `two-column` — clear left/right reading-order challenge
- `ruled-table` — visible grid table with Lao text + numeric values
- `borderless-table` — aligned text/value table without drawn borders
- `receipt` — receipt-like item/amount/total layout
- `form` — label/value form rows

Example:

```bash
lao-ocr generate-capture-pack \
  --corpus training/data/capture-lines.txt \
  --output benchmarks/capture-packs/multi-column-v1 \
  --font /path/to/NotoSansLao-Regular.ttf \
  --pack-id multi-column-v1 \
  --text-license Apache-2.0 \
  --text-provenance "Project-authored Lao benchmark text released Apache-2.0" \
  --template two-column
```

Every page records its template and benchmark tags. For example, a captured two-column phone photo can carry all of:

```text
template:two-column
layout:multi-column
capture:phone-photo
source:capture-pack
source:real-capture
language:lao
```

This lets the same real capture participate in overlapping benchmark slices without duplicating it.

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

For multi-layout campaigns, generate one combined suite instead of separate packs. See [capture-suite.md](capture-suite.md).

## Page-ID QR markers

New capture packs include a `page-id:qr-v1` marker in the top-right header. Keep it visible when printing/capturing. `register-capture-directory` uses it to map raw scanner/phone filenames back to the correct ground truth and rejects filename/QR disagreements.

Before sending printable pages to contributors, package them with `build-capture-kit` so collectors receive no ground truth or digital source pages. See [capture-kit.md](capture-kit.md).

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

## Optical-evidence integrity check

`register-capture` does not accept a digital copy/re-encode of the generated capture-pack page as real benchmark evidence. Before registration it checks:

- visible foreground/content is present
- page aspect similarity
- normalized grayscale mean absolute error
- perceptual dHash distance

Near-identical digital re-encodes are rejected with an instruction to submit a real scan/photo. Blank or nearly blank captures are also rejected.

Accepted capture-pack registrations receive:

```text
capture:optical-evidence
source:real-capture
```

and store the digital-source similarity diagnostics in the sample notes for audit/review.

The check is intentionally conservative: it is meant to catch accidental digital copies, not to prove forensic camera/scanner provenance by itself. Manual review is still required before freezing a public benchmark.

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

Then manually review each accepted sample and record the decision:

```bash
lao-ocr review-dataset-sample \
  --manifest benchmarks/public/manifest.jsonl \
  --dataset-root benchmarks/public \
  --id <sample-id> \
  --status approved \
  --reviewer <reviewer-alias> \
  --notes "Page ID, orientation, ground truth, tags, and release metadata checked."
```

Review image orientation, page ID, exact ground truth, subset/tags, contributor/release metadata, and absence of unrelated private material. See [dataset-review.md](dataset-review.md).

Freeze the final test document IDs only after the readiness gate reports approved review coverage.

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

## Track campaign completion

Use `lao-ocr capture-campaign-report` to see which capture-suite pages still need flatbed/phone/degraded captures before freezing the benchmark test set. See [capture-campaign-report.md](capture-campaign-report.md).

## Bulk registration

For scanner/phone sessions containing many capture-suite pages, use `register-capture-directory`. Filenames must match the printed page IDs. Dry-run/preflight and rollback behavior are documented in [bulk-capture-registration.md](bulk-capture-registration.md).
