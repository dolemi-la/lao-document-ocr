# Dataset format

Phase 1 uses newline-delimited JSON (JSONL). Each line describes one benchmark page.

The format is intentionally reviewable in Git and strict enough to keep benchmark results reproducible.

## Example

```json
{
  "id": "clean-001",
  "document_id": "document-001",
  "split": "test",
  "subset": "clean-print",
  "source": "data/clean-print/clean-001.png",
  "ground_truth": "ground-truth/clean-print/clean-001.txt",
  "language": "lo",
  "license": "CC-BY-4.0",
  "provenance": "Scanned by Example Contributor from an openly licensed source",
  "source_url": "https://example.org/source",
  "license_url": "https://creativecommons.org/licenses/by/4.0/",
  "sha256": "<64 hex characters>",
  "notes": "Optional notes"
}
```

In a `.jsonl` file, each object occupies one line.

## Required fields

### `id`

Stable page identifier. Allowed characters: letters, numbers, `.`, `_`, `-`.

Do not reuse an ID for a different page.

### `document_id`

Stable ID of the original source document.

All pages from one original document must stay in the same train/dev/test split. The validator rejects cross-split document leakage.

### `split`

One of:

- `train`
- `dev`
- `test`

The test split must not be used for model selection or training.

### `subset`

One of:

- `clean-print`
- `noisy-scan`
- `phone-photo`
- `mixed-lao-english`
- `multi-column`
- `simple-table`
- `complex-table`
- `receipt`
- `form`
- `other`

A sample belongs to its dominant document category.

### `source`

Relative path to the source page image.

Phase 1 benchmarks one page per sample so failures and metrics are attributable. Multi-page PDFs should be rendered into page images before entering the fixed benchmark.

### `ground_truth`

Relative path to exact UTF-8 ground-truth text.

Ground truth should preserve intended reading order. Do not silently spell-correct the source document.

### `license`

SPDX identifier when possible, or a precise rights statement.

A sample with unclear redistribution or model-training rights must not be committed to the public dataset.

### `provenance`

Human-readable explanation of where the sample came from and who created or contributed it.

## Optional fields

### `language`

Defaults to `lo`.

### `source_url`

Original public source URL, when applicable.

### `license_url`

URL containing the source/license terms, when applicable.

### `sha256`

SHA-256 of the source page. Public benchmark releases should include hashes.

### `notes`

Anything reviewers need to understand about the sample.

## Path rules

`source` and `ground_truth` must be relative and may not contain parent traversal (`..`).

## Validation

```bash
lao-ocr validate-dataset \
  --manifest benchmarks/manifest.jsonl \
  --dataset-root benchmarks/public
```

Validation checks:

- JSONL syntax and schema
- duplicate sample IDs
- allowed split/subset values
- document-level train/dev/test leakage
- path safety
- source and ground-truth existence
- UTF-8/non-empty ground truth
- optional SHA-256 integrity

## Benchmark

```bash
lao-ocr benchmark \
  --manifest benchmarks/manifest.jsonl \
  --dataset-root benchmarks/public \
  --split test \
  --languages lao+eng \
  --output benchmarks/results/tesseract-baseline.json
```

The report includes:

- CER and WER
- weighted overall metrics
- per-subset metrics
- per-sample metrics
- runtime
- Python/platform metadata

## Split policy

The first public release should freeze its test document IDs.

A reasonable starting point for a sufficiently large collection is:

- train: 80%
- dev: 10%
- test: 10%

Document grouping matters more than exact percentages. Pages from one original document must never be divided across splits.

## Privacy

Do not use documents containing private personal information unless they were deliberately created or sanitized for public redistribution and all required rights and consents exist.
