# Benchmarks

The benchmark suite measures OCR quality without hiding hard document classes behind one aggregate number.

## Metrics

- CER: Character Error Rate
- WER: Word Error Rate

Lower is better.

Overall CER/WER are weighted by the amount of reference text, not the arithmetic mean of per-page percentages.

## Dataset structure

Suggested layout:

```text
benchmarks/public/
├── data/
│   ├── clean-print/
│   ├── noisy-scan/
│   ├── phone-photo/
│   ├── mixed-lao-english/
│   ├── multi-column/
│   ├── simple-table/
│   ├── complex-table/
│   ├── receipt/
│   └── form/
└── ground-truth/
    └── ...
```

The manifest can live separately, for example:

```text
benchmarks/manifest-v1.jsonl
```

See [../docs/dataset-format.md](../docs/dataset-format.md).

## Validate a dataset

```bash
lao-ocr validate-dataset \
  --manifest benchmarks/manifest-v1.jsonl \
  --dataset-root benchmarks/public
```

## Run the Tesseract baseline

Run in an environment with Tesseract Lao and English language data:

```bash
lao-ocr benchmark \
  --manifest benchmarks/manifest-v1.jsonl \
  --dataset-root benchmarks/public \
  --split test \
  --languages lao+eng \
  --output benchmarks/results/tesseract-v1.json
```

## Synthetic smoke benchmark

The repository also includes a deterministic smoke generator. It exists to verify the OCR/benchmark pipeline, not to estimate real-world accuracy.

The Docker API image includes Noto Sans Lao and Noto Sans from Debian's `fonts-noto-core` package.

Build the image:

```bash
docker compose build api
```

Generate the smoke set:

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  -w /workspace \
  lao-document-ocr-api:latest \
  python benchmarks/generate_synthetic_smoke.py \
  --output benchmarks/generated/smoke
```

Run the current reference configuration:

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  -w /workspace \
  lao-document-ocr-api:latest \
  lao-ocr benchmark \
  --manifest benchmarks/generated/smoke/manifest.jsonl \
  --dataset-root benchmarks/generated/smoke \
  --split test \
  --languages lao+eng \
  --psm 6 \
  --output benchmarks/generated/smoke/report.json
```

Current checked-in smoke result:

```text
Tesseract 5.5.0
languages: lao+eng
PSM: 6
samples: 4
CER: 0.0235
WER: 0.1000
```

See [results/synthetic-smoke-tesseract-5.5.0-psm6.json](results/synthetic-smoke-tesseract-5.5.0-psm6.json).

These numbers must never be marketed as document accuracy. The pages are synthetic and intentionally simple.

## What every public real-document report must include

- benchmark manifest version/commit
- number of pages
- number of reference characters
- engine/model version
- preprocessing version/commit
- CER
- WER
- per-subset CER/WER
- hardware/platform information
- runtime

Never publish a rounded "accuracy" percentage without defining the benchmark and metric behind it.

## Dataset policy

- Do not commit private documents.
- Do not redistribute scans without clear rights.
- Keep pages from the same source document in one split only.
- Freeze public test document IDs once a benchmark release is published.
- Keep exact UTF-8 ground truth under review like source code.

See [../docs/dataset-sources.md](../docs/dataset-sources.md) for reviewed/candidate sources.

## Benchmark release bundle

Keep OCR, recognizer, layout, and DOCX fidelity reports as separate JSON files, then create one provenance bundle:

```bash
lao-ocr bundle-benchmarks \
  --ocr benchmarks/results/tesseract.json \
  --layout benchmarks/results/layout.json \
  --docx benchmarks/results/docx-fidelity.json \
  --revision "$(git rev-parse HEAD)" \
  --label baseline-v1 \
  --output benchmarks/results/baseline-v1.bundle.json
```

See [../docs/benchmark-bundles.md](../docs/benchmark-bundles.md).

## Rights-clear capture packs

The preferred real-scan acquisition path is:

```text
approved text source
  -> generate-capture-pack
  -> print
  -> real flatbed/phone capture
  -> register-capture
  -> validate-dataset
```

Source decisions live in `source-registry.json`.

See:

- [../docs/benchmark-source-review.md](../docs/benchmark-source-review.md)
- [../docs/capture-benchmark-workflow.md](../docs/capture-benchmark-workflow.md)

## Coverage report

Track benchmark growth and integrity with:

```bash
lao-ocr dataset-report \
  --manifest benchmarks/public/manifest.jsonl \
  --dataset-root benchmarks/public \
  --output benchmarks/results/dataset-report.json
```

The report includes sample/document counts, split/subset/license coverage, captures per document, missing benchmark categories, and validation errors. Exact duplicate source-image hashes are rejected by dataset validation.

## Multi-axis tag metrics

Dataset samples may carry multiple tags in addition to their primary `subset`. OCR benchmark reports include a `tags` section with independent weighted CER/WER for every tag, so overlapping dimensions can be measured without duplicating samples.

## Layout-labeled coverage

Samples may optionally reference reviewed Document-AST layout ground truth. `lao-ocr dataset-report` includes layout-labeled sample/document counts, coverage ratio, and breakdowns by split/subset so layout-training readiness is visible separately from OCR text coverage.

## Compare owned OCR against Tesseract

Run both engines on the same fixed split, then use the CI-friendly comparison gate:

```bash
lao-ocr compare-benchmarks \
  --baseline benchmarks/results/tesseract-v1.json \
  --candidate benchmarks/results/owned-v1.json \
  --output benchmarks/results/owned-vs-tesseract.json
```

The comparator rejects sample-set/label mismatches, requires overall CER improvement by default, and caps regressions on shared subsets/tags. See [../docs/benchmark-comparison.md](../docs/benchmark-comparison.md).

## Freeze the reviewed test set

Before publishing baseline/candidate numbers, freeze the reviewed test split:

```bash
lao-ocr freeze-benchmark \
  --manifest benchmarks/manifest-v1.jsonl \
  --dataset-root benchmarks/public \
  --split test \
  --output-manifest benchmarks/frozen/test-v1.jsonl \
  --output-lock benchmarks/frozen/test-v1.lock.json \
  --revision "$(git rev-parse HEAD)"
```

Verify the lock before every published benchmark run with `verify-benchmark-freeze`. See [../docs/benchmark-freeze.md](../docs/benchmark-freeze.md).

When publishing baseline/candidate results, use `--freeze-lock <lock.json>` with `lao-ocr benchmark` so the command refuses to run if the frozen test set has changed.

## Benchmark readiness

Before freezing/publishing a test set, verify coverage across real capture/layout dimensions:

```bash
lao-ocr benchmark-readiness \
  --manifest benchmarks/manifest-v1.jsonl \
  --dataset-root benchmarks/public \
  --split test \
  --output benchmarks/results/readiness-v1.json
```

Increase `--min-documents-per-dimension`, `--min-total-documents`, and/or `--min-layout-labeled-documents` for a release-quality gate. See [../docs/benchmark-readiness.md](../docs/benchmark-readiness.md).
