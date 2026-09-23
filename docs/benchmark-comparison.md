# OCR benchmark comparison gate

The Phase 2 quality gate compares the project-owned OCR pipeline against the Tesseract baseline on the same fixed test set.

## Run the Tesseract baseline

```bash
lao-ocr benchmark \
  --manifest benchmarks/manifest-v1.jsonl \
  --dataset-root benchmarks/public \
  --split test \
  --engine tesseract \
  --freeze-lock benchmarks/frozen/test-v1.lock.json \
  --languages lao+eng \
  --psm 3 \
  --output benchmarks/results/tesseract-v1.json
```

## Run the owned full-page pipeline

Use the same manifest, dataset root, and split:

```bash
lao-ocr benchmark \
  --manifest benchmarks/manifest-v1.jsonl \
  --dataset-root benchmarks/public \
  --split test \
  --engine owned \
  --freeze-lock benchmarks/frozen/test-v1.lock.json \
  --model models/recognizer.pt2 \
  --calibration models/calibration.json \
  --device cpu \
  --layout-detector learned \
  --layout-model models/layout.pt2 \
  --reading-order learned \
  --reading-order-model models/reading-order.pt2 \
  --output benchmarks/results/owned-v1.json
```

The benchmark report records the engine and reading-order configuration used.

## Compare

```bash
lao-ocr compare-benchmarks \
  --baseline benchmarks/results/tesseract-v1.json \
  --candidate benchmarks/results/owned-v1.json \
  --output benchmarks/results/owned-vs-tesseract.json
```

Default gate:

- primary metric: CER
- overall candidate CER must be strictly lower than baseline CER
- no shared subset/tag may regress by more than 0.02 absolute CER

## Fixed-set protection

The comparator rejects reports unless they contain the same split, sample IDs, primary subset labels, and multi-axis tags. This prevents accidental improvement claims after changing or dropping difficult test samples.

## Thresholds

Require a minimum absolute overall improvement with `--min-improvement 0.01`.

Change the maximum allowed per-slice regression with `--max-slice-regression 0.01`.

This protects categories such as phone photos, noisy scans, multi-column pages, receipts/forms, tables, and mixed Lao/English even when the overall average improves.

## CER and WER

CER is the default primary metric because Lao word boundaries are not always represented by spaces consistently. WER can still be selected with `--primary-metric wer`.

## Exit codes

`compare-benchmarks` is CI-friendly:

- `0` — comparison gate passed
- `1` — reports are comparable, but the candidate failed the quality gate
- `2` — reports cannot be compared safely because of schema/sample-set/configuration mismatch

Keep the baseline JSON, candidate JSON, comparison JSON, layout/table metrics, DOCX fidelity report, and benchmark release bundle together for releases.
