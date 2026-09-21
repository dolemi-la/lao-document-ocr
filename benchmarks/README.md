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

## What every public report must include

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
