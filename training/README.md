# Training

This directory is reserved for the open Lao recognizer training pipeline.

Current components:

- normalized Lao corpus preparation with Lao-ratio filtering and SHA-256 deduplication
- text/JSONL ingestion through `lao-ocr prepare-corpus`

Example:

```bash

### Phetsarath-compatible training subset

For canonical Phetsarath-first rendering, filter the prepared corpus before synthetic generation:

```bash
lao-ocr prepare-corpus \
  --input training/data/lao-lines.txt \
  --output training/data/lao-lines-phetsarath.txt \
  --font-compatible-with benchmarks/private/fonts/PhetsarathOT-v4.103/PhetsarathOT-Regular.ttf \
  --font-coverage-report training/data/lao-lines-phetsarath.meta.json
```

On the current 10,073-line model-development corpus, Phetsarath OT v4.103 Regular covers 9,694 lines (96.24%) and excludes 379 lines. The largest excluded group contains Lao U+0EBA (຺); a smaller number contain Vietnamese, Greek, CJK, symbols, or emoji. Keep those lines out of the canonical Phetsarath synthetic run unless a separate font is explicitly reviewed for that subset. Do not allow silent fallback/tofu rectangles.
lao-ocr prepare-corpus \
  --input source.jsonl \
  --format jsonl \
  --field text \
  --output training/data/lao-lines.txt \
  --min-lao-ratio 0.5
```

Planned components:

- dataset manifest validator
- synthetic Lao line generator
- font licensing manifest
- augmentations
- recognizer training config
- evaluation
- ONNX/export tooling
- model cards

No model binaries should be committed directly to Git.

## Synthetic line generation

After preparing a corpus, render deterministic labeled line images with one or more Lao-capable fonts:

```bash
lao-ocr generate-synthetic \
  --corpus training/data/lao-lines.txt \
  --output training/generated/v1 \
  --font /path/to/PhetsarathOT-Regular.ttf \
  --variants-per-line 3 \
  --seed 20260921 \
  --require-complete-font
```

Each generated sample records its source text, font, font size, deterministic seed, augmentation parameters, relative image path, and SHA-256. Current augmentations include small rotation, brightness jitter, Gaussian blur, and image noise.

Do not commit generated training images or large corpora to Git. Reproduce them from corpus + config + seed instead.

Full recognizer workflow: [../docs/recognizer-training.md](../docs/recognizer-training.md).
