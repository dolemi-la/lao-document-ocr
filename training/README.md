# Training

This directory is reserved for the open Lao recognizer training pipeline.

Current components:

- normalized Lao corpus preparation with Lao-ratio filtering and SHA-256 deduplication
- text/JSONL ingestion through `lao-ocr prepare-corpus`

Example:

```bash
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
