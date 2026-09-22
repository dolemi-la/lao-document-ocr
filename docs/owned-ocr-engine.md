# Project-owned full-page OCR engine

The project now has an experimental full-page path that does not use Tesseract for recognition:

```text
page image
  -> deterministic morphology-based line detection
  -> line crops
  -> crnn-ctc-v2 recognizer
  -> optional confidence calibration
  -> RecognizedLine objects
  -> Document AST
  -> DOCX / Markdown / TXT / JSON
```

## Status

This is an engineering baseline, not the default engine.

It currently targets:

- clean printed scans
- mostly single-column pages
- reasonably separated text lines

It does **not** claim robust:

- multi-column reading order
- complex tables
- forms
- headers/footers
- arbitrary camera perspective
- handwriting

Those remain Phase 3 work.


## Local document conversion

```bash
lao-ocr convert-document \
  --input scan.pdf \
  --output-dir output \
  --engine owned \
  --model recognizer.pt2 \
  --calibration calibration.json
```

This runs the same page preprocessing, line detection, recognizer, document AST, and exporters as the API path.

## API configuration

The normal API defaults to Tesseract:

```bash
OCR_ENGINE=tesseract
```

To use the project-owned recognizer, install the training/inference dependencies and provide an exported model:

```bash
OCR_ENGINE=owned
OCR_MODEL_PATH=/models/recognizer.pt2
OCR_CALIBRATION_PATH=/models/calibration.json   # optional
```

The API health endpoint reports the selected engine and model metadata.

The base API Docker image intentionally stays lightweight and does not include PyTorch. Use an environment/image that installs `.[train]` when selecting `OCR_ENGINE=owned`.

## Confidence calibration

Raw mean timestep probability is not a reliable probability of OCR correctness.

Fit calibration on a **held-out development set**:

```bash
lao-ocr benchmark-recognizer \
  --manifest dev-lines/manifest.jsonl \
  --model recognizer.pt2 \
  --output dev-report.json

lao-ocr calibrate-recognizer \
  --report dev-report.json \
  --output calibration.json \
  --bins 10
```

Then use it during line inference:

```bash
lao-ocr recognize-line \
  --model recognizer.pt2 \
  --calibration calibration.json \
  --image line.png
```

or benchmarking:

```bash
lao-ocr benchmark-recognizer \
  --manifest test-lines/manifest.jsonl \
  --model recognizer.pt2 \
  --calibration calibration.json \
  --output test-report.json
```

Never fit calibration on the final test set.

The current calibration target is observed character accuracy (`1 - CER`) using quantile bins. This is deliberately simple and reviewable; more sophisticated calibration can be added once the real benchmark is large enough.

## Region-first detection

The owned OCR path now detects text regions before text lines:

```text
page
  -> morphology text regions
  -> independent region crops
  -> line detection inside each region
  -> project-owned recognizer
```

This handles separated columns/sections more safely than applying one horizontal morphology kernel across the whole page.

The detector is behind a `TextRegionDetector` interface so a learned detector can replace the morphology implementation without changing the recognizer/export pipeline.

The current region detector is still deterministic morphology, not a trained layout model.

## Accelerator runtime

Owned recognizer inference supports explicit CPU/CUDA/MPS/auto device selection. The API caches one model instance per model/calibration/device key. See [gpu-worker.md](gpu-worker.md).

## Text-region detector selection

The owned engine supports two text-region detector modes:

```text
OCR_LAYOUT_DETECTOR=morphology   # default
OCR_LAYOUT_DETECTOR=learned
```

For the learned mode also configure:

```text
OCR_LAYOUT_MODEL_PATH=/models/layout-detector.pt2
OCR_LAYOUT_CONFIDENCE=0.55
```

The learned detector is experimental and should remain opt-in until it beats the deterministic baseline on the fixed layout benchmark. See [layout-model-training.md](layout-model-training.md).

## Semantic region hints

When `OCR_LAYOUT_DETECTOR=learned`, layout regions retain their semantic class (`heading`, `paragraph`, `list`, `table`) through line detection and OCR. Lines from one detected region share a stable block/paragraph ID, so multi-line learned paragraphs/lists remain grouped in the AST.

The hints are applied conservatively and do not fabricate missing table cells or list markers.

## Learned visual regions

With `OCR_LAYOUT_DETECTOR=learned`, the same layout model can preserve predicted `image` regions as real image blocks in the document AST/DOCX. The owned engine exposes the layout model as both its text-region and visual-region detector.

Learned visual boxes are de-duplicated against OCR text/table/native-PDF regions before the deterministic raster and diagram fallbacks run.
