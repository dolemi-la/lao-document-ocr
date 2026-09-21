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

## Line detection

The current line detector uses thresholding plus horizontal morphology.

It exists so the project-owned recognizer can run through the complete document pipeline today. It should eventually be replaced or complemented by learned text/layout detection for difficult page structures.
