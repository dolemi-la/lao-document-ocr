# Architecture

## Design goals

- Lao-first
- local-first
- reproducible
- no proprietary OCR dependency
- OCR engine replaceable without changing document exports
- deterministic document reconstruction where possible
- explicit confidence rather than invented structure

## Pipeline

```text
PDF / image
    |
    v
page rendering
    |
    v
preprocessing
- EXIF orientation
- grayscale
- contrast
- conservative deskew
    |
    v
OCR engine
    |
    v
recognized lines
- text
- bbox
- confidence
- grouping ids
    |
    v
structure inference
    |
    v
Document AST
    |
    +--> DOCX
    +--> Markdown
    +--> TXT
    +--> JSON
```

## Components

### `src/lao_document_ocr/preprocessing.py`

Image cleanup that should improve OCR without aggressively altering glyph shapes.

### `src/lao_document_ocr/ocr/`

OCR interface and current Tesseract adapter.

The long-term Lao model should implement the same `OcrEngine` interface.

### `src/lao_document_ocr/structure.py`

Turns OCR lines into semantic blocks. v0.1 intentionally uses conservative heuristics.

### `src/lao_document_ocr/table_detection.py`

Detects clear ruled grids using horizontal/vertical morphology, assigns OCR lines to row/column cells, and emits editable table blocks. Borderless or ambiguous tables intentionally fall back to ordinary text instead of being guessed.

### `src/lao_document_ocr/models.py`

The canonical document AST.

### `src/lao_document_ocr/exporters/`

Output-specific renderers. The DOCX exporter sets Lao language metadata and a Lao-capable default font.

### `services/api`

Thin HTTP layer. v0.1 is synchronous by design.

An async job interface can be added later without coupling it to OCR internals.

### `apps/web`

Small upload/download client.

## Future recognizer

Target architecture:

```text
page
 |
 +--> text detection
 |       |
 |       +--> line crops
 |               |
 |               +--> Lao recognizer
 |
 +--> layout model
 |
 +--> table structure model
 |
 +--> reading-order resolver
         |
         v
     Document AST
```

The detector/layout/recognizer models should have independently versioned benchmark results.

## Trust model

The pipeline must prefer text preservation over making up structure.

For example:

- uncertain table -> paragraph/text blocks
- uncertain heading -> paragraph
- low OCR confidence -> retain text plus confidence
- unsupported content -> expose metadata rather than silently dropping it
