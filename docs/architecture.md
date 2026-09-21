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

### `src/lao_document_ocr/semantic.py`

Classifies OCR paragraph groups into heading/body/list blocks. Heading levels are estimated from line height relative to the page's typical text height. List parsing distinguishes ordered vs bulleted markers, joins wrapped continuation lines into clean list items, and keeps the original OCR text unchanged for CER/WER.

### `src/lao_document_ocr/table_detection.py`

Detects clear ruled grids using horizontal/vertical morphology, assigns OCR lines to row/column cells, and emits editable table blocks. Missing interior border segments can form rectangular row/column spans; non-rectangular/ambiguous merges are rejected. Borderless tables intentionally fall back to ordinary text instead of being guessed.

### `src/lao_document_ocr/borderless_tables.py`

Detects borderless tables only from strong repeated OCR geometry: stable column anchors, tight row spacing, short cell text, and real gaps between columns. Two-column candidates additionally require a consistently numeric/currency/date-style value column so ordinary two-column prose remains reading-order content instead of being converted into a table.

### `src/lao_document_ocr/reading_order.py`

Applies conservative two-column reading order only when a clear central gutter separates at least two blocks per side. Ambiguous/spanning layouts keep normal top-to-bottom ordering instead of guessing.

### `src/lao_document_ocr/layout_metrics.py`

Compares reviewed and predicted document ASTs independently of OCR text quality. It reports normalized block IoU/F1, semantic type accuracy, reading-order accuracy, and table shape/cell-span structure metrics.

### `src/lao_document_ocr/visual_fidelity.py`

Optionally renders generated DOCX files through headless LibreOffice and compares the rendered pages against reference images/PDFs using pixel similarity, foreground IoU, tolerant edge F1, and page-count consistency. LibreOffice is benchmark-only and is not required by the OCR runtime.

### `src/lao_document_ocr/header_footer.py`

Finds exact repeated text in the top/bottom page margins across multi-page documents and tags it as header/footer content. The DOCX exporter moves those tagged blocks into real Word header/footer parts while retaining the role in the document AST.

### `src/lao_document_ocr/embedded_images.py`

Extracts native image objects from PDF pages, normalizes them to PNG, maps their rectangles into rendered-page coordinates, and emits image blocks. Images covering most of a page are intentionally excluded so scan-background images are not duplicated alongside OCR text.

### `src/lao_document_ocr/raster_regions.py`

Masks recognized text and known native-PDF image rectangles, then detects dense photo/logo-like raster regions. It rejects near-full-page regions to avoid duplicating scan backgrounds and crops from the original color page for DOCX preservation.

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
