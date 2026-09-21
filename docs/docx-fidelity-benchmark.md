# DOCX visual-fidelity benchmark

AST layout metrics tell us whether the document structure is correct. They do not tell us whether the generated Word file renders similarly to the source page.

The DOCX fidelity benchmark renders the generated `.docx` through headless LibreOffice, rasterizes the resulting PDF, and compares those rendered pages against a reference image or PDF.

LibreOffice is an **optional benchmark dependency**. It is not required for normal OCR or DOCX export.

## Requirements

Install LibreOffice so one of these is available:

- `libreoffice`
- `soffice`
- `loffice`

You can also pass an explicit binary path.

## Run

```bash
lao-ocr benchmark-docx \
  --reference reference.pdf \
  --docx output.docx \
  --output docx-fidelity.json \
  --dpi 144
```

With an explicit office binary:

```bash
lao-ocr benchmark-docx \
  --reference reference.png \
  --docx output.docx \
  --output docx-fidelity.json \
  --office-binary /path/to/soffice
```

The report records:

- LibreOffice binary
- LibreOffice version, when available
- PyMuPDF version
- render DPI
- page count
- per-page metrics
- document-level averages

## Metrics

### Pixel similarity

```text
1 - mean_absolute_grayscale_error / 255
```

This captures broad visual similarity such as whitespace, dark/light regions, images, and text placement.

### Foreground IoU

Pixels darker than a conservative near-white threshold are treated as foreground.

The metric compares overlap of source and generated foreground regions. It is useful for text density, block placement, table borders, images, and margins.

### Edge F1

Canny edges are compared with a one-pixel neighborhood tolerance. This emphasizes text/glyph edges, table borders, image boundaries, and block geometry.

### Page-count score

```text
min(reference_pages, predicted_pages)
-------------------------------------
max(reference_pages, predicted_pages)
```

Missing or extra rendered pages therefore reduce the document score.

### Composite score

The current transparent weighting is:

```text
page_score =
    0.40 * pixel_similarity
  + 0.30 * foreground_iou
  + 0.30 * edge_f1

document_score =
    mean(page_score) * page_count_score
```

These weights are intentionally simple and documented. Do not silently change them between benchmark releases.

## Alignment

The predicted rendered page is aspect-preserving resized into the reference page canvas and centered.

It is **not** geometrically warped to make the prediction look better. Different aspect ratios therefore receive a real penalty.

## What the score means

This is a rendering-fidelity metric, not an OCR accuracy metric.

A DOCX can have perfect OCR text but weak visual fidelity, or strong visual fidelity but transcription errors.

Publish DOCX fidelity alongside CER/WER, layout block F1/IoU, reading-order accuracy, and table structure metrics.

## Limitations

The benchmark is affected by the rendering engine and installed fonts.

For reproducible public results:

- report the LibreOffice version
- report the benchmark commit
- use the same render DPI
- use the same font environment when possible

The metric does not prove pixel-perfect Microsoft Word rendering. LibreOffice is used as a reproducible open benchmark renderer.

## CI

Because LibreOffice is a large system dependency, normal unit tests test the image metrics and missing-renderer behavior without requiring LibreOffice.

A dedicated benchmark CI job or release workflow can install LibreOffice and run real render comparisons.
