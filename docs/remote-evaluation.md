# Remote source diagnostics

The source registry contains rights-unclear and unapproved remote documents that are useful for understanding real OCR failure modes without copying those files into the public benchmark.

Use `evaluate-remote-sources` for **diagnostic OCR only**. It is intentionally separate from the fixed benchmark pipeline.

## Example

Evaluate one reviewed registry candidate:

```bash
lao-ocr evaluate-remote-sources \
  --source-id ptc-camscanner-2024-02-15 \
  --output reports/remote-ptc.json
```

Evaluate selected pages from more than one source:

```bash
lao-ocr evaluate-remote-sources \
  --source-id ptc-camscanner-2024-02-15 \
  --source-id worldbank-p172774-kpmg-lao-2024-rotated-raster-pages \
  --page 1 \
  --page 3 \
  --max-pages-per-source 3 \
  --max-source-mb 30 \
  --output reports/remote-selected.json
```

Without `--page`, PDFs sample representative pages (first, middle, last by default). Use `--all-remote` only when intentionally probing every `remote-evaluation-*` registry entry; collection pages, blocked hosts, and unverified entries may fail and are reported individually.

## GitHub Actions

For an environment with real Tesseract Lao/English installed, run the manual **Remote Scan Diagnostic** workflow.

Inputs:

- `source_id`: one `remote-evaluation-*` registry ID;
- `page`: optional 1-based page number;
- `max_source_mb`: bounded download limit, default 25 MiB.

The workflow is manual-only, has read-only repository permissions, installs `tesseract-ocr-lao` and `tesseract-ocr-eng`, validates the report privacy boundary, and uploads only the JSON diagnostic report for 14 days.

If the selected remote source fails, the workflow still uploads the report containing the source error and then marks the job failed.

## Safety and storage boundaries

The runner is intentionally bounded:

- only registry entries whose status starts with `remote-evaluation-` are eligible;
- source selection must be explicit unless `--all-remote` is supplied;
- only HTTPS URLs on port 443 are accepted;
- loopback/private/link-local/reserved hosts are rejected before download;
- redirects are revalidated;
- downloads default to a 25 MiB maximum per source;
- requests default to a 20 second timeout;
- PDFs default to a 500-page structural limit;
- rendered pages retain the normal 40 million pixel safety limit;
- downloaded files and sampled page renders live in a temporary directory and are removed after the run.

The JSON report stores source metadata, final URL, SHA-256, byte size, page counts, native-text statistics, OCR statistics, block counts, confidence summaries, rotation metadata, and timings.

It deliberately does **not** store:

- downloaded PDF/image bytes;
- rendered page images;
- extracted native document text;
- OCR-transcribed document text;
- derived ground truth.

## Layer-gap diagnostics

For PDF pages, the report compares only **statistics** from the native PDF text layer and OCR output. It never stores either text body.

Each sampled page gets a `layer_gap.classification`:

- `native-layer-empty`: the PDF page exposes no native text, but OCR recovers text;
- `native-layer-sparse`: the native layer is tiny (for example, only a scanner watermark) while OCR recovers substantially more;
- `lao-missing-from-native-layer`: OCR recovers substantial Lao text while the native layer contains little or no Lao;
- `native-layer-script-anomaly`: the native layer contains a large share of non-Latin/non-Lao letters while OCR recovers mostly Lao/Latin text, which is a signal for garbled embedded text;
- `ocr-much-richer-than-native`: OCR recovers at least about twice as much non-space text as the native layer;
- `no-large-gap-detected`: no large statistical discrepancy was detected;
- `image-no-native-layer`: direct image inputs have no PDF text layer by definition.

The classifier is a triage signal, not ground truth. It helps prioritize pages for visual review and OCR improvement.

Each page also includes an `ocr_quality.band` derived from mean OCR block confidence: `low` below 0.60, `medium` from 0.60 to below 0.80, `high` from 0.80 upward, and `no-confidence` when the engine reports no block confidence. The curated suite aggregates these bands so low-confidence real scans are easy to prioritize.

PDF pages also include `page_media` diagnostics based on embedded-image coverage:

- `full-page-raster-no-text-layer`: a raster image covers at least 90% of the page and the PDF exposes no native text;
- `full-page-raster-sparse-text-layer`: a full-page raster exists with only a tiny native layer;
- `full-page-raster-with-text-overlay`: a full-page raster exists together with a substantial text overlay;
- `partial-raster-content`: a significant embedded raster exists but does not cover the full page;
- `no-large-raster-layer`: no large embedded raster layer was detected;
- `direct-raster-image`: direct PNG/JPEG/TIFF/WebP input.

The suite aggregates these categories as `page_media_classifications`. This helps distinguish born-digital pages from scanned pages that merely happen to carry an OCR overlay.

For selected suite entries, `rotation_probe: true` runs diagnostic OCR at 0°, 90°, 180°, and 270° clockwise. The report stores only per-rotation confidence/text statistics and a conservative recommendation; it never stores rotated page images or OCR text. A non-zero orientation is recommended only when its composite confidence/letter score improves by at least 15%, mean block confidence improves by at least 0.08, and the rotated result retains enough recognized letters.

The curated suite enables the diagnostic rotation probe only for the verified World Bank/KPMG raster-overlay pages. The main OCR pipeline also has an opt-in `--auto-orient-right-angles` mode for `convert-document` and `evaluate-remote-suite`. It uses the same conservative score/confidence/character thresholds and keeps 0° when improvement is not clear. The feature remains off by default while real-suite A/B evidence is collected.

To limit cost, the opt-in production path first runs baseline OCR and skips the 90°/180°/270° probes when mean baseline line confidence is already at least 0.65. Low-confidence pages still receive the full right-angle probe.

## Not benchmark accuracy

A remote diagnostic report has:

```json
"not_benchmark_accuracy": true
```

These sources generally have no cleared project ground truth, so the runner does not calculate CER/WER and its output must not be presented as benchmark accuracy.

For publishable accuracy numbers, use the reviewed/frozen dataset workflow documented in [benchmark-freeze.md](benchmark-freeze.md) and [benchmark-source-review.md](benchmark-source-review.md).

## Exit behavior

The runner continues across per-source failures so one unavailable host does not erase diagnostics from other selected sources.

The CLI exits:

- `0` when all selected sources complete successfully;
- `1` when one or more selected sources fail, while still writing the diagnostic report.

This is useful for source-health checks without silently treating a partial run as complete.

## Curated diagnostic suite

The repository includes:

`benchmarks/remote-diagnostic-suite.json`

This is a page-pinned diagnostic manifest, not a benchmark dataset. It uses verified institutional sources only and intentionally excludes unverified sources, collection-level discovery records, and third-party mirrors/libraries.

Each suite entry pins the source-registry ID, explicit 1-based page numbers, the expected registry text-layer classification, an optional per-source download-size bound, and a short explanation of the failure mode.

Run it locally:

```bash
lao-ocr evaluate-remote-suite \
  --suite benchmarks/remote-diagnostic-suite.json \
  --registry benchmarks/source-registry.json \
  --output reports/remote-suite.json
```

Before any download, the suite validates that every source still has verified `remote-evaluation-candidate-not-approved` status, a verified page count, an in-range pinned page selection, and the expected text-layer classification. Registry drift therefore fails fast instead of silently changing the diagnostic population.

The suite report aggregates source success/error counts, sampled-page count, native vs OCR Lao-character totals, layer-gap classifications, registry text-layer categories, and OCR elapsed time.

It stores no document/OCR text and remains explicitly marked `not_benchmark_accuracy: true`.

The manual **Remote Scan Diagnostic Suite** workflow runs this fixed manifest with Tesseract Lao/English, validates it against the registry before downloading anything, verifies the same privacy boundary, uploads only the suite JSON for 14 days, and preserves partial reports before failing when one or more sources are unavailable.
