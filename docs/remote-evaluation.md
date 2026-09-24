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
