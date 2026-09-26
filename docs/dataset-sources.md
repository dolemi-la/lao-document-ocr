# Dataset and font source review

This file tracks sources considered for benchmark data, synthetic rendering, and recognizer training.

A repository being public is not enough. We only ingest material when redistribution and the intended OCR/model-training use are clear.

## Approved for synthetic rendering

### Phetsarath OT — canonical Lao font

- Source: https://phetsarath.mts.la/
- Pinned release: Phetsarath OT v4.103
- License: SIL Open Font License 1.1
- Status: canonical font for Lao capture packs, generated DOCX defaults, and new recognizer-development rendering
- Use: official/government-style Lao document rendering and synthetic OCR data
- Reproducibility: the canonical capture workflow verifies both the v4.103 ZIP SHA-256 and PhetsarathOT-Regular.ttf SHA-256 before rendering.
- Notes: Phetsarath OT is the standard font used for Lao official documents. Canonical capture generation also requires complete glyph coverage and uses HarfBuzz-backed shaping through PyMuPDF rather than Pillow BASIC layout.

### Noto Lao — optional historical/development font

- Source: https://github.com/notofonts/lao
- License: SIL Open Font License 1.1
- Status: approved but no longer the project default
- Use: historical experiments and optional font-diversity studies only
- Notes: earlier synthetic recognizer experiments used Noto Lao variants. Keep those results labeled as historical development evidence; new canonical Lao rendering should prefer Phetsarath OT.

### Noto Sans

- Source: https://github.com/notofonts/noto-fonts
- License: SIL Open Font License 1.1
- Status: approved for synthetic Latin/digit rendering
- Use: Latin text and ASCII digits in mixed Lao/English generated samples

## Approved as OCR baseline dependency

### Tesseract Lao trained data

- Source: https://github.com/tesseract-ocr/tessdata
- License: Apache-2.0
- Status: approved as the v0.x baseline OCR model
- Use: benchmark/control model while our own Lao recognizer is developed
- Notes: this is a model dependency, not our benchmark ground truth.

## Approved source for synthetic training text

### HPLT v3 Lao (lao_Laoo)

- Source: https://hplt-project.org/datasets/v3.0
- License: CC0 for HPLT packaging; underlying extracted-text rights remain source-dependent
- Status: approved for text-only corpus preparation
- Use: diverse Lao text lines for synthetic OCR rendering and language coverage
- Notes: use bounded, provenance-recorded Lao text as model-development input only. It is not a real scanned-document OCR benchmark and must not be reported as document accuracy. Prefer `lao-ocr sample-hplt-lao` rather than downloading the full language dataset.

Example after downloading/exporting a text or JSONL slice locally:

```bash
lao-ocr prepare-corpus \
  --input /path/to/lao-source.jsonl \
  --format jsonl \
  --field text \
  --output training/data/lao-lines.txt \
  --min-lao-ratio 0.5 \
  --limit 100000
```

## Candidate training-text source — review before ingestion

### Tesseract langdata / langdata_lstm

- Sources:
  - https://github.com/tesseract-ocr/langdata
  - https://github.com/tesseract-ocr/langdata_lstm
- Repository license: Apache-2.0 for `langdata_lstm`
- Status: candidate only
- Potential use: Lao character inventory, language examples, word lists, synthetic-text generation
- Caution: Tesseract documentation notes that exact historical training reproduction may rely on commercially available fonts. We should independently review the specific Lao files and provenance before copying them into our own training corpus.

## Real scanned Lao documents

Status: project-created capture-pack workflow approved; third-party scanned-document sources remain unapproved unless independently reviewed.

The authoritative candidate registry is `benchmarks/source-registry.json`; see [benchmark-source-review.md](benchmark-source-review.md).

For the current collection procedure, see [capture-benchmark-workflow.md](capture-benchmark-workflow.md).

The real-document benchmark should prefer:

1. documents created specifically for this project and released with explicit permission;
2. public-domain Lao documents with clear provenance;
3. openly licensed documents whose terms permit redistribution and OCR/model-training use;
4. sanitized synthetic/semi-synthetic documents for categories where real public data is unavailable.

Do not add random PDFs from government, social media, cloud drives, or search results merely because they are publicly accessible.

## Contribution requirements

Every real benchmark document must record:

- source URL or contributor
- license/rights statement
- original document ID
- page IDs
- SHA-256
- train/dev/test split
- document category
- ground-truth author/reviewer
- any sanitization performed

If rights are unclear, keep the source out of the public dataset until resolved.
