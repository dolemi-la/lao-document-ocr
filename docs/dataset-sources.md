# Dataset and font source review

This file tracks sources considered for benchmark data, synthetic rendering, and recognizer training.

A repository being public is not enough. We only ingest material when redistribution and the intended OCR/model-training use are clear.

## Approved for synthetic rendering

### Noto Lao

- Source: https://github.com/notofonts/lao
- License: SIL Open Font License 1.1
- Status: approved for synthetic rendering
- Use: Lao glyph rendering for generated OCR fixtures/training data
- Notes: the project does not need to vendor the font binary; Debian's `fonts-noto-core` package provides Noto Sans Lao and Noto Looped Lao.

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

Status: not yet approved.

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
