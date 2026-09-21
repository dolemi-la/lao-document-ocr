# Roadmap

## Phase 0 — baseline/local MVP

Status: complete

- image and PDF input
- Lao + English Tesseract baseline
- preprocessing
- Document AST
- DOCX/Markdown/TXT/JSON export
- FastAPI
- web upload UI
- Docker
- unit tests
- reproducible benchmark utilities

Exit condition: a clean Lao scan can be processed locally into editable outputs with no cloud service.

## Phase 1 — Lao dataset + benchmark

Status: in progress

- [x] dataset manifest format
- [x] train/dev/test split rules
- [x] document licensing/provenance metadata
- [ ] clean printed scans
- [ ] noisy scans
- [ ] phone photos
- [ ] mixed Lao/English
- [ ] multi-column documents
- [ ] simple/complex tables
- [ ] receipts/forms as separate subsets
- [x] CER/WER reporting by subset
- [ ] Tesseract baseline report

Exit condition: every model change can be measured on a fixed, legal, documented dataset.

## Phase 2 — own recognizer

Status: started

- [x] synthetic Lao text generator
- [x] font inventory and licensing review (initial Noto Lao path)
- [x] deterministic augmentation pipeline
- [ ] line recognizer training
- [ ] model export for CPU inference
- [ ] confidence calibration
- [ ] versioned weights/checksums
- [ ] regression benchmark

Exit condition: our recognizer beats the published Tesseract baseline on the fixed test set.

## Phase 3 — layout/table reconstruction

- text-region detection
- heading/body/list classification
- reading order
- table detection
- row/column/cell structure
- merged cells
- header/footer detection
- embedded image preservation
- DOCX fidelity benchmark

Exit condition: layout metrics and table metrics are published alongside OCR accuracy.

## Phase 4 — production hardening

- async jobs
- resource limits
- GPU worker option
- batch processing
- cancellation
- observability
- rate limiting for public deployments
- storage adapters
- security review
- accessibility
- localization
- optional deployment presets

The open-source core remains usable locally without authentication or billing.
