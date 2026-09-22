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
- [x] rights-clear printable capture-pack + real-capture registration workflow
- [x] benchmark source review registry/policy
- [x] dataset coverage/integrity report + duplicate-image detection
- [x] multi-axis benchmark tags and per-tag CER/WER
- [x] structured printable capture templates for multi-column/table/receipt/form collection
- [x] combined multi-template capture suite generator
- [x] capture campaign completion report by template/device mode
- [ ] clean printed scans
- [ ] noisy scans
- [ ] phone photos
- [ ] mixed Lao/English
- [ ] multi-column documents
- [ ] simple/complex tables
- [ ] receipts/forms as separate subsets
- [x] CER/WER reporting by subset
- [x] benchmark release bundle with report SHA-256/source revision
- [ ] Tesseract baseline report

Exit condition: every model change can be measured on a fixed, legal, documented dataset.

## Phase 2 — own recognizer

Status: started

- [x] synthetic Lao text generator
- [x] font inventory and licensing review (initial Noto Lao path)
- [x] deterministic augmentation pipeline
- [x] line recognizer training pipeline (CRNN + CTC)
- [x] model export for CPU inference (`torch.export`)
- [x] confidence calibration pipeline (held-out dev report -> calibration artifact)
- [x] versioned checkpoints/artifact checksums
- [x] exported-recognizer regression benchmark pipeline

Exit condition: our recognizer beats the published Tesseract baseline on the fixed test set.

## Phase 3 — layout/table reconstruction

Status: started

- [x] deterministic clean-scan text-line detection baseline
- [x] general deterministic text-region detector interface/baseline
- [ ] learned text-region detector
- [x] heuristic heading hierarchy + body/list classification baseline
- [x] conservative 2–4 column reading-order baseline
- [ ] learned/general reading-order model
- [x] simple ruled-table detection baseline
- [x] row/column/cell structure for clear ruled grids
- [x] conservative aligned-text borderless-table baseline
- [ ] general/complex borderless table detection
- [x] merged-cell baseline for clear ruled rectangular spans
- [ ] merged cells in borderless/complex tables
- [x] repeated header/footer detection baseline and DOCX preservation
- [x] native PDF embedded-image preservation baseline (excluding full-page scans)
- [x] conservative photo-like raster-region detection for image/scanned inputs
- [x] deterministic line-art/diagram region preservation baseline
- [ ] learned/general diagram/illustration segmentation
- [x] layout/table AST benchmark metrics pipeline
- [x] optional DOCX visual fidelity benchmark pipeline (LibreOffice renderer)

Exit condition: layout metrics and table metrics are published alongside OCR accuracy.

## Phase 4 — production hardening

- [x] bounded local async conversion jobs
- [x] upload/page/active-job concurrency limits baseline
- [x] CPU/CUDA/MPS owned-recognizer worker option + NVIDIA deployment preset
- [x] atomic bounded batch job submission baseline
- [x] cooperative job cancellation at page boundaries
- [x] request IDs + Prometheus-format HTTP/job observability baseline
- [x] disabled-by-default per-client submission rate-limit baseline
- [x] filesystem result-storage adapter abstraction baseline
- [x] S3-compatible/object-storage result adapter
- [x] baseline security hardening/review (uploads, parser caps, headers, permissions, non-root API)
- [ ] external deployment penetration/security review
- [x] Lao/English web accessibility baseline
- [x] Lao/English web localization baseline
- [x] local/public/S3 deployment presets

The open-source core remains usable locally without authentication or billing.
