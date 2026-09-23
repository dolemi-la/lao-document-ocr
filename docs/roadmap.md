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
- [x] bounded official HPLT Lao streaming sampler + provenance metadata
- [x] optional reviewed layout AST labels in dataset manifests + coverage reporting
- [x] reproducible layout-training manifest from reviewed AST labels
- [x] categorical mask + ordered box targets from reviewed layout ASTs
- [x] dataset coverage/integrity report + duplicate-image detection
- [x] multi-axis benchmark tags and per-tag CER/WER
- [x] structured printable capture templates for multi-column/table/receipt/form collection
- [x] combined multi-template capture suite generator
- [x] collector-safe reproducible capture-kit packaging
- [x] mobile/browser collector service for blind capture kits
- [x] checksum-bound blind capture submission ZIP handoff
- [x] byte-reproducible combined capture PDF + collector ZIP generation
- [x] manual CI workflow for revision-bound collector ZIP + SHA-256 artifact
- [x] project-authored rights-clear Lao capture corpus + printable 60-page campaign
- [x] capture campaign completion report by template/device mode
- [x] atomic bulk capture-directory registration workflow
- [x] machine-readable page-ID QR markers for capture-suite integrity
- [x] digital capture-suite OCR QA benchmark (explicitly non-real)
- [x] real-data benchmark readiness coverage gate
- [x] machine-readable manual review approval gate before public freeze
- [x] local visual review queue for benchmark capture approval
- [x] transactional batch review decision sheet workflow
- [ ] clean printed scans
- [ ] noisy scans
- [ ] phone photos
- [ ] mixed Lao/English
- [ ] multi-column documents
- [ ] simple/complex tables
- [ ] receipts/forms as separate subsets
- [x] CER/WER reporting by subset
- [x] configurable Tesseract traineddata with model hashes
- [x] benchmark release bundle with report SHA-256/source revision
- [x] frozen test-set manifest/lock with file hashes
- [x] real-source benchmark readiness gate by coverage dimension
- [x] capture optical-evidence integrity check against digital re-encodes
- [x] benchmark command enforces frozen test-set lock
- [x] OCR reports cryptographically bound to verified frozen test-set lock
- [ ] Tesseract baseline report

Exit condition: every model change can be measured on a fixed, legal, documented dataset.

## Phase 2 — own recognizer

Status: started

- [x] synthetic Lao text generator
- [x] bounded HPLT v3 Lao text sampler with provenance metadata
- [x] font inventory and licensing review (initial Noto Lao path)
- [x] deterministic augmentation pipeline
- [x] leakage-safe normalized-text-group train/dev splitting
- [x] clean/noisy/phone synthetic augmentation profiles
- [x] line recognizer training pipeline (CRNN + CTC)
- [x] model export for CPU inference (`torch.export`)
- [x] confidence calibration pipeline (held-out dev report -> calibration artifact)
- [x] optional CTC prefix beam-search decoder + decoder-specific calibration
- [x] versioned checkpoints/artifact checksums
- [x] exported-recognizer regression benchmark pipeline
- [x] fixed-set owned-vs-Tesseract benchmark comparison gate
- [x] optional character n-gram shallow fusion for CTC beam decoding

Exit condition: our recognizer beats the published Tesseract baseline on the fixed test set.

## Phase 3 — layout/table reconstruction

Status: started

- [x] deterministic clean-scan text-line detection baseline
- [x] general deterministic text-region detector interface/baseline
- [x] learned text-region detector training/export/inference pipeline (quality not yet benchmark-ready)
- [x] learned semantic-region hints propagated into owned OCR/AST blocks
- [x] heuristic heading hierarchy + body/list classification baseline
- [x] conservative 2–4 column reading-order baseline
- [x] learned pairwise reading-order train/export/inference pipeline (quality not yet benchmark-ready)
- [x] simple ruled-table detection baseline
- [x] row/column/cell structure for clear ruled grids
- [x] aligned-text borderless-table v2 baseline with stable anchors
- [x] complex aligned borderless-table baseline (spans + sparse blank cells)
- [x] learned layout table-region fallback for sparse/unstructured borderless tables
- [x] merged-cell baseline for clear ruled rectangular spans
- [x] horizontal merged-cell baseline for aligned borderless tables
- [x] vertical + rectangular merged-cell baseline for strongly aligned borderless tables
- [x] repeated header/footer detection baseline and DOCX preservation
- [x] native PDF embedded-image preservation baseline (excluding full-page scans)
- [x] conservative photo-like raster-region detection for image/scanned inputs
- [x] deterministic line-art/diagram region preservation baseline
- [x] learned image/illustration segmentation pipeline via layout image class (quality not yet benchmark-ready)
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
- [x] automated Python/web dependency vulnerability scanning + Dependabot
- [ ] external deployment penetration/security review
- [x] Lao/English web accessibility baseline
- [x] Lao/English web localization baseline
- [x] local/public/S3 deployment presets

The open-source core remains usable locally without authentication or billing.
