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
- [x] bounded remote-source OCR diagnostic runner with temporary downloads and metadata-only reports
- [x] curated page-pinned remote scan diagnostic suite + manual Tesseract CI workflow
- [x] remote scan script-anomaly + OCR-confidence triage diagnostics
- [x] remote PDF full-page-raster/media-layer diagnostics
- [x] diagnostic 0/90/180/270 OCR rotation probe for selected raster pages
- [x] Tesseract OSD orientation hint cross-check against exhaustive rotation diagnostics
- [x] opt-in coordinate-safe right-angle auto-orientation for document conversion + remote A/B diagnostics
- [x] no-text line-axis veto for sideways right-angle candidates, shared with remote probes
- [x] geometry-aware confidence probe skips with unchanged rotation acceptance and safe fallback
- [x] bounded small-angle deskew direction correction with signed-angle and pipeline regressions
- [x] conservative small-angle image-evidence veto for clipping/unsupported alignment; real accuracy validation remains open
- [ ] reliable text-skew proposals and text-retention safeguards — tracked in [#17](https://github.com/dolemi-la/lao-document-ocr/issues/17)
- [x] rotate preserved embedded-image payloads with auto-oriented PDF page geometry
- [x] opt-in right-angle auto-orientation exposed through sync API + async single/batch jobs
- [x] opt-in right-angle auto-orientation exposed for focused remote-source CLI/workflow diagnostics
- [x] focused exhaustive rotation probes exposed for selected remote PDF/image diagnostics
- [x] align remote rotation-probe scoring with production weighted OCR-line metrics
- [x] expose best-vs-runner-up orientation score-margin diagnostics + suite summaries
- [x] bilingual web UI opt-in control for right-angle auto-orientation
- [x] Lao-dominance probe-skip optimization + auto-orientation probe-state reporting
- [x] no-text selected-orientation review summaries in conversion JSON, async jobs and bilingual web warnings
- [x] strict manual rotation forms across sync/async/batch API + bilingual web recovery and acknowledgement checks
- [x] explicit CLI/Python per-page cardinal overrides with validation, source/embedded geometry, EXIF normalization and separate review provenance
- [x] retire production orientation-hint early exit after production-equivalent remote comparison
- [x] plain-OCR vs auto-orient suite performance mode with diagnostic probes independently switchable
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
- [x] verified one-step blind submission registration into the benchmark dataset
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

Tracking: real optical collection + registration is tracked in [#14](https://github.com/dolemi-la/lao-document-ocr/issues/14).

- [x] CER/WER reporting by subset
- [x] configurable Tesseract traineddata with model hashes
- [x] benchmark release bundle with report SHA-256/source revision
- [x] frozen test-set manifest/lock with file hashes
- [x] real-source benchmark readiness gate by coverage dimension
- [x] capture optical-evidence integrity check against digital re-encodes
- [x] benchmark command enforces frozen test-set lock
- [x] OCR reports cryptographically bound to verified frozen test-set lock
- [ ] Tesseract baseline report — tracked in [#15](https://github.com/dolemi-la/lao-document-ocr/issues/15)

Exit condition: every model change can be measured on a fixed, legal, documented dataset.

Current real-scan diagnostic warning: the [pinned Tesseract deskew A/B](tesseract-real-deskew-comparison.md)
completed seven of nine pages, with two Census TLS failures. Corrected deskew
reduced the PTC page-1 recognized count from 1,321 to 280; a no-deskew control
returned 1,390. These counts are not accuracy labels. Reliable deskew proposals
and text-retention safeguards remain unvalidated; do not treat the direction
fix or higher confidence as a full real-scan quality pass.

## Phase 2 — own recognizer

Status: started

- [x] synthetic Lao text generator
- [x] bounded HPLT v3 Lao text sampler with provenance metadata
- [x] canonical Phetsarath OT v4.103 font policy + pinned license/hash review
- [x] HarfBuzz-shaped capture rendering + strict no-tofu glyph coverage gate
- [x] normalized strict font checks on individual capture packs + suite propagation before output writes
- [x] deterministic augmentation pipeline
- [x] deterministic chunked synthetic generation for large corpora
- [x] decorrelate balanced synthetic font and capture-profile schedules
- [x] storage-efficient grayscale synthetic training images
- [x] full-canvas synthetic geometry + saturated pixel conversion with versioned dimensions
- [x] leakage-safe normalized-text-group train/dev splitting
- [x] matched training-only expansion audit with pinned images, fixed dev/vocabulary, and no-text identities
- [x] explicit fresh-evaluation audit against supplied prior manifests, pinned images, and vocabulary
- [x] clean/noisy/phone synthetic augmentation profiles
- [x] line recognizer training pipeline (CRNN + CTC)
- [x] recognizer training device selection (CPU/CUDA/Apple MPS/auto)
- [x] atomic per-epoch recognizer training state + resumable long runs
- [x] explicit weights-only new-experiment initialization with architecture/vocabulary checks and preserved parent provenance
- [x] width-aware dev decoding aligned with exported recognizer inference
- [x] normalized training/benchmark CER parity + explicit retained-state metric migration and best-epoch provenance
- [x] fixed-width train/dev padding aligned with exported bidirectional inference
- [x] upfront CTC-capacity preflight + persisted capacity diagnostics
- [x] shared line-resize planning + no-text width-cap/effective-dimension preflight
- [x] no-text per-epoch prediction-health diagnostics + all-empty development warnings
- [x] model export for CPU inference (`torch.export`)
- [x] device-portable exported recognizer LSTM state for CPU/CUDA/MPS inference
- [x] opt-in padding-invariant unidirectional CRNN v3 with v2 checkpoint compatibility
- [x] confidence calibration pipeline (held-out dev report -> calibration artifact)
- [x] optional CTC prefix beam-search decoder + decoder-specific calibration
- [x] versioned checkpoints/artifact checksums
- [x] exported-recognizer regression benchmark pipeline
- [x] normalized whole-line benchmark metrics + aggregate-only report mode excluding arbitrary metadata
- [x] fixed-set owned-vs-Tesseract benchmark comparison gate
- [x] optional character n-gram shallow fusion for CTC beam decoding
- [x] repeatable normalized held-out-text exclusions for LM training + no-text exclusion fingerprints

Development evidence: the [matched Phetsarath coverage expansion](training-expansion.md)
completed its recorded 720-update budget per arm. Final normalized dev CER was
52.44% for the 72-image baseline and 77.33% for the 288-image expansion. This is
an unfavorable synthetic-development result, not real-data validation or a model
promotion. The recognizer exit condition remains unmet.

A separately recorded [fixed-data optimization-budget study](recognizer-budget-study.md)
then continued only the 288-image expanded model to 80 total epochs / 2,880
updates. Final training CER was 0% and final normalized dev CER was 15.56%
(35/225), versus 77.33% at its earlier 720-update endpoint. This uses four times
the compute budget and the same repeatedly inspected 12-image synthetic dev set;
it does not overturn the equal-update comparison or establish optical accuracy.
The best-on-dev local export remains a development candidate, not a promoted model.

A [frozen-model synthetic challenge](recognizer-fresh-evaluation.md) subsequently
evaluated 160 new exact-label groups in 480 correlated variants. The unchanged
candidate scored 10.30% CER versus 46.41% for the older baseline, but only
123/480 lines were exact. This is restricted short-line synthetic evidence,
not document-independent optical accuracy or a production promotion.

A separately planned [coverage fine-tuning run](recognizer-coverage-finetuning.md)
completed 10 epochs / 1,440 new updates on 1,152 training images with 156 development
images. The fixed final epoch reduced matched expanded-dev CER from 16.70% to
9.60%, and the unchanged 480-image synthetic challenge from 10.30% to 6.30%.
Exact challenge lines increased from 123 to 188. Epoch 9 was best on development
(9.05%) but was not substituted into the final challenge result. Real optical
validation and production promotion remain incomplete.

A separately planned [longer-line fine-tuning run](recognizer-long-line-finetuning.md)
completed eight epochs / 2,016 updates on 2,016 training images, preserving the
156-image development set. Matched dev CER fell from 9.60% to 5.59%, and the
unchanged short challenge from 6.30% to 4.26% (248/480 exact lines). A new
144-image longer-line challenge improved from 14.67% to 7.48%, but only 18/144
lines were exact and none of its 48 phone-style lines was exact. These remain
private synthetic results; the optical benchmark and release criterion stay open.

A fixed [phone-diversity comparison](recognizer-phone-diversity-study.md) then
completed 1,260 updates each for a 2,016-image repetition control and a 2,520-image
phone-variant arm, both retaining 864 text groups. Final development CER was
5.36% versus 4.85%; the phone arm was selected before challenge scoring. Its
short/long challenge CER was 3.99%/6.51%, versus 3.98%/7.13% for the control.
Only one of 48 longer phone-style lines was exact. These remain single-seed,
previously inspected synthetic regression results, not real optical validation.

A fixed-data [input-resolution comparison](recognizer-resolution-study.md) completed
three epochs / 945 updates per arm at 48 × 768 versus 64 × 1,024. The h64 final
model was selected using development CER (4.70% versus 4.85% for its parent),
before challenge evaluation. Long-challenge CER improved from 6.51% to 5.50%,
but short-challenge CER regressed from 3.99% to 4.35%, and exact short lines fell
from 257 to 235. This is a documented tradeoff, not a universal upgrade. Both
new exports and the parent remain private; real optical release gates stay open.

A fixed-budget [short-line replay study](recognizer-short-replay-study.md) compared
ordinary h64 continuation with oversampling existing short training images,
using 1,260 updates and 7,560 presentations per arm. The development-only rule
selected control (4.04% overall dev CER); replay lost one exact short dev line.
Selected-control short-challenge CER improved from 4.35% to 3.62% and longer-line
CER from 5.50% to 5.26%, but exact longer lines fell from 19 to 18. Replay had
more exact challenge lines but was not retroactively selected. Both candidates
remain private synthetic experiments, not a release or optical-validation result.

A fixed-budget [text-coverage study](recognizer-text-coverage-study.md) compared
420 new training text groups (1,260 images) against further repetitions, with
1,260 updates and 7,560 presentations per arm. The expanded final model passed
the short-dev guard and was selected before challenge evaluation: 3.62% dev
CER versus the control's 3.81%. Short-challenge CER improved from the parent's
3.62% to 3.37% (273/480 exact), and long-challenge CER from 5.26% to 4.53%
(27/144 exact). Long phone-style exact lines were only 5/48; the control had
slightly better CER on that subset. Both exports remain private development
artifacts, and the real optical release criterion is still unmet.

A [fixed-data coverage-budget continuation](recognizer-coverage-budget-continuation.md)
then restored the expanded model's exact epoch-2 state and completed epoch 6,
adding 2,520 updates on unchanged 3,780-image training data. Matched dev CER
changed from 3.62% to 3.30%, short-challenge CER from 3.37% to 3.02%, and
long-challenge CER from 4.53% to 4.09%. Exact short challenge lines rose from
273/480 to 286/480, and exact long challenge lines from 27/144 to 29/144.
However, exact short development lines fell from 15/30 to 13/30, so the
pre-recorded safeguard retains the parent. This is additional-compute synthetic
regression evidence, not an equal-compute comparison, optical validation or
production promotion.

A [two-rate comparison](recognizer-learning-rate-progress.md) is paused after
both first epochs completed (630 updates each). Standard epoch 2 was stopped
when free storage fell to approximately 2.2 GiB, below its 4 GiB floor; its
completed epoch-1 state remains intact. Gentle epoch 2 has not started. No new
selection, challenge result, or completion is claimed, and the previously
retained text-coverage export remains selected. Resolve storage headroom before
resuming the two fixed final stages.

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
- [x] process-local expired artifact/workspace cleanup retries with bounded backoff, retained-job admission and aggregate metrics
- [x] same-job web status/download recovery with frozen corrections and stale-response guards
- [x] bounded web conversion requests including response bodies, with safe timeout recovery
- [x] upload/page/active-job concurrency limits baseline
- [x] CPU/CUDA/MPS owned-recognizer worker option + NVIDIA deployment preset
- [x] atomic bounded batch job submission baseline
- [x] cooperative job cancellation at page boundaries
- [x] owned stored-result handoff on cancellation, with coordinated best-effort deletion
- [x] ASGI lifespan-managed idle cleanup with bounded join and unchanged retry/backoff limits
- [x] closed shutdown/scheduling admission, queued-job terminal accounting, and request-owned single/batch rejection cleanup
- [x] storage readiness probes outside the shared job-state lock, with post-probe owner/result revalidation
- [x] bounded active-download leases defer expiry deletion until admitted responses close, without renewing public access
- [x] opt-in namespace-bound SQLite recovery for already-expired cleanup tasks, with commit ordering and single-owner/download-lease guards — see [durable expired cleanup](durable-expired-cleanup.md)
- [x] pre-write ownership for tracked API result publication, persisted terminal retention, and atomic expiry transfer — see [publication ownership](result-publication-ownership.md)
- [x] pre-allocation workspace ownership, cancellation-safe upload/batch aborts, and upload-owner shutdown guards — see [workspace ownership](workspace-ownership.md)
- [x] recoverable exact-key filesystem staging with bounded writer/cleanup locks and crash-retry coverage — see [filesystem staging cleanup](filesystem-staging-cleanup.md)
- [x] bounded read-only filesystem remnant inventory with private-by-default metadata, partial-scan reporting, and no ownership/deletion inference — see [inventory](filesystem-remnant-inventory.md)
- [x] bounded read-only S3 multipart inventory with explicit scope, paired pagination, private metadata and offline real-SDK coverage — see [S3 inventory](s3-multipart-inventory.md)
- [ ] full-lifecycle reconciliation for legacy random temporary files, framework spool files, S3 multipart remnants, and late provider writes
- [x] request IDs + Prometheus-format HTTP/job observability baseline
- [x] disabled-by-default per-client submission rate-limit baseline
- [x] filesystem result-storage adapter abstraction baseline
- [x] S3-compatible/object-storage result adapter
- [x] baseline security hardening/review (uploads, parser caps, headers, permissions, non-root API)
- [x] automated Python/web dependency vulnerability scanning + Dependabot
- [ ] external deployment penetration/security review — tracked in [#16](https://github.com/dolemi-la/lao-document-ocr/issues/16)
- [x] Lao/English web accessibility baseline
- [x] Lao/English web localization baseline
- [x] local/public/S3 deployment presets

The open-source core remains usable locally without authentication or billing.

## Deskew guard validation follow-up

The [image-evidence guard](deskew-evidence-guard.md) prevents the observed PTC
output collapse without additional OCR calls. The same-byte check completed
four sources / seven pinned pages; all five proposed rotations were rejected.
MAF page 1 returns fewer characters and KPMG page 21 changes its right-angle
selection. Issue #17 stays open for those reviews, missing Census sources, and
the genuine capture pilot. This is not a ground-truth accuracy pass.


### Orientation follow-up

The [line-geometry veto](orientation-line-geometry.md) addresses high-confidence
sideways OCR candidates: KPMG pages 8/15/21 now select 90 degrees in the available
same-byte diagnostic, matching public-page visual orientation. Plain OCR output
is unchanged on the seven available pages. This is not transcription-accuracy
validation. Two source failures, MAF's text-retention tradeoff and the genuine
capture pilot keep issue #17 open. That candidate-veto slice did not change
high-confidence probe skipping; the follow-up below covers that separate path.

The [geometry-aware probe-skip follow-up](orientation-probe-skip.md) makes
confident sideways baselines eligible for the existing opt-in probes. Its
project-authored Tesseract control still corrects only 6/8 known rotations,
the same as before: near-equal upright/sideways OCR scores fail the unchanged
acceptance thresholds. Two cases add three probe attempts each without a
correction. This closes the early-exit path, not the broader high-confidence
orientation-selection problem or the outstanding optical-validation gate.

### Manual correction surface

[Explicit per-page corrections](manual-page-rotation.md) now provide a CLI/Python
recovery path after visual review, without weakening automatic acceptance.
The [HTTP/web controls](manual-page-rotation-http.md) now share that contract.
[Same-job web recovery](web-job-recovery.md) retries interrupted status/download
requests without submitting a replacement OCR job or using edited correction
settings. Neither feature resolves issue #17 or the optical benchmark gates.
