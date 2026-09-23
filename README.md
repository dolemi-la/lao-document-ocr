# Lao Document OCR

Open-source, Lao-first OCR for turning scanned images and PDFs into editable documents.

The project is local-first: no account, cloud API, billing system, or external LLM is required for the v0.1 pipeline.

## v0.1

Input:

- PDF
- PNG
- JPEG
- TIFF
- WebP

Output:

- editable DOCX
- Markdown
- plain text
- structured JSON document AST

Current OCR baseline:

- Tesseract
- Lao + English (`lao+eng`)
- conservative image cleanup and deskew
- heuristic heading hierarchy and paragraph classification
- numbered and bulleted lists with clean Word/Markdown list styles
- conservative simple ruled-table reconstruction into editable Word tables
- merged cells when interior ruled borders are clearly absent
- vertical/rectangular merged cells in borderless tables when tall cell geometry clearly spans later row bands
- sparse blank cells in strongly anchored 3+ column borderless tables
- borderless aligned-text table reconstruction with stable column anchors and horizontal merged headers
- region-first text detection that isolates columns/sections before line OCR
- conservative 2–4 column reading order when clear gutters exist
- repeated multi-page headers/footers exported into real Word header/footer parts
- native PDF images/logos/figures preserved as real Word media assets (full-page scan backgrounds excluded)
- learned layout image regions can preserve project-owned illustration/photo predictions before heuristic visual fallbacks
- conservative photo-like raster regions from scanned/image inputs preserved as Word media
- line-art/diagram regions preserved as Word media after text/table/image masking

The OCR engine is behind a small interface so it can be replaced by our own Lao recognizer without changing the API or exporters.

> Accuracy claims must be backed by the benchmark suite. This project does not claim 99% accuracy.

## Quick start with Docker

```bash
docker compose up --build
```

Open:

- Web: http://localhost:5173
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

The API image installs Tesseract Lao/English data plus Noto Lao fonts for reproducible smoke tests.

## Local development

Requirements:

- Python 3.11+
- Node.js 20+
- pnpm
- Tesseract
- Lao and English Tesseract trained data

On Ubuntu 24.04:

```bash
sudo apt install tesseract-ocr tesseract-ocr-lao tesseract-ocr-eng fonts-noto-core
```

On macOS, install Tesseract with your package manager, then make sure both `lao.traineddata` and `eng.traineddata` are available in Tesseract's tessdata directory.

Install backend:

```bash
make install
make api
```

Install web app:

```bash
cd apps/web
pnpm install
pnpm dev
```

## Local CLI conversion

Convert a document without running the web app/API:

```bash
lao-ocr convert-document \
  --input scan.pdf \
  --output-dir output \
  --engine tesseract
```

Use the project-owned recognizer instead:

```bash
lao-ocr convert-document \
  --input scan.pdf \
  --output-dir output \
  --engine owned \
  --model models/recognizer.pt2 \
  --calibration models/calibration.json
```

Both modes write editable `.docx` plus `.md`, `.txt`, and structured `.json`.

## API

### Health

```http
GET /health
```

Reports whether Tesseract and the requested language data are available.

### Parse into structured JSON

```http
POST /v1/parse
Content-Type: multipart/form-data
```

Upload field: `file`.

### Observability

Every response carries an `X-Request-ID`, and Prometheus-format metrics are available at:

```text
GET /metrics
```

The endpoint exposes normalized HTTP request counters/durations plus live/completed async-job metrics without adding a mandatory telemetry SDK. See [docs/observability.md](docs/observability.md).

### Asynchronous jobs

The default web UI uses bounded background jobs for long conversions:

```text
POST   /v1/jobs
POST   /v1/jobs/batch
GET    /v1/jobs/{job_id}
DELETE /v1/jobs/{job_id}
GET    /v1/jobs/{job_id}/download
```

See [docs/async-jobs.md](docs/async-jobs.md).

### Convert

```http
POST /v1/convert
Content-Type: multipart/form-data
```

Returns a ZIP containing:

```text
document.docx
document.md
document.txt
document.json
```

Defaults:

- max file size: 25 MB
- max PDF length: 60 pages
- OCR engine: Tesseract
- OCR languages: Lao + English
- Tesseract page segmentation mode: 3

Override them with:

- `MAX_UPLOAD_BYTES`
- `MAX_PAGES`
- `MAX_PAGE_PIXELS`
- `OCR_ENGINE=tesseract|owned`
- `OCR_LANGUAGES`
- `OCR_PSM`
- `OCR_TESSDATA_DIR` (optional custom Tesseract traineddata directory)
- `OCR_MODEL_PATH` (required for `OCR_ENGINE=owned`)
- `OCR_DECODER=greedy|beam`
- `OCR_BEAM_WIDTH`
- `OCR_LANGUAGE_MODEL_PATH` (optional character n-gram model; beam decoder only)
- `OCR_LANGUAGE_MODEL_WEIGHT`
- `OCR_LANGUAGE_MODEL_TOKEN_BONUS`
- `OCR_CALIBRATION_PATH` (optional for the owned recognizer)
- `OCR_DEVICE` (`cpu`, `cuda`, `mps`, or `auto`; default `cpu`)
- `OCR_LAYOUT_DETECTOR` (`morphology` or `learned`)
- `OCR_LAYOUT_MODEL_PATH`
- `OCR_LAYOUT_CONFIDENCE`
- `OCR_READING_ORDER` (`deterministic` or `learned`)
- `OCR_READING_ORDER_MODEL_PATH`
- `OCR_READING_ORDER_MAX_BLOCKS`
- `CORS_ORIGINS`
- `JOB_MAX_WORKERS`
- `JOB_MAX_ACTIVE`
- `JOB_RETENTION_SECONDS`
- `JOB_ROOT`
- `BATCH_MAX_FILES`
- `RESULT_STORAGE_BACKEND` (`filesystem` or `s3`)
- `RESULT_STORAGE_ROOT`
- `RESULT_STORAGE_S3_BUCKET`
- `RESULT_STORAGE_S3_PREFIX`
- `RESULT_STORAGE_S3_ENDPOINT_URL`
- `RESULT_STORAGE_S3_REGION`
- `RESULT_STORAGE_S3_FORCE_PATH_STYLE`
- `RATE_LIMIT_REQUESTS` (0 disables the in-process limiter)
- `RATE_LIMIT_WINDOW_SECONDS`
- `RATE_LIMIT_MAX_CLIENTS`
- `RATE_LIMIT_TRUST_PROXY_HEADERS`
- `API_BIND_ADDRESS` / `API_PORT`
- `WEB_BIND_ADDRESS` / `WEB_PORT`
- `VITE_API_URL`

The project-owned engine is experimental and requires the optional PyTorch dependencies. See [docs/owned-ocr-engine.md](docs/owned-ocr-engine.md).

S3-compatible storage requires the optional `s3` dependency extra or `INSTALL_S3=true` for the API Docker build. See [docs/storage-adapters.md](docs/storage-adapters.md).

## Document AST

OCR output is normalized into an intermediate representation rather than writing Word files directly:

```text
Document
└── Page
    └── Block
        ├── heading
        ├── paragraph
        ├── list
        ├── table
        └── image
```

Every text block can include:

- text
- bounding box
- OCR confidence
- semantic block type
- metadata

This lets DOCX, Markdown, TXT, JSON, HTML, search indexing, and future RAG integrations share one OCR pass.

## Lao Word output

DOCX output uses:

- language metadata: `lo-LA`
- default font: `Noto Sans Lao`

The project does not bundle Phetsarath OT. If the font is legally installed on the machine opening the document, the exporter can be configured to use `Phetsarath OT` instead.

## Benchmarks

The repository has a strict JSONL dataset format, dataset validator, CER/WER benchmark runner, per-subset and multi-axis benchmark-tag reporting, split-leakage checks, structured capture templates with page-ID QR markers, and a deterministic synthetic Lao smoke benchmark.

Useful commands:

```bash
lao-ocr add-dataset-sample --help
lao-ocr validate-dataset --manifest <manifest.jsonl> --dataset-root <dataset>
lao-ocr freeze-benchmark --manifest <manifest.jsonl> --dataset-root <dataset> --output-manifest frozen.jsonl --output-lock benchmark.lock.json
lao-ocr verify-benchmark-freeze --lock benchmark.lock.json --dataset-root <dataset>
lao-ocr dataset-report --manifest <manifest.jsonl> --dataset-root <dataset> --output dataset-report.json
lao-ocr build-review-queue --manifest <manifest.jsonl> --dataset-root <dataset> --output-dir /tmp/review
lao-ocr apply-review-decisions --manifest <manifest.jsonl> --dataset-root <dataset> --decisions /tmp/review/review-decisions.csv
lao-ocr review-dataset-sample --manifest <manifest.jsonl> --dataset-root <dataset> --id <sample-id> --status approved --reviewer <alias>
lao-ocr benchmark-readiness --manifest <manifest.jsonl> --dataset-root <dataset> --output readiness.json
lao-ocr prepare-layout-training-manifest --manifest <manifest.jsonl> --dataset-root <dataset> --output training/layout.jsonl
lao-ocr prepare-layout-targets --training-manifest training/layout.jsonl --dataset-root <dataset> --output training/layout-targets
lao-ocr train-layout-detector --targets-manifest training/layout-targets/targets.jsonl --dataset-root <dataset> --output training/layout/runs/tiny-unet-v1 --device auto
lao-ocr export-layout-detector --checkpoint training/layout/runs/tiny-unet-v1/layout-detector.pt --output training/layout/runs/tiny-unet-v1/layout-detector.pt2
lao-ocr train-reading-order --training-manifest training/layout.jsonl --dataset-root <dataset> --output training/layout/runs/reading-order-v1 --device auto
lao-ocr export-reading-order --checkpoint training/layout/runs/reading-order-v1/reading-order.pt --output training/layout/runs/reading-order-v1/reading-order.pt2
lao-ocr benchmark --manifest <manifest.jsonl> --dataset-root <dataset> --output report.json
# Optional: add --tessdata-dir /path/to/tessdata for hashed custom weights
lao-ocr compare-benchmarks --baseline baseline.json --candidate candidate.json --output comparison.json
lao-ocr benchmark-layout --reference reference.json --prediction prediction.json --output layout-report.json
lao-ocr benchmark-docx --reference reference.pdf --docx output.docx --output docx-fidelity.json
lao-ocr bundle-benchmarks --ocr ocr.json --layout layout.json --docx docx.json --revision <git-sha> --output bundle.json
lao-ocr prepare-corpus --input <source.txt> --output training/data/lao-lines.txt
lao-ocr sample-hplt-lao --output training/data/hplt-lao-10k.txt --metadata training/data/hplt-lao-10k.meta.json --limit 10000
lao-ocr generate-synthetic --corpus training/data/lao-lines.txt --output training/generated/v1 --font <font.ttf> --augmentation-profile balanced
lao-ocr train-char-lm --corpus training/data/lao-lines.txt --vocabulary training/runs/crnn-v2/vocab.json --output training/runs/crnn-v2/char-lm.json
lao-ocr generate-capture-pack --corpus training/data/lao-lines.txt --output benchmarks/capture-packs/baseline-v1 --font <font.ttf> --pack-id baseline-v1 --text-license CC0-1.0 --text-provenance <source> --template plain
lao-ocr generate-capture-suite --corpus training/data/lao-lines.txt --output benchmarks/capture-packs/baseline-suite-v1 --font <font.ttf> --suite-id baseline-suite-v1 --text-license CC0-1.0 --text-provenance <source>
lao-ocr build-capture-kit --suite-manifest benchmarks/capture-packs/baseline-suite-v1/capture-suite.json --output baseline-suite-v1.collector.zip --revision <git-sha>
lao-ocr benchmark-capture-suite --suite-manifest <capture-suite.json> --engine tesseract --output digital-qa.json
lao-ocr capture-campaign-report --suite-manifest <capture-suite.json> --dataset-manifest <manifest.jsonl> --output campaign-report.json
lao-ocr register-capture --help
lao-ocr register-capture-directory --help
lao-ocr train-recognizer --manifest training/generated/v1/manifest.jsonl --output training/runs/crnn-v2
lao-ocr export-recognizer --checkpoint training/runs/crnn-v2/recognizer.pt --output training/runs/crnn-v2/recognizer.pt2
lao-ocr benchmark-recognizer --manifest training/generated/v1/manifest.jsonl --model training/runs/crnn-v2/recognizer.pt2 --output training/runs/crnn-v2/benchmark.json
lao-ocr calibrate-recognizer --report <dev-report.json> --output <calibration.json>
```

See:

- [benchmarks/README.md](benchmarks/README.md)
- [docs/dataset-format.md](docs/dataset-format.md)
- [docs/dataset-sources.md](docs/dataset-sources.md)
- [docs/dataset-intake.md](docs/dataset-intake.md)
- [docs/dataset-review.md](docs/dataset-review.md)
- [docs/review-queue.md](docs/review-queue.md)
- [docs/owned-ocr-engine.md](docs/owned-ocr-engine.md)
- [docs/layout-benchmark.md](docs/layout-benchmark.md)
- [docs/docx-fidelity-benchmark.md](docs/docx-fidelity-benchmark.md)
- [docs/benchmark-bundles.md](docs/benchmark-bundles.md)
- [docs/benchmark-comparison.md](docs/benchmark-comparison.md)
- [docs/tesseract-models.md](docs/tesseract-models.md)
- [docs/benchmark-freeze.md](docs/benchmark-freeze.md)
- [docs/benchmark-readiness.md](docs/benchmark-readiness.md)
- [docs/benchmark-source-review.md](docs/benchmark-source-review.md)
- [docs/hplt-sampling.md](docs/hplt-sampling.md)
- [docs/capture-benchmark-workflow.md](docs/capture-benchmark-workflow.md)
- [docs/capture-suite.md](docs/capture-suite.md)
- [docs/capture-kit.md](docs/capture-kit.md)

Maintainers can also run the manual GitHub Actions workflow **Build Collector Capture Kit** to generate a revision-bound collector ZIP + SHA-256 artifact without committing generated benchmark files.
- [docs/capture-suite-qa.md](docs/capture-suite-qa.md)
- [docs/project-authored-capture-campaign.md](docs/project-authored-capture-campaign.md)
- [docs/bulk-capture-registration.md](docs/bulk-capture-registration.md)
- [docs/capture-campaign-report.md](docs/capture-campaign-report.md)
- [docs/synthetic-augmentation.md](docs/synthetic-augmentation.md)
- [docs/language-model.md](docs/language-model.md)

Synthetic smoke numbers are pipeline sanity checks only and must not be presented as real-document accuracy.

HPLT bounded streaming is optional and requires:

```bash
pip install -e '.[data]'
```

See [docs/hplt-sampling.md](docs/hplt-sampling.md) for provenance/licensing caveats.


## Architecture

See [docs/architecture.md](docs/architecture.md).

## Roadmap

See [docs/roadmap.md](docs/roadmap.md).

Recognizer development includes an optional CTC prefix beam-search decoder; greedy remains the default.

Recognizer development: [docs/recognizer-training.md](docs/recognizer-training.md).

Layout training data: [docs/layout-training-data.md](docs/layout-training-data.md).

High-level direction:

1. reproducible local baseline
2. Lao OCR dataset and benchmark
3. own Lao recognizer weights
4. layout and table reconstruction
5. production hardening

## Privacy

The default Docker deployment processes documents locally. There is no telemetry or external OCR API in v0.1.

Do not commit private scanned documents to this repository.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0.

Learned layout detector: [docs/layout-model-training.md](docs/layout-model-training.md).

Learned reading order: [docs/reading-order-model.md](docs/reading-order-model.md).
