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
- heuristic heading/list reconstruction
- conservative simple ruled-table reconstruction into editable Word tables
- merged cells when interior ruled borders are clearly absent
- conservative borderless aligned-text table reconstruction for strong row/column geometry
- conservative two-column reading order when a clear gutter exists
- repeated multi-page headers/footers exported into real Word header/footer parts
- native PDF images/logos/figures preserved as real Word media assets (full-page scan backgrounds excluded)
- conservative photo-like raster regions from scanned/image inputs preserved as Word media

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
- `OCR_ENGINE=tesseract|owned`
- `OCR_LANGUAGES`
- `OCR_PSM`
- `OCR_MODEL_PATH` (required for `OCR_ENGINE=owned`)
- `OCR_CALIBRATION_PATH` (optional for the owned recognizer)
- `CORS_ORIGINS`

The project-owned engine is experimental and requires the optional PyTorch dependencies. See [docs/owned-ocr-engine.md](docs/owned-ocr-engine.md).

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

The repository has a strict JSONL dataset format, dataset validator, CER/WER benchmark runner, per-subset and multi-axis benchmark-tag reporting, split-leakage checks, structured capture templates, and a deterministic synthetic Lao smoke benchmark.

Useful commands:

```bash
lao-ocr add-dataset-sample --help
lao-ocr validate-dataset --manifest <manifest.jsonl> --dataset-root <dataset>
lao-ocr dataset-report --manifest <manifest.jsonl> --dataset-root <dataset> --output dataset-report.json
lao-ocr benchmark --manifest <manifest.jsonl> --dataset-root <dataset> --output report.json
lao-ocr benchmark-layout --reference reference.json --prediction prediction.json --output layout-report.json
lao-ocr benchmark-docx --reference reference.pdf --docx output.docx --output docx-fidelity.json
lao-ocr bundle-benchmarks --ocr ocr.json --layout layout.json --docx docx.json --revision <git-sha> --output bundle.json
lao-ocr prepare-corpus --input <source.txt> --output training/data/lao-lines.txt
lao-ocr generate-synthetic --corpus training/data/lao-lines.txt --output training/generated/v1 --font <font.ttf>
lao-ocr generate-capture-pack --corpus training/data/lao-lines.txt --output benchmarks/capture-packs/baseline-v1 --font <font.ttf> --pack-id baseline-v1 --text-license CC0-1.0 --text-provenance <source> --template plain
lao-ocr generate-capture-suite --corpus training/data/lao-lines.txt --output benchmarks/capture-packs/baseline-suite-v1 --font <font.ttf> --suite-id baseline-suite-v1 --text-license CC0-1.0 --text-provenance <source>
lao-ocr capture-campaign-report --suite-manifest <capture-suite.json> --dataset-manifest <manifest.jsonl> --output campaign-report.json
lao-ocr register-capture --help
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
- [docs/owned-ocr-engine.md](docs/owned-ocr-engine.md)
- [docs/layout-benchmark.md](docs/layout-benchmark.md)
- [docs/docx-fidelity-benchmark.md](docs/docx-fidelity-benchmark.md)
- [docs/benchmark-bundles.md](docs/benchmark-bundles.md)
- [docs/benchmark-source-review.md](docs/benchmark-source-review.md)
- [docs/capture-benchmark-workflow.md](docs/capture-benchmark-workflow.md)
- [docs/capture-suite.md](docs/capture-suite.md)
- [docs/capture-campaign-report.md](docs/capture-campaign-report.md)

Synthetic smoke numbers are pipeline sanity checks only and must not be presented as real-document accuracy.

## Architecture

See [docs/architecture.md](docs/architecture.md).

## Roadmap

See [docs/roadmap.md](docs/roadmap.md).

Recognizer development: [docs/recognizer-training.md](docs/recognizer-training.md).

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
