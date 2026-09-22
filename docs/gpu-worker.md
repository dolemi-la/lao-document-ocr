# GPU / accelerator owned-recognizer inference

The project-owned CRNN recognizer can run on:

- CPU
- NVIDIA CUDA
- Apple Metal (MPS) for local non-Docker use
- automatic device selection

Tesseract remains CPU-based.

## Runtime device

The API setting is:

```text
OCR_DEVICE=cpu
```

Supported values:

```text
cpu
cuda
mps
auto
```

CPU is the default so existing deployments stay deterministic.

`auto` chooses:

1. CUDA when available
2. MPS when available
3. CPU otherwise

An explicit unavailable accelerator fails clearly instead of silently falling back.

## Model reuse

The API caches the owned recognizer instance by:

- model path
- calibration path
- device

The exported model is loaded/moved onto the selected accelerator once per API process instead of once per OCR job.

This matters for GPU deployments because repeatedly loading the model can waste time and VRAM.

## CLI

Recognize one line on CUDA:

```bash
lao-ocr recognize-line \
  --model recognizer.pt2 \
  --image line.png \
  --device cuda
```

Benchmark on an accelerator:

```bash
lao-ocr benchmark-recognizer \
  --manifest dev/manifest.jsonl \
  --model recognizer.pt2 \
  --output report.json \
  --device cuda
```

Full document conversion with the owned recognizer:

```bash
lao-ocr convert-document \
  --input scan.pdf \
  --output-dir output \
  --engine owned \
  --model recognizer.pt2 \
  --device cuda
```

For Apple Silicon local Python environments, use `--device mps` when the installed PyTorch build supports MPS.

## NVIDIA Docker preset

Requirements:

- NVIDIA GPU + compatible driver
- NVIDIA Container Toolkit
- Docker/Compose GPU support
- exported `recognizer.pt2` and its sidecar JSON in a model directory

Example model directory:

```text
models/
├── recognizer.pt2
├── recognizer.pt2.json
└── calibration.json   # optional
```

Run:

```bash
docker compose \
  --env-file deploy/presets/gpu.env.example \
  -f docker-compose.yml \
  -f deploy/compose.public.yml \
  -f deploy/compose.gpu.yml \
  up --build -d
```

The GPU overlay:

- uses `Dockerfile.owned-api`
- requests all NVIDIA GPUs from Docker
- sets `OCR_ENGINE=owned`
- sets `OCR_DEVICE=cuda`
- mounts the model directory read-only at `/models`
- defaults to one OCR worker and four active jobs

One worker is conservative because concurrent OCR jobs can multiply GPU memory usage.

Tune only after measuring model/VRAM behavior.

## PyTorch wheel source

`Dockerfile.owned-api` accepts:

```text
TORCH_INDEX_URL
```

If empty, PyTorch is installed from normal PyPI.

If your deployment requires a specific CUDA wheel, set the official PyTorch wheel index URL appropriate for that environment.

The repository intentionally does not hardcode one CUDA version because supported wheel/driver combinations change over time.

## Combine with S3 storage

The GPU overlay can be combined with:

```text
deploy/compose.s3.yml
```

For example:

```bash
docker compose \
  --env-file deploy/presets/gpu.env.example \
  -f docker-compose.yml \
  -f deploy/compose.public.yml \
  -f deploy/compose.gpu.yml \
  -f deploy/compose.s3.yml \
  up --build -d
```

Set the S3 environment variables separately.

## Current performance caveat

The current owned recognizer is still experimental and has not yet beaten a real-document benchmark baseline.

GPU support improves inference throughput/latency; it does not turn an undertrained model into a production-quality OCR model.
