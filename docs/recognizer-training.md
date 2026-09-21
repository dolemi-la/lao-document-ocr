# Lao line recognizer training

The first project-owned recognizer is a CRNN-style model:

```text
grayscale line image
  -> convolutional feature extractor
  -> bidirectional LSTM
  -> character classifier
  -> CTC loss / greedy CTC decoding
```

Model version: `crnn-ctc-v2`. The classifier starts with a negative blank-logit bias to reduce early CTC blank collapse.

This is training infrastructure, not a published high-accuracy Lao model yet.

## 1. Install training dependencies

```bash
pip install -e ".[train]"
```

Or build the dedicated CPU training container:

```bash
docker build -f Dockerfile.train -t lao-document-ocr-train .
```

The training Dockerfile installs PyTorch from the official CPU wheel index so a local CPU training image does not pull CUDA runtime packages. GPU training should use a separate CUDA-specific image/preset later.

## 2. Prepare text

```bash
lao-ocr prepare-corpus   --input source.jsonl   --format jsonl   --field text   --output training/data/lao-lines.txt   --min-lao-ratio 0.5
```

## 3. Render labeled lines

```bash
lao-ocr generate-synthetic   --corpus training/data/lao-lines.txt   --output training/generated/v1   --font /usr/share/fonts/truetype/noto/NotoSansLao-Regular.ttf   --variants-per-line 3
```

The generated manifest is `training/generated/v1/manifest.jsonl`.

## 4. Train

```bash
lao-ocr train-recognizer   --manifest training/generated/v1/manifest.jsonl   --output training/runs/crnn-v2   --epochs 20   --batch-size 32   --dev-ratio 0.1
```

The output directory contains:

- `recognizer.pt` — training checkpoint
- `metadata.json` — architecture/training metadata and checkpoint SHA-256
- `vocab.json` — exact character vocabulary

The train/dev split is deterministic from sample IDs.

## 5. Export for CPU inference

```bash
lao-ocr export-recognizer   --checkpoint training/runs/crnn-v2/recognizer.pt   --output training/runs/crnn-v2/recognizer.pt2
```

The exported `torch.export` artifact uses a fixed padded input width. The original valid line width is retained during decoding so padded pixels do not contribute CTC output.

The sidecar `recognizer.pt2.json` contains:

- model version
- vocabulary
- vocabulary checksum
- artifact SHA-256
- image dimensions
- width downsample factor

## 6. Recognize one cropped line

```bash
lao-ocr recognize-line   --model training/runs/crnn-v2/recognizer.pt2   --image line.png
```

This command is for model development. Full-page OCR still uses the existing OCR engine interface until the project-owned recognizer reaches benchmark quality and is integrated with line detection.

## CTC capacity

A target needs at least one timestep per character, plus an extra timestep where identical adjacent characters require a separating blank. Training rejects samples that cannot fit the available sequence width rather than silently producing invalid CTC targets.

## Accuracy

Do not publish results from tiny synthetic smoke runs as model accuracy. A model release is only meaningful after evaluation against the fixed real-document benchmark described in the Phase 1 roadmap.

## 7. Regression benchmark an exported recognizer

```bash
lao-ocr benchmark-recognizer \
  --manifest training/generated/v1/manifest.jsonl \
  --model training/runs/crnn-v2/recognizer.pt2 \
  --output training/runs/crnn-v2/benchmark.json
```

The report includes aggregate CER/WER plus per-sample hypotheses, timing, and the current uncalibrated probability score. Treat the probability as diagnostic only until confidence calibration is implemented.

## Development sanity result

A small local sanity run was used to verify that the training code can actually learn rather than only execute:

- 12 hand-written Lao/mixed lines
- 5 synthetic variants per line (60 images total)
- deterministic 80/20-style sample split
- 50 epochs on CPU
- model `crnn-ctc-v2`
- best dev CER observed: ~0.494
- full 60-sample manifest CER after export: ~0.514
- several generated Lao lines were recognized exactly

These figures are **not model accuracy claims**. The text set is tiny, synthetic, and contains multiple augmented variants of the same phrases. Its only purpose is a training-pipeline sanity check.

An earlier initialization collapsed entirely to CTC blank predictions even while loss decreased. `crnn-ctc-v2` therefore initializes the blank-class output bias negatively. Keep an explicit blank-collapse sanity test when changing the recognizer architecture or loss setup.
