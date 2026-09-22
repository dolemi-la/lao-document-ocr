# Learned reading-order model

The project includes an experimental project-owned reading-order model trained directly from reviewed document AST order.

It is an **opt-in baseline**. The deterministic 2–4-column resolver remains the default until the learned model is trained on a meaningful reviewed dataset and beats the frozen layout benchmark.

## Model

Current model version:

```text
pairwise-reading-order-v1
```

Architecture:

```text
block A geometry/type
block B geometry/type
relative pair geometry
        ↓
small MLP
        ↓
P(A appears before B)
```

Each block feature includes normalized:

- bounding box edges
- center
- width/height
- area
- semantic block-type one-hot
- header/footer role flags

Each pair also includes relative center/size differences and horizontal/vertical overlap.

Training creates both directions for every reviewed block pair:

```text
(A, B) -> 1
(B, A) -> 0
```

so the binary task stays balanced.

## Training data

The model consumes the reviewed layout-training manifest generated from the public dataset:

```bash
lao-ocr prepare-layout-training-manifest \
  --manifest benchmarks/public/manifest.jsonl \
  --dataset-root benchmarks/public \
  --output training/layout/layout-training.jsonl
```

The order of blocks in each reviewed one-page AST is the reading-order ground truth.

Only blocks with bounding boxes are used for pair generation.

## Train

```bash
lao-ocr train-reading-order \
  --training-manifest training/layout/layout-training.jsonl \
  --dataset-root benchmarks/public \
  --output training/layout/runs/reading-order-v1 \
  --epochs 20 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --hidden-size 64 \
  --device auto
```

Supported devices:

- `cpu`
- `cuda`
- `mps`
- `auto`

Training records:

- train loss
- dev pair accuracy
- dev loss
- train/dev page counts
- train/dev pair counts
- checkpoint SHA-256

The best checkpoint is selected by **dev pair accuracy**.

## Export

```bash
lao-ocr export-reading-order \
  --checkpoint training/layout/runs/reading-order-v1/reading-order.pt \
  --output training/layout/runs/reading-order-v1/reading-order.pt2
```

The `torch.export` artifact uses a fixed 512-pair inference batch.

Runtime prediction pads/chunks pair features into that shape. This avoids thousands of tiny model calls while staying compatible with PyTorch builds that specialize exported batch dimensions.

The sidecar metadata records:

- model version
- feature version
- model config
- fixed inference batch size
- artifact SHA-256

Tampered artifacts are rejected on load.

## Runtime ordering

For each unordered block pair `A/B`, runtime evaluates both directions:

```text
P(A before B)
P(B before A)
```

and combines them:

```text
pair_score(A before B) =
    (P(A before B) + (1 - P(B before A))) / 2
```

Each block accumulates expected pairwise wins. Blocks are then sorted by descending score with original position as a deterministic tie-break.

This is a baseline ranking method, not a claim of globally cycle-free pairwise reasoning.

## Safety / resource bound

Pairwise ordering is O(n²).

The learned resolver therefore has a block cap:

```text
256 blocks
```

by default.

If a page exceeds the cap, the pipeline falls back to deterministic ordering.

CLI override:

```bash
--reading-order-max-blocks 256
```

API setting:

```text
OCR_READING_ORDER_MAX_BLOCKS=256
```

## Use in local conversion

Default deterministic order:

```bash
lao-ocr convert-document \
  --input scan.pdf \
  --output-dir output \
  --reading-order deterministic
```

Learned order:

```bash
lao-ocr convert-document \
  --input scan.pdf \
  --output-dir output \
  --reading-order learned \
  --reading-order-model reading-order.pt2 \
  --reading-order-max-blocks 256 \
  --device cpu
```

The learned ordering model can be used with either:

- Tesseract OCR
- the project-owned recognizer

because it operates on final document blocks rather than character recognition.

## API configuration

```text
OCR_READING_ORDER=deterministic
```

or:

```text
OCR_READING_ORDER=learned
OCR_READING_ORDER_MODEL_PATH=/models/reading-order.pt2
OCR_READING_ORDER_MAX_BLOCKS=256
OCR_DEVICE=auto
```

The API health endpoint validates the selected resolver and exposes its metadata.

## Document metadata

Structured JSON output records the ordering strategy:

```json
{
  "metadata": {
    "reading_order": {
      "name": "ExportedReadingOrderResolver",
      "version": "pairwise-reading-order-v1",
      "feature_version": "reading-order-pair-v1",
      "device": "cpu",
      "max_blocks": 256,
      "inference_batch_size": 512
    }
  }
}
```

Deterministic output records:

```json
{
  "name": "DeterministicReadingOrderResolver",
  "version": "multi-column-v2"
}
```

## Current quality status

The train/export/inference pipeline is functional.

A tiny smoke run with one train page and one dev page produced:

```text
best dev pair accuracy: 0.50
```

and an incorrect page order.

That is expected from a one-page/one-epoch sanity fixture and is **not a model-quality result**.

Do not make the learned resolver the default or publish an accuracy claim until it beats the deterministic baseline on held-out reviewed documents.

## Benchmark gate

The important external metric remains **reading-order accuracy** from:

```bash
lao-ocr benchmark-layout ...
```

The learned resolver should also be checked against:

- block F1/IoU
- semantic block accuracy
- table structure metrics
- DOCX visual fidelity

because an ordering model should not be evaluated in isolation from the complete document pipeline.

## Future improvements

Potential next steps:

- page-context encoder rather than independent block pairs
- transformer/listwise ranking model
- explicit column/group features
- cycle-aware ranking loss
- hard-negative pair sampling
- direct joint layout + reading-order model
