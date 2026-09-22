# Layout training data

The OCR text benchmark and layout benchmark now share one dataset manifest.

A sample can optionally include:

```json
{
  "layout_ground_truth": "layout-ground-truth/clean-print/page-001.json"
}
```

The referenced file is a reviewed one-page Document AST.

## Prepare a training manifest

Build a compact JSONL manifest from only layout-labeled samples:

```bash
lao-ocr prepare-layout-training-manifest \
  --manifest benchmarks/public/manifest.jsonl \
  --dataset-root benchmarks/public \
  --output training/layout/layout-training.jsonl
```

Filter to one or more splits:

```bash
lao-ocr prepare-layout-training-manifest \
  --manifest benchmarks/public/manifest.jsonl \
  --dataset-root benchmarks/public \
  --output training/layout/train-dev.jsonl \
  --split train \
  --split dev
```

Before writing output, the command runs the normal dataset validator. It refuses to build training input from a dataset with:

- split leakage
- duplicate source images
- missing files
- hash mismatches
- invalid UTF-8 ground truth
- invalid layout ASTs
- layout/image dimension mismatches
- invalid table spans/overlaps

## Entry format

Example JSONL entry:

```json
{
  "id": "page-001",
  "document_id": "document-001",
  "image": "data/phone-photo/page-001.jpg",
  "layout_ground_truth": "layout-ground-truth/phone-photo/page-001.json",
  "split": "train",
  "subset": "phone-photo",
  "tags": [
    "capture:phone-photo",
    "layout:multi-column"
  ],
  "sha256": "...",
  "width": 1200,
  "height": 1600,
  "block_counts": {
    "heading": 1,
    "paragraph": 8,
    "table": 1
  }
}
```

Entries are sorted by sample ID for reproducibility.

## Why keep the AST path?

The training manifest does not flatten the layout labels into one model-specific target format.

The reviewed AST remains the source of truth and can feed multiple future tasks:

- text-region detection
- semantic block classification
- reading order
- table region/structure detection
- image/illustration segmentation

Model-specific target generation should derive from the AST rather than creating competing ground-truth formats.

## Review policy

Only reviewed layout labels should be used for training/evaluation.

Do not treat raw model predictions as ground truth without human review.

Useful review checks include:

- every semantic block has a bbox
- bbox is tight enough to represent the intended region
- reading order matches the intended document order
- table row/column metadata is correct
- merged-cell spans are correct
- image/diagram regions are labeled deliberately
- page dimensions exactly match the source image

## Next model step

The next learned-layout stage can consume this JSONL manifest and derive model targets such as:

- categorical region masks
- box/class targets
- reading-order pairs
- table-cell graphs

The reviewed AST should remain unchanged when experimenting with different model architectures.

## Generate model targets

Convert the reviewed AST labels into model-ready categorical masks and box/class targets:

```bash
lao-ocr prepare-layout-targets \
  --training-manifest training/layout/layout-training.jsonl \
  --dataset-root benchmarks/public \
  --output training/layout/targets
```

Output:

```text
targets/
├── classes.json
├── targets.jsonl
├── masks/
│   └── <sample-id>.png
└── boxes/
    └── <sample-id>.json
```

The fixed class map is:

```json
{
  "background": 0,
  "heading": 1,
  "paragraph": 2,
  "list": 3,
  "table": 4,
  "image": 5
}
```

Masks use the original page dimensions and fill each reviewed block bbox with its semantic class ID. The box JSON preserves class, bbox, and AST reading-order index.

If two different semantic classes overlap, target generation fails rather than silently choosing one label. This is an annotation-quality signal that should be resolved in the reviewed AST.

The target step also reopens the source image and checks its dimensions against the layout annotation, protecting against stale/moved manifests.
