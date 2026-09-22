# Learned layout detector

The project now includes an experimental project-owned semantic layout segmentation pipeline.

It is designed to replace the deterministic morphology text-region detector over time, but **morphology remains the default** until the learned model is trained on a meaningful reviewed dataset and beats the fixed layout benchmark.

## Model

Current architecture:

```text
RGB page image
  -> Tiny U-Net encoder/decoder
  -> per-pixel semantic class logits
```

Model version:

```text
tiny-layout-unet-v1
```

Classes:

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

At runtime, heading/paragraph/list/table classes are converted into **separate class-aware regions** rather than one generic text mask. Region-aware line detection preserves the region ID, local line order, and semantic hint into `RecognizedLine`.

Document structure then uses those hints conservatively:

- learned heading hints can promote otherwise-normal text to a heading while heuristic height still selects heading level;
- learned paragraph hints are recorded without demoting strong heuristic headings;
- learned list hints create an unresolved list only when explicit OCR list markers are absent; exporters do not invent bullets/numbers for unresolved lists;
- learned table hints create an unresolved table region only when ruled/borderless geometry has not already reconstructed real cells.

Strong explicit list markers and geometric table reconstruction remain authoritative.

## Prerequisites

Install the optional training dependencies:

```bash
pip install -e ".[train]"
```

Prepare reviewed layout labels first:

1. public dataset manifest
2. reviewed one-page AST layout ground truth
3. layout training manifest
4. categorical masks/box targets

See [layout-training-data.md](layout-training-data.md).

## Train

```bash
lao-ocr train-layout-detector \
  --targets-manifest training/layout/targets/targets.jsonl \
  --dataset-root benchmarks/public \
  --output training/layout/runs/tiny-unet-v1 \
  --epochs 20 \
  --batch-size 4 \
  --learning-rate 0.001 \
  --image-height 256 \
  --image-width 256 \
  --base-channels 32 \
  --device auto
```

Supported devices:

- `cpu`
- `cuda`
- `mps`
- `auto`

Training uses:

- deterministic random seed
- train/dev split from reviewed target manifests
- inverse-frequency class weights with bounded scaling
- cross-entropy segmentation loss
- pixel accuracy
- per-class IoU
- mean IoU
- foreground mean IoU

The best checkpoint is selected by **dev foreground mean IoU**.

## Export

```bash
lao-ocr export-layout-detector \
  --checkpoint training/layout/runs/tiny-unet-v1/layout-detector.pt \
  --output training/layout/runs/tiny-unet-v1/layout-detector.pt2
```

The sidecar metadata includes:

- model version
- model configuration
- semantic class IDs
- artifact SHA-256

Runtime loading rejects an artifact whose SHA-256 no longer matches its metadata.

## Use in local document conversion

The owned OCR engine can use either deterministic morphology or the learned layout detector.

Morphology (default):

```bash
lao-ocr convert-document \
  --input scan.pdf \
  --output-dir output \
  --engine owned \
  --model recognizer.pt2 \
  --layout-detector morphology
```

Learned layout detector:

```bash
lao-ocr convert-document \
  --input scan.pdf \
  --output-dir output \
  --engine owned \
  --model recognizer.pt2 \
  --layout-detector learned \
  --layout-model layout-detector.pt2 \
  --layout-confidence 0.55
```

The recognizer and layout model use the same selected runtime device.

## API configuration

To use the learned detector with `OCR_ENGINE=owned`:

```text
OCR_ENGINE=owned
OCR_MODEL_PATH=/models/recognizer.pt2
OCR_DEVICE=auto

OCR_LAYOUT_DETECTOR=learned
OCR_LAYOUT_MODEL_PATH=/models/layout-detector.pt2
OCR_LAYOUT_CONFIDENCE=0.55
```

Default:

```text
OCR_LAYOUT_DETECTOR=morphology
```

This keeps the stable deterministic detector as the default for local/self-hosted deployments.

## Inference behavior

The learned detector:

1. letterboxes the page into the model input size
2. runs semantic segmentation
3. converts low-confidence pixels to background
4. restores the semantic mask to original page coordinates
5. combines heading/paragraph/list/table pixels into text regions
6. removes tiny components
7. passes those regions into region-aware line detection

It does not yet use the predicted semantic class directly to bypass the downstream structure heuristics.

## Current status

The **training/export/inference pipeline is functional**.

A local smoke run verified:

- train CLI
- checkpoint creation
- `torch.export` artifact creation
- artifact SHA validation
- CPU runtime loading
- learned-region detection

That smoke used only one train page and one dev page and produced dev foreground mIoU of 0.0. This is expected for such a tiny sanity fixture and is **not a model-quality result**.

Do not publish a learned-layout accuracy claim until training/evaluation uses the frozen real layout benchmark.

## Quality gate

The learned detector should not become the default until it beats the deterministic baseline on reviewed held-out data for metrics such as:

- block F1
- mean block IoU
- semantic block type accuracy
- reading-order accuracy
- table-cell structure F1
- DOCX visual fidelity

See:

- [layout-benchmark.md](layout-benchmark.md)
- [docx-fidelity-benchmark.md](docx-fidelity-benchmark.md)

## Next improvements

Likely model work after collecting enough real layout labels:

- stronger backbone
- data augmentation
- class-aware crop sampling
- boundary-aware loss
- direct semantic block extraction
- learned reading order
- learned table structure
- learned image/diagram segmentation

## Learned visual / illustration regions

The same semantic segmentation model also predicts the `image` class. When the exported layout detector is used by the owned OCR engine, runtime can convert connected `image` components into preserved image blocks for DOCX/JSON output.

This path:

1. reuses the same cached page segmentation used for text-region detection;
2. crops from the original color page, not the grayscale OCR image;
3. rejects near-full-page regions so scan backgrounds are not duplicated;
4. suppresses candidates that substantially overlap recognized text, tables, or native PDF images;
5. adds accepted regions as `BlockType.IMAGE` with `source=learned-layout-image`.

The deterministic raster-photo and line-art/diagram detectors then run only on areas not already claimed by learned/native visual blocks.

This completes the **learned visual-region pipeline**, but model quality still depends on reviewed `image` labels in the real layout dataset. It is not yet a benchmark-quality illustration segmentation claim.
