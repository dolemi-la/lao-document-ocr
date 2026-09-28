# Synthetic OCR augmentation profiles

Synthetic training images support deterministic augmentation profiles that approximate common document-capture conditions.

These profiles improve training diversity. They are **not** substitutes for the real scan/phone-photo benchmark.

## Profiles

### `default`

Mild augmentation parameters:

- small rotation
- light Gaussian noise
- mild blur
- brightness jitter

### `clean-scan`

Models relatively clean office scans:

- very small rotation
- low noise/blur
- small brightness/contrast changes
- mild resolution loss
- high JPEG quality

### `noisy-scan`

Models lower-quality scans/copies:

- larger skew
- stronger noise and blur
- stronger brightness/contrast changes
- light uneven shadowing
- stronger resolution loss
- lower JPEG quality

### `phone-photo`

Models handheld document captures:

- larger rotation
- perspective distortion
- uneven lighting/shadow
- moderate noise/blur
- brightness/contrast variation
- resolution loss
- JPEG compression

### `balanced`

Cycles deterministically through:

```text
clean-scan
noisy-scan
phone-photo
```

This is useful when each corpus line has multiple variants and you want a predictable mixture without maintaining separate datasets.

When multiple fonts are supplied, balanced scheduling also rotates font choice across the three capture profiles instead of pairing one font with one fixed degradation mode. Over the deterministic cycle, each font is exercised with clean-scan, noisy-scan, and phone-photo augmentation, avoiding a hidden font/profile correlation in training data.

## Generate a balanced training set

```bash
lao-ocr generate-synthetic \
  --corpus training/data/lao-lines-phetsarath.txt \
  --output training/generated/balanced-v1 \
  --font /path/to/PhetsarathOT-Regular.ttf \
  --variants-per-line 3 \
  --augmentation-profile balanced \
  --require-complete-font \
  --seed 20260921
```

Generated training images are stored as grayscale PNGs. The augmentation pipeline is grayscale before serialization, so this avoids redundant RGB channels while preserving the exact intensity data consumed by recognizer training.

Each manifest entry records:

- resolved `augmentation_profile`
- rotation
- brightness
- contrast
- blur
- noise
- perspective ratio
- shadow strength
- resolution scale
- JPEG quality
- seed
- output image SHA-256

Within the same generator revision and dependency environment, the same corpus/fonts/options/seed produce the same profile schedule and augmentation metadata.

## Geometry and pixel-range integrity (version 2)

Rotation and perspective transforms expand and translate the output canvas to
contain the complete transformed source rectangle, with a small interpolation
margin. They do not deliberately crop edge characters or rescale the geometry
to force it back into the old dimensions. A no-op transform keeps the input size.
Degenerate one-pixel-wide/high inputs skip perspective distortion. A transform
crossing a projective singularity is rejected rather than producing invalid bounds.

Cubic interpolation can create values outside the grayscale range. Before
serialization, values are clamped to `[0, 255]` and rounded to the nearest integer.
This prevents unsigned-byte conversion from wrapping bright pixels to dark and
negative values to white. Rounding also prevents floating-point noise around
pure white from introducing artificial gray backgrounds.

New augmentation metadata contains `geometry_version: 2`, `source_width`,
`source_height`, `output_width`, and `output_height`. Old unversioned images must
not be relabeled as version 2. Regenerating images with this implementation can
change dimensions and hashes; retain old artifacts for audit and use a separate
output directory. Chunk/full-run determinism is still tested within this version.

Preserving a long rotated line can increase the canvas height. Re-run strict
font coverage, image hash checks, and CTC capacity checks on new datasets. Do not
shorten labels or accept impossible CTC alignments to hide a capacity failure.
The corrected 300-image Phetsarath diagnostic set has two capacity failures at
height 48 / max width 768; at height 64 / max width 768 all 300 pass. That is a
capacity observation, not a trained-model quality result. A changed image size
or dataset requires a new compatible training run, not an exact resume from the
old dataset/checkpoint.

The measured defects and paired training diagnostics are recorded in
[recognizer-training-health.md](recognizer-training-health.md). Correcting
augmentation integrity is not itself evidence of higher OCR accuracy.

## Benchmark policy

Do not report accuracy on these synthetic variants as real-world OCR accuracy.

Use real registered capture-pack scans/photos for the fixed benchmark. Synthetic augmentation is training data and pipeline sanity data only.
