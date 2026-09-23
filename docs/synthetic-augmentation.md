# Synthetic OCR augmentation profiles

Synthetic training images support deterministic augmentation profiles that approximate common document-capture conditions.

These profiles improve training diversity. They are **not** substitutes for the real scan/phone-photo benchmark.

## Profiles

### `default`

Backward-compatible mild augmentation:

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

## Generate a balanced training set

```bash
lao-ocr generate-synthetic \
  --corpus training/data/lao-lines.txt \
  --output training/generated/balanced-v1 \
  --font /path/to/NotoSansLao-Regular.ttf \
  --variants-per-line 3 \
  --augmentation-profile balanced \
  --seed 20260921
```

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

The same corpus/fonts/options/seed produce the same profile schedule and augmentation metadata.

## Benchmark policy

Do not report accuracy on these synthetic variants as real-world OCR accuracy.

Use real registered capture-pack scans/photos for the fixed benchmark. Synthetic augmentation is training data and pipeline sanity data only.
