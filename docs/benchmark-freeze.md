# Frozen benchmark test set

Before publishing baseline or model-quality numbers, freeze the reviewed test split so the evaluated sample set cannot drift silently.

## Freeze

```bash
lao-ocr freeze-benchmark \
  --manifest benchmarks/manifest-v1.jsonl \
  --dataset-root benchmarks/public \
  --split test \
  --output-manifest benchmarks/frozen/test-v1.jsonl \
  --output-lock benchmarks/frozen/test-v1.lock.json \
  --revision "$(git rev-parse HEAD)"
```

The command validates the dataset first, then writes:

- a sorted frozen JSONL manifest for the selected split
- a lock JSON containing hashes/provenance for the frozen set

## Lock contents

The lock records:

- frozen split
- source Git revision when supplied
- source manifest hash
- frozen manifest hash
- sample/document counts
- document IDs
- subset/tag coverage
- source image SHA-256 for every sample
- ground-truth SHA-256 for every sample
- optional layout-ground-truth SHA-256

The frozen manifest also rewrites each sample's source `sha256` to the actual verified source hash at freeze time.

## Verify later

```bash
lao-ocr verify-benchmark-freeze \
  --lock benchmarks/frozen/test-v1.lock.json \
  --dataset-root benchmarks/public
```

Verification detects:

- frozen manifest changes
- changed source images
- changed text ground truth
- changed layout annotations
- missing locked files
- paths escaping the dataset root

A verification failure exits nonzero.

## Recommended release flow

1. finish capture collection
2. validate the dataset
3. review test IDs and licenses
4. freeze the test split
5. verify the freeze
6. run the Tesseract baseline on the frozen manifest
7. run the owned candidate on the same frozen manifest
8. compare with `compare-benchmarks`
9. publish reports + lock file + revision

Do not regenerate a lock merely because a candidate model performs poorly on the frozen set. Any benchmark-content change should create a new benchmark version with reviewed reasons.
