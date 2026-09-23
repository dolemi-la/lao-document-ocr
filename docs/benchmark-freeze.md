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

## Public freeze policy

The CLI is strict by default. Every selected sample must have:

- verified real-source evidence (`source:real-document`, or `source:real-capture` plus `capture:optical-evidence`)
- `review.status=approved` from the manual review workflow

For development-only fixtures/smoke tests, opt out explicitly with `--allow-unverified-sources` and/or `--allow-unreviewed`. Do not use these bypass flags for published results.

The lock stores the review record for every frozen sample in addition to content hashes.

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
3. approve every publication sample with `review-dataset-sample`
4. run `benchmark-readiness` and resolve pending/rejected samples
5. freeze the test split
6. verify the freeze
7. run the Tesseract baseline on the frozen manifest
8. run the owned candidate on the same frozen manifest
9. compare with `compare-benchmarks`
10. publish reports + lock file + revision

Do not regenerate a lock merely because a candidate model performs poorly on the frozen set. Any benchmark-content change should create a new benchmark version with reviewed reasons.

## Enforce the lock during OCR benchmarking

After freezing, pass the lock directly to every baseline/candidate run:

```bash
lao-ocr benchmark \
  --manifest benchmarks/frozen/test-v1.jsonl \
  --dataset-root benchmarks/public \
  --split test \
  --freeze-lock benchmarks/frozen/test-v1.lock.json \
  --engine tesseract \
  --output benchmarks/results/tesseract-v1.json
```

The command verifies the frozen manifest and every locked source/ground-truth/layout hash before initializing the OCR engine. A tampered benchmark therefore fails before producing a report.
