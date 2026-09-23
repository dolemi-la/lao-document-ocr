# Benchmark readiness gate

Before publishing OCR baseline/model numbers, verify that the real test split covers the document conditions the project claims to support.

The readiness gate measures **unique source documents**, not only sample count, so many captures of one printed page cannot masquerade as broad coverage.

## Run

```bash
lao-ocr benchmark-readiness \
  --manifest benchmarks/manifest-v1.jsonl \
  --dataset-root benchmarks/public \
  --split test \
  --output benchmarks/results/readiness-v1.json
```

Default requirement: at least one unique test document in every required coverage dimension.

That default is intentionally only a completeness check, not a claim that one document is statistically sufficient for publication.

## Coverage dimensions

The gate currently checks:

- clean print / flatbed scan
- noisy/degraded scan
- phone photo
- mixed Lao/English
- multi-column layout
- simple ruled table
- complex/borderless table
- receipt
- form

A sample can satisfy multiple dimensions through tags. For example a phone photo of a mixed-language form can count toward:

```text
capture:phone-photo
language:mixed
document:form
```

This is why benchmark readiness uses multi-axis tags rather than forcing every sample into one exclusive category.

## Real-source requirement

By default, publication readiness only counts samples with accepted real-source evidence:

```text
source:real-document
```

or capture-pack registrations carrying both:

```text
source:real-capture
capture:optical-evidence
```

Digital capture-pack pages and synthetic renders do **not** satisfy the release gate by themselves, even if they carry layout/capture-style tags. A `source:real-capture` tag without `capture:optical-evidence` is also excluded. This prevents synthetic/digital copies from being reported as real-world benchmark evidence.

For local tooling smoke checks only, this guard can be bypassed with:

```bash
--allow-unverified-sources
```

Do not use that override for a published benchmark release.

## Stricter release targets

Projects can require more independent documents per dimension:

```bash
--min-documents-per-dimension 20
```

Require a minimum total number of unique test documents:

```bash
--min-total-documents 150
```

Require reviewed layout annotations for a subset of test documents:

```bash
--min-layout-labeled-documents 40
```

These values are project/release decisions. The tool does not pretend there is one universal statistically valid threshold for every OCR deployment.

## Validation is part of readiness

The gate also runs normal dataset validation, including:

- file existence
- SHA-256 checks
- split leakage
- duplicate source-image detection
- layout-ground-truth validation

A coverage-complete dataset is still **not ready** if validation fails.

## Exit code

- `0` — coverage and validation targets passed
- `1` — dataset is valid to inspect but not publication-ready under the requested thresholds

The JSON report includes every dimension, its sample/document count, required minimum, missing dimensions, and overall readiness.

## Recommended publication sequence

1. collect real capture-pack scans/photos
2. run `dataset-report`
3. run `capture-campaign-report`
4. run `benchmark-readiness` with the release thresholds
5. manually review licenses/ground truth
6. freeze the test set
7. verify the freeze lock
8. run the Tesseract baseline
9. run the owned candidate
10. compare with `compare-benchmarks`

Do not weaken readiness thresholds after seeing model scores just to make a release pass.
