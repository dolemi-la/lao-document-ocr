# Manual benchmark review

Real optical evidence and valid hashes are necessary, but they do not replace human review.

Before a sample can enter a frozen public benchmark, a reviewer should verify the actual capture, its identity, metadata, and ground truth, then record an explicit approval in the dataset manifest.

## Build a visual review queue

Before recording decisions, generate a local thumbnail/ground-truth queue with `lao-ocr build-review-queue`. It defaults to unreviewed/rejected test samples and does not mutate the manifest. See [review-queue.md](review-queue.md).

## Approve a sample

```bash
lao-ocr review-dataset-sample \
  --manifest benchmarks/public/manifest.jsonl \
  --dataset-root benchmarks/public \
  --id project-authored-lao-v1-plain-p0001-phone-a \
  --status approved \
  --reviewer "reviewer-alias" \
  --notes "QR/page ID, orientation, ground truth, subset/tags, and release metadata checked."
```

Approval is refused unless that sample passes dataset validation with source SHA-256 verification.

The manifest entry receives a review object such as:

```json
{
  "review": {
    "status": "approved",
    "reviewer": "reviewer-alias",
    "reviewed_at": "2026-09-23T08:00:00Z",
    "notes": "QR/page ID, orientation, ground truth, subset/tags, and release metadata checked."
  }
}
```

`reviewed_at` is generated in UTC and must be timezone-aware.

## Reject a sample

A reviewer can record rejection even when the sample is currently invalid:

```bash
lao-ocr review-dataset-sample \
  --manifest benchmarks/public/manifest.jsonl \
  --dataset-root benchmarks/public \
  --id project-authored-lao-v1-plain-p0001-phone-a \
  --status rejected \
  --reviewer "reviewer-alias" \
  --notes "Wrong page framing; recapture required."
```

This makes rejection/review state visible in Git instead of leaving it in chat or an external spreadsheet.

## Review checklist

At minimum, verify:

- the capture is the intended printed page
- QR/page ID is correct
- orientation is correct
- the full relevant page content is visible
- no unrelated private/confidential material is in frame
- ground truth matches the visible source exactly
- primary subset and multi-axis tags are appropriate
- contributor/release metadata is credible and complete
- optical-evidence checks are present for real capture-pack images
- optional layout annotations match the capture when supplied

Approval is a human attestation that these checks were performed; it is not a claim that OCR output is correct.

## Dataset coverage report

`dataset-report` includes:

- approved count
- rejected count
- unreviewed count
- approved coverage ratio
- IDs in each review state

Example:

```bash
lao-ocr dataset-report \
  --manifest benchmarks/public/manifest.jsonl \
  --dataset-root benchmarks/public \
  --output benchmarks/results/dataset-report.json
```

## Readiness and freeze policy

`benchmark-readiness` requires approved manual review by default.

`freeze-benchmark` also requires, by default:

- verified real-source evidence
- approved manual review for every selected sample

Development-only smoke workflows can opt out explicitly:

```text
--allow-unverified-sources
--allow-unreviewed
```

Do not use those bypass flags for a published benchmark release.

## Updating a review

Running `review-dataset-sample` again replaces the previous review record for that sample. Manifest updates are written atomically.

If the capture or ground truth changes after approval, its hashes/manifest diff change and the sample should be reviewed again before publication.
