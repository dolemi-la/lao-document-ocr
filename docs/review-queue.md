# Local benchmark review queue

Manual review is easier when reviewers can see the capture, transcription, tags, and current review state together.

`build-review-queue` creates a local static review bundle without changing the dataset manifest.

## Build the default queue

```bash
lao-ocr build-review-queue \
  --manifest benchmarks/public/manifest.jsonl \
  --dataset-root benchmarks/public \
  --output-dir /tmp/lao-ocr-review
```

Default behavior:

- split: `test`
- status filter: `needs-review`
- includes unreviewed + rejected samples
- excludes already approved samples

Output:

```text
/tmp/lao-ocr-review/
├── index.html
├── review-queue.json
└── thumbnails/
    ├── <sample-id>.jpg
    └── ...
```

Open `index.html` locally in a browser.

## What each card shows

- capture thumbnail
- sample/document ID
- current review status
- primary subset
- multi-axis tags
- source and ground-truth paths
- exact UTF-8 ground truth
- current reviewer/notes
- file/preview problems when detected
- example approval command

All manifest/ground-truth text is HTML-escaped before rendering.

## Filter the queue

Examples:

```bash
# Only unreviewed test samples
lao-ocr build-review-queue \
  --manifest benchmarks/public/manifest.jsonl \
  --dataset-root benchmarks/public \
  --output-dir /tmp/review-unreviewed \
  --status unreviewed

# Recheck rejected samples
lao-ocr build-review-queue \
  --manifest benchmarks/public/manifest.jsonl \
  --dataset-root benchmarks/public \
  --output-dir /tmp/review-rejected \
  --status rejected

# Audit approved dev samples
lao-ocr build-review-queue \
  --manifest benchmarks/public/manifest.jsonl \
  --dataset-root benchmarks/public \
  --output-dir /tmp/review-approved-dev \
  --split dev \
  --status approved
```

Allowed status filters:

- `needs-review`
- `unreviewed`
- `rejected`
- `approved`
- `all`

## Thumbnail bounds

For large phone photos, bounded JPEG thumbnails keep the queue manageable:

```bash
--thumbnail-width 900
--thumbnail-height 1200
```

The originals are never modified.

## Record a decision

After reviewing a sample:

```bash
lao-ocr review-dataset-sample \
  --manifest benchmarks/public/manifest.jsonl \
  --dataset-root benchmarks/public \
  --id <sample-id> \
  --status approved \
  --reviewer <reviewer-alias> \
  --notes "Checked page identity, orientation, ground truth, tags, and release metadata."
```

Or record a rejection with `--status rejected` and a reason.

Rebuild the queue after decisions are recorded. Approved samples disappear from the default `needs-review` queue.

## Privacy and sharing

The review bundle may contain thumbnails and full ground-truth text from the benchmark dataset.

Treat it as local review material. Do not publish/share the generated queue unless the underlying benchmark content is already approved for that audience.
