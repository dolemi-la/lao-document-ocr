# Capture submission bundles

A capture submission is a blind, checksum-bound ZIP used to transfer one collector session from a contributor to a maintainer.

It contains only captured images and non-answer provenance. It does not contain OCR ground truth.

## From the mobile collector

After capturing one or more pages, use the **Download capture submission ZIP** button.

The ZIP contains:

```text
capture-submission.json
SHA256SUMS
captures/
  <page-id>.jpg
  <page-id>.png
  ...
```

The manifest records:

- suite ID
- capture-kit SHA-256
- capture-kit source revision
- capture ID
- capture mode
- QR requirement
- expected/captured page counts
- completeness flag
- capture filenames/sizes/SHA-256 hashes

## Pack from the command line

If captures already exist in a collector output directory:

```bash
lao-ocr pack-capture-submission \
  --capture-dir /path/to/captures/phone-a \
  --output phone-a.submission.zip \
  --require-complete
```

The archive is built deterministically from the same capture bytes and session metadata.

## Verify before import

```bash
lao-ocr verify-capture-submission \
  --submission phone-a.submission.zip
```

Verification checks:

- bounded ZIP size/member count
- safe member paths
- complete SHA256SUMS coverage
- manifest schema
- capture metadata/hash/size consistency
- unique page IDs
- capture mode/suite/session identifiers
- image validity and decoded pixel limit

## Extract safely

```bash
lao-ocr extract-capture-submission \
  --submission phone-a.submission.zip \
  --output-dir /path/to/incoming/phone-a
```

Extraction verifies first, refuses a non-empty destination directory, writes the directory as `0700` and files as `0600` where supported, and produces:

```text
/path/to/incoming/phone-a/
  <page-id>.jpg
  <page-id>.png
  capture-submission.json
```

The resulting directory is compatible with the existing `register-capture-directory` workflow.

## Preferred maintainer import

Use the one-step importer to verify the archive, bind it to the intended suite/kit revision, extract it into a temporary private directory, run the existing optical-evidence checks, and atomically register the captures:

```bash
lao-ocr register-capture-submission \
  --submission phone-a.submission.zip \
  --suite-manifest benchmarks/capture-packs/project-authored-lao-v1/capture-suite.json \
  --contributor "Contributor Alias" \
  --release-license CC0-1.0 \
  --dataset-root benchmarks/public \
  --dataset-manifest benchmarks/public/manifest.jsonl \
  --expected-kit-sha256 <collector-kit-sha256> \
  --expected-source-revision <git-revision> \
  --require-complete \
  --dry-run
```

The importer derives `capture_id` and capture mode from the verified submission instead of accepting retyped values. It also requires the submission suite ID and expected page count to match the supplied suite manifest. `--expected-kit-sha256` and `--expected-source-revision` are optional but recommended for maintainer handoffs.

After reviewing the dry-run report, rerun without `--dry-run` and add `--confirm-release`. Temporary extracted files are removed automatically after the import attempt.

## Manual maintainer workflow

The lower-level commands remain available when manual staging is useful:

1. receive the submission ZIP
2. run `verify-capture-submission`
3. inspect the manifest/session provenance
4. run `extract-capture-submission` into a private staging directory
5. run `register-capture-directory --dry-run`
6. review optical-evidence/authenticity results
7. register with `--confirm-release`
8. run manual visual review and readiness gates

A valid checksum only proves archive integrity. It does not replace contributor rights confirmation, optical-evidence checks, or manual benchmark review.
