# Public benchmark dataset intake

The public benchmark should contain only material that can be redistributed and used for OCR/model evaluation.

Do not commit a scan merely because it is publicly reachable on the web.

## Add one reviewed page

Prepare:

- one page image
- exact UTF-8 ground-truth text
- a stable document ID
- the dominant benchmark subset
- a license/rights statement
- provenance describing where the page came from

Then run:

```bash
lao-ocr add-dataset-sample \
  --dataset-root benchmarks/public \
  --manifest benchmarks/public/manifest.jsonl \
  --id clean-001 \
  --document-id document-001 \
  --subset clean-print \
  --image /path/to/page.png \
  --ground-truth /path/to/page.txt \
  --license CC-BY-4.0 \
  --provenance "Scanned from Example Collection by Example Contributor" \
  --source-url https://example.org/source \
  --license-url https://creativecommons.org/licenses/by/4.0/ \
  --confirm-redistributable
```

The command:

- requires explicit rights confirmation
- validates the image and UTF-8 ground truth
- copies both files into the dataset tree
- calculates the source SHA-256
- prevents duplicate sample IDs/accidental overwrites
- keeps every page from one source document in the same split
- appends one reviewable JSONL manifest entry

## Split assignment

If `--split` is omitted, the split is derived deterministically from `document_id` using SHA-256 and the project default 80/10/10 train/dev/test ratios.

If a document already exists in the manifest, every new page automatically reuses that document's existing split.

This prevents page-level leakage between train/dev/test.

You may explicitly choose a split:

```bash
--split test
```

but the command refuses to move later pages of an existing document into a different split.

## Ground truth policy

Ground truth is an exact transcription of the visible source in intended reading order.

Do not:

- silently correct spelling
- rewrite grammar
- normalize numbers into another representation
- translate text
- omit difficult sections because the OCR model struggles with them

Unicode normalization may be applied consistently by evaluation code, but the stored ground truth should remain a faithful transcription.

## Rights review

Good candidates include:

- documents created specifically for this project and explicitly released for reuse
- public-domain material with clear provenance
- Creative Commons/openly licensed documents permitting redistribution
- sanitized documents deliberately created for public benchmarking

Do not add:

- private documents
- identity documents
- bank statements or confidential records
- random social-media images
- random government/company PDFs with no clear redistribution terms

The `--confirm-redistributable` flag is an explicit contributor assertion, not an automated legal determination.

## Validate after intake

```bash
lao-ocr validate-dataset \
  --manifest benchmarks/public/manifest.jsonl \
  --dataset-root benchmarks/public
```

Review the manifest diff before committing any new benchmark data.
