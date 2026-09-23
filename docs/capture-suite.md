# Capture suites

A capture suite combines multiple structured printable packs into one contributor-friendly campaign.

Instead of generating and printing one layout at a time, one command can produce:

- plain text pages
- two-column pages
- ruled tables
- borderless tables
- receipts
- forms

plus one combined PDF.

## Generate all templates

```bash
lao-ocr generate-capture-suite \
  --corpus training/data/lao-lines.txt \
  --output benchmarks/capture-packs/baseline-suite-v1 \
  --font /path/to/NotoSansLao-Regular.ttf \
  --suite-id baseline-suite-v1 \
  --text-license CC0-1.0 \
  --text-provenance "independently cleared Lao corpus" \
  --dpi 150 \
  --lines-per-page 8 \
  --max-pages-per-template 20
```

By default all available templates are included.

## Generate selected templates

Repeat `--template`:

```bash
lao-ocr generate-capture-suite \
  --corpus training/data/lao-lines.txt \
  --output benchmarks/capture-packs/forms-and-receipts-v1 \
  --font /path/to/NotoSansLao-Regular.ttf \
  --suite-id forms-and-receipts-v1 \
  --text-license CC0-1.0 \
  --text-provenance "Reviewed CC0 Lao corpus" \
  --template receipt \
  --template form
```

## Output layout

```text
baseline-suite-v1/
├── baseline-suite-v1.pdf
├── capture-suite.json
├── plain/
│   ├── capture-pack.json
│   ├── pages/
│   └── ground-truth/
├── two-column/
├── ruled-table/
├── borderless-table/
├── receipt/
└── form/
```

The combined PDF concatenates every pack in the same order recorded by `capture-suite.json`.

Each pack remains independently usable with `register-capture`.

## Why keep separate pack manifests?

A contributor may capture only part of a suite.

Keeping one capture-pack manifest per template means:

- page IDs remain stable
- exact ground truth stays local to the source pack
- capture registration remains unchanged
- template tags remain explicit
- individual layout packs can be regenerated/reviewed separately

The suite manifest is an orchestration/provenance layer, not a replacement for page-level manifests.

## Recommended campaign

For an initial rights-clear benchmark campaign:

- 20 plain pages
- 20 two-column pages
- 20 ruled-table pages
- 20 borderless-table pages
- 20 receipt pages
- 20 form pages

Then capture each page with at least:

- one flatbed scan
- one phone photo

A smaller degraded-scan subset can be added later.

Do not count the generated digital pages as real optical benchmark evidence. The benchmark samples are the registered real captures.

## Track collection progress

After captures start entering the public dataset, compare the suite against the dataset with `capture-campaign-report`. See [capture-campaign-report.md](capture-campaign-report.md).
