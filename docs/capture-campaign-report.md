# Capture campaign coverage report

A capture suite defines the pages we want contributors to photograph/scan. The public dataset tells us which captures have already been registered.

The campaign report compares those two sources and shows what is still missing.

## Default coverage target

By default every capture-suite page is expected to have:

- one `flatbed-scan`
- one `phone-photo`

Run:

```bash
lao-ocr capture-campaign-report \
  --suite-manifest benchmarks/capture-packs/baseline-suite-v1/capture-suite.json \
  --dataset-manifest benchmarks/public/manifest.jsonl \
  --output benchmarks/results/baseline-suite-v1.campaign.json
```

## Custom required capture modes

Repeat `--require-mode` to override the defaults:

```bash
lao-ocr capture-campaign-report \
  --suite-manifest benchmarks/capture-packs/baseline-suite-v1/capture-suite.json \
  --dataset-manifest benchmarks/public/manifest.jsonl \
  --require-mode flatbed-scan \
  --require-mode degraded-scan \
  --require-mode phone-photo \
  --output benchmarks/results/baseline-suite-v1.campaign.json
```

## Report contents

The report includes:

- expected suite pages
- required page/mode capture pairs
- completed page/mode capture pairs
- overall completion ratio
- completion by capture mode
- completion by template
- registered dataset samples belonging to the suite
- dataset samples unrelated to the suite
- explicit missing page IDs and capture modes

Example missing item:

```json
{
  "page_id": "baseline-suite-v1-receipt-p0004",
  "template": "receipt",
  "missing_modes": ["flatbed-scan"],
  "present_modes": ["phone-photo"]
}
```

## Backward compatibility

New dataset entries should use multi-axis tags such as:

```text
capture:phone-photo
capture:flatbed-scan
```

For older manifests without tags, the campaign reporter maps legacy primary subsets:

- `clean-print` → `flatbed-scan`
- `noisy-scan` → `degraded-scan`
- `phone-photo` → `phone-photo`

## Recommended collection gate

Before freezing a benchmark test set:

1. generate the campaign report
2. reach the agreed capture target for every test page
3. review every missing item
4. validate the dataset
5. freeze document IDs
6. run Tesseract/owned-model baselines

Do not silently drop difficult pages merely to reach 100% completion.
