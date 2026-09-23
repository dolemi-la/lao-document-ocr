# Digital capture-suite QA benchmark

Generated capture-suite pages are useful for verifying OCR/template behavior before real scans arrive, but they are **not** real optical benchmark evidence.

Use this command only as a digital pipeline sanity check:

```bash
lao-ocr benchmark-capture-suite \
  --suite-manifest benchmarks/capture-packs/baseline-suite-v1/capture-suite.json \
  --engine tesseract \
  --output benchmarks/results/baseline-suite-v1.digital-qa.json
```

The command supports the same owned-recognizer, decoder, language-model, layout-detector, device, and reading-order options as the normal full-page benchmark command.

Every report includes:

```json
{
  "qa": {
    "kind": "digital-capture-suite",
    "not_real_benchmark": true
  }
}
```

and prints a warning that the result must not be reported as real scan/photo accuracy.

## What it is good for

Use digital suite QA to catch regressions in:

- page-template rendering
- Lao/English OCR plumbing
- multi-column ordering
- ruled/borderless table handling
- receipt/form handling
- model/export/runtime compatibility
- template-specific catastrophic failures

## What it cannot prove

It does not measure robustness to:

- camera perspective
- lighting variation
- blur
- paper texture
- printer/scanner artifacts
- compression noise
- folds/shadows
- real device differences

Those require registered real captures and the normal frozen benchmark workflow.

## Recommended workflow

1. generate a structured capture suite
2. run digital QA against Tesseract and/or the owned stack
3. fix obvious template/pipeline failures
4. print the suite
5. collect real flatbed/phone captures
6. register captures into the public benchmark
7. pass readiness + optical-evidence checks
8. freeze the real test split
9. run/publish the real Tesseract baseline
10. compare owned OCR against that frozen baseline
