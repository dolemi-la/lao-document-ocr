# Benchmarks

Do not use private documents in the public benchmark set.

## Metrics

- CER: Character Error Rate
- WER: Word Error Rate

Lower is better.

## Suggested public dataset layout

```text
benchmarks/data/
├── clean-print/
├── noisy-scan/
├── phone-photo/
├── mixed-lao-english/
├── multi-column/
└── tables/
```

Each sample should include:

- source image/PDF page
- exact UTF-8 ground truth
- license/provenance metadata
- optional layout annotations

## Report format

Always report results per subset and include:

- number of pages
- number of characters
- engine/model version
- preprocessing version
- CER
- WER
- hardware
- runtime

Never publish a rounded "accuracy" percentage without the underlying benchmark definition.
