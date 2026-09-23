# Project-authored benchmark text corpora

This directory contains text created specifically for Lao Document OCR development and printable capture campaigns.

## `project-authored-lao-v1.txt`

- language: Lao with deliberate Lao/English, numeric, date/time, reference-code, and currency examples
- license: CC0-1.0
- intended use: public capture-pack source text, public benchmark ground-truth source for those generated pages, synthetic rendering, and OCR regression fixtures
- provenance: project-authored; no third-party corpus is required to use this file
- integrity/provenance metadata: `project-authored-lao-v1.meta.json`

The text file is **not** real OCR evidence. To create real benchmark samples:

1. generate a capture pack/suite from this corpus;
2. print the generated pages;
3. create real flatbed scans or phone photos;
4. register those optical captures with `register-capture`.

Do not count the generated digital PNG/PDF pages as real scan/phone-photo benchmark results.
