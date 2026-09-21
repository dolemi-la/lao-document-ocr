# Benchmark source review

The public benchmark must be redistributable, reviewable, and usable for OCR/model evaluation without relying on ambiguous source-document rights.

A document being publicly downloadable does **not** automatically make it suitable for inclusion in this repository.

The machine-readable review registry is:

`benchmarks/source-registry.json`

## Current decisions

### HPLT v2 Lao — approved for text only

Source:

https://hplt-project.org/datasets/v2.0

HPLT publishes the Lao `lao-Laoo` corpus under CC0.

Approved uses:

- Lao training text
- synthetic OCR rendering
- printable capture-pack source text

Not treated as:

- a real scanned-document benchmark
- evidence for camera/scan robustness by itself

For real optical evaluation, render/print project-created pages from reviewed CC0 text and capture those pages with real scanners/phones.

### KhamLao MOE textbook-derived corpus — not approved

Source:

https://github.com/nouvath07/KhamLao

The repository is MIT, but its corpus tooling describes harvesting Lao Ministry of Education textbooks.

The repository software license does not by itself establish redistribution/model-training rights for the underlying textbook content.

Project policy:

- do not copy the scraped textbook pages
- do not copy OCR ground truth derived from those pages into the public benchmark
- revisit only if the underlying source rights are independently clarified

### Lao-SABAIDEE — pending/unavailable

Catalog reference:

https://github.com/xinke-wang/OCRDatasets

The catalog lists roughly 500 Lao handwritten samples but does not provide a public download link or clear license.

Project policy:

- do not use it today
- revisit if an authoritative distribution source and explicit license become available

### NayanaOCR Corpus 2025 — not selected

Source:

https://huggingface.co/datasets/Cognitive-Lab/NayanaOCR_Corpus_2025

It is a large synthetic multilingual OCR/layout corpus released under CC BY-NC 4.0 with supplemental terms.

Project policy:

- do not use it as the default/public benchmark source
- do not make it a default training dependency
- prefer permissive sources compatible with downstream open/commercial use

## Preferred real-benchmark acquisition

Use the project capture-pack workflow:

1. start with reviewed redistributable Lao text (CC0 preferred)
2. generate printable pages with deterministic IDs and ground truth
3. print the pages
4. create real flatbed scans and phone photos
5. contributors explicitly release their capture images
6. register captures through the CLI
7. freeze test IDs before publishing benchmark numbers

This gives us real:

- paper texture
- printer artifacts
- scanner noise
- camera perspective
- lighting variation
- blur/compression
- device differences

without depending on unclear source-document copyright.

## Review rule

New sources should enter `benchmarks/source-registry.json` before data is downloaded into the public benchmark.

The registry should record:

- authoritative URL
- license or rights statement
- allowed uses
- rejected/pending uses
- review notes

If rights are ambiguous, the default decision is **do not ingest**.
