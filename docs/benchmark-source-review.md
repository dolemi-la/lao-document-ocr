# Benchmark source review

The public benchmark must be redistributable, reviewable, and usable for OCR/model evaluation without relying on ambiguous source-document rights.

A document being publicly downloadable does **not** automatically make it suitable for inclusion in this repository.

The machine-readable review registry is:

`benchmarks/source-registry.json`

## Current decisions

### Project-authored Lao capture text v1 — approved for public capture packs

Source:

`resources/corpora/project-authored-lao-v1.txt`

License: Apache-2.0.

This corpus is written specifically for the project and is the preferred default source for public printable capture campaigns. Generated digital pages are source material only; real benchmark readiness still requires registered optical scans/photos carrying `capture:optical-evidence`.


### HPLT v3 Lao — approved for text only

Source:

https://hplt-project.org/datasets/v3.0

HPLT includes Lao `lao_Laoo` in v2. HPLT states that its dataset packaging is licensed under CC0, while it does not own the underlying extracted text; downstream users remain responsible for applicable source rights and legal obligations.

Project-policy allowed uses:

- provenance-recorded training/model-development text, subject to downstream source-rights responsibility

Not approved by HPLT packaging terms alone for:

- republishing sampled lines as public capture-pack source text
- public benchmark ground truth
- a real scanned-document benchmark
- evidence for camera/scan robustness by itself

For real optical evaluation, render project-created pages from independently redistributable/self-authored text and capture those pages with real scanners/phones. For the bounded HPLT training-text workflow and provenance rules, see [hplt-sampling.md](hplt-sampling.md).

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

## Remote real-world scanned-document candidates — not approved for ingestion

On 2026-09-24, several official/publicly hosted Lao PDFs were reviewed specifically for real camera/scanner failure modes. They are recorded in `benchmarks/source-registry.json` as `remote-evaluation-candidate-not-approved`.

Verified examples include:

- Lao Census CamScanner notice: three raster pages; useful PDF text is effectively absent
- Lao Census ID-card delivery notice: visible Lao scan with a present but garbled/non-Lao text layer
- Lao Census registration guidance: four pages with zero extracted text lines
- Lao Census household-book notices: image-only pages with handwriting, stamps/signatures, and scan noise
- PTC CamScanner notice: three pages whose extracted text is effectively only the CamScanner watermark
- HPC CamScanner legislation: 21 image-only pages with seals/signatures and zero extracted text lines
- Lao National Assembly CamScanner invitation: document published directly as a scanned image
- MAF/forestry CamScanner document: 15 raster pages, zero native text lines, handwritten metadata and scan noise
- MPWT Council of Lao Architects and Engineers: one-page PDF whose native layer is effectively only the CamScanner watermark
- NAP Laos Paris Agreement: 41-page CamScanner document with a present but noisy/segmented Lao OCR layer
- LaoWIS Salavan groundwater plan: 64-page document containing rotated CamScanner pages whose OCR/table layer is heavily fragmented
- LaoWIS Savannakhet groundwater plan: 55-page mixed document with embedded CamScanner lab sheets whose raster table/stamp/signature content is absent from native extraction

These are valuable for understanding real OCR failure modes, but public availability does not establish redistribution/model-training rights. Project policy therefore remains:

- do not vendor the PDF/image bytes
- do not add derived ground truth to the public benchmark
- do not use the documents as default training data
- retain authoritative URLs plus review evidence for source discovery/rights follow-up
- promote a source only after an independent rights/license review

The useful failure-mode categories discovered so far are `image-only`, `scanner-watermark-only`, `present-but-garbled-text-layer`, `present-but-noisy-ocr-layer`, and `present-but-fragmented-layout-layer`, and `mixed-native-and-raster-camera-pages`, and `present-but-severely-garbled-ocr-layer`, and `rotated-raster-with-garbled-text-layer`.

### Third-party mirror examples — discovery only

Two especially useful scan-failure examples were also found on non-authoritative mirrors:

- Savannakhet DPWT five-year plan (AnyFlip): 44 pages with repeated CamScanner markers and severely garbled Lao extraction
- Xaythany groundwater-quality thesis (PubHTML5): 124 CamScanner-derived pages with a noisy Lao OCR layer and broken spacing/combining marks
- Lao44 2019 official Lao letter: one directly verified CamScanner page whose native PDF layer contains only the scanner watermark while the raster page contains the complete Lao letter, stamp, signature, and handwriting

These are recorded with the stricter status `remote-evaluation-candidate-third-party-mirror-not-approved`. They are useful for understanding failure modes, but the mirror is not proof of provenance or redistribution rights. Do not download them into the public dataset, generate public ground truth from them, or use them as training data without locating and clearing an authoritative source.

Additional discovery sources:

- a second MAF/forestry PDF is search-indexed as CamScanner 08-22-2022 13.28, but remains unverified because direct inspection is blocked by the host
- the Lao Official Gazette states that authoritative legislation is distributed as image-form PDF; treat it as a collection-level discovery source and review individual files separately
- ST Bank Laos Q4/2024 is search-indexed with a `Scanned with CS CamScanner` footer and dense Lao financial tables, but remains unverified because the original PDF endpoint timed out during direct inspection
- MAF Forestry Strategy to 2035 / Vision 2050 is search-indexed with heavily corrupted Lao text plus a CamScanner marker, but remains unverified because the official PDF fetch timed out during direct rendering
- World Bank Lao PDR disaster-risk-management ESMP includes a Lao project attachment indexed with tables, red stamp, handwritten signatures, and a `Scanned with CamScanner` marker; it remains unverified because the PDF exceeds the verifier's size limit
- World Bank Lao PDR Priority Skills for Growth audited financial statement identifies KPMG Lao and Vientiane and includes a CamScanner-indexed audit page; it remains unverified because the original PDF also exceeds the verifier's size limit
- A second World Bank Lao PSG 2024 audit PDF is directly renderable: 22 pages where later financial statements are rotated raster pages with red stamps/signatures while native extraction collapses into garbage text

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
