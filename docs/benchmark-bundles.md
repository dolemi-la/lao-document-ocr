# Benchmark release bundles

A public OCR result is only useful if people can tell exactly which code and which report files produced the headline numbers.

The benchmark bundle groups the project’s independent benchmark reports into one small release manifest.

Supported report kinds:

- OCR page benchmark (CER/WER)
- project recognizer benchmark
- layout/table AST benchmark
- DOCX visual-fidelity benchmark

## Create a bundle

```bash
lao-ocr bundle-benchmarks \
  --ocr benchmarks/results/tesseract.json \
  --recognizer benchmarks/results/recognizer.json \
  --layout benchmarks/results/layout.json \
  --docx benchmarks/results/docx-fidelity.json \
  --revision "$(git rev-parse HEAD)" \
  --label baseline-v1 \
  --output benchmarks/results/baseline-v1.bundle.json
```

At least one report is required.

The source revision is intentionally required by the CLI so published results are not detached from the code that produced them.

## Bundle contents

Each bundled report records:

- report kind
- original report filename
- SHA-256 of the report JSON
- report schema version
- a compact metric summary

The top-level bundle records:

- bundle schema version
- generation timestamp
- optional release label
- source Git revision

## Example

```json
{
  "schema_version": "1",
  "label": "baseline-v1",
  "source_revision": "abc123...",
  "reports": [
    {
      "kind": "ocr",
      "file": "tesseract.json",
      "sha256": "...",
      "schema_version": "1",
      "summary": {
        "samples": 100,
        "cer": 0.04,
        "wer": 0.17
      }
    }
  ]
}
```

## Why reports stay separate

The bundle does not collapse CER, layout F1, and visual fidelity into one “magic score.”

They measure different things:

- OCR: transcription correctness
- layout: document structure and reading order
- table metrics: structural reconstruction
- DOCX fidelity: rendered visual similarity

The bundle only provides provenance and a compact summary.

## Release guidance

For a public benchmark release:

1. freeze the benchmark dataset/test IDs
2. run each relevant benchmark from the same source revision
3. keep the full individual report JSON files
4. create the bundle with hashes
5. commit or attach all reports and the bundle to the release
6. never publish only a rounded “accuracy” percentage
