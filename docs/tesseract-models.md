# Tesseract traineddata selection

The project defaults to the Tesseract language data installed by the operating system/container.

For reproducible experiments, you can explicitly select another tessdata directory.

## CLI

```bash
lao-ocr benchmark \
  --manifest benchmarks/frozen/test-v1.jsonl \
  --dataset-root benchmarks/public \
  --engine tesseract \
  --languages lao+eng \
  --psm 3 \
  --tessdata-dir /path/to/tessdata \
  --output benchmarks/results/tesseract-custom.json
```

The same `--tessdata-dir` option is supported by:

- `convert-document`
- `benchmark`
- `benchmark-capture-suite`

## API

Set:

```text
OCR_TESSDATA_DIR=/models/tessdata
```

When using Docker/Compose, mount that directory into the API container and point `OCR_TESSDATA_DIR` at the mounted path.

## Reproducibility

When a custom tessdata directory is used, Tesseract engine metadata includes SHA-256 hashes for each required `.traineddata` file.

Example:

```json
{
  "tessdata": {
    "source": "custom",
    "traineddata_sha256": {
      "lao": "...",
      "eng": "..."
    }
  }
}
```

The local filesystem path itself is not included in benchmark metadata.

When no custom directory is supplied, metadata records:

```json
{
  "source": "system-default",
  "traineddata_sha256": {}
}
```

For published baseline results, prefer an explicit tessdata directory so the model weights are identified by hash.

## Digital QA observation

On the project-authored **generated digital capture suite** (not real scan/photo evidence), stronger traineddata can materially change OCR results. This is why model hashes belong in benchmark metadata.

In one reproducible digital QA run on the 60-page project-authored generated suite, Tesseract 5.5.0 + PSM 3 with explicitly selected stronger Lao/English traineddata produced CER 0.4800 and WER 0.9169. The report and exact traineddata hashes are committed under `benchmarks/results/project-authored-lao-v1-digital-qa-tessdata-best-psm3.json`.

Digital QA numbers must never replace the frozen real-capture benchmark.
