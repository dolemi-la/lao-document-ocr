# Bounded HPLT v3 Lao text sampling

The project can stream a **bounded** sample of Lao text from the official HPLT v3 sorted Lao distribution without downloading the complete shard first.

Official sources used by the command:

```text
Dataset / terms:
https://hplt-project.org/datasets/v3.0

Sorted Lao language map:
https://data.hplt-project.org/three/sorted/lao_Laoo.map
```

The current language map points to the official `lao_Laoo` sorted shards. The command reads the map at runtime and processes higher WDS quality bins first rather than hard-coding one shard filename.

## Rights note

HPLT states that it licenses its **dataset packaging** under CC0 and also states that it does not own the underlying extracted text. Downstream users remain responsible for applicable rights and legal obligations for source material.

Therefore this project records HPLT as:

```text
CC0-1.0-packaging; underlying-source-rights-source-dependent
```

Project policy allows bounded HPLT Lao text as provenance-recorded model-development/training input, subject to downstream responsibility for applicable source rights. It is not automatically approved for public capture-pack source text or benchmark ground truth.

Preserve the sampler metadata file with any derived corpus/capture campaign.

## Install the optional data dependency

Normal OCR installs do not require zstd support.

```bash
pip install -e '.[data]'
```

This installs the optional `zstandard` package used for streaming `.jsonl.zst` shards.

## Sample a small corpus

```bash
lao-ocr sample-hplt-lao \
  --output training/data/hplt-lao-10k.txt \
  --metadata training/data/hplt-lao-10k.meta.json \
  --limit 10000 \
  --max-lines-per-document 4 \
  --max-documents 10000 \
  --min-chars 8 \
  --max-chars 180 \
  --min-lao-ratio 0.5 \
  --max-lines-per-document 4
```

The sampler:

1. fetches the official Lao v3 sorted language map;
2. orders shards by WDS quality bin (highest first) and opens them over HTTPS;
3. streams zstd JSONL incrementally;
4. splits document text into line-like segments;
5. applies the normal corpus normalization/Lao-ratio filter;
6. deduplicates accepted normalized lines by default;
7. caps accepted lines per source document to improve source diversity;
8. stops after a hard maximum number of source documents;
9. fails if the requested line count cannot be satisfied within that bound.

It does not intentionally download the rest of the shard after the requested sample is collected.

## Provenance metadata

The metadata JSON records:

- HPLT dataset/terms URL;
- official Lao map URL;
- selected shard URL(s);
- expected full-shard MD5 when the official `.md5` sidecar is available;
- the fact that the full shard was **not** downloaded/verified by a bounded stream;
- filter settings;
- documents/segments observed;
- maximum accepted lines per source document;
- maximum documents allowed to be streamed;
- highest-WDS-bin-first shard ordering;
- accepted-line count;
- output corpus SHA-256;
- the HPLT packaging-vs-underlying-text rights note.

Do not reinterpret `expected_full_shard_md5` as a checksum of the bounded sample. The generated corpus has its own SHA-256.

## Public capture-pack policy

Do **not** automatically copy sampled HPLT lines into public capture packs or public benchmark ground truth. HPLT's packaging license does not establish redistribution rights for every underlying extracted source.

Use the bounded sampler for provenance-recorded model-development/training text where the downstream user accepts responsibility for applicable source rights.

For public capture packs, use text whose underlying redistribution rights are independently clear, for example:

- project-authored text explicitly released for the benchmark;
- source-specific CC0/public-domain text with documented provenance;
- contributor text released under a compatible license.

Then feed that independently cleared corpus to `generate-capture-pack` / `generate-capture-suite`.

## Reproducibility note

A bounded stream is reproducible with respect to the source content available at the recorded official shard URL, filter configuration, and requested line limit. The output metadata and corpus SHA-256 are the authoritative record for a generated model-development/training corpus.

For a release-quality benchmark, freeze the resulting real-capture test set by hash; do not rely on re-streaming HPLT later to recreate benchmark ground truth.
