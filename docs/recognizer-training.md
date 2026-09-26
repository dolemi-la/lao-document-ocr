# Lao line recognizer training

The first project-owned recognizer is a CRNN-style model:

```text
grayscale line image
  -> convolutional feature extractor
  -> bidirectional LSTM
  -> character classifier
  -> CTC loss / greedy CTC decoding
```

Model version: `crnn-ctc-v2`. The classifier starts with a negative blank-logit bias to reduce early CTC blank collapse.

This is training infrastructure, not a published high-accuracy Lao model yet.

## 1. Install training dependencies

```bash
pip install -e ".[train]"
```

Or build the dedicated CPU training container:

```bash
docker build -f Dockerfile.train -t lao-document-ocr-train .
```

The training Dockerfile installs PyTorch from the official CPU wheel index so a local CPU training image does not pull CUDA runtime packages. GPU training should use a separate CUDA-specific image/preset later.

## 2. Prepare text

For a bounded HPLT v3 Lao source sample (optional `data` dependency):

```bash
pip install -e '.[data]'
lao-ocr sample-hplt-lao \
  --output training/data/hplt-lao-10k.txt \
  --metadata training/data/hplt-lao-10k.meta.json \
  --limit 10000 \
  --max-lines-per-document 4 \
  --max-documents 10000
```

HPLT's CC0 statement applies to dataset packaging, not automatic ownership of the underlying extracted text. Preserve the sampler metadata, accept downstream responsibility for applicable source rights, and do not automatically republish sampled lines as public benchmark ground truth. See [hplt-sampling.md](hplt-sampling.md).

```bash
lao-ocr prepare-corpus   --input source.jsonl   --format jsonl   --field text   --output training/data/lao-lines.txt   --min-lao-ratio 0.5
```

## 3. Render labeled lines

```bash
lao-ocr generate-synthetic   --corpus training/data/lao-lines.txt   --output training/generated/v1   --font /usr/share/fonts/truetype/noto/NotoSansLao-Regular.ttf   --variants-per-line 3   --augmentation-profile balanced
```

The generated manifest is `training/generated/v1/manifest.jsonl`. For augmentation profiles and their reproducibility metadata, see [synthetic-augmentation.md](synthetic-augmentation.md).

For large corpora, generation can be split into deterministic chunks with `--start-line` plus `--max-samples`. The offset is zero-based in the prepared corpus; sample IDs, seeds, font rotation, and balanced augmentation profiles stay aligned with an equivalent single full run. This lets interrupted local generation resume in bounded directories that can be merged into one training manifest without changing sample identity.

## 4. Train

```bash
lao-ocr train-recognizer   --manifest training/generated/v1/manifest.jsonl   --output training/runs/crnn-v2   --epochs 20   --batch-size 32   --dev-ratio 0.1   --device auto
```

`--device auto` prefers CUDA, then Apple MPS, then CPU. You can also select `cpu`, `cuda`, or `mps` explicitly. The resolved runtime device is recorded in `metadata.json`.

The default recurrent encoder remains the bidirectional `crnn-ctc-v2` model for checkpoint compatibility. New experiments can opt into `--unidirectional`, which records `crnn-ctc-v3` and removes right-padding dependence from the recurrent prefix. A bidirectional LSTM can change valid-prefix predictions when extra white padding is appended on the right, while the forward-only v3 prefix is stable. Keep v3 opt-in until it is evaluated on the fixed real benchmark; current synthetic sanity evidence is development-only.

The training default `--max-width 768` is sized for the current bounded model-development corpus (up to 180 normalized characters); narrower custom widths remain available for shorter-line datasets.

Before model/optimizer work starts, recognizer training now preflights every sample against CTC timestep capacity using its prepared image width. A run fails immediately if any target cannot fit; successful runs persist the sample count, minimum timestep margin, and maximum required/available timesteps in training state, checkpoint metadata, and final metadata.

Train and dev batches are now padded to that configured fixed width rather than only to each batch maximum. This matches the fixed-width exported inference path for bidirectional v2, so the backward LSTM sees the same right-padding regime during development evaluation and deployment. The padding semantics are versioned as `fixed-max-width-v1` and are part of the strict resumable-training state contract.

The output directory contains:

- `recognizer.pt` — training checkpoint
- `metadata.json` — architecture/training metadata and checkpoint SHA-256
- `vocab.json` — exact character vocabulary
- `training-state.pt` — atomic latest/best training state for exact long-run resume

The train/dev split is deterministic from sample IDs.

Development CER decoding is width-aware: each sample is decoded only through its valid CTC timesteps, excluding batch padding exactly like exported recognizer inference. The current metric semantics are recorded as `valid-timestep-v1` in training state/checkpoints. When an older weights-only checkpoint lacks that version, its baseline dev CER is recomputed before new best-checkpoint decisions are made.

Long runs also write `training-state.pt` atomically after every completed epoch. It contains the latest weights, best weights, optimizer state, deterministic shuffle-generator state, Python/NumPy/Torch RNG state, training history, the exact ordered training-sample checksum, and the resolved runtime device. To continue, set `--epochs` to the new total epoch target and pass `--resume-from`.

Exact `training-state.pt` resume is intentionally strict: the training samples, model/vocabulary, compatible training settings, resolved device, optimizer state, shuffle-generator state, and RNG state must match and restore successfully. A mismatch is rejected rather than silently continuing a different experiment.

Legacy `crnn-ctc-v2` states created before the `bidirectional` config field existed are normalized as `bidirectional=true`, matching v2 semantics. An explicit conflicting value is still rejected.

Example: `lao-ocr train-recognizer --manifest training/generated/v1/manifest.jsonl --output training/runs/crnn-v2 --epochs 50 --batch-size 32 --device auto --resume-from training/runs/crnn-v2/training-state.pt`.

A legacy/final `recognizer.pt` can also be used as a weights-only fallback. In that case optimizer, shuffle-generator, and RNG state cannot be restored; metadata records that limitation and continuation starts from the best stored epoch rather than pretending to restore a later state.

## 5. Export for portable inference

```bash
lao-ocr export-recognizer   --checkpoint training/runs/crnn-v2/recognizer.pt   --output training/runs/crnn-v2/recognizer.pt2
```

The exported `torch.export` artifact uses runtime-device-relative recurrent state, so the same artifact can run on supported CPU, CUDA, or Apple MPS runtimes.

The exported `torch.export` artifact uses a fixed padded input width. The original valid line width is retained during decoding so padded pixels do not contribute CTC output.

The sidecar `recognizer.pt2.json` contains:

- model version
- vocabulary
- vocabulary checksum
- artifact SHA-256
- image dimensions
- width downsample factor

## 6. Recognize one cropped line

```bash
lao-ocr recognize-line   --model training/runs/crnn-v2/recognizer.pt2   --image line.png
```

This command is for cropped-line model development. An experimental full-page engine now combines deterministic line detection with the project-owned recognizer; see [owned-ocr-engine.md](owned-ocr-engine.md). Tesseract remains the default until the owned model beats the fixed real-document baseline.

## CTC capacity

A target needs at least one timestep per character, plus an extra timestep where identical adjacent characters require a separating blank. Training rejects samples that cannot fit the available sequence width rather than silently producing invalid CTC targets.

## Accuracy

Do not publish results from tiny synthetic smoke runs as model accuracy. A model release is only meaningful after evaluation against the fixed real-document benchmark described in the Phase 1 roadmap.

## 7. Regression benchmark an exported recognizer

```bash
lao-ocr benchmark-recognizer \
  --manifest training/generated/v1/manifest.jsonl \
  --model training/runs/crnn-v2/recognizer.pt2 \
  --output training/runs/crnn-v2/benchmark.json
```

The report includes aggregate CER/WER plus per-sample hypotheses, timing, and the raw probability score. Raw confidence is diagnostic only until calibrated on a held-out development set.

## 8. Calibrate confidence on held-out dev data

```bash
lao-ocr calibrate-recognizer \
  --report training/runs/crnn-v2/dev-report.json \
  --output training/runs/crnn-v2/calibration.json \
  --bins 10
```

The current calibrator maps raw mean timestep probability to observed character accuracy using quantile bins. Do not fit it on the final test set.

Use the calibration file with `recognize-line`, `benchmark-recognizer`, or the full-page owned OCR engine.

## Development sanity result

A small local sanity run was used to verify that the training code can actually learn rather than only execute:

- 12 hand-written Lao/mixed lines
- 5 synthetic variants per line (60 images total)
- deterministic 80/20-style sample split
- 50 epochs on CPU
- model `crnn-ctc-v2`
- best dev CER observed: ~0.494
- full 60-sample manifest CER after export: ~0.514
- several generated Lao lines were recognized exactly

These figures are **not model accuracy claims**. The text set is tiny, synthetic, and contains multiple augmented variants of the same phrases. Its only purpose is a training-pipeline sanity check.

### Larger synthetic development run

A later bounded development run used 900 unique normalized text lines from the approved local model-development corpus, three reviewed Noto Lao font families, and the balanced clean/noisy/phone augmentation schedule. This remains synthetic development evidence only.

For the bidirectional `crnn-ctc-v2` candidate on its deterministic 92-line dev split:

- epoch-80 greedy CER: about `0.2083`
- epoch-80 greedy WER: about `0.6932`
- beam width 10 without a language model changed CER only slightly to about `0.2054`
- character 3-gram shallow fusion improved the selected dev configuration to about `0.1799` CER / `0.5841` WER at language-model weight `0.4` and token bonus `+0.05`
- the matching confidence calibration changed mean absolute confidence error only slightly, from about `0.0618` to `0.0616`

The opt-in unidirectional `crnn-ctc-v3` architecture was also run on the same 900-line dataset for 20 epochs. Its best dev CER remained about `0.9830`, so this experiment does not support replacing v2 with v3. The v3 path remains experimental while its padding-invariance benefit is weighed against the current quality gap.

Do not compare these numbers with Tesseract as publishable accuracy. The real-model decision still waits for the frozen rights-clear optical benchmark and its published Tesseract baseline.

An earlier initialization collapsed entirely to CTC blank predictions even while loss decreased. `crnn-ctc-v2` therefore initializes the blank-class output bias negatively. Keep an explicit blank-collapse sanity test when changing the recognizer architecture or loss setup.

## Decoder selection

Greedy CTC decoding remains the default:

```bash
lao-ocr recognize-line \
  --model training/runs/crnn-v2/recognizer.pt2 \
  --image line.png \
  --decoder greedy
```

An optional prefix beam-search decoder is available for model-development experiments:

```bash
lao-ocr recognize-line \
  --model training/runs/crnn-v2/recognizer.pt2 \
  --image line.png \
  --decoder beam \
  --beam-width 10
```

The same `--decoder` / `--beam-width` options are supported by `benchmark-recognizer`, full-page `benchmark`, and owned-model document conversion.

Beam decoding uses log-space CTC prefix beam search and returns the highest summed CTC prefix probability rather than merely collapsing the single most likely path.

### Confidence calibration is decoder-specific

Greedy and beam decoders produce different raw confidence distributions. Calibration artifacts therefore record the decoder they were fitted from.

- legacy calibration files without a decoder field are treated as `greedy`;
- a greedy calibration cannot be loaded with beam decoding;
- fit a separate calibration from a beam-decoded held-out development report before using calibrated beam confidence.

### Current sanity result

On the existing 60-image synthetic overfit sanity set, `beam-width=10` produced the same CER as greedy (`~0.5143`), slightly worse WER, and roughly twice the runtime. This is **not a real-world accuracy result** and is not evidence to change the default decoder. Decoder choice should be decided only on the frozen rights-clear real benchmark.

## Character n-gram shallow fusion

After the recognizer vocabulary is frozen, you can train an optional character n-gram language model:

```bash
lao-ocr train-char-lm \
  --corpus training/data/lao-lines.txt \
  --vocabulary training/runs/crnn-v2/vocab.json \
  --output training/runs/crnn-v2/char-lm.json
```

Use it only with beam decoding and tune fusion parameters on held-out dev data. Confidence calibration must be refit for the exact LM checksum/weight/token bonus. See [language-model.md](language-model.md).

## Leakage-safe train/dev split

Recognizer training splits by **normalized ground-truth text group**, not by generated sample ID. All augmented/rendered variants of the same normalized text therefore stay together in either train or dev.

The current strategy is recorded as:

```text
normalized-text-group-sha256-v1
```

in checkpoint metadata and exported recognizer metadata.

This prevents augmentation leakage such as:

```text
train: line-0001 variant A -> "ສະບາຍດີ"
dev:   line-0001 variant B -> "ສະບາຍດີ"
```

which would make dev CER look better without testing generalization to unseen text.

At least two unique normalized text groups are required. Tiny datasets containing multiple images of only one text string are rejected instead of manufacturing a leaked dev split.
