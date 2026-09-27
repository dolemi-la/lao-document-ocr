# Character n-gram shallow fusion

The project-owned recognizer supports an optional character n-gram language model during CTC prefix beam search.

This is **decoder-side shallow fusion**. It does not change recognizer weights.

## Why character n-grams

Lao text does not always use spaces as word boundaries consistently, so a character model is a simple language prior that does not require committing to a word tokenizer.

The LM can learn local patterns such as:

- common Lao character sequences
- spaces/punctuation where present
- common mixed Lao/English character transitions

It is intentionally small and portable JSON rather than a second neural model.

## Train

Use the recognizer's exact exported/training vocabulary:

```bash
lao-ocr train-char-lm \
  --corpus training/data/lao-lines-phetsarath.txt \
  --exclude-corpus training/data/dev-labels.txt \
  --vocabulary training/runs/crnn-v2/vocab.json \
  --output training/runs/crnn-v2/char-lm.json \
  --order 3 \
  --alpha 0.1
```

The artifact stores:

- n-gram order
- add-alpha smoothing value
- recognizer vocabulary checksum
- vocabulary size
- training statistics
- observed context/next-token counts

Lines containing characters outside the recognizer vocabulary are skipped and counted in `training_stats` rather than silently stripping characters.

## Exclude held-out text before training

The recognizer and its language model must both exclude dev/calibration/test labels from their training text. Splitting image samples alone does not protect a separately trained LM. `--exclude-corpus` accepts a plain-text label file and may be repeated for multiple held-out splits. It removes **every** normalized exact match, including duplicate occurrences, before vocabulary encoding or n-gram counting.

Generate the recognizer dev-label exclusion file from the same complete manifest and `dev_ratio` used for training, not by taking a prefix or randomly selecting images:

```python
from pathlib import Path
from lao_document_ocr.normalization import normalize_lao_text
from lao_document_ocr.training_manifest import (
    deterministic_split,
    load_training_manifest,
)

samples = load_training_manifest("training/generated/v1/manifest.jsonl")
_, dev = deterministic_split(samples, dev_ratio=0.1)
labels = sorted({normalize_lao_text(sample.text) for sample in dev})
output = Path("training/data/dev-labels.txt")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("\n".join(labels) + "\n", encoding="utf-8")
```

Add `--exclude-corpus` for separate calibration/test labels when those sets exist. Each explicit exclusion file must contain usable text; missing or empty files fail rather than silently disabling the guard. A corpus with no encodable training lines after exclusions fails before writing an LM artifact. The library equivalent is `train_character_ngram_language_model(..., exclude_texts=labels)`.

Artifacts and runtime model metadata record a `text_exclusions` object containing the strategy (`normalized-exact-text-v1`), a deterministic SHA-256 fingerprint of the sorted unique normalized exclusions, their count, and the number of training lines removed. They do not store the held-out labels themselves. Reordering exclusion files or repeating labels does not change that fingerprint. `training_stats.excluded_lines` is separate from the count of unknown-vocabulary lines skipped.

Legacy artifacts remain loadable, but absent exclusion metadata means **unverified**, not zero leakage. The CLI warns when no exclusion corpus is supplied. The supplied files still need review: this exact-text guard does not detect paraphrases, line segmentation differences, substrings within longer paragraphs, or document-level overlap. Preserve document/source-level split isolation as well. A changed LM has a new checksum and requires new decoder-bound calibration.

### Historical result correction

The 900-image Noto-era experiment's LM counts match a model trained on the entire 10,073-line corpus, which contains all 92 dev labels. Its reported `0.1799` fusion CER is contaminated and must not be treated as held-out LM improvement. The calibration MAE reported on that same fitted report is in-sample only. Those old artifacts remain historical diagnostics; no production default is justified by them.

## Use with beam decoding

```bash
lao-ocr recognize-line \
  --model training/runs/crnn-v2/recognizer.pt2 \
  --image line.png \
  --decoder beam \
  --beam-width 10 \
  --language-model training/runs/crnn-v2/char-lm.json \
  --language-model-weight 0.25 \
  --language-model-token-bonus 0.0
```

The recognizer rejects the LM if its vocabulary checksum does not match the recognizer vocabulary.

A language model requires `--decoder beam` and a positive LM weight.

## Scoring

Beam ranking uses:

```text
CTC acoustic log probability
+ LM weight * character n-gram log probability
+ token bonus * output length
```

The returned OCR confidence remains based on the selected prefix's **acoustic CTC probability**, not the fused ranking score.

`recognize-line` prints the decoder ranking score and LM log probability as diagnostics when beam decoding is used.

## Token bonus

Character-language-model log probabilities are negative and therefore naturally penalize longer prefixes. `--language-model-token-bonus` can compensate for excessive deletion/shortening.

Do not assume a universal value. Tune it on held-out development data.

## Tune on dev only

Recommended sequence:

1. freeze the recognizer weights
2. train the LM from approved training text with dev/calibration/test labels excluded
3. evaluate a small grid of LM weights / token bonuses on **dev**
4. select one decoder configuration
5. generate a fresh dev recognizer report
6. fit confidence calibration on its designated calibration split; use a separate split to measure calibration quality
7. freeze the configuration
8. evaluate once on the frozen test set

Example candidate grid:

```text
LM weight:   0.10, 0.20, 0.30, 0.40
Token bonus: 0.00, 0.03, 0.06, 0.10
```

Those are search examples, not recommended universal values.

## Calibration binding

Confidence calibration artifacts are bound to:

- decoder (`greedy` or `beam`)
- LM artifact SHA-256, when used
- LM weight
- LM token bonus

A calibration file produced for plain beam search cannot be reused with LM-guided beam search, and changing the LM or fusion parameters requires recalibration.

## Full-page/API use

CLI options are available on:

- `convert-document`
- `benchmark`
- `recognize-line`
- `benchmark-recognizer`

API environment variables:

```text
OCR_DECODER=beam
OCR_BEAM_WIDTH=10
OCR_LANGUAGE_MODEL_PATH=/models/char-lm.json
OCR_LANGUAGE_MODEL_WEIGHT=0.25
OCR_LANGUAGE_MODEL_TOKEN_BONUS=0.0
```

The GPU deployment preset exposes the same variables. Beam search and n-gram scoring currently run on CPU after neural logits are produced, so measure latency as well as CER.

## Data policy

Use only text whose training/reuse rights are documented. The project permits provenance-recorded HPLT v3 Lao for model-development workflows under its source-policy caveats; HPLT CC0 applies to dataset packaging, while underlying extracted-text rights remain source-dependent. See `benchmark-source-review.md` and `hplt-sampling.md`.

Do not train the LM on dev, calibration, or frozen test-set ground truth. Keep their labels out of every upstream training corpus, including corpus text used to synthesize recognizer images. That would otherwise leak benchmark answers into decoding.

## Release gate

LM fusion should only be enabled in a published default if the fixed-set benchmark gate shows a real improvement without unacceptable regressions on important subsets/tags.
