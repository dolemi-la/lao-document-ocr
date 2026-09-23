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
  --corpus training/data/lao-lines.txt \
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
2. train the LM from approved training text
3. evaluate a small grid of LM weights / token bonuses on **dev**
4. select one decoder configuration
5. generate a fresh dev recognizer report
6. fit confidence calibration from that report
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

Use only text whose training/reuse rights are documented. The current benchmark-source policy approves HPLT v2 Lao as CC0 **text input**; see `benchmark-source-review.md`.

Do not train the LM on the frozen test-set ground truth. That would leak benchmark answers into decoding.

## Release gate

LM fusion should only be enabled in a published default if the fixed-set benchmark gate shows a real improvement without unacceptable regressions on important subsets/tags.
