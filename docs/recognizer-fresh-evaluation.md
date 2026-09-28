# Frozen-model Phetsarath synthetic evaluation

## Purpose and limits

The [optimization-budget study](recognizer-budget-study.md) produced a locally
exported candidate selected at epoch 72. Its 13.78% CER was measured on only 12
development images that had been inspected repeatedly. This follow-up freezes
that artifact and evaluates separately selected text without training, decoder
tuning, language-model fitting, or checkpoint reselection.

This is a **private synthetic challenge**, not an optical benchmark or a proof
of document-independent generalization. "New" means no normalized exact-label
match against the supplied inventory of earlier image manifests. It does not
mean unseen source documents, no paraphrases, or complete provenance for every
historical experiment. Once measured, this challenge must not be described as
uninspected evidence in later tuning.

## Reusable input audit

```bash
lao-ocr audit-recognizer-holdout \
  --manifest training/runs/challenge/manifest.jsonl \
  --exclude-manifest training/runs/prior/train.jsonl \
  --exclude-manifest training/runs/prior/dev.jsonl \
  --vocabulary training/runs/frozen-model/vocab.json \
  --output training/runs/challenge/holdout-audit.json
```

`--exclude-manifest` is required and repeatable. Supply actual training,
validation, calibration, previously inspected evaluation, and other relevant
manifests; do not use empty placeholders. All supplied image files must still
exist. The Python entry point is
`training_expansion.audit_recognizer_holdout(excluded_manifests, evaluation, vocabulary)`.

The audit reuses the pinned-image verification and ordered identity convention
of [training-expansion.md](training-expansion.md). It requires nonempty inputs,
valid lowercase image SHA-256 pins, matching actual image bytes, unique IDs
within each manifest, and nonempty normalized labels. It rejects evaluation
labels that occur in any supplied exclusion manifest, byte-identical images
across those sets, duplicate evaluation image bytes, and evaluation characters
outside the supplied vocabulary. It never silently drops or rewrites a sample.

Different images of the same evaluation text are permitted and explicitly
counted as one text group. IDs are local to each exclusion manifest; repeated
IDs across different prior manifests do not hide either manifest's labels.
The successful report contains counts and length-framed fingerprints, not
labels, predictions, sample IDs, or image paths. Fingerprints are reproducibility
identities, not anonymization guarantees. Output reports cannot be overwritten;
validation failures do not create an output directory.

This is an explicit audit command, not an automatic guard inside training or
benchmarking. Passing it only establishes normalized exact-label and
byte-identical-image isolation against the supplied inputs. It does not inspect
a model's training history, prove the vocabulary belongs to the model, certify
rights, detect perceptual duplicates or text substrings, or establish
source/document separation. The report marks these boundaries explicitly.
The experiment below separately pins the vocabulary, model sidecars, artifacts,
source corpus, renderer, and exclusion inputs.

## Selection recorded before inference — 2026-09-29

Starting source revision: `c954318a25b388bc8e647aecd7e054aa903fae8b`.
No checkpoint was updated. Primary model: the epoch-72 candidate from the
288-image, 2,880-update budget study. Secondary model: the older epoch-79
baseline trained on 72 images for 720 updates. These have the same 93-class
vocabulary but **different training data and compute budgets**; this comparison
cannot isolate a data-only or compute-only effect.

The input corpus is the existing 9,694-line Phetsarath-compatible local corpus.
Selection excluded every label in seven known prior full manifests, including
the Noto-era 900-image set, earlier diagnostic datasets, both Phetsarath 300-image
versions, and the expanded manifest. Their union contains 1,142 normalized text
groups. Exclusions were checked before model inference, not after observing
scores. Existing source-registry HPLT private-model-development caveats apply;
this work does not authorize publishing corpus text, synthetic images, font
files, or model weights.

Eligible candidates had 8–32 normalized characters, fitted the frozen vocabulary,
and belonged to the existing reserved hash bucket below 1,000 out of 10,000.
There were 167 eligible groups. The first 160 under a fixed salted SHA-256 order
were selected. Selection did not use either model's predictions or confidence.

Each text was rendered in Phetsarath OT Regular at font size 48, with strict glyph
coverage, then received one deterministic variant per profile: `clean-scan`,
`noisy-scan`, and `phone-photo`. The three variants share their clean source
render. Seed 20260929 and geometry version 2 were fixed. The result is **160 text
groups / 480 correlated images**, not 480 independent documents. Each profile
has 3,354 reference characters; all profiles together have 10,062.

The pinned inputs pass the reusable audit and CTC preflight at height 48 / width
768: zero capacity failures, minimum timestep margin 14, zero width-capped
images, and resized widths 103–387. This is capacity/identity evidence, not a
visual legibility or real-capture certification. The same-corpus, known-vocabulary,
short-line selection and fixed render size limit generalization claims.

## Frozen-candidate results

Both artifacts used greedy decoding, no LM, no calibration, and MPS inference.
All eight predeclared 60-image chunks completed for each artifact. No score-based
sample removal, retraining, parameter changes, or checkpoint reselection occurred.
CER normalizes both reference and hypothesis using the existing benchmark policy.

| Synthetic condition | Images / text groups | Candidate edits / characters | Candidate CER | Older baseline CER | Candidate exact lines |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean-scan profile | 160 / 160 | 265 / 3,354 | 7.9010% | 41.7412% | 49 / 160 |
| Noisy-scan profile | 160 / 160 | 288 / 3,354 | 8.5868% | 44.2457% | 47 / 160 |
| Phone-photo profile | 160 / 160 | 483 / 3,354 | 14.4007% | 53.2499% | 27 / 160 |
| All profiles | 480 / 160 | 1,036 / 10,062 | **10.2962%** | **46.4122%** | **123 / 480** |

The candidate makes 3,634 fewer character edits than the older baseline on this
specific set. Both have zero normalized-empty predictions. Only 25.625% of
candidate lines are completely correct, and only 19 of the 160 text groups are
exact in all three profiles. The baseline has no completely correct lines on
this challenge. Lower CER is not equivalent to reliable whole-line transcription.

The aggregate whitespace-token WER is 467/1,029 (45.3839%) for the candidate and
973/1,029 (94.5578%) for the baseline. These are whitespace-token metrics, not
validated Lao linguistic word segmentation. Phone-style degradation is the
hardest profile here, but these images are synthetic, not actual phone photos.

The challenge's 10.30% CER and the older 12-image development score of 13.78%
use different samples and denominators. Their difference is **not** evidence of
a new training improvement: the candidate weights were identical throughout.
There are no confidence-interval or statistical-significance claims; three
variants of each text are correlated. The selected candidate remains local and
experimental. No Tesseract comparison or production promotion was performed.

## Integrity and CPU cross-check

Every twentieth image was also evaluated on CPU: 24 predetermined cross-checks
per model, covering all three profiles. There were zero normalized-prediction
mismatches in those checks. This does not assert all 480 images were evaluated
on CPU. All model, sidecar, vocabulary, input-manifest, corpus, font, protected
source-code, and previous completion-report hashes stayed unchanged.

Chunk and completion reports contain counts, metrics, and fingerprints, not
reference or OCR strings. Labeled manifests and images remain local and ignored.
The CLI audit independently reproduced the preflight audit. Total new files are
small; no new checkpoint, export, or training process was needed.

Local evidence:

```text
training/runs/fresh-evaluation-c954318/
  prepare.py
  experiment-plan.json
  manifest.jsonl
  images/
  preflight.summary.json
  cli-holdout-audit.summary.json
  evaluate.py
  evaluation-runner.json
  candidate-<offset>.summary.json
  baseline-<offset>.summary.json
  complete.py
  completed.summary.json
```

| Evidence | SHA-256 |
| --- | --- |
| Pre-inference plan | `842e228f53041a0202cf9864c237333488b937c9b4d12772b58dbc8b48e7b067` |
| Evaluation manifest | `7f70bf8fa2719bab74ba9ad207c23e1252f062e13c8ecca6a0323a55d5022026` |
| Completion summary | `13f949ff20c03bde32e8e3b6152d90a567d08e56c0f9193dfc35d673e6a28df1` |
| Frozen candidate export | `66f6bbd8cb2ea078cb2bbb1b14ec247329d30bc96da556bc8740fa0d2605ad0f` |
| Older baseline export | `4c65340d48e106ccf2bc8b58e866a008d1fc2c71636c5bfa4ddd6baaf5fb5c2b` |

## Next decision

The expanded candidate performs better on these new exact-label groups than the
older small-data baseline, but whole-line errors and synthetic-only scope still
prevent a release claim. Preserve these frozen inputs/results. Diagnose the
phone-profile failures without tuning on this challenge; any revised model
needs a separately isolated evaluation set. Longer lines, additional character
coverage, source-level separation, and genuine reviewed optical captures remain
unvalidated. The real benchmark and published Tesseract baseline are still open
requirements, not satisfied by this result.
