# Phetsarath longer-line fine-tuning: completed eight-epoch experiment

## Fixed scope and endpoint

This is a new private synthetic-development experiment based on source revision
`d30349386cd437f1086553b566cafc97c2af2a79`. It initializes from the **fixed final
epoch-10 weights** of the [coverage experiment](recognizer-coverage-finetuning.md),
not that experiment's development-selected epoch-9 checkpoint. Its optimizer,
shuffle state, history, and checkpoint selection start anew. Earlier models,
training runs, and their reported endpoints remain unchanged.

The plan was recorded before image generation or training. It fixed **eight
epochs / 2,016 optimizer updates**, batch size 8, AdamW learning rate 0.0002,
seed 20261002, and Apple MPS. The existing bidirectional CRNN v2, 93-class
vocabulary including blank, 48-pixel input height, 768-pixel padded width,
and normalized valid-timestep greedy decoding remain unchanged. No language
model, beam-search, calibration, architecture, or production default changed.

The primary endpoint is final epoch 8. Development-based selection remains a
separate criterion; in this run it also selected epoch 8. Neither challenge
was used for epoch selection, learning-rate changes, extra stages, or stopping.
All eight foreground stages completed, totaling 16,128 training sample
presentations. No stage remains running.

## Training-only expansion and a separate longer-line challenge

The parent had 1,152 training images covering 576 normalized text groups and
156 development images covering 60 groups. The **entire ordered development
set is unchanged**, including identifiers, normalized labels, and image hashes.
All 1,308 parent training/development images were preserved.

The existing private Phetsarath-compatible corpus provided 288 previously unused
training text groups: 144 with 33–48 characters and 144 with 49–64 characters.
Each was rendered in clean-scan, noisy-scan, and phone-photo synthetic styles,
adding 864 images. The expanded training set has **2,016 images / 864 groups**.

A separate **144-image longer-line challenge** contains 48 unused 49–64-character
groups rendered in the same three styles. It was frozen before training and is
not included in the training/development manifest. The existing 480-image
short-line challenge remains unchanged and excluded from training.

Selection used salted SHA-256 ordering after excluding normalized labels from
every supplied previous manifest, including the parent dataset and old challenge.
The established normalized-text hash buckets still determine train/development
eligibility. Checks found zero exact-label overlap between training/development,
zero exact-label overlap with either challenge, and zero training/challenge
image-byte overlap. This is not proof of source/document separation or absence
of semantic near-duplicates.

Phetsarath OT Regular remains the only font. Strict shaped-font coverage checks
passed. All 2,172 training/development images and all 144 longer-line challenge
images passed CTC capacity preflight. Two training images reached the configured
width cap and were resized proportionally to an effective height as low as 46
pixels; they were retained and recorded, not silently removed. No new challenge
image was width capped. Minimum timestep margins were 15 for the combined
training/development set and 20 for the longer-line challenge.

These are generated images, not genuine scans or camera photographs. Source
material retains the existing HPLT private-model-development limitations. No
corpus text, generated images, font, or model weights are redistributed.

## Matched parent versus final results

Both models use the same native-model aggregate evaluator, fixed-width
preprocessing, and normalized valid-timestep decoder. The parent re-evaluation
exactly reproduced its preceding development and short-challenge results,
including per-profile counts, not just rounded headline scores.

| Evaluation set | Images | Reference characters | Parent edits | Final edits | Parent CER | Final CER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unchanged development | 156 | 5,490 | 527 | 307 | 9.60% | 5.59% |
| Unchanged short-line challenge | 480 | 10,062 | 634 | 429 | 6.30% | 4.26% |
| New longer-line challenge | 144 | 8,214 | 1,205 | 614 | 14.67% | 7.48% |

Completely correct lines on the short challenge increased from **188/480
(39.167%)** to **248/480 (51.667%)**. On the longer challenge they increased
from **3/144 (2.083%)** to **18/144 (12.5%)**. No empty predictions were produced
on either challenge. Development exact lines increased from 24/156 to 39/156.

| Short-line synthetic challenge profile | Images | Parent CER | Final CER | Final exact lines |
| --- | ---: | ---: | ---: | ---: |
| Clean-scan style | 160 | 5.16% | 3.61% | 87/160 |
| Noisy-scan style | 160 | 5.46% | 3.61% | 88/160 |
| Phone-photo style | 160 | 8.29% | 5.58% | 73/160 |

The new longer-line challenge exposes a substantial remaining weakness:

| Longer-line synthetic challenge profile | Images | Final character edits | Final CER | Final exact lines |
| --- | ---: | ---: | ---: | ---: |
| Clean-scan style | 48 | 100/2,738 | 3.65% | 10/48 |
| Noisy-scan style | 48 | 113/2,738 | 4.13% | 8/48 |
| Phone-photo style | 48 | 401/2,738 | 14.65% | 0/48 |

**None of the 48 longer phone-style challenge lines was completely correct.**
The lower aggregate CER must not hide this result. The previously observed
phone-style development subset improved from 276/1,755 (15.73%) to 149/1,755
(8.49%), but that subset and the new longer-line challenge are different sets
with different length distributions; their percentages are not interchangeable.

Final training-set CER was 996/73,477 (1.356%), with 1,576/2,016 exact lines.
That is an in-sample fit result, not independent recognition accuracy.
Whitespace-token WER was 263/1,029 (25.56%) on the short challenge and 202/408
(49.51%) on the longer challenge. These whitespace-token measurements are not
a comprehensive linguistic word-segmentation evaluation for Lao.

## Epoch history

| Epoch | Optimizer updates | Development CER |
| ---: | ---: | ---: |
| 1 | 252 | 8.0510% |
| 2 | 504 | 7.1767% |
| 3 | 756 | 6.7031% |
| 4 | 1,008 | 6.7031% |
| 5 | 1,260 | 6.3206% |
| 6 | 1,512 | 5.9016% |
| 7 | 1,764 | 5.9745% |
| 8, fixed final endpoint | 2,016 | 5.5920% |

The tie at epoch 4 and regression at epoch 7 remain in the history. Each stage
verified actual weight changes, optimizer-step counts, initialization provenance,
unchanged earlier history, and strict restoration of optimizer, shuffle, RNG,
device, sample-identity, and padding state when resuming the next epoch.

## Private artifacts and reproducibility

Files remain local and Git-ignored under:

```text
training/runs/long-line-finetune-d303493/
  experiment-plan.json
  prepared.summary.json
  common.py
  prepare.py
  manifest.jsonl
  long-challenge.jsonl
  new-images/
  train_stage.py
  epoch-001.summary.json ... epoch-008.summary.json
  evaluate.py
  parent-evaluation.summary.json
  final-evaluation.summary.json
  complete.py
  completed.summary.json
  model/training-state.pt
  model/recognizer.pt
  model/recognizer.pt2
  final/recognizer.pt
  final/recognizer.pt2
```

Plan SHA-256:
`e4adb94d0fc3065e198403734f1197f5e8e7bb7a900072fd56f3bb844bf69f46`.
Final export SHA-256:
`d26557244eb725f2270be4a01f25f8a1de2d5945586b470416b9ccddfe3e9584`.
The final and development-selected exports have the same hash because both use
epoch-8 weights. The latest resumable training state remains in `model/`.

Twenty-four deterministically selected images, eight each from development and
both challenges, produced zero native-CPU/exported-CPU prediction mismatches and
zero native-CPU/MPS prediction mismatches. This checks selected inputs and these
devices, not every possible image or deployment platform.

Completion reverified **89 protected path hashes** and **2,796 manifest image
hashes** across training, development, and both challenges. This count includes
the 1,308 preserved parent images. All run scripts are fingerprinted by the plan;
aggregate reports contain no reference or prediction strings.

## Interpretation and next decision

This run improves the matched synthetic metrics and crosses 50% exact lines on
the short challenge, but remains far from reliable longer phone-photo-style
recognition. One seed, correlated style variants, fixed known vocabulary, and a
shared corpus limit generalization claims. The old challenge has been inspected
in earlier experiments; the new challenge has unseen exact labels relative to
the supplied manifests, not independently established document sources.

More training data, a new learning rate, and further optimizer updates changed
together. This is not a controlled causal estimate of any one factor. The
production model and runtime defaults remain unchanged, and the recognizer
release criterion is still unmet: no rights-reviewed real optical benchmark
or matched Tesseract comparison was completed here.

The eight-epoch experiment is complete. The next useful model step is a
separately planned diagnosis of longer phone-style errors and real scan/photo
validation, not quietly extending this budget or moving either challenge into
training. Preserve this endpoint and its unfavorable whole-line result.

## Software gates and independent closing audit

The full local Python suite passed 1,761 tests, with two optional S3 SDK tests
skipped because that extra was not installed, and six dependency deprecation
warnings. Source Ruff checks, web lint, all 145 web tests, the production web
build, and `git diff --check` passed. Local documentation links resolve.

A separate closing audit ran the existing `audit_training_expansion` implementation
against the parent and expanded manifests, verified the unchanged ordered
156-image development set and fixed vocabulary, checked challenge exclusions,
and rehashed all 2,796 manifest images and 89 protected paths. It also checked
completion-report bindings, the final export hashes, the eight recorded stages,
and the unfavorable zero-exact-line result for the longer phone-style subset.

Only aggregate documentation is committed by this experiment; production and
training source code are unchanged. Local gate logs and review evidence are
retained in `benchmarks/private/model-long-lines-d303493/`. Neither passing
software tests nor export parity establishes real-document OCR accuracy.
