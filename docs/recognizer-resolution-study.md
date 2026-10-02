# Input-resolution transfer: completed matched-data study

## Question and fixed design

This private synthetic experiment tested whether larger input images help the
remaining longer-line recognition errors. It starts from the final phone-diversity
checkpoint selected in the [previous study](recognizer-phone-diversity-study.md),
not a replacement for that study or its data. Base source revision:
`3c6ecf2c4811fefc6b7c260682d915245b025099`.

The new [explicit initialization-resize option](recognizer-initialization-resize.md)
transfers the same selected weights while permitting only input height and padded
width to change. Architecture, vocabulary and tensor keys/shapes remain checked;
exact resume is unchanged. Both arms start fresh optimizer, shuffle/RNG and
history state. The parent remains unmodified.

Before training, the plan fixed both arms at **three epochs, 945 optimizer
updates and 7,560 sample presentations each**:

| Arm | Input height | Padded width | Training images | Development images |
| --- | ---: | ---: | ---: | ---: |
| h48 control | 48 | 768 | 2,520 | 156 |
| h64 transfer | 64 | 1,024 | 2,520 | 156 |

Both arms use AdamW at learning rate 0.0001, batch size 8, seed 202610023,
bidirectional CRNN v2, 93 vocabulary classes including blank, Apple MPS and two
Torch CPU threads. The frozen order alternates arms after each epoch. There is
no early stopping, learning-rate revision, extra stage, decoder/LM change or
production promotion. Only the fixed final epoch is the primary endpoint.

The same existing Phetsarath images, labels and ordered split are used in both
arms: **864 training text groups**, the existing development set, and the unchanged
480-image short and 144-image long challenges. No image, label, font or vocabulary
was generated, changed, dropped or transferred between splits. Holdout audits
found zero normalized-label or image overlap with the supplied training/other
challenge exclusions. These audits do not establish source/document independence.

All 2,676 training/development images passed CTC capacity at both geometries.
The two existing width-capped examples remained in each arm. Minimum timestep
margins were 15 at h48 and 23 at h64. The h64 setting changes both resizing and
sequence/padding length, not just CNN height. Equal optimizer steps and source
images are **not** equal FLOPs, memory, elapsed compute or adaptation opportunity.

## Final results, including regressions

Both models use the same counting/normalization/valid-timestep greedy procedure
on the same raw images, with each model's own declared input geometry. The parent
was re-evaluated at its original 48 × 768 geometry and reproduced all earlier
per-profile and overall counts exactly. Lower CER is better.

| Evaluation set | Images | Parent CER | h48 control CER | h64 transfer CER |
| --- | ---: | ---: | ---: | ---: |
| Unchanged development | 156 | 4.85% | 4.88% | 4.70% |
| Frozen short challenge | 480 | 3.99% | 4.06% | 4.35% |
| Frozen long challenge | 144 | 6.51% | 6.20% | 5.50% |

| Evaluation set | Parent exact lines | h48 control exact lines | h64 transfer exact lines |
| --- | ---: | ---: | ---: |
| Unchanged development | 41/156 | 43/156 | 40/156 |
| Frozen short challenge | 257/480 | 249/480 | 235/480 |
| Frozen long challenge | 20/144 | 22/144 | 19/144 |

The h64 model has **452/8,214** longer-challenge character edits, versus
535/8,214 for the parent and 509/8,214 for h48. However, its short-challenge
edits **increase to 438/10,062**, versus 401/10,062 for the parent. Its exact
short lines fall from **257 to 235**, and exact long lines fall from **20 to 19**.
Lower total character error on long lines does not mean more error-free lines.

Development CER improves by only eight edits versus the parent (258 versus 266
of 5,490 characters). The control has 268 development edits. The h64 model also
has fewer exact development lines than either parent or control. These are
small, metric-dependent differences—not a claim of statistical significance.

| Synthetic phone subset | Parent CER | h48 control CER | h64 transfer CER |
| --- | ---: | ---: | ---: |
| Unchanged development, phone-style subset | 6.95% | 6.44% | 5.53% |
| Frozen short challenge, phone-style subset | 4.89% | 4.83% | 4.92% |
| Frozen long challenge, phone-style subset | 12.09% | 11.43% | 9.39% |

Long phone-style character error falls from **12.09% to 9.39%**, with exact lines
increasing from **1/48 to 3/48**. The control has 11.43% CER and 2/48 exact lines.
Most long phone-style lines therefore still contain errors. On the short
phone-style subset, h64 is slightly worse than both alternatives. These are
rendered distortions, not real camera captures.

The h64 final training-set CER is 2.06%,
versus 1.22% for the h48 control.
That is an in-sample fit measure, not held-out accuracy. This short transfer
budget does not determine the best achievable result after longer adaptation.
No extra epochs were added after seeing these results.

## Selection and full stage history

The pre-recorded rule selects the lowest **final overall development CER** among
parent, h48 and h64, with ties preferring that order. It selected **h64**. The
selection receipt was saved before any new challenge evaluation. Challenge
regressions are retained and cannot retroactively alter that experiment's
selection rule. This is the next private experimental candidate, **not a
recommended universal replacement or promoted production model**. The parent
and control remain available for comparison.

| Arm | Epoch | Updates | Development CER |
| --- | ---: | ---: | ---: |
| h48 | 1 | 315 | 4.90% |
| h64 | 1 | 315 | 5.83% |
| h48 | 2 | 630 | 5.06% |
| h64 | 2 | 630 | 5.52% |
| h48 | 3 | 945 | 4.88% |
| h64 | 3 | 945 | 4.70% |

All six planned stages returned successful process receipts. Initialization
source hashes, actual optimizer step counts, changed tensors, prior histories
and strict-resume metadata were checked. Both arms' final epochs also achieved
their own lowest development CER. Every stage regression remains recorded.

Observed training time was 326.71 seconds for h48 and
441.99 seconds for h64. These are invocation timings
under varying shared-machine load, not controlled throughput or inference-latency
benchmarks. No wall-clock equality or production speed claim follows.

## Artifacts and verification

The primary high-resolution export and exact-resume state are private and local:

```text
training/runs/resolution-study-3c6ecf2/h64/final/recognizer.pt2
training/runs/resolution-study-3c6ecf2/h64/model/training-state.pt
```

The corresponding h48 artifacts are under the sibling `h48/` directory. Use each
export with its own metadata and input geometry; do not pair the 64 × 1,024 model
with the older 48 × 768 preprocessing. No runtime model path was changed.

| Arm | Input geometry | Final export SHA-256 |
| --- | --- | --- |
| h48 | 48 × 768 | `9c43a69ed269bc3a2b818963d03f5dbc0911e8842021bbdb89b38a24b138fee5` |
| h64 | 64 × 1024 | `06d38ed47ca29f831ea2ffca46008beb04eecc8da7665195211c8365c9041606` |

Each final export passed **24** native/export and CPU/MPS prediction checks with
zero mismatches, using deterministic samples from development and both challenges.
These spot checks do not establish device parity on every input. The final
native checkpoints contain the actual final tensors, not a different selected
intermediate epoch.

The closing experiment check verified **151 protected-path hashes**
and **3300 image hashes**. The trainer and CLI were intentionally
updated for the new opt-in feature, with before/after hashes recorded separately;
previous datasets, fonts, models and frozen evidence were not modified. Private
plans, stage receipts, selected/final state, evaluations and exports are under
`training/runs/resolution-study-3c6ecf2/`. Gate and independent-review evidence
are under `benchmarks/private/model-resolution-3c6ecf2/`.

Plan SHA-256: `66415716ebc83d616d34b9c6e4af8c64f36ec997c1a9d8817b5ccac67b91bb10`.

## Limits and software gates

This is a one-seed, three-epoch transfer experiment on a restricted known
vocabulary and shared-corpus synthetic images. The development set and both
challenges have been inspected in prior studies; they are regression evidence,
not fresh confirmatory tests. Correlated styles of one text are not independent
documents. Source-rights restrictions remain unchanged: no corpus, image, font
or weights are redistributed. Real optical validation and a fixed-set Tesseract
comparison are still required before release.

The full local suite passed **1,782 tests**, with two optional S3 SDK tests skipped
and six dependency deprecation warnings. All **113** focused initialization,
resume, expansion and holdout tests passed. This change adds **21** resize/CLI
regression cases. Ruff, web lint, all **145** web tests and the production web
build passed. Software test results do not establish recognition quality.
