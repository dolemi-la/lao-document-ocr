# Phone-style diversity versus additional repetitions

## Fixed question and scope

This private synthetic experiment starts from the completed
[eight-epoch longer-line model](recognizer-long-line-finetuning.md). It compares
additional repetitions of its existing training images against fresh phone-style
renderings of existing training text, using equal completed optimizer-update and
sample-presentation budgets. Neither the parent run nor its artifacts is overwritten.
The starting source revision is `cd729abd8d740c480703b347d312fb11a1e0f4ad`.

Before choosing the new training plan, a diagnostic reproduced the parent's
307/5,490 development edits and examined training/development subsets only.
Existing 49–64-character training phone images had 490/8,116 edits (6.04%) and
28/144 exact lines. Development phone images had 149/1,755 edits (8.49%) versus
3.76% for clean and 4.27% for noisy styles. Development rotation/perspective bins
were small, unequal and not a causal isolation of one augmentation operation.
No new challenge predictions were used by that diagnostic.

## Recorded design

The plan was frozen before new image generation or optimizer work:

| Arm | Training images | Text groups | Epochs | Updates | Sample presentations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control: original images | 2,016 | 864 | 5 | 1,260 | 10,080 |
| Phone: original plus 504 new images | 2,520 | 864 | 4 | 1,260 | 10,080 |

Both arms initialize from the exact same parent final weights and start fresh
AdamW state at learning rate 0.0001, batch size 8 and seed 202610021. The model
remains bidirectional CRNN v2 with 93 vocabulary classes including blank,
48-pixel input height and 768-pixel fixed padded width. Training uses Apple MPS
and two Torch CPU threads. The frozen stage order alternates the arms, with the
control's fifth epoch last. There is no learning-rate adjustment, early stopping,
challenge-based continuation, language model, beam search or preprocessing change.

The new images are two standard phone-profile variants for each of 252 existing
training text groups: 126 of length 33–48 and 126 of length 49–64, selected by
salted hash ordering rather than individual recognition errors. Phetsarath OT
Regular is the only font. The existing generator uses font sizes 40–56 and
geometry version 2. No text group or vocabulary character is added.

Phone-style training images increase from 648 to 1,152. Clean/noisy counts stay
648 each, with 72 retained legacy samples whose profile field is absent. Thus
this intervention changes phone/text weighting as well as visual diversity;
it is not an isolated estimate of one distortion's effect.

The ordered 156-image development set stays identical. Both existing challenge
sets remain excluded by normalized label and image identity. The official
training-expansion and holdout audits passed. All 2,676 combined training/dev
images passed CTC capacity checks. The two previously width-capped images remain;
no new example was dropped to improve a score.

## Matched final results

Only each arm's fixed final epoch is compared below. The parent scores were
reproduced by the same evaluator on the same images. Lower CER is better.

| Evaluation set | Images | Parent CER | Control CER | Phone-diversity CER |
| --- | ---: | ---: | ---: | ---: |
| Unchanged development | 156 | 5.59% | 5.36% | 4.85% |
| Frozen short challenge | 480 | 4.26% | 3.98% | 3.99% |
| Frozen long challenge | 144 | 7.48% | 7.13% | 6.51% |

| Evaluation set | Parent exact lines | Control exact lines | Phone-diversity exact lines |
| --- | ---: | ---: | ---: |
| Short challenge | 248/480 (51.67%) | 265/480 (55.21%) | 257/480 (53.54%) |
| Long challenge | 18/144 (12.50%) | 18/144 (12.50%) | 20/144 (13.89%) |


The phone-diversity arm has 51 fewer long-challenge character edits than the
control (535 versus 586). On short lines, however, it has one additional edit
(401 versus 400) and eight fewer exact lines (257 versus 265). The development
choice therefore represents a documented tradeoff, not superiority on every
metric. All endpoint results, including the control's last-epoch regression,
remain recorded.

| Synthetic phone subset | Parent CER | Control CER | Phone-diversity CER |
| --- | ---: | ---: | ---: |
| Development phone-style | 8.49% | 7.92% | 6.95% |
| Short phone-style | 5.58% | 5.10% | 4.89% |
| Long phone-style | 14.65% | 13.95% | 12.09% |

Long phone-style exact-line counts are parent: 0/48; control: 0/48; phone: 1/48. These are generated line
images, not genuine camera captures or full documents. Low character error
must not be confused with error-free whole-line transcription.

The carry-forward choice is **Phone-diversity arm**, determined by the lowest
final overall development CER among parent/control/phone, with the recorded tie
rule preferring parent, then control. That decision was saved before any new
challenge evaluation; challenge scores cannot retroactively change it. The
trainer's separate best-on-development checkpoints are retained privately but
were not substituted for the fixed endpoints. No production model is promoted.

## Full stage history

| Arm | Completed epoch | Optimizer updates | Development CER |
| --- | ---: | ---: | ---: |
| control | 1 | 252 | 5.48% |
| phone | 1 | 315 | 5.45% |
| control | 2 | 504 | 5.46% |
| phone | 2 | 630 | 5.19% |
| control | 3 | 756 | 5.36% |
| phone | 3 | 945 | 5.08% |
| control | 4 | 1008 | 5.28% |
| phone | 4 | 1260 | 4.85% |
| control | 5 | 1260 | 5.36% |

There was one interrupted first-epoch phone invocation: it returned no exit
status, checkpoint or completion report, and its output directory was empty.
No stage process remained when checked. Its cause and discarded update count
are unknown. The identical frozen configuration was restarted from the parent,
not from partial weights, and completed successfully. Completed model-update
sequences are matched; total hardware work is not, because it includes that
unrecorded discarded attempt. No interrupted attempt was counted as a completed epoch.

## Artifacts, verification and limits

Both final exports and both resumable states remain private under
`training/runs/phone-diversity-cd729ab/`. The final exports are
`control/final/recognizer.pt2` and `phone/final/recognizer.pt2`; the respective
`model/training-state.pt` files retain exact-resume state and best checkpoints.
The development-selected carry-forward export is:

```text
training/runs/phone-diversity-cd729ab/phone/final/recognizer.pt2
```

| Arm | Final epoch | Export SHA-256 |
| --- | ---: | --- |
| control | 5 | `9782b75f3ffb947db606bfb38d510cc1abf004c5370b67fe056d6fd37e7881ea` |
| phone | 4 | `a8b95e922c15626a328ea751d2c001a6e055d8bf087ece9766783c63a624b664` |

Each final export passed 24 native/export and CPU/MPS prediction comparisons
with zero mismatches. The completion audit verified 116
protected path hashes and 3,300 prepared/challenge image
hashes. Previous weights, Phetsarath, source data, production settings and OCR
implementation code remain unchanged. The plan hash is
`2cf83d302ce53ff0fe9a3930daa3fbca9763341490f489ad5c1e043c25e2f91f`. Aggregate diagnosis, stage reports, selection,
evaluations, export checks and completion hashes are retained in the private run.
Invocation logs and independent review are under
`benchmarks/private/model-phone-diversity-cd729ab/`.

These are one-seed exploratory results, not a replicated causal finding.
Equal update counts do not imply equal elapsed compute, and three synthetic
views of related text are not independent documents. Both challenges have been
inspected in earlier work; they are regression checks, not new confirmatory
sets. Source/document independence and semantic near-duplicates remain unverified.
Existing private-model-development rights restrictions remain: no source text,
images, font or weights are redistributed. Real scan/photo transcription and a
fixed-set Tesseract comparison are still required for a release decision.

## Software gates

The full local Python suite passed 1,761 tests, with two optional S3 SDK tests
skipped and six dependency deprecation warnings. All 102 focused training,
resume, expansion and holdout cases passed. Ruff, web lint, all 145 web tests,
the production web build and local documentation links passed. No runtime source
or dependency configuration was changed by this experiment. The independent
closing review rechecked both actual optimizer step counters, all protected
hashes, export identities and training/holdout separation.
