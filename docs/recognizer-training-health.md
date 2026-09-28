# Recognizer prediction-health diagnostics

A falling CTC training loss does not prove that the recognizer is learning to
produce usable text. The Phetsarath 300-image development run demonstrated this:
at epoch 20 all 28 development predictions were empty, while loss had fallen.
These diagnostics expose that failure without publishing recognized text.

## Recorded on each new epoch

`train-recognizer` adds `dev_prediction_diagnostics` to each new history row in
`training-state.pt`, `recognizer.pt`, and `metadata.json`. Programmatic
`evaluate(...)` returns the same object as `prediction_diagnostics`.

The version is `valid-timestep-greedy-v1`. Fields are:

| Field | Meaning |
| --- | --- |
| `samples` | Number of evaluated samples. |
| `empty_predictions` | Number whose decoded string is empty. |
| `empty_prediction_ratio` | Empty predictions divided by samples. |
| `predicted_characters` | Total Unicode characters in decoded strings. |
| `reference_characters` | Total Unicode characters in the references used for CER. |
| `predicted_to_reference_character_ratio` | Predicted characters divided by reference characters; can exceed one. |
| `blank_timesteps` | Valid timesteps whose highest-scoring token is CTC blank. |
| `valid_timesteps` | Total timesteps allowed by prepared input widths. |
| `blank_timestep_ratio` | Blank timesteps divided by valid timesteps. |

Ratios use aggregated counts, not an unweighted average of batch ratios. Right
padding outside each sample's valid width never contributes. A zero denominator
produces `null`, not an invented successful measurement. The diagnostics contain
no reference strings, hypotheses, sample IDs, or image paths.

When a nonempty development set has **all** predictions empty, training emits a
`RuntimeWarning` naming the epoch and sample count. This warning does not stop
training, alter gradients, select another checkpoint, or change decoding. Partial
collapse remains visible in the counts without an arbitrary acceptance threshold.
A high blank-timestep ratio alone is not a release decision: CTC blank tokens can
also be part of valid alignments.

## Raw health versus normalized selection CER

Prediction health retains its raw `valid-timestep-greedy-v1` semantics. New
checkpoint-selection CER uses `normalized-valid-timestep-v2`, which normalizes
references and predictions like the exported benchmark. Raw health character
counts are not a substitute for the normalized CER denominator. A spaces-only
prediction is raw nonempty output, not evidence of CTC blank output.

See [recognizer-metric-normalization.md](recognizer-metric-normalization.md) for
explicit migration of old training states, unchanged history, and the retained
height-48/64 checkpoint audit. Earlier measurements below remain historical and
are not retroactively normalized.

## Resume compatibility

This is additive observational metadata, not a change to CER, padding, model
architecture, or optimizer behavior. Existing strict resume contracts still
apply. Older epoch records without diagnostics remain without them; missing
historical telemetry does **not** mean zero empty predictions. New epochs record
current diagnostics. Interrupted-versus-uninterrupted training equivalence is
covered by the existing deterministic CPU regression.

## Bounded Phetsarath diagnostic — 2026-09-28

The investigation used four short, project-authored Lao phrases rendered with
the project's pinned Phetsarath OT Regular font, strict glyph coverage, and no
augmentation. It reused the standard bidirectional CRNN architecture, 48-pixel
input height, fixed 256-pixel padded width, AdamW learning rate 0.001, and seed
20260928. The same four images were repeatedly used for training and evaluation.
**This is deliberately an in-sample overfit diagnostic, not held-out accuracy.**

With 15 output classes including blank:

- CPU and MPS both reached 3/4 exact greedy predictions after 100 optimization
  steps (CER 2/22, approximately 0.0909).
- The MPS run still had 3/4 exact greedy predictions at 200 steps.
- Plain CTC prefix beam search, width 10 and **no language model**, decoded all
  four phrases exactly from that same 200-step model.

A separate MPS control used the existing 148-class Phetsarath recognizer
vocabulary with the same four images. It reached 3/4 exact greedy predictions at
steps 150 and 175 but regressed to 1/4 at step 200 (CER 5/22). It produced no empty
predictions at the recorded evaluation steps. The final result must not be
replaced by the best intermediate result in reporting.

These observations demonstrate learning in this small clean setup and do not
support a claim that Phetsarath rendering or MPS training is universally broken.
They do **not** identify the cause of the broader dataset's failure or justify
changing the production decoder, vocabulary, preprocessing, or model.

For comparison, evaluation of the unchanged 300-image epoch-20 state reported:

| Split | Samples | Empty predictions | Predicted characters | Blank / valid timesteps |
| --- | ---: | ---: | ---: | ---: |
| Training | 272 | 268 | 4 | 35,634 / 35,638 |
| Development | 28 | 28 | 0 | 3,114 / 3,114 |

That checkpoint SHA-256 is
`30aeac6bf31d31026d84b507be983b2c3985067c7c4f5a6c1e7a12ce814b77ce`.
The comparison describes two different diagnostic conditions; it is not an A/B
accuracy improvement on a shared held-out set.

The bounded runner, four generated images, local states, and no-text summaries
are kept under the Git-ignored directory:

```text
training/runs/blank-diagnostic-c5a4ca4/
  probe.py
  cpu.summary.json
  mps.summary.json
  mps-full-vocabulary.summary.json
  small-vocab-beam.summary.json
  existing-300-health.summary.json
```

The preserved runner SHA-256 is
`c5e98a17b9e478a8bd667b0b2a31ff34ae939307095e5797e2f9d8a767dc9cf6`.
The source baseline before this telemetry change was `c5a4ca4`. No existing model,
collector kit, optical dataset, or production default was replaced.

Next, isolate clean versus augmented inputs and short versus longer lines while
keeping the vocabulary, sample identities, and other training settings fixed.
Continue to exclude held-out text from any language-model training. Real OCR
claims still require the reviewed, frozen optical benchmark.

## Paired rendering investigation — 2026-09-28

The follow-up used source baseline `f43644e` and preserved the old synthetic
images before changing the renderer. Two concrete generator defects were found:

1. Rotation and perspective used the original narrow canvas. Projecting the
   dark-pixel centers (`< 128`) from the clean source under each recorded rotation
   placed some ink outside the canvas for 32 of the 300 Phetsarath samples. The
   worst sample placed approximately 44.81% of its dark-pixel centers outside.
   This measures geometric clipping risk to existing source ink, not character
   recognition accuracy, and does not attribute every failure to clipping.
2. Cubic warping produced grayscale values outside `[0, 255]`, followed by an
   unclamped unsigned-byte cast. A controlled black/white rectangle produced
   values from about -50.95 to 283.31; 2,152 overshooting white pixels wrapped to
   dark values. Saturating conversion removed that wraparound in the control.

The correction expands transformed bounds and saturates/rounds output intensity.
Edge-marker, no-op, combined-warp, uniform-white, and overflow regressions cover
these behaviors. This changes only synthetic augmentation, not production OCR,
font policy, optical capture packs, model weights, or decoding defaults.

### Matched 100-step in-sample controls

Six arms used the same 148-class vocabulary, identical initial model-state hash,
seed 20260928, bidirectional CRNN, AdamW learning rate 0.001, input height 48,
fixed padded width 768, batch size four, and exactly 100 optimization steps on
MPS. Every evaluation reused that arm's training images. There was no language
model, held-out tuning, production promotion, or real optical data.

The short arm used the same four project-authored phrases as the earlier probe.
The long arm rotated their order and repeated them five times to make four
129-character strings. Each clean/legacy/corrected triplet used identical labels.
Augmented arms used fixed clean-scan/noisy-scan/phone-photo/phone-photo profiles
and paired seeds. The corrected augmentation changes canvas dimensions and thus
prepared image widths; the length condition also changes glyph scale after the
standard aspect-preserving resize. This is not an isolated linguistic-length
causality test or a representative document benchmark.

| Input condition | Final greedy CER | Predicted / reference characters | Empty predictions |
| --- | ---: | ---: | ---: |
| Short, clean | 0.363636 | 16 / 22 | 0 / 4 |
| Short, legacy augmentation | 0.318182 | 16 / 22 | 0 / 4 |
| Short, corrected augmentation | 0.772727 | 9 / 22 | 0 / 4 |
| Long, clean | 0.992248 | 4 / 516 | 0 / 4 |
| Long, legacy augmentation | 0.980620 | 10 / 516 | 0 / 4 |
| Long, corrected augmentation | 0.992248 | 4 / 516 | 0 / 4 |

These final results do **not** show a recognition improvement from the fix.
Long clean lines also failed to learn usefully in this budget, so augmentation
alone does not explain the larger dataset's collapse. Short corrected inputs
were worse in this one seeded control; do not substitute a better intermediate
step or turn this tiny experiment into a release decision.

A separate regenerated 300-image set preserves all labels, passes font/image hash
checks, and records geometry version 2. At height 48 / max width 768, two samples
need 153 vs 152 and 168 vs 166 CTC timesteps respectively. At height 64 / max width
768 all 300 pass, with minimum timestep margin 11. No training was started on
that incompatible height-48 set, and no samples or labels were silently removed.

Local artifacts remain Git-ignored:

```text
training/runs/paired-augmentation-f43644e/
  paired_probe.py
  old-rotation-clipping.summary.json
  new-data.summary.json
  comparison.summary.json
  <short|long>-<clean|legacy|fixed>/summary.json
training/generated/phetsarath-smoke-300-geometry-v2/manifest.jsonl
```

The comparison summary fingerprints the probe, renderer, font, initial states,
and input manifests without recording reference/predicted text. Historical
images, checkpoints, collectors, and the frozen-data blockers remain unchanged.
Next investigate a bounded short-line curriculum or image-resolution control
using correctly rendered inputs and an unchanged, leakage-safe held-out split.


## Short-line resolution control — 2026-09-28

Starting from `cd5c6a3`, a fresh experiment selected all 84 corrected Phetsarath
images whose normalized labels contain 3–32 characters, in their original order.
The original normalized-text split memberships were preserved: 72 training
images and 12 development images, with zero exact label overlap. These are
synthetic development samples, not a frozen optical benchmark or independent
test set. Selection was based on label length, not observed recognition scores.

Two arms used identical source images, labels, 93-class vocabulary (including
blank), initial model-weight hash, seed 20260928, bidirectional CRNN, batch size
8, fixed padded width 768, AdamW learning rate 0.001, and MPS. Only configured
image height differed: 48 or 64. The vocabulary is a shared known character
inventory derived from the selected corpus; no language model was used. Neither
short-line arm was width-capped, and both passed CTC preflight.

The initial equal 20-epoch budget yielded 100% dev CER in both arms. The plan was
then explicitly extended to 80 epochs for **both** arms (720 optimizer updates
each), before continuing either arm. The saved plan and epoch-20 summary retain
that exploratory decision; this is not a preregistered confirmatory experiment.

| Input height | Latest train CER | Latest dev CER | Best recorded dev CER | Best epoch |
| --- | ---: | ---: | ---: | ---: |
| 48 | 0.016031 | 0.520000 | 0.462222 | 79 |
| 64 | 0.756489 | 0.875556 | 0.862222 | 78 |

All values in this table use the existing training evaluator's raw greedy
strings. Neither latest model has empty dev predictions at epoch 80, but
nonempty output is not equivalent to accurate recognition. In particular,
the 48-pixel model's 1.60% training CER versus 52.00% dev CER indicates a large
generalization gap on this small split. The 64-pixel arm learns more slowly in
this run; this single-seed result does not establish a universal resolution rule.

Both selected best checkpoints were exported and evaluated on CPU and MPS. All
12 raw predictions matched across those devices for each artifact. The exported
benchmark normalizes predictions before CER: its best-checkpoint CER is 0.462222
at height 48 and **0.866667** at height 64. Normalization changed one height-64
prediction, increasing edit count from 194 to 195 out of 225 reference characters.
This accounts for the difference from the raw training-selection value 0.862222;
do not silently equate the two metric conventions. Checkpoint-selection
normalization should be aligned with benchmark normalization in a separately
versioned change before further model selection.

The equal-budget controls demonstrate that the short-line model can leave the
blank-output regime with more optimization steps, not that it has become a
useful general Lao recognizer. No font, production model, decoder, input-height
default, optical dataset, or collector artifact was replaced. No beam/LM tuning
was performed on these 12 development labels.

### Width cap and effective image resolution

A dimension-only audit of all 300 corrected images found:

| Configured height | Width-capped images | Minimum resized image height | Images resized below 32 pixels tall |
| --- | ---: | ---: | ---: |
| 48 | 45 / 300 | 20 | 17 / 300 |
| 64 | 85 / 300 | 20 | 17 / 300 |

The width limit remains 768. Once width-limited, increasing the padded input
height alone does not enlarge that image's actual resized content. These are
**whole image dimensions**, including source margins and augmentation canvas,
not measured glyph/ink heights or a visual quality score. CTC capacity and
legibility are separate questions.

Preflight and pixel preparation now share `plan_line_resize(...)`. The existing
resize and centering policy is unchanged: old/new arrays were compared exactly
on all 300 corrected images at both heights, in addition to pixel-identity,
EXIF-orientation, and rounding-boundary regression tests. Training after the
observational refactor therefore used unchanged prepared pixel arrays.
Successful `ctc_preflight` reports also persist configured dimensions,
width-capped count, and min/max resized dimensions. No labels, hypotheses, IDs,
or image paths are added to that aggregate report. Legacy checkpoint fields
remain historical; strict resume behavior is unchanged.

Local, Git-ignored artifacts:

```text
training/runs/short-resolution-cd5c6a3/
  experiment-plan.json
  epoch20.summary.json
  short-lines.jsonl
  preprocessing.summary.json
  comparison.summary.json
  h48/{training-state.pt,recognizer.pt,recognizer.pt2,metadata.json}
  h64/{training-state.pt,recognizer.pt,recognizer.pt2,metadata.json}
```

The selected input manifest SHA-256 is
`8ff7a7990e3c6780696340912508f810378fd55bb60eccdb241fd6dba39390e2`.
Both initial model-state hashes are
`2ef596b76aa51d895105310813ec3a004279da1a3461fdfea07180876863971c`.
The comparison summary fingerprints the selected checkpoints and retains final
and selected-best measurements rather than replacing final values with a
favorable intermediate result.

## Completed matched coverage expansion — 2026-09-28

The separately recorded [training-expansion experiment](training-expansion.md)
now completed 720 optimizer updates and 5,760 sample presentations in both arms,
after the disk-space interruption. It retained the same 12-image development set,
93-class vocabulary, seed, initial weights, and normalized metric policy.

At the final epoch the 72-image baseline had 1.60% training CER and 52.44% dev
CER; the 288-image expanded model had 74.66% training CER and 77.33% dev CER.
Both had zero raw-empty dev predictions, showing why nonempty output alone is
not a quality gate. The expanded arm was still poor on its own training images;
the baseline's much better training fit did not transfer to a low dev error.
These results come from a single-seed, repeatedly inspected synthetic-development
comparison, not an independent test or evidence of a real-document improvement. No model,
collector kit, or production default was replaced. The next budget experiment
must preserve this completed report rather than move its stopping point.
