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
