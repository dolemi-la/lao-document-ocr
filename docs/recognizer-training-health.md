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
