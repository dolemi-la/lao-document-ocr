# Matched recognizer training expansion

Expanding a synthetic corpus must not silently change the development set used
for a comparison. `audit-training-expansion` checks two manifests with the same
normalized-text split algorithm and reports whether the second adds training
images while retaining the original evaluation conditions.

```bash
lao-ocr audit-training-expansion \
  --baseline-manifest training/runs/experiment/baseline.jsonl \
  --expanded-manifest training/runs/experiment/expanded.jsonl \
  --dev-ratio 0.1 \
  --output training/runs/experiment/expansion-audit.json
```

This command is a separate, explicit audit. It does not generate images, train a
model, change production defaults, or run automatically inside `train-recognizer`.
Use it before starting both arms of a matched comparison. The Python entry point
is `lao_document_ocr.training_expansion.audit_training_expansion`.

## Checks

Every image must have a lowercase 64-character SHA-256 pin, and the current file
bytes must match it. The audit checks the bytes again, including for programmatic
callers; a filename or a supplied hash string is not proof of image identity.
Duplicate or empty sample IDs and empty normalized labels are rejected.

All baseline sample IDs must remain with the same image bytes and normalized
labels. The expanded manifest must produce the **same ordered development IDs**
under the same `dev_ratio`, and must add at least one training image. This detects
new dev examples as well as fallback-driven changes in tiny deterministic splits.
Renaming a path without changing image bytes is allowed; altering an existing
image or its normalized label is not.

The character vocabulary must be unchanged. No normalized label or byte-identical
image may occur on both sides of the expanded train/dev split. Additional variants
of an existing training label are allowed and counted separately from new unique
training text groups. Additional variants of a dev label are rejected.

The command never overwrites an existing report. Validation errors do not create
a report directory. No input manifest, image, or historical artifact is modified.

## Report contract and limits

The report includes sample/text-group counts, split ratio and strategy, vocabulary
checksum, and length-framed SHA-256 identities for the ordered baseline training,
expanded training, and unchanged development sets. Identity inputs are sample ID,
normalized label, and image hash. The report contains **no labels, hypotheses,
sample IDs, or image paths**. Fingerprints are for reproducibility, not a claim of
anonymization or protection against guessing short known labels.

A passing report is deliberately narrow. It does not establish:

- document/source-level separation, near-duplicate isolation, or absence of a dev
  line inside a longer training paragraph;
- rights clearance or permission to republish the training corpus;
- matching seeds, initial weights, optimizer updates, preprocessing, or language
  model exclusions.

Source/document separation and training rights are explicitly marked unverified
or not assessed. These checks remain the experiment author's responsibility.
Additional training text must follow the source registry. HPLT-derived local
training text stays private under the caveats in [hplt-sampling.md](hplt-sampling.md).
Dev/calibration/test labels must also be excluded from any separately trained
language model; see [language-model.md](language-model.md).

## Bounded Phetsarath expansion — 2026-09-28

The baseline was the corrected 84-image short-line experiment under
`training/runs/short-resolution-cd5c6a3/`: 72 training and 12 dev images. Its complete
93-class vocabulary (including blank) was held fixed. This is synthetic model
development, not a real optical benchmark or an independent held-out test.

From the existing 9,694-line Phetsarath-compatible local corpus, selection kept
unique normalized lines of length 3 through 32 that fit that vocabulary. All
labels from the parent 300-image manifest and the deterministic dev hash bucket
were excluded. Among the 1,496 eligible text groups, the first 216 in SHA-256
order were selected before training. Selection did not use predictions or CER.

The selected text was rendered with pinned Phetsarath OT Regular, strict glyph
coverage, balanced augmentation, and geometry version 2. The original 84 images
were referenced unchanged, not regenerated. The result is **288 training images
and the same 12 dev images**. The full 300-image expanded manifest passes the audit
and CTC preflight at height 48 / width 768: minimum timestep margin 15, no
width-capped samples, and resized widths 93 through 432.

The saved plan specifies identical initial model weights, vocabulary, seed
20260928, AdamW learning rate 0.001, batch size 8, input height 48, fixed padded
width 768, normalized CER, and greedy decoding with no language model. The
planned comparison is equal **720 optimizer updates / 5,760 sample presentations**:
80 baseline epochs versus 20 expanded epochs. Equal update budgets intentionally
mean different repetitions of each training example. Best-on-dev checkpoint
selection also has different numbers of epoch evaluations; final-epoch scores
are the primary comparison, not cherry-picked minima.

### Paused stages and completed continuation

The first session reached 180 updates per arm before the 1.5 GiB disk guard
interrupted further training. After space briefly recovered, the baseline reached
360 updates while the expanded arm remained at 180. Those unequal checkpoints
were not used to claim an accuracy change. The paused status and equal-180-update
summary are preserved, rather than overwritten as though the interruption never
happened.

The continuation resumed the saved expanded state to 360 updates first, preserved
matched summaries at 360 and 540 updates, and completed the unchanged planned
budget of **720 updates / 5,760 sample presentations in each arm**. All original
training/dev identities, source image pins, vocabulary, initial-weight hash,
training settings, and the recorded plan were rechecked. No samples were added,
removed, regenerated, or moved between splits during continuation.

### Final normalized results

The following are final-epoch measurements, not the best intermediate results.
Both arms use `normalized-valid-timestep-v2` CER and greedy decoding without an LM.
Each development measurement has 225 reference characters across the same 12
images. Training measurements use each arm's own training set.

| Arm | Training images | Epochs | Updates | Final training CER | Final dev CER | Raw-empty dev predictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 72 | 80 | 720 | 21/1,310 = 0.016031 | 118/225 = 0.524444 | 0/12 |
| Expanded | 288 | 20 | 720 | 4,270/5,719 = 0.746634 | 174/225 = 0.773333 | 0/12 |

The expanded arm's final dev CER is **24.8889 percentage points higher**, not an
improvement. The baseline fits its training images closely but has a large
training/development gap. The expanded model has high error even on its training
images: 1,009/1,310 (0.770229) on the original 72 and 3,261/4,409 (0.739623) on the
216 additions. This is consistent with underfitting under this budget; it does
not prove a single cause, that the additional data are useless, or that more
training will necessarily solve the problem.

Retained best-on-dev checkpoints are reported separately. The baseline selected
epoch 79 at CER 104/225 (0.462222); the expanded run selected epoch 20 at 174/225
(0.773333). These selections had 80 versus 20 epoch evaluations and are not the
primary equal-update comparison. Both selected checkpoints exported successfully.
For each artifact, CPU and MPS produced identical normalized predictions on all
12 dev images, and exported-benchmark CER matched the selected training score.
This verifies export/scoring consistency, not independent model accuracy.

The same tiny development set has been examined repeatedly. There is one seed,
no independent test, no verified document/source separation, and no real optical
benchmark. Equal sample presentations are not equal repetitions per image or an
equal character-token budget. Do not turn this result into a general claim that
more data hurts OCR or use it to change a production default. **No model was
promoted.** The next optimization-budget experiment must be recorded separately;
it must not silently extend this completed 720-update comparison.

The aggregate-only completion report is `completed-720.summary.json`, SHA-256:
`f56b87806d4bfa0b50afedc46fae6aee1bdfc292a7837d6157a480383b2a97c8`.
It fingerprints the plan, audit runner, input identities, final training states,
selected checkpoints, and exports without publishing labels or predictions.
The audit confirmed source checkpoint bytes were unchanged during evaluation.

Local, Git-ignored artifacts:

```text
training/runs/expansion-7022f3b/
  prepare.py
  run.py
  experiment-plan.json
  baseline.jsonl
  expanded.jsonl
  preflight.summary.json
  cli-audit.summary.json
  progress-disk-pause.summary.json
  progress-180-updates.summary.json
  matched-360.summary.json
  matched-540.summary.json
  completed-720.summary.json
  complete.py
  progress.summary.json
  baseline/training-state.pt
  expanded/training-state.pt
  baseline/recognizer.pt2
  expanded/recognizer.pt2
```

The development identity fingerprint is
`65508a69d60af13692ed6e41099b849496de2fb82c0b4c6fd36a6b7bf69b725d`.
The parent manifest, historical models, production OCR defaults, capture kits,
and collectors were not replaced. The recorded matched budget is now complete.
Preserve these states and reports; a future optimization-budget study must
identify its new budget and limitations explicitly rather than relabel this
comparison.
