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

Only the first **180 updates per arm** completed in this session. The baseline
reached epoch 20 with normalized dev CER 1.0 and 10/12 raw-empty predictions. The
expanded arm reached epoch 5 with dev CER 224/225 (approximately 0.995556) and
0/12 raw-empty predictions. Nonempty output is not useful recognition: these
partial values do not establish an improvement from broader coverage.

Before the next stage, available disk space fell below the runner's 1.5 GiB
floor. The guard refused to start more training. After space briefly recovered,
the baseline completed epoch 40 (360 updates, dev CER approximately 0.977778),
but the guard blocked the expanded arm again before epoch 6. The final saved
stages are therefore **baseline: 360 updates; expanded: 180 updates**. Do not
compare those unequal-budget scores as evidence for or against broader coverage.
The earlier equal-180-update summary is preserved separately. Both checkpoints
are saved, and the planned 720-update comparison remains **incomplete**. No
production model was promoted. Resume the expanded arm to epoch 10 first, once
storage permits, rather than silently changing the recorded comparison budget.

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
  progress.summary.json
  baseline/training-state.pt
  expanded/training-state.pt
```

The development identity fingerprint is
`65508a69d60af13692ed6e41099b849496de2fb82c0b4c6fd36a6b7bf69b725d`.
The parent manifest, historical models, production OCR defaults, capture kits,
and collectors were not replaced. A future continuation should use these saved
states and the unchanged recorded budget, not silently start a larger experiment.
