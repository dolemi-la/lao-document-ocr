# Normalized recognizer checkpoint-selection metrics

## One text policy for training and benchmark CER

New recognizer development evaluations use `normalized-valid-timestep-v2`.
Greedy decoding still stops at each sample's valid timestep count. Before
computing edit distance and the reference-character denominator, both reference
and prediction pass through the existing `normalize_lao_text` function, exactly
as in `benchmark_recognizer`.

That function performs NFC normalization, horizontal whitespace consolidation,
line-ending normalization, trimming, and the existing excess-blank-line rule.
It does not correct spelling, remove arbitrary internal spaces, transliterate,
or substitute Lao characters. CER is aggregated as total character edits divided
by total normalized reference characters, not an average of sample CERs. For a
zero total reference length, the benchmark's existing convention is retained:
0 if there are no edits, otherwise 1.

This changes checkpoint-selection measurements, not the CTC training targets,
loss, gradients, image preprocessing, model architecture, raw inference strings,
or the production decoder. It does not imply better recognition.

The version is stored in epoch history, training states, and checkpoints. Export
sidecars preserve the checkpoint's evaluation version, selected `best_epoch`,
and metric-migration trail. A legacy export with no known metric version reports
`null`; exporting old weights does not relabel their historical evaluation.

## Prediction health remains raw

`dev_prediction_diagnostics` continues to use `valid-timestep-greedy-v1`.
Its character counts and empty-string counts describe **raw decoded strings**;
blank-token counts describe the model's valid CTC path. In particular, a string
of spaces is not the same as CTC blank output. These raw counts may differ from
normalized CER's character denominator and must not be substituted for it.
No labels or predictions are added to health or migration metadata.

## Exact resume and explicit metric migration

Normal exact resume requires the current metric version. An old
`valid-timestep-v1` training state is rejected by default rather than mixing
unnormalized scores with normalized checkpoint selection.

To migrate a compatible old state explicitly, retain all its training settings
and supply the new switch with a new output directory:

```bash
lao-ocr train-recognizer \
  --manifest training/generated/v1/manifest.jsonl \
  --output training/runs/crnn-v2-normalized \
  --epochs 81 \
  --batch-size 8 \
  --dev-ratio 0.1 \
  --seed 20260928 \
  --image-height 48 \
  --max-width 768 \
  --device mps \
  --resume-from training/runs/crnn-v2/training-state.pt \
  --recompute-resume-metrics
```

The numbers above illustrate an 80-epoch source; use the source state's actual
settings and a total epoch target greater than its completed epoch. The library
option is `train_recognizer(..., recompute_resume_metrics=True)`.

Migration validates the normal sample, vocabulary, model, padding, configuration,
device, optimizer, shuffle-generator, and RNG contracts. It does not bypass these
guards. Unknown or unversioned **training states** are not accepted by this switch.
The switch without a resume source is an error. Rejected resumes do not create
an output directory or rewrite existing files.

For a valid migration:

1. Re-evaluate the retained latest and retained best weights on the same dev set
   using normalized CER. Select the lower score; ties retain the previous best.
2. Restore the latest weights for optimization and preserve RNG/shuffle state
   around the extra evaluations. Continue from the original completed epoch,
   not from the selected best epoch.
3. Preserve old history entries and their old metric version. Record the new
   baseline separately under `metric_migrations` and use the new metric for new
   epochs and selection decisions.

The migration trail contains source checksum, source/completed epoch, metric
versions, retained-state scores, selected state/epoch, and
`selection_scope: retained-states-only`. It persists across subsequent exact
resumes and exports. **This is not a retroactive search of all earlier epochs:**
weights no longer retained cannot be rescored. Historical scores are not rewritten.

New artifacts record `best_epoch` directly. Legacy weights-only checkpoints
without it use the first history epoch matching the retained minimum, consistent
with strict-improvement selection, rather than the last equal-scoring epoch.
When the retained epoch cannot be identified from available metadata/history,
weights-only continuation fails instead of inventing an epoch. As before,
weights-only continuation is not exact optimizer/RNG restoration; a known older
weights-only metric is re-evaluated before new selection, with this limitation
recorded. Unknown future metric versions are rejected.

## Bounded Phetsarath audit — 2026-09-28

The audit reused the unchanged corrected 84-image short-line dataset and its
72/12 normalized-text-group train/dev split. It scored the retained latest and
best states from the previous 80-epoch height-48 and height-64 runs; it did not
retrain them or search unseen historical weights. All scores below are synthetic
development measurements on 225 normalized reference characters, not optical
benchmark accuracy.

| Retained state | Previous raw CER | Normalized CER | Normalized edits |
| --- | ---: | ---: | ---: |
| Height 48, latest epoch 80 | 0.520000 | 0.524444 | 118 / 225 |
| Height 48, retained best epoch 79 | 0.462222 | 0.462222 | 104 / 225 |
| Height 64, latest epoch 80 | 0.875556 | 0.880000 | 198 / 225 |
| Height 64, retained best epoch 78 | 0.862222 | 0.866667 | 195 / 225 |

For each best artifact, normalized training evaluation agreed with the existing
exported benchmark. CPU and MPS also agreed; each best artifact had zero
normalized-prediction mismatches across the 12 development images.

A separate copy of the height-48 state was explicitly migrated and continued for
one epoch with unchanged settings. All 80 historical rows remained unchanged.
Epoch 81 normalized dev CER was 114/225 (0.506667), so the retained epoch-79
checkpoint remained selected at 104/225 (0.462222). Its new export matched the
selection metric. This verifies migration and scoring, **not a new accuracy gain**.

Every historical experiment file was hashed before and after; none changed.
No optical data, collector kit, production model, decoder default, or font policy
was changed. Phetsarath remains canonical. No language model or calibration was
used in the audit.

The runner and aggregate-only audit summary are Git-ignored:

```text
training/runs/normalized-metric-12e8667/
  audit.py
  audit.summary.json
  h48-migrated-e81/
```

The summary fingerprints the tested training code, source revision, manifest,
and source states. Regression coverage includes normalization equivalence,
raw-health semantics, explicit migration, retained-state comparisons and ties,
unchanged history, RNG/optimizer/shuffle restoration, later exact resume,
weights-only epoch identity, guard failures, CLI forwarding without PyTorch,
and export provenance. Deterministic CPU continuation tests compare tensors and
optimizer state, not just whether the commands finish.
