# Short-line regression diagnosis and proposed BatchNorm experiment

Status: completed and independently verified on 2026-10-04; selection retains parent.
See [all endpoint results and regressions](recognizer-batch-norm-study.md).
Frozen plan SHA256: `6dad1587491ebeffc72778d63d19aaf99d9fd857b022b5c3bcb3fb22cbaf39ed`.
Private runner: `training/runs/batch-norm-d09f644/experiment.py`.
Preparation verified 287 protected files and 4,560 distinct image hashes,
passed holdout and CTC audits, and reproduced the parent development guard counts.
Source checkout: `d09f644d1e7250c4ff82cd0bfc3745c1c0ba45b3`.
The [completed learning-rate study](recognizer-learning-rate-progress.md) and its
selection remain unchanged. The retained model is still
`training/runs/text-coverage-ed769db/expanded/final/recognizer.pt2`.

## What actually regressed

An inference-only diagnostic reproduced the saved development edit counts and
exact-line counts for parent, standard and gentle. It used the same ordered 156
development images, MPS, batch size 8, fixed 64 × 1,024 preprocessing and normalized
valid-timestep greedy decoding as the existing standalone evaluator. Image hashes
were checked by the manifest loader, and input checkpoint/manifest/selection hashes
were checked before and after inference. No optimizer updates, challenge scoring,
new images, saved hybrid weights, or production changes occurred.

Both candidates lost the **same three** parent-exact short images and gained one
other exact short image, giving the reported net change from 15/30 to 13/30.
The three losses cover only two normalized text groups: two phone-style images
and one clean-scan image. Each loss is one edit: two one-character insertions
and one substitution. Removing all whitespace does not fix any of the losses.
Thus this is not simply a whitespace-normalization discrepancy.

Standard improves edit distance on three short images and worsens three; gentle
improves two and worsens three. Aggregate edit counts can improve while errors
spread onto previously exact lines. The safeguard is detecting that distinction.
These 30 images represent only 18 labels, so the three losses are not three
independent examples of a general failure mode.

## BatchNorm diagnostic

The recognizer contains one `BatchNorm2d` layer in `features.7`. Ordinary training
calls `model.train()` at each epoch, updating that layer's running statistics.
The forward pass uses stored statistics in evaluation mode. This behavior was
checked in the installed PyTorch implementation and the local training path.

The diagnostic swapped only `running_mean`, `running_var` and
`num_batches_tracked` in memory. Learned parameters, including BatchNorm affine
weight and bias, stayed with the named model. Counter swapping is included for
buffer completeness; it does not affect an evaluation forward pass.

| In-memory diagnostic model | Overall dev edits / 5,490 | Short edits / 507 | Short exact / 30 |
| --- | ---: | ---: | ---: |
| Parent | 199 | 23 | 15 |
| Standard | 183 | 22 | 13 |
| Gentle | 190 | 23 | 13 |
| Standard parameters + parent BN buffers | 192 | 23 | 13 |
| Gentle parameters + parent BN buffers | 187 | 22 | 15 |
| Parent parameters + standard BN buffers | 199 | 23 | 13 |
| Parent parameters + gentle BN buffers | 197 | 23 | 13 |

Restoring parent buffers to gentle recovers two of its three lost exact images.
Restoring them to standard recovers none. Conversely, replacing parent buffers
with either candidate's buffers reduces short exact lines to 13/30. This supports
a contribution from running-statistic changes and an interaction with learned
parameters; it does not establish that BatchNorm is the sole cause.

The gentle/parent-buffer hybrid would satisfy the numerical development guard,
but it was constructed after inspecting these failures. It is **not selected,
exported, or promoted**, and does not replace either fixed endpoint. Freezing
statistics during training is also different from swapping them after training;
its effect must be measured prospectively.

## Recommended next experiment

Compare ordinary BatchNorm updates against frozen parent BatchNorm statistics
at learning rate 0.00002. This targets the diagnostic signal without adding data,
changing the short-image weighting, or searching more learning rates. Prior
[short replay](recognizer-short-replay-study.md) also failed the exact-line guard,
so another replay run is not the first proposed intervention.

| Setting | Proposed contract |
| --- | --- |
| Initialization | Same retained text-coverage final native weights for every run; fresh AdamW |
| Control | Existing BatchNorm training behavior |
| Treatment | BatchNorm layer in evaluation mode during training; parent running buffers fixed; affine parameters remain trainable |
| Learning rate | 0.00002 in both arms |
| Seeds | Three paired seeds: 2026100401, 2026100402, 2026100403 |
| Fixed endpoint | 2 epochs, 1,260 updates, 7,560 presentations per run |
| Total budget | 6 runs, 7,560 updates, 45,360 presentations |
| Dataset | Existing 3,780 training images / 1,284 groups; 156 unchanged dev images |
| Short training exposure | Existing 1,350 short images, presented twice; no reweighting |
| Geometry / vocabulary | 64 × 1,024 fixed padding; unchanged 93 classes |
| Device / batching | Apple MPS; batch size 6; two Torch CPU threads; zero loader workers |
| Decoder and loss | Existing CTC objective and normalized valid-timestep greedy decoding |
| Data changes | None; neither diagnosed dev labels nor challenge labels enter training |

All other training and optimizer settings remain identical between paired arms.
Changing BatchNorm mode also changes training-time normalization, not just whether
buffers are saved. The experiment estimates that policy's combined effect; it
must not be described as isolating only a storage/update mechanism.

For seeds 1 and 3, run control epoch 1, treatment epoch 1, control epoch 2,
treatment epoch 2. For seed 2, reverse the arm order within each epoch. Complete
all twelve stages in that order. Resume only each run's own durable checkpoint.
Keep every endpoint; do not replace it with an intermediate minimum or choose
the seed with the best result.

## Prospective selection and replication rule

Seed 2026100401 is the predeclared source of the two selectable new checkpoints.
The other seeds test repeatability, not extra chances to select a lucky model.
For each run, the existing parent-relative guard means:

- Fewer than 199 overall development character edits.
- At most 23 short-development character edits.
- At least 15 exact short-development lines.

A policy is selectable only if its primary-seed endpoint passes that guard and
at least two of its three seeds pass. This replication veto is a **proposed new
rule for the next experiment**, not a reinterpretation of the completed study.
Among selectable primary-seed endpoints, choose lowest overall dev edits; ties
prefer parent, then control, then frozen-statistics treatment. Retain parent if
neither qualifies. Report every seed's metrics and paired differences; three
seeds on the same tiny dev set do not establish statistical or optical validity.

Persist the full development measurements, eligibility and chosen artifact hash
before scoring challenges. Then evaluate parent and all six fixed endpoints on
both unchanged synthetic regression challenges, reporting all adverse subsets.
Challenge results cannot revise the saved choice. Export both primary endpoints
with native/export/device parity checks; keep all six resumable states. Secondary
seeds are descriptive replicas and must not silently become selected exports.

## Implementation and validation required before training

The trainer now exposes `--freeze-batch-norm` and serializes the
`freeze_batch_norm` setting (default `False`). An absent field in older
checkpoints retains existing behavior; exact resume rejects policy changes.
Weights-only initialization may intentionally choose the new policy.

Local validation on 2026-10-04 passed 44 focused training/resume cases,
including frozen-policy interrupted/resumed equivalence, affine updates,
dropout mode, buffer preservation, legacy compatibility and mismatch rejection.
Scoped Ruff and CLI flag parsing passed. Scrutinize traced CLI configuration,
epoch transitions, metric migration and checkpoint compatibility without
material findings. Full repository gates and prospective training remain pending.

Apply the policy after every call that enters training mode, especially the
per-epoch `model.train()` call. Set only the BatchNorm layer to evaluation mode;
do not disable LSTM dropout or freeze the entire model. Keep BatchNorm affine
parameters in AdamW. Never patch the completed study's frozen runner.

Focused behavioral tests must establish that:

- Control updates running statistics and the counter; treatment preserves all
  three buffers bit-for-bit while affine parameters still receive gradients.
- The rest of the model stays in training mode and the policy survives the next
  epoch's mode transition.
- The setting round-trips in saved training state, compatible old checkpoints
  retain existing behavior, and incompatible exact resumes are rejected.
- Interrupted/resumed treatment preserves counters, history, RNG, shuffle state,
  dataset identity and frozen buffers across the epoch boundary.

Freeze the executable plan, seed order, source/data/code hashes, treatment
semantics, selection rule and budgets before optimizer work. Audit all protected
artifacts and all 4,560 train/dev/challenge image identities. Verify disjoint
normalized labels and image hashes without altering any split or regenerating
images. Reproduce parent development counts before proceeding.

Start with sustained free space preferably above 10 GiB. Maintain the 4 GiB
stop floor with two-second monitoring during every long action. Preserve failed
receipts and last durable states; never clean up unrelated resources to continue.
Recent epochs took roughly 3.5–4 minutes each: allow about 45–50 minutes for the
twelve training epochs, plus evaluation/export/check time; this is an estimate,
not a completion condition. Run one experiment stage at a time.

Finish with the focused tests, repository software gates, independent endpoint
and selection verification, export metadata/parity checks, and Scrutinize review.
Software success does not establish OCR quality. No production promotion is in
scope, and rights-reviewed optical validation remains outstanding.

## Diagnostic evidence

Private reproducible script and aggregate-only result:

- `benchmarks/private/short-regression-d09f644/diagnose.py`
- `benchmarks/private/short-regression-d09f644/diagnosis.json`

The script asserts reproduction of all three saved overall/short development
edit and exact-line counts, verifies source hashes before/after inference, and
saves no private labels, predictions, images, fonts, or hybrid checkpoints.
This document contains aggregate evidence only. All fixed endpoints, saved selection, challenge measurements and sampled export
parity checks are recorded in the completed study linked above.
