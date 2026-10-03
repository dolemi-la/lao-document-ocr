# Learning-rate comparison: paused at a low-disk safety boundary

## Current status

This experiment is **paused, not complete**. The retained model is still
`training/runs/text-coverage-ed769db/expanded/final/recognizer.pt2`.
No new candidate was selected, neither challenge was scored, and no production
setting was changed. The previous [coverage-budget continuation](recognizer-coverage-budget-continuation.md)
remains a separate completed experiment with its own rejected final endpoint.

The starting revision is `7263730ef6e85917cfe0c87d744ff616825f0860`.
The new plan was frozen before training at SHA-256
`7e0812160ce26fa70a3512b9c2b57d935eb592cdaa34e08fecb5c320074ed1f6`.

## Frozen experiment

Both arms initialize from the retained text-coverage epoch-2 native weights,
not the unselected coverage-budget epoch-6 state. They start fresh AdamW state
with the same seed 2026100302. Each is planned for two epochs, 1,260 optimizer
updates and 7,560 sample presentations. Only learning rate differs:
standard uses 0.00005; gentle uses 0.00002.

The 3,780 training images, 1,284 training text groups, 156 ordered development
images, 93-class vocabulary, Phetsarath font, and 64 by 1,024 input geometry
remain unchanged. Batch size is 6 and device is Apple MPS. No new images or
text were generated. The existing 480-image short and 144-image long challenges
remain excluded from training and from selection.

The existing decision rule is unchanged: a final endpoint must improve overall
development character edits without more short-development edits or fewer
short-development exact lines than the retained parent. Lowest eligible overall
dev edits wins; ties prefer parent, then standard, then gentle. Selection must
be saved before any challenge scoring. The planned two-epoch endpoints must not
be replaced with the interim results below.

## Completed work and interruption

| Arm | Learning rate | Completed epochs | Durable optimizer updates | Interim dev CER |
| --- | ---: | ---: | ---: | ---: |
| Standard | 0.00005 | 1 of 2 | 630 | 3.5701% |
| Gentle | 0.00002 | 1 of 2 | 630 | 3.6066% |

For context, the retained parent has 3.6248% overall development CER. These
interim numbers do not establish whole-line quality, final eligibility, or a
winning learning rate.

During standard epoch 2, observed free disk space fell to approximately
**2.2 GiB**, below the recorded **4 GiB** safety floor. The training process was
terminated with SIGTERM. Its supervisor recorded return code -15. The on-disk
training state still hashes exactly to the completed standard epoch-1 state.
No epoch-2 report exists. The number of discarded in-memory updates is unknown
and is not included in the completed budget.

The gentle epoch-2 stage was never started. Both completed first-epoch states
remain intact. At the closing input audit, free space had recovered to
5.90 GiB, but training was not restarted: a transient
recovery does not establish enough sustained headroom. No unrelated files,
models, datasets, fonts, or caches were deleted to make room. The cause of the
disk-space fluctuation has not been established.

## Exact resume handoff

The remaining order is **standard epoch 2, then gentle epoch 2**. Obtain sustained
storage headroom before restarting; 4 GiB is the stop floor, not a comfortable
starting target. Do not weaken the floor or silently change batch size, device,
learning rate, data, or the fixed budget to continue this experiment.

From the repository root, after resolving storage pressure:

```bash
.venv/bin/python -B benchmarks/private/model-learning-rate-7263730/supervise.py \
  start standard-002-retry1 train --arm standard --epoch 2
```

A supervisor exit code of 75 means the action is still running, not successful.
Observe that same name with `wait standard-002-retry1`; do not advance until it
has a successful process receipt and the matching epoch-2 summary. Keep the
original `standard-002.result.json` SIGTERM receipt rather than overwriting it.
Then run the planned `gentle-002` stage. Both stages resume their own completed
epoch-1 states; do not start from the rejected earlier experiment.

Once both final stages have passed, the frozen experiment runner's `select`,
`evaluate` for parent/standard/gentle, and `complete` actions still remain.
The independent gate helper `verify.py` must account for the preserved original
failure and the successful standard-epoch-2 retry receipt; its original receipt
name must not be turned into a fake success. Record the resource interruption
in the eventual final report. `paused.summary.json` is historical evidence,
not a completion report to overwrite.

All private run files are under `training/runs/learning-rate-7263730/`.
Resume states are `standard/model/training-state.pt` and
`gentle/model/training-state.pt`. Logs and review helpers are under
`benchmarks/private/model-learning-rate-7263730/`.

## Verification completed so far

The closing audit verified **259 protected-file hashes**,
**4,560 distinct image hashes**, and exact matches between both durable
states and their successful first-epoch receipts. The input holdout audits and
CTC preflight passed. All 3,936 train/development images fit CTC capacity; the
two previously width-capped inputs were retained.

The full Python suite passed 1,782 tests with two optional S3 SDK tests skipped
and six dependency deprecation warnings. Python lint, web lint, all 145 web
tests, and the production web build passed. Those are software gates, not proof
that an unfinished training experiment has improved OCR accuracy.

No final export or new challenge result exists for this study. The final-result
writer was prepared but not run. The source corpus and synthetic profiles retain
the prior private-development restrictions; no source text, font, image,
prediction, or model weight is redistributed. Real scan/photo validation and the
fixed-set Tesseract comparison remain open.
