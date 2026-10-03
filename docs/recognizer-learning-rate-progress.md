# Matched learning-rate comparison after the coverage-budget safeguard failure

## Question, parent selection, and recorded decision

The [previous coverage-budget continuation](recognizer-coverage-budget-continuation.md)
improved aggregate scores but lost two exact short development lines. Its saved
decision retained the original epoch-2 text-coverage model. This study starts
from that retained model, not the rejected epoch-6 weights.

It asks whether smaller optimizer steps can improve overall development error
without losing short-development whole-line correctness. The source revision is
`7263730ef6e85917cfe0c87d744ff616825f0860`. Before optimizer work, the plan fixed two
new two-epoch experiments with the same starting weights, dataset, seed,
model geometry, batch size, stopping point, and decision rule. The learning
rate is the only configured difference between the two arms.

| Arm | AdamW learning rate | Epochs | Optimizer updates | Sample presentations |
| --- | ---: | ---: | ---: | ---: |
| Standard | 0.00005 | 2 | 1,260 | 7,560 |
| Gentle | 0.00002 | 2 | 1,260 | 7,560 |

Both initialize through the existing weights-only path with fresh AdamW state
and seed 2026100302. This is not exact continuation of the previous optimizer.
The second epoch of each arm uses strict resume of its own state, preserving
optimizer, shuffle generator, RNG, dataset identity, padding strategy, and device.
Batch size is 6; execution uses Apple MPS and two Torch CPU threads.

## Unchanged data and preprocessing

Both arms use the same 3,780 training images and 1,284 normalized training text
groups. The ordered 156-image development set and the 480-image short and
144-image long challenges are unchanged. No text or image was generated, moved
between splits, or relabeled. Phetsarath OT Regular and the existing 93-class
vocabulary including blank remain unchanged.

Inputs remain 64 pixels high with 1,024-pixel fixed padding. The model is the
existing bidirectional CRNN v2, using normalized valid-timestep greedy decoding.
No architecture, beam search, language model, calibration, or runtime source changed.
All 3,936 combined train/development images passed CTC capacity checks. The two
existing width-capped images were retained rather than filtered after seeing scores.

Exact normalized training/development labels and image bytes remain separated.
Both challenge audits check labels and image hashes against training/development
and the other challenge. These checks do not establish source/document independence,
absence of semantic near-duplicates, or real optical performance.

## Development-only selection

A new final endpoint must reduce overall-development edits without increasing
short-development edits or losing short-development exact lines relative to
parent. Lowest eligible overall-development edits wins, with ties retaining
parent, then standard, then gentle. The rule is unchanged from the earlier
safeguard; it was not adjusted after observing these results.

| Endpoint | Overall dev edits | Short dev edits | Short dev exact lines | Eligible |
| --- | ---: | ---: | ---: | --- |
| parent | 199/5490 | 23/507 | 15/30 | true |
| standard | 183/5490 | 22/507 | 13/30 | false |
| gentle | 190/5490 | 23/507 | 13/30 | false |

The saved carry-forward decision is **parent**. It was recorded before
any new challenge evaluation, then hash-bound into each evaluation report.
Both new endpoints reduce overall development character edits, but each loses
two exact short-development lines (15/30 to 13/30), so neither qualifies.
The short safeguard is based on only 30 previously inspected images representing
18 labels. This limits its evidential strength; passing it is not release validation.

## Matched fixed-final results

Only fixed final epoch-2 weights are compared, never an intermediate best checkpoint.
The same evaluator reproduces the parent counts exactly. Standalone development
counts agree with the training-loop results. Lower CER is better.

| Evaluation set | Images | Parent CER | Standard CER | Gentle CER |
| --- | ---: | ---: | ---: | ---: |
| Unchanged development | 156 | 3.6248% | 3.3333% | 3.4608% |
| Frozen short challenge | 480 | 3.3691% | 3.1306% | 3.2101% |
| Frozen long challenge | 144 | 4.5289% | 4.0662% | 4.2732% |

| Evaluation set | Parent exact lines | Standard exact lines | Gentle exact lines |
| --- | ---: | ---: | ---: |
| Unchanged development | 57/156 | 61/156 | 57/156 |
| Frozen short challenge | 273/480 | 292/480 | 281/480 |
| Frozen long challenge | 27/144 | 32/144 | 29/144 |

| Synthetic subset | Parent CER | Standard CER | Gentle CER |
| --- | ---: | ---: | ---: |
| Unchanged development: clean-scan | 2.8490% | 2.6781% | 3.0199% |
| Unchanged development: noisy-scan | 2.8490% | 2.7350% | 2.7350% |
| Unchanged development: phone-photo | 4.8433% | 4.3875% | 4.3875% |
| Frozen short challenge: clean-scan | 3.1306% | 2.9517% | 3.0411% |
| Frozen short challenge: noisy-scan | 3.2499% | 2.9517% | 2.9517% |
| Frozen short challenge: phone-photo | 3.7269% | 3.4884% | 3.6374% |
| Frozen long challenge: clean-scan | 2.5201% | 2.5566% | 2.6297% |
| Frozen long challenge: noisy-scan | 3.2505% | 2.9218% | 2.8853% |
| Frozen long challenge: phone-photo | 7.8159% | 6.7202% | 7.3046% |

Profile regressions relative to parent (CER increase or fewer exact lines):

- standard, frozen long challenge / clean-scan: CER 2.5201% → 2.5566%; exact lines 13 → 15.
- gentle, unchanged development / clean-scan: CER 2.8490% → 3.0199%; exact lines 19 → 18.
- gentle, frozen short challenge / phone-photo: CER 3.7269% → 3.6374%; exact lines 86 → 85.
- gentle, frozen long challenge / clean-scan: CER 2.5201% → 2.6297%; exact lines 13 → 13.

Long phone-style exact lines are parent: 5/48; standard: 6/48; gentle: 5/48.
All results, including subset regressions, are retained. A lower character
error rate is not equivalent to better whole-line correctness. The private
summaries also report whitespace-token WER; this is not a complete linguistic
word-segmentation assessment for Lao.

## Stage history and verification

| Arm | Epoch | Optimizer updates | Development CER |
| --- | ---: | ---: | ---: |
| standard | 1 | 630 | 3.5701% |
| gentle | 1 | 630 | 3.6066% |
| standard | 2 | 1260 | 3.3333% |
| gentle | 2 | 1260 | 3.4608% |

The original standard epoch-2 attempt was stopped with SIGTERM when free disk
space fell to approximately 2.2 GiB, below the frozen 4 GiB stop floor. Its
original return-code -15 receipt and `paused.summary.json` remain unchanged.
Discarded in-memory updates are unknown and excluded from the completed budget.
After storage recovered, `standard-002-retry1` resumed the verified own-arm
epoch-1 state. Both epoch-2 runs were observed with two-second disk checks.

All four stages now have successful process receipts (including that retry)
and observed changes to model
weights. Independent review checks actual AdamW step counters and learning rates,
fresh parent initialization, strict own-arm resume chains, unchanged histories,
and identical final shuffle-generator states between arms. The final native
checkpoints match the latest fixed-epoch weights, not merely the best dev weights.

The audit verified 259 protected-file hashes and
4,560 distinct image hashes. Each export passed 24
native/export and CPU/MPS prediction checks with zero mismatches. This is sampled
export/device parity, not a guarantee for every input or hardware backend.

## Local artifacts and boundaries

The carry-forward export is:

```text
training/runs/text-coverage-ed769db/expanded/final/recognizer.pt2
```

Both new final exports remain under
`training/runs/learning-rate-7263730/<arm>/final/recognizer.pt2` with metadata.
Exact-resume states are at `<arm>/model/training-state.pt`. They require their
own export metadata and 64 by 1,024 input geometry. Parent and rejected-epoch-6
artifacts remain intact.

| Arm | Final export SHA-256 |
| --- | --- |
| standard | `d33963a01c5fec7e95e1e18965c9f279539eb45d2735d787cb2c1ca277dda580` |
| gentle | `73412f5a6cd740632b35de370e1573db0180f295851deaf872c46264aeae0ded` |

Frozen plan SHA-256: `7e0812160ce26fa70a3512b9c2b57d935eb592cdaa34e08fecb5c320074ed1f6`.
Plans, hashes, private runners, stage histories, selection and evaluations remain
under the Git-ignored run directory. Independent review and execution/gate logs
are under `benchmarks/private/model-learning-rate-7263730/`. No font, source text,
images, private predictions, or model weights are redistributed by this commit.

## Software gates and interpretation

The full local Python suite passed 1,782 tests with two optional S3 SDK cases
skipped and six dependency deprecation warnings. The focused training/resume/
holdout suite also passed all 90 tests. Web lint, all 145 web tests,
production web build, Python lint, and local documentation-link checks passed.
No production code, deployment setting, default OCR model, or dependency changed.

This is one paired seed on a shared synthetic source corpus, not a replicated
causal result. Both optimizers are reset, so these results are not directly
interchangeable with the prior exact-resume budget continuation. Equal update
counts do not guarantee equal wall time. Repeatedly inspected development and
challenge sets are regression evidence, not fresh confirmatory tests.

Rights-reviewed real scan/photo transcriptions and a fixed-set Tesseract
comparison remain required for a release decision. No production promotion
is claimed. All planned training and evaluation steps are complete.
