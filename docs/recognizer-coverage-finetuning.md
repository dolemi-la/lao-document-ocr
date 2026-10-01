# Phetsarath coverage fine-tuning: completed ten-epoch experiment

## Scope and fixed decision

This is a new private synthetic-development experiment, initialized from the
selected epoch-72 checkpoint of the [budget study](recognizer-budget-study.md).
It does not overwrite or resume that study's completed 80-epoch endpoint.
The optimizer, shuffle seed, history, and checkpoint selection start afresh.
The base source revision is `3e1203f3180a1c1f0183307ca26abf242d3faf5b`, with the
new explicit `--initialize-from` implementation fingerprinted in the private plan.

Before generating images or training, the plan fixed **10 epochs / 1,440 optimizer
updates**, batch size 8, AdamW learning rate 0.0003, seed 20261001, and Apple MPS.
Input height remains 48, padded width 768, with the same bidirectional CRNN v2,
93-class vocabulary including blank, and normalized valid-timestep greedy decoding.
There is no language model, beam search, calibration change, or production promotion.

The primary endpoint is the **final epoch 10**, not the lowest development score.
All ten foreground stages completed and produced weight updates; the retained
state records 11,520 training sample presentations. No stage remains running.

## Expanded data and exclusions

The original 288 training images and 12 development images were preserved.
The existing private Phetsarath-compatible corpus supplied 288 unused training
text groups: 144 of length 8–32 and 144 of length 33–48 characters. Each was
rendered in clean-scan, noisy-scan, and phone-photo **synthetic** profiles, adding
864 training images. The result is **1,152 training images / 576 text groups**.

The development extension uses 48 unused text groups, six short and 42 longer,
also rendered in three profiles. Together with the original development images,
this gives **156 development images / 60 text groups**. This development set
is deliberately longer-line weighted; it is not a representative document sample.

Selection used salted SHA-256 ordering after excluding normalized labels in all
seven supplied prior manifests and the frozen 480-image challenge. Existing
normalized-text hash buckets still determine train versus development membership.
The audit found no exact-label overlap between train/development, and no label
or image-byte overlap between the new combined dataset and the frozen challenge.
This is not a proof of semantic near-duplicate or document/source independence.

Phetsarath OT Regular remained the only font. Strict shaped-font coverage checks
passed. All 1,308 combined images passed CTC capacity checks; none was width
capped, and the smallest available-minus-required timestep margin was 15.
The generation seed, geometry version 2, font hash, corpus hash, image hashes,
selection identity, and training implementation hashes are retained locally.

The source retains the existing HPLT private-model-development caveats. No corpus
text, generated images, font, or model weights are redistributed by this change.
These generated scan/photo styles are not genuine optical scans or phone photos.

## Matched final-model results

Both the parent and final model were evaluated by the same native-model evaluator
with identical fixed-width preprocessing and normalized valid-timestep decoding.
The parent result on the frozen challenge exactly reproduces its previous report,
including 1,036 character edits and 123 completely correct lines.

| Evaluation set | Images | Parent character edits | Final character edits | Parent CER | Final CER |
| --- | ---: | ---: | ---: | ---: | ---: |
| Expanded development | 156 | 917 / 5,490 | 527 / 5,490 | 16.70% | 9.60% |
| Newly added development only | 144 | 886 / 5,265 | 504 / 5,265 | 16.83% | 9.57% |
| Original development only | 12 | 31 / 225 | 23 / 225 | 13.78% | 10.22% |
| Frozen synthetic challenge | 480 | 1,036 / 10,062 | 634 / 10,062 | 10.30% | 6.30% |

On the challenge, exact lines increased from **123/480 (25.625%)** to
**188/480 (39.167%)**. Whitespace-token WER decreased from 467/1,029 (45.38%)
to 362/1,029 (35.18%). Whitespace tokenization is not a comprehensive linguistic
word-segmentation evaluation for Lao. Neither model produced empty challenge
predictions.

| Frozen challenge profile | Images | Parent CER | Final CER | Final exact lines |
| --- | ---: | ---: | ---: | ---: |
| Synthetic clean-scan | 160 | 7.90% | 5.16% | 72 / 160 |
| Synthetic noisy-scan | 160 | 8.59% | 5.46% | 68 / 160 |
| Synthetic phone-photo | 160 | 14.40% | 8.29% | 48 / 160 |

The final training-set CER is 546/31,834 (1.715%). This is an in-sample fit metric,
not a replacement for the development or challenge result. New development
phone-style samples still have 276/1,755 (15.73%) character errors; longer,
distorted lines remain a weakness despite the short-line challenge improvement.

## Epoch history and checkpoint selection

| Epoch | Optimizer updates | Development CER |
| ---: | ---: | ---: |
| 1 | 144 | 13.8251% |
| 2 | 288 | 12.2769% |
| 3 | 432 | 11.3843% |
| 4 | 576 | 11.4936% |
| 5 | 720 | 11.2022% |
| 6 | 864 | 10.0911% |
| 7 | 1,008 | 9.8543% |
| 8 | 1,152 | 9.7450% |
| 9 | 1,296 | 9.0528% |
| 10, fixed endpoint | 1,440 | 9.5993% |

The retained best-on-development checkpoint is **epoch 9**. Its development score
is reported separately; it was not substituted for epoch 10 in the challenge
comparison and has not been assigned that final model's challenge score.
Epoch-4 and epoch-10 regressions remain in the history. The challenge was not
used to choose an epoch, learning rate, stopping point, or additional stage.
It was already inspected in earlier work, so it is not a fresh confirmatory test.

## Artifacts and verification

All experiment artifacts are private and Git-ignored under:

```text
training/runs/coverage-finetune-3e1203f/
  experiment-plan.json
  prepared.summary.json
  prepare.py
  manifest.jsonl
  new-images/
  train_stage.py
  epoch-001.summary.json ... epoch-010.summary.json
  evaluate.py
  parent-evaluation.summary.json
  final-evaluation.summary.json
  complete.py
  completed.summary.json
  model/training-state.pt       # latest epoch 10 and selected epoch 9; resumable
  model/recognizer.pt           # development-selected epoch 9
  model/recognizer.pt2          # development-selected epoch 9 export
  final/recognizer.pt           # fixed final epoch 10 weights
  final/recognizer.pt2          # fixed final epoch 10 export; results reported above
```

The plan hash is
`3d9a0b0e41a6b39db4e267c7595865cb615df90788b3cf0ec35f52487270d23f`.
The final export hash is
`0920292ed2bdabb74fb8fae754a1b8e92c8dcd9b9a9e02a86af4681d190a8b61`.
The development-selected export hash is
`8b1ff175328da379634550cbef03196a2bd2967b8fe15eea82dbee212d9d0eac`.

Twenty-four deterministically selected development/challenge images had zero
normalized prediction mismatches between native CPU and exported CPU inference,
and zero between native CPU and MPS. This checks selected export/device behavior,
not every possible input or GPU platform. Parent artifacts, the previous
challenge, Phetsarath, and protected runtime components remained unchanged.

The new trainer option loads only a selected inference checkpoint with restricted
weights-only deserialization, verifies architecture, full vocabulary identity,
finite tensors and matching state keys/shapes, and requires a new or empty output
directory. It is mutually exclusive with strict resume. Initialization provenance
survives subsequent exact resumes without importing the parent's optimizer or
history. Tests exercise actual initial weights before the first forward call,
reset state, resume provenance, invalid inputs, source preservation, and CLI routing.

## Interpretation and next model work

The final candidate improves these matched synthetic metrics, but most challenge
lines are still not completely correct. One seed, repeated related text profiles,
a restricted vocabulary, and a shared source corpus limit generalization claims.
More data, changed learning rate, and additional updates changed together: this
experiment does not isolate a causal benefit of any one factor.

The recognizer has not been promoted to production or compared with Tesseract on
a rights-reviewed real optical test set. Real scan/photo collection and verified
transcriptions remain required. A follow-up training budget must be recorded as
a new decision, preserving this completed endpoint and keeping the frozen
challenge out of training. This run is complete, not a background training job.

## Software gates and preservation

The full local suite passed 1,761 tests with two optional S3 SDK tests skipped
(the local S3 extra was unavailable) and six dependency deprecation warnings.
All 54 focused trainer regression cases passed, including 12 new initialization
and CLI cases. Ruff, web lint, all 145 web tests, the web build, and local
documentation links passed. Fifty protected path hashes and all 1,788 combined
training/development/challenge image hashes were checked after completion.
Local gate logs and review evidence remain under
`benchmarks/private/model-training-3e1203f/`.
