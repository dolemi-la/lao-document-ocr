# Fixed-data coverage-budget continuation: epochs 2 through 6

## Question and preserved endpoint

This is an exploratory optimization-budget extension of the completed
[text-coverage study](recognizer-text-coverage-study.md), not a rewrite of its
matched-budget result. That study's selected expanded model stopped at epoch 2
and 1,260 optimizer updates. Its data, selected weights, final export, two-epoch
history and all previous experiment files remain unchanged.

The new source revision is `d40e29f3586dfb0126f7e59e6de174ae024c22e7`. Before optimizer work,
a separate plan fixed four additional epochs, ending at **six total epochs /
3,780 cumulative optimizer updates**. It restores the parent's actual latest
state into a new output directory. The parent's fixed-final checkpoint was
verified to contain the same weights as that latest state before continuation.

The question is whether additional optimization on the same expanded images
improves the final model. This is not an equal-compute comparison: it adds
**2,520 updates and 15,120 sample presentations**, for 22,680 presentations
within this fine-tuning lineage. Earlier ancestor pretraining is not included
in these counts. No stopping point was extended after viewing the new results.

## Unchanged inputs and training contract

The run keeps all 3,780 training images representing 1,284 normalized text groups,
the ordered 156 development images, and the same 93-class vocabulary including
blank. The two frozen challenges contain 480 short-line and 144 longer-line
synthetic images; neither is included in training. All 4,560 distinct image
hashes were rechecked. Exact normalized-label and image-byte holdout audits pass;
source/document independence and semantic near-duplicates remain unverified.

Phetsarath OT Regular remains the only font for the existing synthetic dataset.
No images, text labels, font settings, augmentation profiles or manifests were
generated, replaced, relabeled or moved between splits. CTC preflight covered
3,936 training/development images with zero capacity failures. The two existing
width-capped cases remain; no example was dropped to improve reported scores.

Training remains bidirectional CRNN v2, 64 by 1,024 fixed input geometry, batch
size 6, AdamW learning rate 0.00005, seed 20261003, Apple MPS and two Torch CPU
threads. Exact resume restores optimizer, shuffle-generator and RNG state,
checks dataset/device/padding identity, and preserves inherited initialization
provenance. It does not reset optimizer state or discard earlier history.

The existing four-GiB free-space floor is checked before each stage. No old
models, files or datasets are removed to create space. New artifacts are limited
to the continuation state/checkpoint, final export and aggregate experiment
reports. Runtime code, dependency configuration and production defaults are
unchanged.

## Development-only safeguard

The pre-recorded rule compares the final epoch 6 with the preserved parent.
To carry forward the final model, it must lower overall development edits,
not increase short-development edits, and not reduce short-development exact
lines. Otherwise the parent remains the candidate. The short guard covers only
30 images representing 18 labels and is not a strong real-document validation.

| Endpoint | Overall-dev edits | Short-dev edits | Short-dev exact lines | Eligible |
| --- | ---: | ---: | ---: | --- |
| parent | 199/5490 | 23/507 | 15/30 | true |
| final | 181/5490 | 23/507 | 13/30 | false |

The saved carry-forward choice is **parent**. Selection was
persisted before scoring either challenge and is hash-bound into the evaluation
reports. Only the fixed final endpoint is compared; a better intermediate
checkpoint is not substituted after seeing challenge scores. Production
promotion remains false.

## Matched final-model evaluation

The unchanged native evaluator reproduced the parent's counts on every reported
set. All models use identical preprocessing and normalized valid-timestep greedy
decoding. No language model, beam search, calibration or postprocessing changed.
Lower CER is better; whole-line correctness is a separate outcome.

| Evaluation | Images | Parent edits | Final edits | Parent CER | Final CER |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unchanged development | 156 | 199/5490 | 181/5490 | 3.6248% | 3.2969% |
| Frozen short challenge | 480 | 339/10062 | 304/10062 | 3.3691% | 3.0213% |
| Frozen long challenge | 144 | 372/8214 | 336/8214 | 4.5289% | 4.0906% |

| Evaluation | Parent exact lines | Final exact lines |
| --- | ---: | ---: |
| Unchanged development | 57/156 | 62/156 |
| Frozen short challenge | 273/480 | 286/480 |
| Frozen long challenge | 27/144 | 29/144 |

| Synthetic subset | Parent CER | Final CER | Parent exact lines | Final exact lines |
| --- | ---: | ---: | ---: | ---: |
| Unchanged development: clean-scan | 2.8490% | 2.7920% | 19/48 | 21/48 |
| Unchanged development: noisy-scan | 2.8490% | 2.5641% | 20/48 | 22/48 |
| Unchanged development: phone-photo | 4.8433% | 4.2165% | 13/48 | 15/48 |
| Frozen short challenge: clean-scan | 3.1306% | 2.7728% | 99/160 | 102/160 |
| Frozen short challenge: noisy-scan | 3.2499% | 2.7430% | 88/160 | 98/160 |
| Frozen short challenge: phone-photo | 3.7269% | 3.5480% | 86/160 | 86/160 |
| Frozen long challenge: clean-scan | 2.5201% | 2.6297% | 13/48 | 13/48 |
| Frozen long challenge: noisy-scan | 3.2505% | 2.8853% | 9/48 | 11/48 |
| Frozen long challenge: phone-photo | 7.8159% | 6.7568% | 5/48 | 5/48 |

The final model transcribes 5/48
long phone-style challenge lines exactly, with 6.7568% CER.
These are generated line images, not genuine photographs. A lower character
error does not establish reliable whole-line or full-document transcription.
All subset regressions remain in these tables and the private summaries.

Training-fit counts are in-sample results, not independent accuracy evidence:

| Training subset | Images | Parent CER | Final CER |
| --- | ---: | ---: | ---: |
| original_training_fit | 2520 | 1.1650% | 0.7651% |
| new_training_fit | 1260 | 2.5998% | 1.6200% |

The summaries also retain whitespace-token WER. That tokenization is not a
comprehensive linguistic word-segmentation assessment for Lao.

## Complete stage history

| Total epoch | Total optimizer updates | Additional updates | Development CER |
| --- | ---: | ---: | ---: |
| 2, preserved parent | 1260 | 0 | 3.6248% |
| 3 | 1890 | 630 | 3.5883% |
| 4 | 2520 | 1260 | 3.3333% |
| 5 | 3150 | 1890 | 3.2969% |
| 6 | 3780 | 2520 | 3.2969% |

The retained best-on-development checkpoint is epoch
5 with 3.2969% CER.
This selected score is distinct from the primary fixed epoch-6 endpoint.
All four additional stages have successful process receipts, observed weight
changes and verified cumulative optimizer counters. The original two history
entries are unchanged. All stages finished; no ongoing training is implied.

A duplicate start request for epoch 3 was rejected by an exclusive action claim.
The already-launched invocation was observed to successful completion instead;
no second epoch-3 training result was created or counted.

## Private artifacts and reproducibility

All new experiment files remain Git-ignored under
`training/runs/coverage-budget-d40e29f/`. The final epoch-6 export is
`final/recognizer.pt2` and the exact-resume state is `model/training-state.pt`.
The development-selected carry-forward path is:

```text
training/runs/text-coverage-ed769db/expanded/final/recognizer.pt2
```

Use the accompanying export metadata and 64 by 1,024 preprocessing. The new
final checkpoint and export remain available even when the parent is retained
by the safeguard. No historical model is overwritten or promoted to production.

| Evidence | SHA-256 |
| --- | --- |
| Frozen experiment plan | `1edb1e5f830377d4a4311ca8a1f969dec8270a918e7c6614844eefbe3db2c91b` |
| Completion report | `1a53d24570b128a02e3c12f70cc49abf90851b9e52bc746a43411d048496802a` |
| New final export | `d3f37394767303a75cc1deee4229a4a4096c24020a958a98c7fafe56a5d8ccbb` |
| New resumable state | `163e2fe0e2f02a8b4d53152d42514d5aae3038759fb01085e9085500956eef77` |

The final export passed 24 native/export and CPU/MPS
prediction checks, with 0 native/export and
0 CPU/MPS mismatches. The closing self-review verified
238 protected-path hashes, 4560
distinct image hashes, the complete resume-source chain, actual AdamW counters,
selection logic, parent-count reproduction and final export identity.
Gate logs and closing audit are retained under
`benchmarks/private/model-coverage-budget-d40e29f/`.

## Interpretation and next evidence

This is a single-seed exploratory continuation decided after inspecting the
previous endpoint. The same small development set and related synthetic
challenges have been inspected repeatedly; they are regression evidence, not
fresh confirmatory tests. More optimizer work does not prove a benefit from
new data, independent document generalization or real camera robustness.

The fixed budget is complete regardless of the direction of the results.
Any additional experiment requires its own recorded decision and must preserve
these endpoints and exclusions. Reviewed genuine scan/photo transcriptions
and a fixed-set Tesseract comparison remain prerequisites for the recognizer
release decision. Existing source restrictions remain: no private text,
images, font files or model weights are redistributed by this documentation.

## Software gates

The full local Python suite passed 1782 tests, with 2 optional S3 SDK
tests skipped and 6 existing dependency deprecation warnings. Ruff,
web lint, all 145 web tests and the production web build passed. These software
checks are separate from the OCR experiment and do not establish optical accuracy.
No runtime source or dependency configuration changed in this continuation.
