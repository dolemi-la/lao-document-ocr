# Fixed-data Phetsarath optimization-budget study

## Scope and pre-recorded stopping point

This is a separate exploratory continuation of the completed
[720-update coverage comparison](training-expansion.md), not a revision of its
stopping point or unfavorable result. That comparison remains complete: the
288-image expanded arm had final normalized development CER 174/225 (77.33%)
after 20 epochs, versus 118/225 (52.44%) for the 72-image baseline after 80 epochs.
Both had used 720 optimizer updates.

Before starting this continuation on 2026-09-28, a new plan fixed the expanded
arm's endpoint at **80 total epochs / 2,880 optimizer updates**. It resumed the
saved epoch-20 latest state into a new directory, leaving the completed study
unchanged. The hypothesis was that the expanded model's high training error
could decrease with a larger optimization budget. The plan was motivated by
observed results, so this is not a preregistered confirmatory experiment.

The primary comparison is the same 288-image model-development dataset at 20
versus 80 epochs: **four times the total compute budget**, not equal compute.
The old baseline at 80 epochs is context only. Equal epochs across different
training-set sizes do not mean equal optimizer updates or sample presentations.

## Fixed inputs and settings

The source revision was `11d5894c9401d6a5be8839ffe9ff8a8b178c2dd7`.
The continuation retained all 288 training images, all 12 ordered development
images, the 93-class vocabulary including blank, and the exact normalized-text
split. No image was added, removed, regenerated, relabeled, or moved between
splits. The development denominator is 225 normalized reference characters.

Inputs remain short Phetsarath OT Regular synthetic lines with corrected
geometry-version-2 augmentation. Existing image hashes and the matched-expansion
audit were checked before every stage. The data retain the source registry's
private-model-development restrictions; this study does not establish source
rights or authorize republishing the text, images, font, or model.

Other settings remain unchanged: bidirectional CRNN v2, input height 48, maximum
padded width 768, batch size 8, AdamW learning rate 0.001, seed 20260928, MPS,
greedy decoding, and `normalized-valid-timestep-v2` CER. No language model,
beam search, calibration, learning-rate schedule, or production default changed.

Six foreground stages ended at epochs 30, 40, 50, 60, 70, and 80. Each stage
restored optimizer, data-loader shuffle, and Python/NumPy/Torch RNG states under
the strict resume contract. It verified the protected parent artifacts and
unchanged training source files, preserved the preceding history, and checked
actual optimizer step counters against the recorded budget. A 2 GiB free-space
floor was checked before each stage. All stages completed; none remains running.

The old 20 epochs remain unchanged in the new history. The completed parent
comparison, its plan and reports, its training/checkpoint files, and its exports
were not overwritten. Twenty-nine parent artifacts were fingerprinted before
continuation and rechecked afterward.

## Results at the recorded endpoints

The primary results are final-epoch scores, not selected intermediate minima.
Both rows use exactly the same expanded training images and development images.

| Expanded model | Total epochs | Optimizer updates | Sample presentations | Training CER | Development CER |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original endpoint | 20 | 720 | 5,760 | 4,270/5,719 = 74.6634% | 174/225 = 77.3333% |
| New budget endpoint | 80 | 2,880 | 23,040 | 0/5,719 = 0% | 35/225 = 15.5556% |

The continuation adds **2,160 updates**. Development CER is 61.7778 percentage
points lower than at the original endpoint. The final model has zero normalized
training edits on both the original 72 training images and the 216 additions,
and no raw-empty predictions on either training or development images.

This shows that the earlier underfit state was not a permanent inability to
learn these fixed images. It does not isolate a general benefit of more data or
establish real-document accuracy. Zero training error is an in-sample fit result,
not proof of generalization. The remaining train/dev gap and the small repeatedly
inspected dev set are reasons not to promote this model yet.

Recorded stage scores are preserved even when they regress:

| Epoch | Total optimizer updates | Normalized development CER |
| ---: | ---: | ---: |
| 20, inherited starting point | 720 | 77.3333% |
| 30 | 1,080 | 16.8889% |
| 40 | 1,440 | 15.1111% |
| 50 | 1,800 | 16.4444% |
| 60 | 2,160 | 15.5556% |
| 70 | 2,520 | 14.6667% |
| 80, final endpoint | 2,880 | 15.5556% |

The retained best-on-development checkpoint is epoch **72**, with 31/225
normalized character edits (**13.7778% CER**). This selected score is separate
from the primary final-epoch result. The history contains 80 epoch-level
selection evaluations. It must not be presented as an independent test result.

## Export and consistency verification

The selected checkpoint was exported locally without changing the production
model. CPU and MPS gave identical normalized predictions for all 12 development
images. Both exported-benchmark scores were 31/225 CER, matching checkpoint
selection, with 12/16 whitespace-token edits (75% WER). The tiny whitespace-token
count should not be treated as a reliable word-level Lao accuracy estimate.

The final latest state was independently re-evaluated on training and development
images. Its MPS training counts were 0/5,719 character edits; its CPU and MPS
development evaluators both reported 35/225. Evaluating the original and added
training subsets separately produced counts that sum to the full training set.
Source checkpoint hashes were unchanged after evaluation.

## Evidence and preservation

All local experiment files remain Git-ignored under:

```text
training/runs/expansion-budget-11d5894/
  experiment-plan.json
  run.py
  complete.py
  epoch-030.summary.json
  epoch-040.summary.json
  epoch-050.summary.json
  epoch-060.summary.json
  epoch-070.summary.json
  epoch-080.summary.json
  completed-2880.summary.json
  model/{training-state.pt,recognizer.pt,recognizer.pt2,metadata.json,vocab.json}
```

The stage and completion summaries contain aggregate counts and hashes, not
reference strings, predictions, sample IDs, or image paths. The private plan also
records local source-artifact paths for integrity checks. These fingerprints are
reproducibility identifiers, not a claim of anonymization.

| Evidence | SHA-256 |
| --- | --- |
| New pre-recorded plan | `bbc92445ec5fbe19e02f8c52f154757c70ff4f62ab67de84590bceb72b2bd75a` |
| New completion summary | `ea72009291b685180056c9c30dd3add628c1c3b987d40c3432a48c356c483fb5` |
| Preserved parent completion summary | `f56b87806d4bfa0b50afedc46fae6aee1bdfc292a7837d6157a480383b2a97c8` |
| Unchanged development identity | `65508a69d60af13692ed6e41099b849496de2fb82c0b4c6fd36a6b7bf69b725d` |
| Selected local export | `66f6bbd8cb2ea078cb2bbb1b14ec247329d30bc96da556bc8740fa0d2605ad0f` |

The completion report additionally binds the source state, final state, selected
checkpoint, runner, audit code, individual stage reports, and model vocabulary.
The original 720-update study must remain cited as a separate completed result.

## Limits and next decision

There is only one seed and the same twelve synthetic development images have
been examined across several experiments. No new independent test was used.
Normalized exact labels and image bytes are isolated across training/dev, but
document/source separation and near-duplicate independence remain unverified.
These results cover short lines and a restricted known vocabulary, not long
pages, arbitrary document layouts, or unknown character inventories.

No production model, font policy, OCR default, collector session, capture kit,
or real-data benchmark was replaced. The real optical benchmark and published
Tesseract baseline remain prerequisites for the recognizer release decision.

The planned optimization budget is complete. With training error now zero,
blindly adding epochs is not the next priority. Preserve the candidate and assess
a separately defined, larger unseen evaluation set with reviewed provenance,
while continuing the genuine optical-capture campaign. Any further development
must retain this endpoint and distinguish new experiments from this one.

## Follow-up: frozen-model challenge

The [separate synthetic evaluation](recognizer-fresh-evaluation.md) freezes this
study's selected epoch-72 artifact and evaluates new exact-label groups without
retraining or parameter tuning. It scored 1,036/10,062 character edits (10.30%)
on 160 text groups rendered in three profiles. This uses a new denominator,
not a revision of the 12-image development result above. Whole-line correctness
was 123/480; source/document independence and real optical performance remain
unverified. The candidate remains local and experimental.
