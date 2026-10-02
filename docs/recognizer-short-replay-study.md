# Matched-budget short-line replay at higher input resolution

## Question and fixed experiment

This private synthetic-development experiment starts from the final h64 model
of the [resolution study](recognizer-resolution-study.md). That model improved
longer-line aggregate error but regressed on the short challenge. This follow-up
tests extra repetitions of existing short training images against ordinary
continued training, without using challenge labels or predictions for training.
The source checkout was `f83479b4ad82616788716211c5b6d5aa7e6af09d`.

A pre-plan diagnostic reproduced the parent development result: 258/5,490
character edits across 156 images. Its short subset contains only 30 images
representing 18 normalized text groups, with 28/507 edits and 11 exact lines.
This is a small, repeatedly inspected development subset, not a fresh test.
No challenge was newly scored for the diagnostic.

Before any optimizer work, the plan fixed these endpoints:

| Arm | Manifest training entries | Distinct training images | Epochs | Updates | Presentations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control | 2,520 | 2,520 | 3 | 1,260 | 7,560 |
| Short replay | 3,780 | 2,520 | 2 | 1,260 | 7,560 |

Both arms start from the same h64 weights and fresh AdamW state, with learning
rate 0.00005, batch size 6, seed 202610024, input height 64, fixed padded width
1,024, Apple MPS, and two Torch CPU threads. Model layers and the 93-class
vocabulary remain unchanged. The smaller batch and lower learning rate relative
to the parent apply equally to both new arms. The comparison is not an estimate
of those changes alone. Equal updates/presentations do not establish equal
elapsed time, memory, or hardware work.

The primary endpoints are the fixed final epochs, not intermediate minima.
The frozen order is control-1, replay-1, control-2, replay-2, control-3. No
learning-rate adjustment or extra stage was introduced after seeing results.

## Repeated entries are not new images

The original dataset has 720 training images whose normalized labels contain at
most 32 characters, and 1,800 longer training images. The replay manifest retains
all original entries and adds 1,260 references to existing short images: a full
additional pass over the 720, followed by the first 540 in salted SHA-256 order.
Repeated entries have distinct manifest IDs but the exact same image bytes and
labels. No image, text group, vocabulary character, font, or augmentation was added.

The replay arm therefore presents 1,980 short and 1,800 longer entries per epoch,
versus 720 and 1,800 for control. Over the completed budgets, short-image
presentations are 3,960 versus 2,160; longer-image presentations are 3,600 versus
5,400. This tests weighting toward short examples, not visual diversity.
All 864 original training text groups and all original images remain available.
Phetsarath and all source restrictions are unchanged.

The ordered 156-image development set and both frozen challenges remain
byte-identical. Official expansion and holdout audits verify unchanged
vocabulary, preserved original samples, and no normalized-label or image overlap
across training and either challenge. The two pre-existing width-capped examples
remain; no difficult sample was silently removed. There are 3,300 distinct
images across original training, development, and the two challenges, not 4,560
new independent cases. The replay duplicates intentionally increase manifest
entries without increasing independent data.

## Selection before challenge evaluation

The recorded rule adds a short-development guard. A new candidate must have no
more short-development character edits than the parent, no fewer exact short
development lines, and fewer overall development edits than the parent.
Among qualifying fixed endpoints, lowest overall development edits wins;
ties prefer parent, then control, then replay. Parent is retained when neither
candidate qualifies. The decision was persisted before scoring either challenge.

| Endpoint | Short-dev edits | Short-dev CER | Short-dev exact lines | Eligible |
| --- | ---: | ---: | ---: | --- |
| parent | 28/507 | 5.5227% | 11/30 | true |
| control | 27/507 | 5.3254% | 11/30 | true |
| replay | 27/507 | 5.3254% | 10/30 | false |

The carry-forward decision is **control**. This is a development-only experiment
choice, not a claim that the chosen model dominates every metric or is suitable
for production. The 30-image guard is weak evidence about real short documents.

## Matched fixed-endpoint results

All models use the same existing native evaluator and their 64 by 1,024 geometry.
Parent counts were reproduced exactly; final standalone development counts
agree with each trainer's final recorded count despite the evaluator's batch
size of 8 versus training/development-loader batch size 6. Lower CER is better.

| Evaluation | Images | Parent CER | Control CER | Replay CER |
| --- | ---: | ---: | ---: | ---: |
| Unchanged development | 156 | 4.6995% | 4.0437% | 4.3352% |
| Frozen short challenge | 480 | 4.3530% | 3.6176% | 3.6076% |
| Frozen long challenge | 144 | 5.5028% | 5.2593% | 5.3324% |

| Evaluation | Parent exact lines | Control exact lines | Replay exact lines |
| --- | ---: | ---: | ---: |
| Short challenge | 235/480 | 255/480 | 263/480 |
| Long challenge | 19/144 | 18/144 | 20/144 |

| Synthetic subset | Parent CER | Control CER | Replay CER |
| --- | ---: | ---: | ---: |
| Development phone-style | 5.5271% | 5.4131% | 5.4131% |
| Short phone-style | 4.9195% | 4.2039% | 4.1145% |
| Long phone-style | 9.3864% | 8.6560% | 9.0212% |

Long phone-style exact lines are 3/48
for parent, 2/48 for control,
and 3/48 for replay.
All results, including adverse changes, are retained. Character error and
whole-line correctness are different endpoints; neither proves optical quality.
Whitespace-token WER in the private summaries is not a full linguistic Lao
word-segmentation evaluation.

## Observed tradeoff

The selected control improves short-challenge character edits from 438 to 364
and exact lines from 235 to 255. Its long-challenge edits decrease from 452 to
432, but exact lines decline from 19 to 18 and whitespace-token edits increase
from 177 to 181. The short replay arm has one fewer short-challenge character
edit than control (363 versus 364), eight more exact short lines, and two more
exact longer lines. It nevertheless failed the previously fixed development
guard, so those challenge observations did not replace the saved selection.
Neither arm is uniformly superior on all measured endpoints.

## Complete stage history

| Arm | Epoch | Optimizer updates | Overall development CER |
| --- | ---: | ---: | ---: |
| control | 1 | 420 | 4.4262% |
| replay | 1 | 630 | 4.5173% |
| control | 2 | 840 | 4.2987% |
| replay | 2 | 1260 | 4.3352% |
| control | 3 | 1260 | 4.0437% |

All five stages have successful process receipts, saved checkpoints, observed
weight updates and exact optimizer step counters. Resumed stages restore
optimizer, shuffle generator, RNG, dataset identity, padding strategy and device.
Earlier histories are preserved, including any intermediate regression.

## Artifacts and verification

Private artifacts remain under `training/runs/short-replay-f83479b/`. Each arm
has a final export at `<arm>/final/recognizer.pt2` and exact-resume state at
`<arm>/model/training-state.pt`. The next development candidate is:

```text
training/runs/short-replay-f83479b/control/final/recognizer.pt2
```

It requires the accompanying export metadata and 64 by 1,024 preprocessing.
Older models and both new arm outputs are preserved separately.

| Arm | Final epoch | Export SHA-256 |
| --- | ---: | --- |
| control | 3 | `017eee10674331ab5a6784033a003cca4f23f5a49744c6fa05b0c44498ada8c0` |
| replay | 2 | `10a25e59442edf28f3a20d64411100018ea9a73eb9df1bf2236d13608378400a` |

Each final export passed 24 deterministically chosen native/export and CPU/MPS
prediction comparisons, with zero mismatches. Independent review checked
179 protected path hashes, 3,300 distinct
image hashes, exact stored optimizer counters, frozen development identity,
replay multiplicities, and the saved selection rule. The plan hash is
`1cef4a72e836eb2c692158abc60d3a431cda34f1480b82d0f574c5f2757fca82`. Completion summary SHA-256 is
`b0f5751d92dfd8253b5f5b99dc7bb423e17fa930ee1fb2c67689a81b922ccd43`. Process receipts, gates and the closing review
are retained in `benchmarks/private/model-short-replay-f83479b/`.

## Limits and remaining work

This is one-seed exploratory synthetic evidence with correlated renderings and
a shared corpus. Development and challenge sets have already been inspected;
these are regression checks, not fresh confirmatory tests. Document/source
independence and semantic near-duplicates are unverified. Weighting existing
short images does not add missing characters or establish robustness to actual
phone captures. No data, font or weights are redistributed. Production defaults
and runtime source code were not modified.

Real rights-reviewed scans/photos, verified transcriptions and a fixed-set
Tesseract comparison remain required for the recognizer release criterion.

## Software gates

The full local Python suite passed 1,782 tests, with two optional S3 SDK tests
skipped and six existing dependency deprecation warnings. Ruff, all 145 web
tests, web lint and the production web build passed. The new private experiment
script also passed Ruff. These software checks are not evidence of real OCR
accuracy. No runtime source or dependency configuration changed in this slice.
