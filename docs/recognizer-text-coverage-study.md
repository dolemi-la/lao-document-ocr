# Text-coverage expansion versus additional repetitions

## Question and fixed design

This is a completed private synthetic-development experiment starting from the
selected control of the [short-line replay study](recognizer-short-replay-study.md).
It asks whether additional unused text groups help more than further training on
the existing images at the same completed optimizer-update budget. The parent
checkpoint and all earlier data, reports and exports are preserved.

The plan was frozen before generating additions or updating model weights.
Source revision: `ed769dba0ccdf06a715c8ef2f1b798d3839775f1`. Plan SHA-256:
`90c31ba71cd2cc97f11347958a066cf55ba065d01bac34814bb506f9edc6eab6`.

| Arm | Training images | Text groups | Epochs | Updates | Sample presentations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control: original images | 2,520 | 864 | 3 | 1,260 | 7,560 |
| Expanded: old plus new text images | 3,780 | 1,284 | 2 | 1,260 | 7,560 |

Both arms initialize from the same selected parent final weights and start new
AdamW optimizers at learning rate 0.00005, batch size 6, seed 20261003. Geometry
remains 64 by 1,024, with the same bidirectional CRNN v2 and 93 vocabulary classes
including blank. Training uses Apple MPS and two Torch CPU threads. There is no
learning-rate adjustment, early stopping, new decoder, language model, or
production-default change. All five planned stages completed.

The primary result is each arm's fixed final endpoint, not its best intermediate
checkpoint. Equal updates and sample presentations do not mean equal wall time,
FLOPs or exposure counts for individual images. The expanded arm presents each
original image twice rather than the control's three times.

## Added data, exclusions and resource limits

The expansion adds 420 normalized text groups: 210 of length 8-32, 105 of length
33-48 and 105 of length 49-64 characters. They are selected by salted SHA-256
ordering from the existing private corpus after excluding labels in all supplied
prior manifests and retaining only the existing training hash bucket. The
eligible pools contained 1,062 short, 490 medium and 473 long text groups.

Each addition is rendered once in each of three synthetic styles: clean scan,
noisy scan and phone photo. Thus 1,260 distinct new images are created, not repeat
references to old images. Phetsarath OT Regular is the only font. The existing
40-56 font-size schedule, balanced profiles, geometry version 2 and strict
font-coverage checks are unchanged. No new vocabulary character is introduced.

The profile mix also changes: clean/noisy training images rise from 648 each
to 1,068 each, while phone-style images rise from 1,152 to 1,572. Short images
rise from 720 to 1,350, medium images from 1,116 to 1,431, and long images from
684 to 999. Thus this is not a pure isolated test of text count: profile/length
weighting and old-image repetition change alongside text coverage.

The official expansion audit preserves all 2,676 old training/development images
and the ordered 156-image development set. It verifies 420 new training groups
and no cross-split exact-label/image overlap. Both frozen challenges remain
excluded by normalized labels and image hashes. An independent audit also checks
that each new label has exactly one rendering per profile and did not occur in
any supplied exclusion manifest.

All 3,936 expanded training/development images pass CTC capacity preflight. The
two pre-existing width-capped images remain, with no new image removed for poor
model performance. The minimum timestep margin is 12.
Free space is checked against a 4 GiB floor before generation and every stage;
no old data or checkpoint is deleted to make room.

Unused exact labels do not establish unseen documents, semantic independence,
complete historical exclusion coverage, or source rights. Existing HPLT
private-model-development caveats remain. Source text, images, fonts and model
weights are not redistributed by this documentation commit.

## Development-only choice

Before challenge scoring, a candidate must improve overall development character
edits without increasing short-development edits or losing short-development
exact lines relative to the parent. Among eligible candidates, the lowest
overall development edit count wins; ties retain parent, then control, then
expanded. This is the prior study's safeguard, not a rule chosen after results.

| Endpoint | Overall-dev edits | Short-dev edits | Short-dev exact lines | Eligible |
| --- | ---: | ---: | ---: | --- |
| parent | 222/5490 | 27/507 | 11/30 | true |
| control | 209/5490 | 22/507 | 14/30 | true |
| expanded | 199/5490 | 23/507 | 15/30 | true |

The recorded carry-forward choice is **expanded**. The selection receipt was
saved before any challenge evaluation and is hash-bound into each evaluation
report. Only 30 short-development images representing 18 labels support the
short-line guard. A development choice is not production validation.

## Matched fixed-endpoint results

The unchanged native evaluator reproduces the parent's previous counts exactly.
All models use 64 by 1,024 preprocessing and normalized valid-timestep greedy
decoding. Lower character error rate is better.

| Evaluation set | Images | Parent CER | Control CER | Expanded CER |
| --- | ---: | ---: | ---: | ---: |
| Unchanged development | 156 | 4.0437% | 3.8069% | 3.6248% |
| Frozen short challenge | 480 | 3.6176% | 3.5778% | 3.3691% |
| Frozen long challenge | 144 | 5.2593% | 4.6506% | 4.5289% |

| Evaluation set | Parent exact lines | Control exact lines | Expanded exact lines |
| --- | ---: | ---: | ---: |
| Unchanged development | 46/156 | 50/156 | 57/156 |
| Frozen short challenge | 255/480 | 262/480 | 273/480 |
| Frozen long challenge | 18/144 | 26/144 | 27/144 |

Unchanged development: expanded minus control is -10 character edits and +7 exact lines.
Frozen short challenge: expanded minus control is -21 character edits and +11 exact lines.
Frozen long challenge: expanded minus control is -10 character edits and +1 exact line.

| Synthetic subset | Parent CER | Control CER | Expanded CER |
| --- | ---: | ---: | ---: |
| Unchanged development phone-style | 5.4131% | 4.6154% | 4.8433% |
| Frozen short challenge phone-style | 4.2039% | 4.1443% | 3.7269% |
| Frozen long challenge phone-style | 8.6560% | 7.7794% | 7.8159% |

The expanded model does not win every subset: its phone-style development CER
is 4.8433% versus the control's 4.6154%, and long phone-style CER is 7.8159%
versus 7.7794% (one additional character edit). Its aggregate development and
challenge advantages must not be presented as universal superiority.

Long phone-style exact-line counts are parent: 2/48; control: 3/48; expanded: 5/48.
These are generated text-line images, not actual scans or camera captures.
Character error and whole-line correctness are distinct endpoints; neither
establishes full-page document quality. All adverse changes are retained.

Final training-fit counts are stored separately for original images and, for
the expanded model, the new images. They are in-sample results, not independent
accuracy evidence. Whitespace-token WER is not a complete linguistic word-level
Lao evaluation.

## Complete stage history

| Arm | Epoch | Optimizer updates | Development CER |
| --- | ---: | ---: | ---: |
| control | 1 | 420 | 4.0801% |
| expanded | 1 | 630 | 3.7158% |
| control | 2 | 840 | 3.8980% |
| expanded | 2 | 1260 | 3.6248% |
| control | 3 | 1260 | 3.8069% |

One premature control-epoch-2 invocation was refused by the stage-order guard
before loading weights or training. After the preceding stage finished, the
unchanged command was retried successfully. The original refusal receipt is
preserved, and the retry source hash matches the control epoch-1 checkpoint.
No optimizer updates from the refused invocation are counted.

Every stage has a successful process receipt, a saved checkpoint and observed
weight changes. Independent review verifies the actual stored AdamW counters,
unchanged earlier histories and restored optimizer, shuffle-generator, RNG,
dataset identity, padding strategy and device on resumed stages. No training
stage remains running after completion.

## Artifacts and checks

Both candidates remain private under `training/runs/text-coverage-ed769db/`.
Each has `<arm>/final/recognizer.pt2` plus accompanying metadata and
`<arm>/model/training-state.pt` for exact resume. The selected development
candidate is:

```text
training/runs/text-coverage-ed769db/expanded/final/recognizer.pt2
```

This candidate requires its own export metadata and 64 by 1,024 preprocessing.
A retained best-on-dev checkpoint is not automatically interchangeable with the
fixed-final artifact used in this comparison.

| Arm | Final epoch | Export SHA-256 |
| --- | ---: | --- |
| control | 3 | `7f0d170744df312e705c14e7997f0e3e840f2440206bb068190cb9ccaf11a5d4` |
| expanded | 2 | `af60fdb859e23560f7ad778eea898785822374496c07253589d6e058b2ed6a8d` |

Each final export passed 24 deterministic native/export and CPU/MPS prediction
comparisons with zero mismatches. The closing review verified
208 protected paths and 4,560
distinct image hashes. The experiment runners, frozen plan, generated manifest,
selection, evaluations, stage summaries and completion report remain local.
Gate logs and independent audit evidence are in
`benchmarks/private/model-text-coverage-ed769db/`.

## Software gates and limits

The full local suite passed 1782 tests with 2 optional
S3 SDK tests skipped and 6 dependency deprecation warnings. The
110 focused initialization, resume, expansion and holdout tests passed. Ruff,
web lint, all 145 web tests and the production web build passed. No runtime
source, dependency, production model or service configuration changed.

This remains single-seed exploratory evidence. New text groups, new visual
renderings, length weighting and repetition counts changed together. The same
source corpus and correlated profiles limit independence. Development and both
challenges have been inspected repeatedly; they are regression sets, not fresh
confirmatory tests. Future evaluation must not silently turn them into training
data. Rights-reviewed real scan/photo transcriptions and a fixed-set Tesseract
comparison remain prerequisites for any recognizer release decision.
