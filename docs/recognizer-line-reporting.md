# Whole-line recognizer reporting and decoder development control

CER measures character edits, not the probability of an entirely correct line.
For example, one deletion in a 100-character line gives 1% CER but zero exact
lines. Report both measures rather than translating low CER into reliable
whole-line transcription.

## Reusable benchmark metrics

`benchmark-recognizer` and the Python `benchmark_recognizer(...)` function now
include the following fields in `overall`:

| Field | Definition |
| --- | --- |
| `exact_lines` | Number of normalized hypotheses equal to their normalized references. |
| `exact_line_ratio` | `exact_lines / samples`. |
| `empty_predictions` | Number of hypotheses empty after benchmark normalization. |
| `empty_prediction_ratio` | `empty_predictions / samples`. |

Existing character/word edit counts, denominators, CER and WER retain their
semantics. Both reference and hypothesis use `normalize_lao_text`: Unicode NFC,
existing whitespace handling, and line-ending normalization. It does not correct
Lao words or substitute characters. New reports mark this additive contract as
`metric_version: normalized-line-metrics-v1`; historical reports are not rewritten.
`RecognitionMetricCounts.add(...)` is a lower-level accumulator and expects its
caller to supply normalized strings, as the benchmark does.

Whole-line ratios use image counts, not character counts. Images with the same
label still count individually; these ratios do not establish the number of
independent text groups or documents. A zero-sample accumulator reports `null`
for line ratios. An empty benchmark input is still rejected.

**Normalized-empty predictions differ from raw training-health emptiness.** A
spaces-only prediction counts as empty here. It is not necessarily an all-blank
CTC token path. Training health retains its original raw semantics, and this
change does not alter checkpoint selection, normalization, decoding, or training.

The default detailed report still includes model metadata and per-image IDs,
reference text, hypotheses, timing, and confidence. It adds `exact_match` and
`empty_prediction` booleans to those rows. The existing confidence-calibration
input remains available in this mode; calibration still requires separate data
and decoder-compatible settings.

## Aggregate-only reports

```bash
lao-ocr benchmark-recognizer \
  --manifest training/runs/development/manifest.jsonl \
  --model training/runs/candidate/recognizer.pt2 \
  --output training/runs/development/aggregate.json \
  --summary-only
```

Python callers use `benchmark_recognizer(samples, recognizer, summary_only=True)`.
This mode constructs an allowlisted report containing only:

- `schema_version`, `report_type: recognizer-aggregate-only`, and `metric_version`;
- total benchmark elapsed seconds;
- aggregate metrics in `overall`.

It does not construct per-image result rows or access arbitrary recognizer
metadata. The returned JSON contains no reference/predicted strings, sample IDs,
image paths, vocabulary, or per-image confidence. Model metadata is intentionally
omitted in full: copying a nominally harmless metadata object could expose
paths, vocabulary, or text supplied by a custom recognizer. Regression tests
include metadata/confidence properties that raise if the summary accesses them.

The mode does not anonymize the inputs, hide console output from arbitrary
recognizer implementations, or certify source rights. Inference necessarily
handles labels and predictions in process memory. It is a report-content boundary,
not a global logging or memory-erasure guarantee. CLI status still prints the
chosen output path and aggregate counts, not document text.

Aggregate reports cannot fit confidence calibration: they deliberately contain
no usable per-image calibration points. The existing calibration loader rejects
them. They also do not identify a model or freeze a dataset by themselves. Keep
separate verified artifact/input hashes and the holdout audit with an experiment.
The existing report writer is unchanged and can overwrite its destination; use
new output paths to preserve historical evidence. No remote diagnostic command
or its privacy policy was changed by this addition.

## Predeclared paired decoder control — 2026-09-29

The starting repository revision was `b90eeeb`. The candidate artifact remained
fixed at:

```text
66f6bbd8cb2ea078cb2bbb1b14ec247329d30bc96da556bc8740fa0d2605ad0f
```

This study used the **old twelve development labels**, already examined and used
for checkpoint selection. It is explicitly development, not a new holdout or an
independent accuracy test. Their labels are disjoint from both the 288 training
labels and the frozen 160-group challenge. No challenge image was run through a
model during this study, and every existing file in its directory was fingerprinted
and verified unchanged.

The plan was saved before inference. It fixed greedy decoding versus plain CTC
prefix beam search at width 10, with no language model, token bonus, calibration,
weight update, or checkpoint reselection. Both decoders used the same exported
candidate, MPS, and the same images/preprocessing. No beam-width search followed.

Four cases each contained the same twelve text groups:

1. The original twelve development images, referenced unchanged.
2. Twelve new `clean-scan` profile images.
3. Twelve new `noisy-scan` profile images.
4. Twelve new `phone-photo` profile images.

New profiles shared each label's Phetsarath OT Regular clean render at font size
48. Seeds were fixed as `20260929 + text_index * 3 + profile_index`, using current
geometry-version-2 augmentation and strict font coverage. Each new image had a
pinned hash and passed capacity preflight at input height 48 / maximum width 768.
The training-exclusion audit passed for every case. There are **48 correlated
images of only 12 reused labels**, with one new seed per label/profile. Original
images and regenerated profiles need not have identical font size or geometry.

## Results and limits

Each case has 225 normalized reference characters. The table reports both
predeclared decoders; no favorable intermediate settings replaced the plan.

| Development case | Greedy CER | Beam-10 CER | Exact lines, greedy | Exact lines, beam |
| --- | ---: | ---: | ---: | ---: |
| Original images | 31/225 = 13.7778% | 31/225 = 13.7778% | 2/12 | 2/12 |
| New clean-scan profile | 30/225 = 13.3333% | 30/225 = 13.3333% | 3/12 | 3/12 |
| New noisy-scan profile | 27/225 = 12.0000% | 27/225 = 12.0000% | 2/12 | 2/12 |
| New phone-photo profile | 34/225 = 15.1111% | 34/225 = 15.1111% | 2/12 | 2/12 |

**All 48 normalized predictions were identical between decoders**, not merely
aggregate scores. No exact line was repaired or broken; neither decoder produced
normalized-empty output. Both yielded 9/48 exact lines across the four cases.
The regenerated profiles alone yielded 7/36. These are correlated development
counts, not estimates of real-document reliability. The noisy-profile result
being numerically better than clean in this small control is not evidence that
adding noise improves OCR generally.

Benchmark elapsed time summed across the four cases was approximately 0.645 s
for greedy and 2.356 s for beam, after identical unscored warm-up and excluding
model loading. Greedy ran first in each case; this single local timing is not a
controlled performance benchmark or a promised latency. It showed extra work
without a prediction change in this setting.

The first and last image of each case were also decoded on CPU for each decoder:
**16 predetermined parity checks, zero normalized-prediction mismatches**. This
does not claim all 48 images were evaluated on CPU. The actual CLI summary-mode
smoke check reproduced original-case greedy metrics. Default detailed-report
calibration compatibility and normalized metric parity have regression coverage.

The result does not show that beam search is always useless, nor does it solve
phone-profile recognition errors. It provides no basis to replace greedy as the
production default. Do not tune on the preserved 480-image challenge to manufacture
a decoder gain. Further diagnostics should isolate rendering/preprocessing and
training robustness on designated development data; revised candidates need
separate evaluation. Genuine reviewed optical captures remain outstanding.

## Local evidence

All new images, manifests, and experiment scripts remain private and Git-ignored:

```text
training/runs/decoder-development-b90eeeb/
  prepare.py
  experiment-plan.json
  preflight.summary.json
  original.jsonl
  clean-scan.jsonl
  noisy-scan.jsonl
  phone-photo.jsonl
  images/
  evaluate.py
  evaluation-runner.json
  <case>.summary.json
  cli-summary.json
  completed.summary.json
```

The completion summary contains aggregate counts and fingerprints, not reference
strings, hypotheses, sample IDs, or image paths. The private plan also retains
protected local paths to verify file identity; it is not a public anonymization
artifact. The four case reports, runner, benchmark code, artifact, and plan are
fingerprinted. All 519 protected input/code files were verified unchanged.

| Evidence | SHA-256 |
| --- | --- |
| Pre-inference plan | `588e6032371217da31bc544540f1334df26edaafd34ccf35f4ca9d2d7bafc09e` |
| Completion summary | `739c3b5bbdd97d97c85a7ac10577d9ed7c5a323c6a5bd59167f3a7eb8a74aa9e` |

No training was started. Production models, OCR defaults, font policy, capture
kits, and collector sessions were not modified.


## Subsequent page-cleanup correction

A separate [deskew investigation](preprocessing-deskew.md) found that existing
page cleanup could compound small tilt rather than undo it. The angle correction
is now covered by pixel and pipeline regressions. Its old-development-data
comparison reduced overall errors relative to old cleanup, but did not improve
the phone subset; direct line inference still had fewer character errors than
either cleanup path. No page preprocessing was added to exported line inference,
and the frozen challenge was not used for this investigation.
