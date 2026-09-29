# Small-angle deskew direction correction

## Reproduced defect

Inspection at source revision `501dade713b5835c031b1c3042dfbb9b2a65e406`
found that page cleanup could increase a small tilt instead of correcting it.
The existing `_deskew` negated the angle from the minimum-area ink rectangle
after a partial angle-convention adjustment. On controlled printed-row patterns,
a -3 degree rotation became approximately +6 degrees of residual image-axis
tilt, and a +3 degree rotation became approximately -6 degrees. The magnitude
was roughly doubled in both directions. These are image-geometry measurements,
not OCR accuracy estimates.

Rectangle axes are interchangeable modulo 90 degrees. Cleanup now maps the
reported rectangle angle to the nearest horizontal/vertical axis using:

```python
angle = (rectangle_angle + 45.0) % 90.0 - 45.0
```

That signed angle is passed directly to `getRotationMatrix2D`, without the
additional negation. Regression tests exercise actual rotated pixels as well
as equivalent positive/negative rectangle-angle representations.

## Production scope

This fixes the already-enabled small-angle cleanup in
`preprocess_image`, which `process_document` applies before engine recognition.
It is **not** a new 90/180/270-degree orientation decision. Right-angle
`auto_orient_right_angles` remains opt-in and defaults to false.

The existing guards are retained: fewer than 50 detected foreground points,
absolute tilt below 0.15 degrees, or tilt above 12 degrees do not rotate.
Page dimensions, interpolation, border handling, autocontrast, and contrast
adjustment are unchanged. Upright, blank, and sparse controls remain unchanged
by the deskew step. Pipeline tests verify that corrected pixels reach the OCR
engine and that the page canvas retains its size.

This remains a minimum-area-rectangle estimator over thresholded page ink, not
a robust text-baseline estimator or perspective rectifier. Graphics, margins,
borders, and asymmetric text can influence it. The unchanged same-size warp
can also clip content near a page boundary. This correction does not address
those separate limitations or redesign source-image/embedded-asset alignment.

Direct exported **line** recognition does not call page preprocessing. The
change therefore does not alter the frozen line model's normal inference path,
weights, vocabulary, font policy, or decoder defaults.

## Predeclared development comparison — 2026-09-29

A new plan was saved before inference. It reused only the old twelve
development labels and the existing four cases from
[the decoder development control](recognizer-line-reporting.md): original
images plus regenerated clean-scan, noisy-scan, and phone-photo variants.
No new image, text label, or training step was introduced. There are 48
correlated images of 12 repeatedly inspected development labels, not an
independent evaluation set.

Three paths were fixed before scoring:

1. Direct line inference with no page cleanup, matching the prior decoder study.
2. Full page cleanup from the original source revision, followed by line inference.
3. Full page cleanup with the corrected angle, followed by line inference.

All paths use the same pinned Phetsarath candidate export, greedy decoding,
normalized benchmark scoring, and MPS. No language model, calibration, beam
search, checkpoint selection, or tuning was performed. Old versus corrected
cleanup isolates the angle change. No-cleanup versus either cleanup also
changes contrast and autocontrast, so it does not isolate deskew alone.

The original source cleanup was preserved in the ignored experiment directory.
The new plan pins both implementations, all inputs, and the model. The frozen
160-text-group challenge was **not** used for inference; its files and results
were included in protected-file verification.

### Results

Each case has 225 normalized reference characters. Exact-line counts are
image-level counts, not independent document observations.

| Development case | No-cleanup CER | Old-cleanup CER | Corrected-cleanup CER | Exact lines: none / old / corrected |
| --- | ---: | ---: | ---: | ---: |
| Original images | 31/225 = 13.7778% | 47/225 = 20.8889% | 41/225 = 18.2222% | 2 / 2 / 2 |
| Clean-scan variants | 30/225 = 13.3333% | 34/225 = 15.1111% | 33/225 = 14.6667% | 3 / 3 / 3 |
| Noisy-scan variants | 27/225 = 12.0000% | 39/225 = 17.3333% | 36/225 = 16.0000% | 2 / 2 / 3 |
| Phone-photo variants | 34/225 = 15.1111% | 46/225 = 20.4444% | 46/225 = 20.4444% | 2 / 2 / 2 |
| All 48 images | 122/900 = 13.5556% | 166/900 = 18.4444% | 156/900 = 17.3333% | 9 / 9 / 10 |

Corrected cleanup makes ten fewer character edits overall than old cleanup and
repairs one previously nonexact line. Across individual images it has fewer
edits on 14, more on eight, and equal edit counts on 26. No previously exact
old-cleanup line becomes nonexact. The phone subset's total edit count does not
improve. Both cleanup paths still have more character errors than direct line
recognition in this control. Thus the bug fix is **not** evidence for adding
page cleanup to exported line recognition, claiming a general phone-photo gain,
or changing that model's preprocessing contract.

All no-cleanup metrics exactly reproduce the previous decoder study. None of
the paths produces a normalized-empty string. The first and last image of each
case were also checked on CPU for each path: 24 predetermined checks, zero
normalized-prediction mismatches. This does not claim all images were scored on
CPU.

## Geometry and integrity checks

Known-rotation printed-row controls use +/-1, +/-3, +/-6, and +/-10 degrees,
plus an upright case. The original cleanup approximately doubles each nonzero
tilt; the corrected cleanup has zero residual rectangle angle in this run.
Pixel regression tests allow a small rasterization tolerance instead of relying
on exact floating-point angles. Separate tests cover the public preprocessing
function, document-pipeline engine input, angle guards, empty/sparse pages,
and positive/negative angle conventions.

All 572 protected model, source, input, and prior-study files were verified
unchanged before and after inference. Both preprocessing implementations have
separate hashes. The summary contains aggregate measurements and hashes, not
reference strings, OCR text, sample IDs, or image paths. The private plan
contains local paths for identity checks; hashes are reproducibility identifiers,
not anonymization guarantees.

Local evidence remains private and Git-ignored:

```text
training/runs/deskew-direction-501dade/
  prepare.py
  legacy_preprocessing.py
  experiment-plan.json
  evaluate.py
  completed.summary.json
```

| Evidence | SHA-256 |
| --- | --- |
| Pre-inference plan | `6495837a979e2d762f4ed9671d159b20c860ef56600831cae24c3d8f0c67aae9` |
| Completion summary | `cd62e0bc6236200c7e1922486c06c4dad4fab754156740edba2667b9135f9749` |
| Frozen candidate export | `66f6bbd8cb2ea078cb2bbb1b14ec247329d30bc96da556bc8740fa0d2605ad0f` |

The local Tesseract executable is unavailable, so no new Tesseract remote-suite
A/B was run. Earlier remote diagnostic results remain historical; these
synthetic controls do not replace optical evidence. No document was downloaded,
no production model was promoted, and capture kits/collector sessions were not
modified. A future real-suite comparison must pin the preprocessing revision
as well as the same source pages and OCR settings.
