# Conservative image-evidence guard for small-angle deskew

## Scope

Issue [#17](https://github.com/dolemi-la/lao-document-ocr/issues/17) followed the
[pinned Tesseract comparison](tesseract-real-deskew-comparison.md): a mathematically
correct rotation direction does not make the whole-foreground rectangle a
reliable estimate of text alignment. On PTC page 1, applying that proposal
returned 280 non-space OCR characters, versus 1,390 with only deskew disabled.
These were output-count diagnostics, not reviewed transcription accuracy.

This change retains the corrected direction but vetoes unsupported proposals.
It changes the existing page-cleanup path, not recognizer weights, training,
line-recognizer preprocessing, or the default for right-angle auto-orientation.

## Acceptance rule

The existing Otsu foreground mask, minimum of 50 foreground points, and
0.15-to-12-degree angle limits remain. Non-finite proposals now return the
unrotated grayscale image. For an in-range proposal, both checks must pass:

1. Every threshold-foreground pixel center must remain inside the original
   canvas after rotation. Projecting the foreground convex hull is sufficient
   for this affine bound check. A failing proposal is rejected before even
   rasterizing the candidate mask.
2. Horizontal row alignment must improve strictly. For each mask row, count
   foreground pixels and divide by the mask's total foreground count. Include
   zero rows above and below, and sum squared first differences of this profile.
   Compare the original mask to a same-sized, nearest-neighbor rotated binary
   mask. Ties and decreases reject the rotation.

The rule was recorded before this run's remote inference. It uses no fitted
percentage threshold, OCR confidence, labels, or source-specific exception.
It is a conservative heuristic, not an optimal skew estimator or an accuracy
criterion. No alternate angles are searched and no additional OCR is invoked.

On rejection, return the same grayscale pixel array that entered deskew.
Autocontrast and contrast adjustment still happen normally. On acceptance,
retain the established same-size cubic grayscale warp and border behavior.
Page dimensions and downstream source/embedded-image handling are unchanged;
this is not a redesign of arbitrary-angle coordinate propagation. Right-angle
orientation remains optional and runs after this small-angle cleanup as before.

### Limits of the guard

Foreground here means the Otsu threshold mask, not every faint or antialiased
pixel and not identified text. A mask pixel's projected center remaining inside
the canvas does not guarantee its full interpolation footprint is retained.
Projection sharpness can be influenced by graphics, rules, borders, and noise.
The guard can reject a useful rotation, or accept a harmful one that passes these
checks. It does **not** guarantee nondecreasing recognized-character counts,
correct line segmentation, or improved accuracy. Pixel scores are not comparable
to confidence scores or CER.

Whole-foreground angle proposals remain imperfect. This veto is deliberately
conservative; genuine optical captures and better text-specific proposals remain
necessary follow-up work. Additional mask processing adds some compute even
though it adds no OCR calls.

## Rights-clear regression fixtures

Tests create simple printed-row geometry and corner graphics from scratch; no
PTC document, remote page image, or remote transcription is copied into tests.
An upright row fixture with an off-center circular graphic causes the old
estimator to propose an in-range rotation. The guard leaves its pixels unchanged.
This reproduces the estimator's geometric failure mode, not a claim of matching
PTC's layout or reproducing its exact Tesseract edit counts.

Coverage includes positive and negative supported tilts, rejection of boundary
clipping before warp, safe-bounds but worsened alignment, ties, non-finite angles,
empty masks, foreground-count versus intensity semantics, no input mutation,
and page-pipeline behavior with right-angle auto-orientation both on and off.
The original direction-convention tests isolate that calculation from the new
veto; end-to-end known-tilt tests still require actual corrected pixels.

## Same-byte real-scan diagnostic — 2026-09-29

Starting checkout: `d6863589c58c91760e8590aea207620fc94bba90`.
Before inference, the plan recorded the guard rule and three arms:

- **No deskew:** retain the existing contrast cleanup but bypass rotation.
- **Unguarded:** the corrected-direction implementation at the starting checkout,
  not the older wrong-sign implementation.
- **Guarded:** the same implementation with this image-evidence veto.

Each arm ran with right-angle auto-orientation off and then on. Extra diagnostic
rotation probes were off. Tesseract 5.5.3, PSM 3, Lao/English, and pinned
`tessdata_fast` 4.1.0 files were unchanged from the earlier comparison. Each
successful source was downloaded once for this run, matched its historical
SHA-256, and supplied identical page pixels to all arms/conditions. Raw PDFs and
page PNGs lived only in temporary directories and were removed afterward.

Four of six sources succeeded, covering seven of nine pinned pages. Both Census
downloads failed again with `RemoteEvaluationError`; this run records the error
class without response/error-body text and does not newly diagnose the network
cause. No substitute pages, transport-security bypass, or source-registry
verification changes were made. No genuine optical pilot is available yet.

### Plain OCR, right-angle auto-orientation off

Counts are recognized non-space characters, not ground-truth edits.

| Source / page | Unguarded characters | Guarded characters | Unguarded confidence | Guarded confidence | Guard result |
| --- | ---: | ---: | ---: | ---: | --- |
| PTC / 1 | 280 | 1,390 | 0.698132 | 0.770005 | Reject projected foreground clipping |
| MAF / 1 | 1,295 | 1,168 | 0.575202 | 0.561618 | Reject decreased row alignment |
| MAF / 15 | 704 | 724 | 0.764893 | 0.785533 | Reject decreased row alignment |
| LaoWIS / 55 | 867 | 867 | 0.636229 | 0.636229 | No in-range proposal |
| KPMG / 8 | 747 | 756 | 0.430509 | 0.427868 | Reject decreased row alignment |
| KPMG / 15 | 600 | 600 | 0.333263 | 0.333263 | No in-range proposal |
| KPMG / 21 | 1,612 | 1,813 | 0.340536 | 0.379049 | Reject decreased row alignment |

PTC's rejected proposal is approximately -5.1166 degrees. Its guarded image and
OCR statistics match the no-deskew control exactly, including 1,390 recognized
characters. That resolves this specific observed output-collapse case, not the
question of how many recognized characters are correct.

All five proposed small-angle rotations on these seven pages were rejected;
the other two pages had no in-range proposal. Consequently **all guarded outputs
match no-deskew outputs in this real-scan run**. It does not validate an accepted
rotation on a real scan. Accepted rotations are covered by geometric fixtures.

The tradeoff is explicit: MAF page 1 returns 127 fewer characters, with lower
confidence, under the guard. No claim is made that those removed characters are
wrong or that the guard improves that page. This is not a no-regression accuracy
certificate.

### Right-angle auto-orientation on

PTC, MAF, and LaoWIS retain zero-degree orientation. KPMG pages 8 and 15 retain
180 and 90 degrees respectively. KPMG page 21 changes from **90 degrees in the
unguarded arm to 180 degrees in both the guarded and no-deskew arms**. Its
weighted confidence changes from 0.801102 to 0.893649 and recognized-character
count from 2,120 to 2,147. Without reviewed ground truth or visual orientation
review, this does not certify the changed selection as correct. Keep it as an
open validation item, not a claimed orientation improvement.

Plain OCR used seven engine recognition calls per arm; auto-orientation used
19 per arm. No extra calls were added by the guard. Total plain pipeline times
were approximately 9.64 s without deskew, 10.04 s unguarded, and 10.11 s guarded.
Auto-orientation totals were approximately 22.24, 23.69, and 22.87 s respectively.
Order was reversed on even pages, but this single local timing remains
observational, not a controlled performance benchmark or promised latency.

## Evidence, preservation, and remaining work

Local ignored evidence is under `training/runs/deskew-guard-d686358/`:
`experiment-plan.json`, the preserved `unguarded_preprocessing.py`, `run.py`,
`runner.identity.json`, six `source-XX.summary.json` files, `complete.py`, and
`completed.summary.json`. Reports contain source identifiers, numeric counts,
fixed decision reasons, and hashes, not OCR/native strings, sample transcriptions,
raw URLs, or persisted page-image paths. Error messages are not serialized.
No remote content entered training or the public benchmark.

Plan SHA-256:
`a5b8efdf5c03a6c9bdb979739a75db809346cbd96a8bf480fc75631f29cea789`.
Completion summary SHA-256:
`3b9bf2ae3f4329f7afc19b106a092fcd617cadc2b1c6475face6d750baaa3b8c`.
The completion audit verifies source hashes, identical input pixels, unchanged
pinned runtime/models, no-extra-OCR counts, and exact no-deskew pixels/statistics
when a proposal is rejected. Previous studies and the frozen candidate/challenge
identities remain unchanged. No recognizer training or model promotion occurred.

**Issue #17 remains open.** The engineering veto and PTC reproduction are done,
but two source failures, MAF page-1 output reduction, KPMG page-21 orientation
review, and the six-page genuine capture pilot remain unresolved. The project
still has no reviewed, frozen optical benchmark or published real Tesseract
accuracy baseline. Do not substitute these heuristic diagnostics for those gates.


## Right-angle line-axis follow-up

The [orientation geometry follow-up](orientation-line-geometry.md) separately
checks OCR line axes before ranking already-probed right-angle candidates.
Tesseract can return confident text in vertical boxes; the new veto prevents
those rotated candidates from winning purely on confidence/character count.
It shares versioned numeric geometry with remote rotation probes, preserves the
existing score thresholds and baseline skip rules, and adds no OCR calls.
KPMG pages 8 and 21 change from 180 to 90 degrees, while page 15 remains at 90.
Plain OCR and the deskew guard are unchanged. MAF transcription review, missing
source coverage and the genuine optical pilot remain unresolved; see the linked
report for the visual-review scope, same-result replay method and limitations.
