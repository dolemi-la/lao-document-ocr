# Right-angle candidate veto from OCR line geometry

## Finding and bounded fix

The [deskew-guard comparison](deskew-evidence-guard.md) left a specific question:
KPMG page 21 selected 180 degrees rather than 90 after an unsupported deskew
proposal was rejected. The same problem also occurred on page 8. Higher OCR
confidence did not establish a correct page orientation.

A targeted diagnostic at `3a01033f561ce0869235814a58f6a9d915e2bb23` examined
0/90/180/270 degrees with pinned Tesseract 5.5.3, Lao/English, PSM 3, and the
existing `tessdata_fast` 4.1.0 files. Each successful PDF matched its historical
SHA-256. No source document, rendered page or transcription was retained.

At 180 degrees, Tesseract returned high-confidence text whose line boxes were
vertical in the input image. At 90 degrees, the line boxes were horizontal:

| KPMG PDF page | Clockwise angle | Weighted confidence | Non-space characters | Horizontal lines | Vertical lines | Original score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 90 | 0.944768 | 803 | 18 | 0 | 6.320117 |
| 8 | 180 | 0.942855 | 877 | 0 | 21 | 6.390335 |
| 15 | 90 | 0.917945 | 653 | 37 | 0 | 5.951137 |
| 15 | 180 | 0.916266 | 650 | 0 | 38 | 5.936041 |
| 21 | 90 | 0.890414 | 2,098 | 34 | 0 | 6.810967 |
| 21 | 180 | 0.893649 | 2,147 | 0 | 34 | 6.856338 |

The table counts all nonempty OCR line boxes; the veto below uses only its
more restricted multi-character, non-square subset.

The score combines confidence and recognized-character count. It slightly
favored the sideways candidate on pages 8 and 21. The result demonstrates why
successful text recognition is not by itself a reliable page-orientation signal.
It does not establish the recognizer's internal mechanism for handling sideways
text or the accuracy of every recognized character.

The assistant also visually inspected rendered pages 8, 15 and 21 of the public
[World Bank PDF](https://documents1.worldbank.org/curated/en/099090525045519534/pdf/P172774-541a8e9e-e74f-4890-88e3-b6c6a037e69d.pdf).
Their main printed text/table layout needs a 90-degree clockwise correction.
This is a narrow visual orientation observation, not a human-reviewed benchmark
annotation or permission to publish document images or ground-truth text.

## Shared geometry rule

`orientation_geometry.orientation_line_geometry(...)` emits only a version,
counts, and a boolean decision. It never emits text, coordinates or sample IDs.
The version is `multi-character-line-axis-v1`.

A line contributes only when it contains at least two non-space characters and
has finite, positive, non-square bounding-box dimensions. Width greater than
height supplies horizontal support; height greater than width supplies vertical
support. Single-character, square, empty and invalid boxes are unknown and are
counted separately as ignored lines. Support is weighted by recognized character
count, not by an unweighted vote over line boxes.

A nonzero orientation candidate is vetoed when both conditions hold:

- at least two informative lines are vertical;
- vertical character support is strictly greater than horizontal character support.

This is an exploratory horizontal Lao/English layout heuristic, chosen after
inspecting the failure. It is not a source-specific exception, trained classifier,
or proven universal rule. It can be uninformative or wrong for unusual layouts,
vertical scripts, fragmented characters, or unreliable OCR boxes.

## Selection and compatibility

Only candidates already probed by opt-in right-angle auto-orientation are
filtered. The baseline at zero degrees remains available even when its geometry
looks sideways. Remaining candidates retain the existing score/confidence/count
ordering and existing improvement thresholds. If the winner still fails those
thresholds, return the unchanged baseline. No confidence threshold is relaxed,
no extra OCR is invoked, and no source ID or expected angle is hard-coded.

The existing high-confidence and Lao-dominant probe-skip rules are unchanged.
**A high-confidence sideways baseline may still be skipped.** This patch does
not claim to solve that separate case. Nor can line-axis evidence distinguish
upright from upside-down horizontal text; existing OCR ranking still decides
between those cases. Right-angle auto-orientation remains off by default.

Diagnostics add `candidate_geometry_policy`, `candidate_geometry`, and
`geometry_vetoed_degrees`. Retained best/runner-up score margins now describe
eligible candidates, not rejected sideways candidates. The geometry list keeps
the successful baseline/probe evidence available without document strings.
Failed probes do not acquire fabricated geometry or serialized exception text.

Optional `ocr_line_stats` metadata adds `orientation_geometry`. The standalone
remote rotation diagnostic applies the same veto when that versioned geometry
is available. Legacy numeric-only stats remain supported without inventing box
evidence. The remote geometry reader accepts a fixed set of integer fields,
recomputes the veto, and excludes arbitrary metadata keys. This does not change
the standalone diagnostic's existing rotate-then-cleanup procedure into the
production cleanup-then-rotate procedure; sharing a veto is not a claim that
every diagnostic image is pixel-equivalent to a production probe.

## Paired pipeline validation

The six-source, nine-page suite was requested. Four sources / seven pages were
available again; both Census fetches returned `RemoteEvaluationError`. The
exception class was recorded without message bodies. No substitute source or
transport-security bypass was used. There is still no registered optical pilot.

Current production code performed real OCR. The historical pipeline was loaded
from the exact previous commit and received replayed recognition results after
verifying every OCR input pixel hash. This isolates selection from OCR variation
and verifies equal call counts; it is not a timing comparison of independent
OCR executions. Both arms used identical source bytes, rendered page pixels,
preprocessing, traineddata, engine settings and raw recognition results.

With auto-orientation disabled, all seven output `Page` objects were unchanged,
including text, blocks and geometry. Additional line-stat metadata is
observational. With auto-orientation enabled:

| Source / page | Before clockwise angle | After clockwise angle | After recognized characters | Real OCR calls |
| --- | ---: | ---: | ---: | ---: |
| PTC / 1 | 0 | 0 | 1,390 | 1 |
| MAF / 1 | 0 | 0 | 1,168 | 1 |
| MAF / 15 | 0 | 0 | 724 | 1 |
| LaoWIS / 55 | 0 | 0 | 867 | 4 |
| KPMG / 8 | 180 | 90 | 803 | 4 |
| KPMG / 15 | 90 | 90 | 653 | 4 |
| KPMG / 21 | 180 | 90 | 2,098 | 4 |

The KPMG result now agrees with the public-page visual orientation observation.
Pages 8 and 21 return fewer characters than their old sideways winners; this is
not evidence of worse or better transcription accuracy. Page dimensions and
preserved image handling use the existing right-angle coordinate transforms.
The upright available cases retain zero degrees. There were seven plain OCR
calls and nineteen auto-orientation calls, with zero additional calls needed by
the geometry check. The historical comparison replayed the same twenty-six
results rather than invoking Tesseract again.

## MAF remains unresolved

The separate targeted MAF page-1 check reproduced 1,168 characters and 32
horizontal lines under guarded cleanup, versus 1,295 characters and 36 lines
under unguarded cleanup (35 horizontal, one single-character vertical line).
The public MAF page has upright but degraded printed content. Visual inspection
of its layout is not a verified transcription and does not determine which
extra/missing OCR characters are correct. This fix neither disables the deskew
guard nor claims to resolve MAF's text-retention tradeoff.

Issue #17 remains open for MAF transcription review, the unavailable Census
cases, the genuine capture pilot, and broader validation of accepted deskew
proposals. No owned model, training dataset, capture kit, collector session or
font policy was changed. Phetsarath remains canonical.

## Evidence and tests

Local, Git-ignored evidence is under `training/runs/orientation-axis-3a01033/`:

- `experiment-plan.json`, `probe.py`, and two targeted source summaries;
- `validation-plan.json`, `validate.py`, and `historical_pipeline.py`;
- six `validation-source-XX.summary.json` reports;
- `completed.summary.json`.

Validation plan SHA-256:
`0909c1654f865cde2fae484f82aa1158dba1eca1a8fba82304b7f51a8f0f0ed3`.
Completion summary SHA-256:
`2abb3f948a7453e95afdcb81560b9fa37dc0184d909ecf521ce4ef36e1dcea7c`.

Reports retain fixed source identifiers, aggregate counts, geometry decisions
and hashes, not OCR/native text, line coordinates, image paths or error-message
bodies. Temporary remote bytes/pages were removed. Protected model, source,
previous experiment and challenge-summary hashes were checked unchanged.
These are diagnostic observations, not a real held-out OCR accuracy benchmark.

Regression tests cover character-weighted evidence, sparse/ambiguous/invalid
boxes, confidence-only wrong winners, safe baseline fallback, unchanged score
thresholds and skip rules, default-off behavior, failed probes, report privacy,
and standalone-diagnostic compatibility. Generated fake-engine fixtures contain
no copies of the institutional remote documents.
