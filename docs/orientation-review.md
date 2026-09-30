# Review signals for unresolved output orientation

## Scope

The [geometry-aware probe-skip control](orientation-probe-skip.md) showed that
confident sideways and upright OCR results can have essentially equal scores.
Probing more angles is not the same as correcting the page. This addition makes
unresolved output visible to users without relaxing rotation acceptance rules.
It does not choose a new angle, change OCR, add calls, or certify accuracy.

The review signal examines only the geometry associated with the **selected**
angle in each page's existing orientation diagnostics. A vetoed or losing
sideways candidate must not trigger a warning about an upright selected result.
The same `multi-character-line-axis-v1` sideways criterion remains in effect.

## Versioned, no-text summary

`orientation_review.build_orientation_review(...)` returns a frozen validated
`OrientationReview`. Its `to_dict()` emits this fixed shape:

```json
{
  "version": "selected-line-axis-review-v1",
  "status": "review-required",
  "page_count": 3,
  "review_pages": [2],
  "unassessed_pages": [3]
}
```

Page numbers are one-based and local to the individual input document. They are
sorted, unique, bounded by the actual processed page count, and disjoint across
the two lists. Each serialization returns fresh lists. No text, predictions,
line boxes, source paths, confidence, vocabulary or arbitrary model metadata
is included. Page count/page numbers themselves are document metadata, not an
anonymization guarantee.

| Status | Meaning |
| --- | --- |
| `not-requested` | Right-angle auto-orientation was disabled. Every page is unassessed by this optional check. |
| `not-assessed` | No supported enabled/disabled orientation contract was provided, for example by a legacy/custom processor. |
| `review-required` | At least one selected output retains sideways-dominant geometry. There may also be unassessed pages. |
| `incomplete` | No selected page triggered the sideways flag, but some evidence is missing/invalid/non-informative, or there are zero pages. |
| `no-sideways-evidence` | Every page had usable selected evidence and none triggered the heuristic. This is **not** a claim of upright or accurate OCR. |

Missing or duplicate page records, invalid/unknown angles, duplicate selected
candidate records, unsupported geometry versions, invalid counts, inconsistent
line/character support, and entirely non-informative geometry stay unassessed.
A reported `sideways_dominant` boolean is not trusted independently: the existing
geometry sanitizer recomputes it from known integer counts. Raw diagnostic or
engine strings are never forwarded into the summary.

The check cannot distinguish horizontal upside-down text from upright text. It
can miss unusual layouts, unreliable boxes, sparse labels, or faint content.
The absence of a warning is never a successful orientation/accuracy test.

## Production surfaces

The document pipeline stores the summary at
`metadata.auto_orientation.review`. It is included in `/v1/parse` JSON and the
JSON inside synchronous conversion archives, asynchronous archives, and CLI
conversion outputs. Existing page content and export processing are unchanged.
DOCX/Markdown/TXT contents are not modified to insert warning text.

The async worker separately recomputes the typed summary from the selected
geometry, not from an arbitrary embedded `review` object. The job manager
exposes it as `orientation_review` only after marking the job `succeeded`.
Uploading, queued, running, failed and cancelled jobs expose `null`. A legacy
runner that supplies no typed summary also exposes `null`, rather than inventing
assessment results. Each batch member has its own summary and page numbering.

The worker computes the result before returning its stored artifact. The
manager's existing success transition is still protected by its lock; a poll
cannot expose an in-progress summary as a completed result. Cancellation and
failure paths suppress the result even when the worker had populated it.

The web UI displays persistent, page-specific Lao/English cautions in its
existing polite live region after a job succeeds. Sideways pages and unassessed
pages have different wording. Conversion/download remains available; the UI
does not imply the source was corrected merely because a ZIP was created.
Switching language re-renders the warning, and starting another upload clears
it with the previous job. This original reporting slice does not change automatic selection. The later
[HTTP/web manual-correction integration](manual-page-rotation-http.md) adds a
separate operator control without weakening automatic thresholds.

The frontend validates the known version, status, page count and disjoint
bounded integer arrays before constructing local messages. It never renders
message text supplied by the summary. Missing/legacy/future/malformed summaries
are ignored for backward compatibility without generating an "all upright"
message. Default-off jobs do not gain an orientation warning.

## Repeated known-rotation control

Starting source revision: `e5da6227e93792b63b6901b6d0e7d11994bfffed`.
The new plan was recorded before inference. It reused the previous construction:
twelve project-authored Lao lines and a mixed page made from six of those lines
plus the same six project-authored English lines. Phetsarath OT Regular at size
32, layout and once-before-cardinal-rotation preprocessing were unchanged.
These are eight correlated variants of two synthetic pages, not optical data.

Pinned Tesseract 5.5.3, PSM 3, Lao/English `tessdata_fast` 4.1.0, and a shared
pixel-hash-keyed in-memory OCR cache supplied both the previous and current
selection implementations. There were eight actual Tesseract executions and
26 logical calls in each arm. All selected pixels, recognized-line objects,
selection diagnostics and requested OCR input hashes were equal.

| Control family / clockwise input | Selected correction | Required correction | Review status |
| --- | ---: | ---: | --- |
| Lao / 0 | 0 | 0 | No sideways evidence |
| Lao / 90 | 0 | 270 | Review required |
| Lao / 180 | 180 | 180 | No sideways evidence |
| Lao / 270 | 90 | 90 | No sideways evidence |
| Mixed / 0 | 0 | 0 | No sideways evidence |
| Mixed / 90 | 0 | 270 | Review required |
| Mixed / 180 | 180 | 180 | No sideways evidence |
| Mixed / 270 | 90 | 90 | No sideways evidence |

Both versions still correct 6/8 orientations. The two known unresolved sideways
outputs now raise review flags; they are **not fixed**. No general warning
sensitivity/specificity, real optical accuracy, or new transcription improvement
is inferred from this small repeatedly inspected control. No remote source or
frozen challenge image was used for inference.

Private evidence remains Git-ignored under
`training/runs/orientation-review-e5da622/`: `check.py`, `historical_pipeline.py`,
`experiment-plan.json`, and `completed.summary.json`. The plan pins the runner,
font, source corpus, engine models, old/new pipeline, summary helper, and prior
study/model identities. Summaries contain only counts, fixed case/policy names,
page numbers and hashes. Images and recognized strings were not saved.

Completion SHA-256:
`17f67e1cc4dbf1b1d29d1f6809cb5f90c55b419d93fe038858fa9e7ca0aba1c6`.

## Regression coverage and remaining work

Python tests exercise selected-versus-rejected geometry, unknown evidence,
malformed/duplicate records, immutable bounded public data, equal-score fallback,
unchanged default behavior, real API processing/export paths, async status and
download agreement, batch membership and all lifecycle states. Fake OCR engines
use authored fixture text, not institutional documents.

The web has dependency-free `pnpm test` coverage using Node's test runner and
TypeScript stripping (CI Node 22). Tests cover both locales, terminal-state
handling, malformed or missing responses, page ordering, unsafe metadata, and
App/live-region wiring. The App-wiring check is source-level, not a browser
interaction or visual audit. Web lint and TypeScript/Vite build remain separate
gates. Lao message glyph coverage was checked against the pinned Phetsarath
font; no font is bundled by this change.

Issue #17 remains open. Equal-confidence automatic selection needs a separately
justified design. MAF transcription review, missing-source coverage and the
actual printed/scanned/photographed pilot remain outstanding. Model weights,
font policy, capture kits, collector sessions and optional-orientation defaults
are unchanged.

## Explicit correction follow-up

[Manual page rotation](manual-page-rotation.md) adds CLI/Python `--rotate-page`
and `page_rotations` overrides. It records the operator's correction separately
and reviews the selected output without claiming the angle is correct. A bad
manual choice can still trigger a sideways warning. The later [HTTP/web integration](manual-page-rotation-http.md) adds the request
controls and preserves warnings for manually corrected outputs.


## HTTP/web manual-correction follow-up

[Manual HTTP/web corrections](manual-page-rotation-http.md) now expose the same
CLI/Python rules through multipart requests and the bilingual UI. For jobs with
explicit overrides, the public review is recomputed from selected page geometry
and matches `manual_page_rotations.review`, even when auto-orientation is off.
Unspecified default-off pages remain unassessed. The earlier CLI-only scope
statement describes that earlier implementation; it no longer describes the
current request schema. No warning is treated as a verified transcription or
certificate of correct operator choice.
