# Geometry-aware orientation probe skipping

## Bounded change

The [line-axis veto](orientation-line-geometry.md) excluded sideways rotated
candidates, but its original implementation at `ecc22dd` still let confidence
shortcuts suppress every probe of a sideways baseline. That creates a separate
gap: the optional orientation path can stop before looking for a qualifying
horizontal candidate.

The skip policy is now `confidence-with-line-axis-v1`. First compute the existing
high-confidence or Lao-dominant skip reason, with all thresholds unchanged. If
that shortcut would apply but the baseline already satisfies the shared
`multi-character-line-axis-v1` sideways criterion, disregard the shortcut and
run the existing 90/180/270-degree probes.

The sideways criterion is unchanged: at least two informative vertical lines
with strictly more vertical than horizontal character support. Single-character
labels, one vertical line, square/invalid boxes and equal support remain
insufficient to override the shortcut. Horizontal high-confidence and
horizontal Lao-dominant pages keep their one-call fast paths.

This uses no document-specific exception, new confidence threshold or extra
search angles. Affected opt-in pages can incur **three additional OCR attempts**
compared with the old early return. Existing low-confidence pages already
probed, and default-off requests still perform only baseline recognition.

## Probing is not acceptance

The zero-degree baseline remains an eligible fallback. Rotated candidates still
pass the existing sideways veto and ranking, then must meet all previous
acceptance requirements: at least 15% score improvement, at least 0.08 confidence
improvement, and the existing recognized-character retention floor. None is
relaxed by baseline geometry. Probe errors and insufficient improvements retain
the baseline; cancellation remains checked before each probe.

Consequently, this fixes the **suppressed-probe path**, not every confidently
sideways page. Tesseract can return similarly confident text from upright and
sideways images. When their scores/confidences are nearly equal, the existing
acceptance rules intentionally reject a rotation even after geometry enables
probing. More probing must not be reported as a successful correction.

New diagnostics add:

- `probe_skip_policy: confidence-with-line-axis-v1`;
- `probe_skip_override_reason: sideways-baseline` only when geometry overrode a
  shortcut that otherwise would have applied, or `null` otherwise.

`probe_skipped` is false and `probe_skip_reason` is null on that overridden path.
Existing per-candidate geometry, eligible score margins, selected angle and
selected statistics remain available. The additional fields are fixed policy
labels, not OCR/reference text, boxes, paths or error-message contents. A
baseline that was already low confidence is not mislabeled as a skip override.

## Paired project-authored Tesseract control

Starting revision: `ecc22dd6476ac1931f54d1f79fc5ac02086d3dca`.
The plan was saved before inference, after the failing regression tests and
bounded skip-rule patch. It fixed two synthetic page families: twelve eligible
project-authored Lao corpus lines, and six of those lines interleaved with six
newly authored English fixture lines. Phetsarath OT Regular, font size 32,
strict coverage, padding and line spacing were fixed in the preserved runner.

Each page received the existing cleanup once, followed by exact 0/90/180/270
clockwise input rotations. Both selection versions consumed these same image
pixels. The required correction follows directly from construction:
`(360 - input_rotation) % 360`. This is a known-rotation geometry control, not
transcription ground truth or real optical evidence. There are **eight correlated
variants of two synthetic pages**, not eight independent documents.

The engine was local Tesseract 5.5.3, PSM 3, `lao+eng`, with the existing pinned
`tessdata_fast` 4.1.0 files. Historical/current selection shared an in-memory
OCR cache keyed by complete pixel/mode/dimension hashes. Eight unique images
required eight actual OCR executions; logical call counts describe how many
recognition requests each selection path would make without cache reuse. This
is not an independent runtime comparison. No OCR strings or images were saved.

| Family / clockwise input | Required correction | Old selected | New selected | Old calls | New calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lao / 0 | 0 | 0 | 0 | 1 | 1 |
| Lao / 90 | 270 | 0 | 0 | 1 | 4 |
| Lao / 180 | 180 | 180 | 180 | 4 | 4 |
| Lao / 270 | 90 | 90 | 90 | 4 | 4 |
| Mixed / 0 | 0 | 0 | 0 | 1 | 1 |
| Mixed / 90 | 270 | 0 | 0 | 1 | 4 |
| Mixed / 180 | 180 | 180 | 180 | 4 | 4 |
| Mixed / 270 | 90 | 90 | 90 | 4 | 4 |

Both implementations select the constructed correction in **6/8 cases**.
**No orientation-accuracy improvement was demonstrated by this control.** The
two previously skipped sideways baselines are now probed, but remain unresolved:

- Lao / 90 has baseline confidence 0.559825 and 466 recognized characters,
  versus 0.560746 and 465 in the upright image. It formerly took the Lao-dominant
  shortcut; neither relative score nor confidence improves enough to rotate.
- Mixed / 90 has baseline confidence 0.816828 and 500 recognized characters,
  effectively the same as the upright image. It formerly took the high-confidence
  shortcut; the unchanged acceptance requirements again prevent rotation.

The logical call total increases from 20 to 26 on these controls. All existing
OCR inputs were hash-identical, and all selected images and recognized-line
objects on unchanged probe paths were equal. Supported high-confidence
corrections are demonstrated by deterministic fake-engine regressions, not by
an invented successful Tesseract case. No settings were tuned after viewing
these results and no additional rule was introduced to make the table favorable.

## Tests, artifacts, and limitations

Regression tests cover both overridden shortcuts, unchanged horizontal and
ambiguous fast paths, unchanged score/confidence/character gates, default-off
behavior, selected dimensions through `process_document`, cancellation,
all-failed probes and no-text diagnostics. The earlier test that explicitly
preserved skipping of a vertical high-confidence baseline is replaced by
horizontal compatibility coverage and the new separate skip-policy tests.

Local evidence remains Git-ignored:

```text
training/runs/orientation-fastpath-ecc22dd/
  check.py
  historical_pipeline.py
  experiment-plan.json
  completed.summary.json
```

Completion SHA-256:
`bff318afd516fcf34389c3c073ba681dd59f35b99c8554830877388a67aab0e7`.
The plan pins the runner, font, source corpus, model/data files, old/new pipeline,
preprocessing, geometry helper and protected earlier summaries. Completion
contains only numeric observations, hashes and fixed policy/case identifiers,
not OCR strings, labels, coordinates or image paths. The private plan retains
local paths for preservation checks; fingerprints are not anonymization.

No remote document was fetched for this slice, and the institutional remote
suite was not rerun. No model, language model, training dataset, optical capture
kit, collector session or production default was replaced. The frozen challenge
was not used for inference. Phetsarath remains canonical and auto-orientation
remains opt-in.

Issue #17 remains open. A separate future selection design must address
high-confidence equal-score sideways versus horizontal candidates without
silently weakening accuracy/retention safeguards. MAF transcription review,
missing-source coverage and the genuine capture pilot are still outstanding.

## User-visible unresolved-output follow-up

The [selected-output review signal](orientation-review.md) now flags retained
sideways geometry in conversion JSON, successful job status and the bilingual
web UI. It does not relax any rule above. The repeated eight-case control still
corrects 6/8 orientations; both unresolved cases now request review. Missing
geometry remains unassessed rather than being called upright.
