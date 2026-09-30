# Explicit per-page rotation for CLI and Python conversion

## Purpose and scope

The [orientation review warnings](orientation-review.md) flag output that may
remain sideways. Equal-confidence automatic decisions can still retain that
output rather than make an unsupported guess. An operator who has inspected the
input now has a separate, explicit correction path. No automatic confidence,
score, geometry, or character-retention threshold is weakened.

This slice exposes corrections in `convert-document`, `process_document(...)`,
and `convert_document_to_outputs(...)`. The HTTP request schema and web UI do
**not** yet accept manual corrections. Their automatic option and review warning
behavior remain unchanged. Do not send a manual-rotation form field and assume
that an existing endpoint will apply it.

## CLI

Use the original document as input and a new output directory:

```bash
lao-ocr convert-document \
  --input ./scanned-document.pdf \
  --output-dir ./reviewed-conversion \
  --rotate-page 2:90 \
  --rotate-page 5:270
```

`PAGE:DEGREES` uses one-based document page numbers and clockwise corrections.
Accepted angles are exactly `0`, `90`, `180`, and `270`. For an image input the
only page is 1. The existing image loader still processes its single supported
page; this option does not add multi-frame TIFF support.

A page shown rotated 90 degrees clockwise from upright needs a **270-degree
clockwise correction**, not another 90 degrees. The values are relative
corrections to the displayed input, not target absolute page orientations.

Manual and automatic correction can coexist:

```bash
lao-ocr convert-document \
  --input ./scanned-document.pdf \
  --output-dir ./reviewed-conversion \
  --auto-orient-right-angles \
  --rotate-page 1:0 \
  --rotate-page 2:270
```

Here page 1 is explicitly kept at its displayed cardinal orientation, page 2
is manually rotated 270 degrees clockwise, and other pages retain automatic
processing. **Explicit zero is an override.** Automatic orientation cannot
replace any listed page's choice, including zero. It invokes recognition once
on each overridden page and does not run cardinal probes for that page.

Zero does not disable the existing contrast cleanup or guarded small-angle
deskew. Those operations still run after the explicit cardinal correction.

## Python

```python
from lao_document_ocr.conversion import convert_document_to_outputs
from lao_document_ocr.ocr.tesseract import TesseractEngine

outputs = convert_document_to_outputs(
    "scanned-document.pdf",
    "reviewed-conversion",
    engine=TesseractEngine(languages="lao+eng"),
    auto_orient_right_angles=True,
    page_rotations={1: 0, 2: 270},
)
```

The lower-level `process_document(...)` accepts the same `page_rotations`
mapping. `None` or an empty mapping leaves the manual path disabled. The mapping
is copied before loading/recognition so later caller mutations cannot change
page choices during conversion.

Keys and values must be actual integers: booleans, floats, strings, zero or
negative page numbers, and unsupported angles fail rather than being coerced.
The CLI additionally rejects malformed specifications and repeated page numbers,
even if the repeated values agree. It does not silently wrap angles or apply
the last duplicate.

Syntax/type validation precedes document loading in the pipeline and engine
construction in the CLI. Actual page bounds are checked once the document is
loaded, before the first OCR call. Conversion creates its output directory only
after processing succeeds. Invalid page selections leave existing exports
untouched and do not create a new output directory. This is not a new atomic
transaction for all exporters: an unrelated export-time failure retains the
existing exporter behavior. Use fresh output paths to preserve prior results.

## Order and coordinate contract

1. Load the displayed page. PDF rendering honors its existing page rotation.
   Image EXIF orientation is normalized once for both OCR and preserved source
   pixels, and the consumed orientation tag is cleared when transformed.
2. Apply the operator's cardinal correction before preprocessing. Rotate the
   page image and preserved embedded-image payloads and bounding boxes together.
   Quarter turns swap page dimensions and recalculate embedded width ratios.
3. Run the existing grayscale/contrast cleanup and guarded small-angle deskew.
4. Recognize overridden pages directly. Unspecified pages follow their normal
   automatic-or-default-off path. Subsequent layout reconstruction and image
   extraction receive the corrected source-image frame.

The EXIF change also fixes an existing mismatch without manual options: cleanup
previously transposed tagged camera images while preserved source pixels could
remain in the raw frame. Tests cover all eight EXIF orientation values on JPEG
fixtures. Untagged/default-orientation behavior remains unchanged.

This guarantees the new cardinal transform is propagated consistently. It does
not redesign the existing arbitrary-angle deskew coordinate handling, establish
that a chosen angle is correct, or guarantee that every faint glyph survives
preprocessing. The input files are read only and are not rewritten. The
existing page/pixel limits, cancellation checks, and OCR-error propagation remain.

## Provenance and review metadata

Only a nonempty manual mapping adds this document metadata:

```json
{
  "manual_page_rotations": {
    "version": "explicit-page-rotation-v1",
    "basis": "loaded-page-before-cleanup",
    "pages": [
      {"page": 1, "degrees_clockwise": 0},
      {"page": 2, "degrees_clockwise": 270}
    ],
    "review": {
      "version": "selected-line-axis-review-v1",
      "status": "incomplete",
      "page_count": 3,
      "review_pages": [],
      "unassessed_pages": [3]
    }
  }
}
```

The example assumes page 3 had no automatic assessment. The review uses existing
selected-line geometry from overridden pages and any other automatically
assessed pages. Unspecified default-off pages remain unassessed. A bad manual
choice can still produce `review-required`; it is not exempt from the sideways
heuristic. `no-sideways-evidence` does not certify upright text, transcription
quality, or ground truth, and cannot detect all upside-down horizontal output.

The separate `auto_orientation.enabled` flag continues to mean the automatic
option was requested. Its page `degrees_clockwise` value describes **additional
automatic correction**, and remains zero on manually overridden pages. The
actual manual angle appears only in `manual_page_rotations.pages` rather than
being misrepresented as an automatic decision. Diagnostics on those pages use
`probe_strategy: manual-override`, `probe_skip_reason: manual-override`, and an
empty `probed_degrees` list. Their selected candidate geometry uses angle zero
relative to the already manually corrected image.

When auto-orientation is off its existing review remains `not-requested`; read
`manual_page_rotations.review` for the explicit-correction result instead. The
new metadata contains numeric angles/page numbers and fixed policy/review
labels, not OCR strings or private source paths. The document/export JSON as a
whole still intentionally contains its recognized content. No remote diagnostic
report is changed by this feature.

## Validation

Regression coverage includes strict parsing/types and duplicate rejection,
actual page bounds before OCR/output writes, explicit-zero precedence, other
pages' automatic path, caller-mapping mutation, source/input preservation,
selected-output warnings, cancellation and errors, all cardinal transformations,
EXIF normalization, PDF display rotation/page numbering, actual embedded-image
block payloads/boxes, and CLI generation of all four export types.

A foreground integration check used the existing project-authored Lao corpus
and six new English fixture lines rendered with local Phetsarath OT Regular.
It constructed Lao and mixed pages, each at 0/90/180/270 degrees. The plan was
saved before conversion and pinned the runner, font, corpus, relevant code,
traineddata and protected earlier experiment/model summaries.

Pinned Tesseract 5.5.3 with Lao/English `tessdata_fast` 4.1.0 performed two upright
reference conversions and eight explicit-correction conversions through the
actual CLI dispatcher. Every explicit case also enabled the automatic option.
All eight outputs matched their upright references' complete page AST and TXT
exports, produced DOCX/Markdown/TXT/JSON, preserved source hashes, recorded the
specified angle, and applied no additional automatic rotation.

These are **known, operator-supplied corrections on eight correlated variants
of two synthetic pages**. The result verifies CLI/Python recovery and coordinate
handling, not automatic orientation accuracy, correct transcription, independent
OCR generalization, or real optical capture quality. No remote document or
frozen evaluation image was used for inference, and no training occurred.
Temporary fixture images and exports were removed after verification.

Private evidence remains Git-ignored under
`training/runs/manual-rotation-bd3fdb1/`: `check.py`, `experiment-plan.json`, and
`completed.summary.json`. Completion SHA-256:
`b8401068dc260d9219c6aa76821186e6226f6f5f73e141a7df8a8461f627db84`.
The aggregate summary contains checks/counts and hashes rather than recognized
text. The plan retains local paths for integrity checks; hashes do not anonymize
input content.

Issue #17 remains open for automatic equal-confidence selection, MAF review,
missing-source coverage, and genuine optical validation. HTTP and web manual
controls are a separate follow-up, not part of this completed CLI/Python slice.
Phetsarath remains canonical. No model weights, corpus, capture kit, collector
session, or default auto-orientation setting was changed.
