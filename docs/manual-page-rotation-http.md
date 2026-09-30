# Manual page corrections in HTTP conversion and the web app

## Scope

The existing [CLI/Python rotation contract](manual-page-rotation.md) is now
available through HTTP and the bilingual web app. This is an operator-directed
recovery path for unresolved orientation, not a change to automatic selection,
a new orientation classifier, or evidence of improved OCR accuracy.

The starting checkout was `527923026c2623b0854d6f45b054c3d53566cdb1`.
No model weights, font policy, capture kit, collector session, remote source,
training dataset, or automatic acceptance threshold is changed by this slice.

## HTTP request

These multipart endpoints accept the same optional, repeated string field:
`POST /v1/parse`, `POST /v1/convert`, `POST /v1/jobs`, and
`POST /v1/jobs/batch`. The field name is **`rotate_page`**, not
`page_rotations` and not the CLI flag `--rotate-page`.

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/jobs \
  -F 'file=@original.pdf' \
  -F 'rotate_page=1:0' \
  -F 'rotate_page=2:270' \
  -F 'auto_orient_right_angles=true'
```

Each value must be exactly `PAGE:DEGREES`: a one-based ASCII page number and
clockwise degrees in `0`, `90`, `180`, or `270`. The server does not normalize
negative/full-turn angles, accept leading-zero page numbers, trim specifications,
parse comma-separated fields, coerce booleans, or use the last duplicate. Even
identical duplicate pages fail with HTTP 422. Omit the field for no overrides;
a supplied empty string is invalid.

Explicit zero is retained: it prevents cardinal auto-orientation on that page.
Other pages follow the existing auto-or-default-off setting. Corrections are
relative to the displayed original upload after PDF rotation/EXIF normalization,
and applied before existing contrast cleanup and guarded small-angle deskew.
They do not disable those cleanup steps or rewrite the uploaded file.

The core pipeline, geometry transforms, manual provenance and selected-output
review rules are reused unchanged. Each overridden page invokes recognition
once; it does not run automatic cardinal probes.

### Bounds and failure behavior

Syntax, duplicate pages, the configured `MAX_PAGES` bound, and at most
`MAX_PAGES` specifications are checked before application job reservation or copying the upload to a working
directory. Framework multipart parsing still precedes the endpoint.
Individual specifications are capped at 32 characters before converting their
integer text. Existing upload byte, content, page and pixel checks remain.

Upload validation now returns the page count supported by the loader. This is
the PDF page count or **one** for image uploads, including the currently
single-page TIFF path. Actual bounds are then checked before OCR, enqueueing,
or export writes. No second PDF rendering pass is needed for this preflight.
An invalid selection produces HTTP 422, not a successful ignored override or a
generic asynchronous conversion failure.

### Batch semantics

**The same rotation map applies independently to every file in a batch.**
Page numbers restart at 1 for each input. Different maps require separate
single-job submissions; there is no filename-keyed map or cross-document page
numbering in this contract.

All uploaded members are content/bounds-validated before the first enqueue.
If, for example, `2:90` is valid for a two-page PDF but invalid for another
member's image upload, the entire submission is rejected and its reservations
and workspaces are discarded. No member starts OCR. Existing unrelated jobs
are not discarded. This does not promise transactional rollback of unrelated
runtime failures after a valid batch has been enqueued.

## Job acknowledgement and result review

Each job snapshots its mapping into an independently copied, read-only mapping.
Mutating the caller's original dictionary, a sibling job's mapping, or a public
response cannot alter these stored choices. Public job responses acknowledge
only the sorted page/angle pairs, in every lifecycle state:

```json
{
  "page_rotations": [
    {"page": 1, "degrees_clockwise": 0},
    {"page": 2, "degrees_clockwise": 270}
  ]
}
```

An omitted mapping is acknowledged as `[]`. This is request provenance, not an
assessment of whether the selected angles are correct. It adds no document
strings, paths, OCR predictions, or arbitrary engine metadata.

For manual jobs, the worker recomputes `orientation_review` from the pipeline's
selected page geometry, including manual pages even when auto-orientation is
false. It does not forward an embedded `manual_page_rotations.review` object or
trust an arbitrary supplied summary. The result matches the JSON archive's
`metadata.manual_page_rotations.review`. Unspecified default-off pages stay
unassessed; an incorrect manual choice can still trigger review.

Without manual overrides, the existing automatic-review behavior is preserved.
Uploading, queued, running, failed and cancelled jobs still expose no completed
review. The pipeline's separate `metadata.auto_orientation.review` retains its
automatic-request semantics, including `not-requested` when auto-orientation is
disabled. Absence of a sideways flag never certifies upright or accurate text.

## Web workflow

After choosing the original file, enter comma-separated corrections in
**Manual page corrections (optional)**, for example `1:0, 2:270`, then convert.
The bilingual instructions explain clockwise direction, one-based pages,
explicit zero, and precedence over auto-orientation. The browser trims each
comma-separated token and submits separate canonical `rotate_page` fields.
It checks syntax, duplicate pages, safe integer values, and image-page bounds.
PDF bounds and deployment-specific page limits remain server-validated.

The field has a visible label, associated help/error text, invalid-state
semantics, keyboard focus styling and localized errors. It is disabled before
file selection and while processing. Invalid input blocks submission. Selecting
a new file clears both the corrections and previous result warnings. Editing
corrections for the same file does not alter its previous export; converting
again uploads the original file bytes with the new choices.

Existing page-specific review warnings remain visible after successful manual
jobs. Downloads are not blocked by a heuristic review warning. No preview,
per-page thumbnail editor, export-file mutation, or guess at a manual angle is
introduced here.

The UI checks exact acknowledgement at job creation **and** before downloading
success. A stale API may otherwise ignore unknown multipart fields silently.
Missing/mismatched acknowledgement for a nonempty manual request prevents its
automatic download and shows a localized update/retry error. On creation-time
mismatch the UI attempts to cancel only that submitted job; it does not claim
cancellation succeeded. Ordinary requests with no overrides remain compatible
with legacy responses lacking the new acknowledgement.

## Verification and limits

New Python tests cover all four actual endpoints, every cardinal angle with
auto-orientation on and off, malformed/duplicate/oversized/out-of-range fields,
pre-enqueue batch rejection, per-file page counts, immutable job snapshots,
OpenAPI, full output-archive contents, and status/archive review agreement.
They run real workers, pipeline and exporters with **fixture OCR engines**;
these are software-contract tests, not new OCR model evaluations.

Dependency-free web tests cover parser boundaries, exact repeated FormData,
explicit zero, image limits, both locales, stale/mismatched acknowledgements,
and App integration. A separate local headless Chrome interaction check used
mocked HTTP responses at 1280x900 and 390x844: it verified validation, Lao errors,
processing locks, original upload bytes, persistent warnings, new-file reset,
mobile horizontal overflow, and the stale-server no-download/cancel-attempt
path. It is not a screenshot/visual accessibility audit, an optical benchmark,
or a browser test against a running Tesseract server.

The isolated browser profile and mock/Vite services are stopped and their
temporary files removed. Its runner and aggregate-only result remain ignored:
`benchmarks/private/manual-rotation-http-5279230/`. Source-registry, suite,
project-corpus, frozen model/challenge-summary and collector ZIP hashes are
checked unchanged around that smoke test. New Lao catalog characters pass
coverage checks against the existing private Phetsarath font; no font is bundled.

Issue #17 remains open: equal-confidence automatic selection, MAF transcription
review, unavailable Census sources, and the genuine capture pilot are not
resolved by a manual operator option. Automatic orientation remains opt-in.

## Same-job network recovery

[Web job recovery](web-job-recovery.md) adds explicit retries for interrupted
status or archive requests. These reuse the acknowledged job and frozen manual
correction map, not edited form values, and refresh identity/acknowledgement
before downloading. Applying new angles still requires a new conversion.
