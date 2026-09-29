# Pinned Tesseract real-scan deskew comparison

## Decision

Tesseract Lao/English execution is restored locally. The planned paired comparison
completed **four of six sources, seven of nine pinned pages**. Two Census URLs
failed HTTPS negotiation from this host. This is a partial remote diagnostic,
not a complete suite pass, a ground-truth accuracy benchmark, or deployment
approval.

The PTC page shows a serious character-retention warning: the corrected deskew
produces 280 recognized non-space characters versus 1,321 with the old deskew.
Its confidence rises despite that loss. A separately recorded control disabling
only deskew produces 1,390 characters. Do not declare the cleanup validated on
real documents from confidence alone. Investigate deskew proposal reliability
and text retention before claiming a general OCR quality improvement.

The mathematical direction fix remains correct in its known-rotation tests.
That does not prove that a bounding rectangle over all foreground pixels is a
reliable estimate of text skew on every document. This experiment does not
identify which page features caused the PTC proposal, establish correct OCR
transcriptions, or justify restoring the wrong-sign rotation. No production
setting, model, or source registry was changed by this comparison.

## Runtime setup — 2026-09-29

Homebrew Tesseract **5.5.3**, with Leptonica **1.87.0**, was installed along with
its required dependencies. Only the necessary Lao model was added to the default
language directory; the existing Homebrew English and OSD files were preserved.
Both the default and pinned configurations report `eng`, `lao`, and `osd`, and
`TesseractEngine(languages="lao+eng").is_available()` succeeds.

The actual comparison uses a separate explicit directory:

```text
benchmarks/private/tessdata-fast-4.1.0/
  lao.traineddata
  eng.traineddata
  osd.traineddata
  LICENSE
  provenance.json
```

All three models come from the official `tesseract-ocr/tessdata_fast` release tag
`4.1.0`, resolved to commit `65727574dfcd264acbb0c3e07860e4e9e9b22185`.
The local provenance file records the exact source URLs, sizes, and SHA-256
values. This pins downloaded bytes; it is not a claim of an independent signed
upstream attestation. The bulk `tesseract-lang` package was not installed.

| Model | Bytes | SHA-256 |
| --- | ---: | --- |
| Lao | 6,386,744 | `20124962e93e68121e02c49a949d1f9df5db87dd62e5e4aa578362ca532444f8` |
| English | 4,113,088 | `7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2` |
| OSD | 10,562,727 | `9cf5d576fcc47564f11265841e5ca839001e7e6f38ff7f7aacf46d15a96b00ff` |

OSD is installed and fingerprinted, but production right-angle selection still
uses the existing OCR-verified exhaustive path, not an OSD early exit.

To run a new corrected-path diagnostic using this local setup, choose a new
output path:

```bash
.venv/bin/lao-ocr evaluate-remote-suite \
  --suite benchmarks/remote-diagnostic-suite.json \
  --registry benchmarks/source-registry.json \
  --engine tesseract --languages lao+eng --psm 3 \
  --tessdata-dir benchmarks/private/tessdata-fast-4.1.0 \
  --no-rotation-probes \
  --output training/runs/new-remote-check/summary.json
```

That command is not itself the historical paired comparison below. Its normal
report/writer behavior is unchanged. Do not overwrite historical evidence.

## Paired design recorded before inference

Source baseline: `6f1d84a8071d7c8d5314b13efb014793e084995f`.
The legacy preprocessor was loaded from
`501dade713b5835c031b1c3042dfbb9b2a65e406`; the corrected preprocessor was read
from the current checkout. Both implementations and the installed executable,
model files, Python dependencies, registry, and suite manifest were fingerprinted.
No historical checkout or production file was modified to run the old code.

The source IDs and pages are exactly those in the canonical
`benchmarks/remote-diagnostic-suite.json`. Each reachable source was downloaded
**once for the primary comparison**, into a temporary directory. All four arms
used copies of the same verified bytes. The rendered input pixels and native
numeric statistics were also checked for pairwise equality.

Two conditions were fixed before inference:

- **Plain:** right-angle auto-orientation off in both arms.
- **Auto:** right-angle auto-orientation on in both arms.

Extra diagnostic rotation probes were disabled throughout. The same Tesseract
`lao+eng`, PSM 3, explicit traineddata directory, renderer, page selection,
processing limits, and layout pipeline were used. Line-stat reporting was enabled
in both arms without adding OCR passes or changing the OCR decision logic.
Source indices alternated the legacy/corrected execution order. This was one run
per condition, not a statistically controlled timing benchmark.

The retained reports allowlist numeric page metrics, source/code hashes,
small-angle corrections, right-angle decisions, and elapsed times. They do not
retain native/OCR strings, rendered pages, source documents, credentials, or
source URLs. Private setup/experiment plans also contain local paths and model
provenance URLs; those are not anonymized public artifacts.

## Coverage and unavailable sources

Completed cases:

| Source | Pinned pages | Download SHA-256 |
| --- | --- | --- |
| PTC CamScanner | 1 | `3e38937b2ec29559eeebd253db52f2044411e172088ecfaa0ad2977526f522db` |
| MAF forestry CamScanner | 1, 15 | `f0f72bc69fcc334aaca4665706c48d734ad0db3260f3300117e33d753de00cec` |
| LaoWIS appendix | 55 | `4dfdd2beb39d39381355312b77ca6ce4fdc7edd9f561aa9358d3abce06b563f7` |
| World Bank/KPMG | 8, 15, 21 | `4737dd16777882c423eb105d9355d266e1a0a32bbf16e3a0034095e9cc2d3294` |

The registration-guidance and ID-card-delivery Census sources did not download.
The Python transport returned a TLS internal-alert error during the network
investigation; both endpoints also failed the explicitly recorded TLS-1.2-only
retry and curl transport check, with curl exit code 35 and no HTTP response.
Certificate and hostname validation were not disabled. No alternate URL,
replacement document, page, or insecure HTTP transport was substituted.

These are failures observed from this host, not proof that the sites are globally
offline. The original failed records, retry plan, failed retry records, and
transport checks are preserved. Missing pages are not counted as successes and
have no invented OCR measurements.

## Plain condition: no right-angle auto-orientation

Confidence below is the character-weighted OCR-line confidence, **not accuracy**.
Character counts are recognized non-space characters, not verified correct text.

| Page | Old confidence | Corrected confidence | Old characters | Corrected characters |
| --- | ---: | ---: | ---: | ---: |
| PTC 1 | 0.601841 | 0.698132 | 1,321 | 280 |
| MAF 1 | 0.569311 | 0.575202 | 1,286 | 1,295 |
| MAF 15 | 0.791585 | 0.764893 | 709 | 704 |
| LaoWIS 55 | 0.636229 | 0.636229 | 867 | 867 |
| KPMG 8 | 0.346610 | 0.430509 | 537 | 747 |
| KPMG 15 | 0.333263 | 0.333263 | 600 | 600 |
| KPMG 21 | 0.303495 | 0.340536 | 1,670 | 1,612 |

LaoWIS 55 and KPMG 15 received no small-angle rotation in either arm and produced
identical cleaned pixels. Other pages had opposite-sign corrections under the
two implementations. Confidence and character changes are mixed; more characters
can include errors, and higher mean confidence can reflect recognition of only
a smaller, easier part of the page.

## Auto-orientation condition

The selected right-angle rotations agree between the old and corrected arms:
PTC, MAF, and LaoWIS remain at 0 degrees; KPMG pages 8, 15, and 21 select 180,
90, and 90 degrees clockwise respectively. These are matched algorithm decisions,
not independently reviewed orientation labels.

| KPMG page | Selected angle, both | Old confidence | Corrected confidence | Old characters | Corrected characters |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 180 | 0.861941 | 0.893000 | 858 | 872 |
| 15 | 90 | 0.917945 | 0.917945 | 653 | 653 |
| 21 | 90 | 0.852529 | 0.801102 | 2,059 | 2,120 |

The PTC character-loss warning remains in this condition. The auto-orientation
path does not repair it. Auto-orientation remains off by default.

## Post-hoc PTC isolation control

After observing the warning, a separate plan fixed one additional control on
PTC page 1: disable only `_deskew`, while retaining the same grayscale,
autocontrast, contrast enhancement, page raster, Tesseract models, PSM 3, and
right-angle-off setting. The source was fetched again transiently; its hash and
rendered input pixels matched the primary comparison exactly. No threshold sweep
or production behavior change was made.

| Cleanup | Small-angle correction | Confidence | Recognized characters | OCR lines | Recognized Lao characters |
| --- | ---: | ---: | ---: | ---: | ---: |
| Old deskew | +5.1166 degrees | 0.601841 | 1,321 | 26 | 1,234 |
| Corrected deskew | -5.1166 degrees | 0.698132 | 280 | 9 | 264 |
| Deskew disabled, other cleanup unchanged | 0 | 0.770005 | 1,390 | 26 | 1,305 |

This localizes the observed output difference to the deskew treatment in this
control. It does not determine correct words, prove that every extra character
is valid, or identify a universal deskew threshold. It motivates a conservative,
content-aware proposal/retention investigation rather than another unverified
default change.

## Runtime observations

Totals use only the seven pages completed in both arms. Engine time measures
`TesseractEngine.recognize`; pipeline time additionally includes cleanup and
reconstruction, but excludes network fetch and the initial PDF rasterization.

| Condition | Old / corrected recognize calls | Old engine seconds | Corrected engine seconds | Old pipeline seconds | Corrected pipeline seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain | 7 / 7 | 8.487 | 8.678 | 9.478 | 9.769 |
| Auto | 19 / 19 | 19.628 | 22.265 | 20.650 | 23.345 |

These single local timings do not establish a statistically meaningful speed
change. They are not comparable with historical runs using different models,
Tesseract builds, preprocessing revisions, or page coverage.

## Preservation and next action

All downloaded source documents and rendered page images were confined to
managed temporary directories and removed before source/control reports were
saved. Rights-unclear documents were not ingested into training or the public
benchmark. Source permissions were not upgraded by being reachable. The public
optical manifest is still absent; this does not satisfy the frozen benchmark or
published Tesseract accuracy baseline.

Local, Git-ignored evidence:

```text
training/runs/real-deskew-6f1d84a/
  experiment-plan.json
  legacy_preprocessing.py
  run-primary.py
  run.py
  source-00.summary.json ... source-05.summary.json
  network-retry-plan.json
  source-00-tls12.summary.json
  source-03-tls12.summary.json
  download-recheck.summary.json
  curl-transport-check.summary.json
  ptc_control.py
  ptc-no-deskew-plan.json
  ptc-no-deskew.summary.json
  complete.py
  completed.summary.json
```

Primary plan SHA-256:
`d511bc05dd344e1d177448f830dbd9eb83ec77b64d4a96ae6e2e2b689108d1ee`.
Completion summary SHA-256:
`fa9fef5a1faf18090400a3b804cb641693b70a59ebb6490ab47f340f568af332`.
PTC control summary SHA-256:
`706751a5681e67e7a5683ff84b1c2ffd896bffd4c4c95b1ecf7096c500c57519`.

The next engineering task is to reproduce and guard against text loss from
unreliable deskew proposals without changing the fixed comparison evidence or
mistaking confidence for accuracy. The six-page physical pilot remains the next
optical intake step; collectors, capture kits, Phetsarath policy, and owned-model
weights were left unchanged. This comparison changes documentation and local
runtime setup only, not production preprocessing behavior.

Follow-up engineering is tracked in [issue #17](https://github.com/dolemi-la/lao-document-ocr/issues/17).

## Conservative deskew evidence follow-up

The [image-evidence guard](deskew-evidence-guard.md) now vetoes a small-angle
proposal if threshold foreground would cross the page boundary or horizontal
row alignment does not improve. A new same-byte Tesseract comparison reproduces
PTC output at 1,390 rather than 280 characters, but has mixed results elsewhere
and incomplete source coverage. No ground-truth accuracy claim is made. Issue
#17 remains open, including MAF page-1 retention and KPMG page-21 orientation
review. Historical measurements in this document are not overwritten.
