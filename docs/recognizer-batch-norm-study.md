# Replicated BatchNorm policy experiment

Completed 2026-10-04. Saved development-only selection: **parent**.

The [diagnostic and prospective plan](recognizer-short-regression-plan.md) compares
ordinary BatchNorm updates with fixed parent running statistics. Affine parameters
remain trainable; dropout remains active. This changes training-time normalization
as well as running-statistic updates. No architecture or data changes were made.

All six runs initialize from the same retained text-coverage weights with fresh
AdamW, learning rate 0.00002, batch size 6, MPS and two Torch CPU threads.
Each finishes two epochs, 1,260 updates and 7,560 image presentations.
Total budget: 7,560 updates and 45,360 presentations across three paired seeds.
Training uses 3,780 images / 1,284 groups; development has 156 images.
The 93-class vocabulary, 64 × 1,024 geometry and all splits are unchanged.

## Development-only decision

Each endpoint must have fewer than 199 overall edits, at most 23 short edits,
and at least 15 short exact lines. Only seed 2026100401 can supply a selected
candidate, and its policy must pass in at least two of three seeds. Among eligible
primary endpoints, lowest overall edits wins; ties prefer parent, control, frozen.
The saved selection preceded every challenge evaluation; all evaluations bind its hash.

| Endpoint | Dev edits / 5,490 | Short edits / 507 | Short exact / 30 | Run guard | Selectable |
| --- | ---: | ---: | ---: | --- | --- |
| parent | 199 | 23 | 15 | True | True |
| control-2026100401 | 193 | 23 | 14 | False | False |
| frozen-2026100401 | 197 | 25 | 12 | False | False |
| control-2026100402 | 197 | 25 | 13 | False | False |
| frozen-2026100402 | 193 | 23 | 15 | True | False |
| control-2026100403 | 194 | 25 | 13 | False | False |
| frozen-2026100403 | 196 | 25 | 12 | False | False |

## Paired effects

Differences below are frozen minus control; lower edits and higher exact counts are better.

| Seed | Overall dev edit difference | Short edit difference | Short exact difference |
| --- | ---: | ---: | ---: |
| 2026100401 | +4 | +2 | -2 |
| 2026100402 | -4 | -2 | +2 |
| 2026100403 | +2 | +0 | -1 |

## Fixed endpoint evaluation

CER is character edits divided by reference characters. Exact counts measure whole lines.

| Endpoint | Dev CER / exact | Short challenge CER / exact (480) | Long challenge CER / exact (144) |
| --- | --- | --- | --- |
| parent | 3.6248% / 57 | 3.3691% / 273 | 4.5289% / 27 |
| control-2026100401 | 3.5155% / 57 | 3.2300% / 278 | 4.4315% / 30 |
| frozen-2026100401 | 3.5883% / 55 | 3.2002% / 278 | 4.4558% / 29 |
| control-2026100402 | 3.5883% / 54 | 3.1008% / 284 | 4.5654% / 25 |
| frozen-2026100402 | 3.5155% / 55 | 3.1405% / 281 | 4.4315% / 27 |
| control-2026100403 | 3.5337% / 56 | 3.2697% / 282 | 4.2975% / 28 |
| frozen-2026100403 | 3.5701% / 54 | 3.1604% / 281 | 4.4558% / 25 |

### All adverse profile comparisons with parent

A subset is listed if CER increases or exact lines decrease; neither metric alone describes all behavior.

- control-2026100401, fixed_dev / clean-scan: CER 2.8490% → 3.0199%; exact 19 → 19.
- control-2026100401, frozen_long_challenge / clean-scan: CER 2.5201% → 2.7757%; exact 13 → 13.
- frozen-2026100401, fixed_dev / unknown: CER 6.2222% → 6.2222%; exact 5 → 4.
- frozen-2026100401, fixed_dev / clean-scan: CER 2.8490% → 3.0769%; exact 19 → 18.
- frozen-2026100401, frozen_short_challenge / clean-scan: CER 3.1306% → 3.0411%; exact 99 → 98.
- frozen-2026100401, frozen_long_challenge / clean-scan: CER 2.5201% → 2.8488%; exact 13 → 12.
- control-2026100402, fixed_dev / unknown: CER 6.2222% → 6.6667%; exact 5 → 4.
- control-2026100402, fixed_dev / clean-scan: CER 2.8490% → 2.9060%; exact 19 → 19.
- control-2026100402, fixed_dev / noisy-scan: CER 2.8490% → 2.9630%; exact 20 → 19.
- control-2026100402, fixed_dev / phone-photo: CER 4.8433% → 4.5014%; exact 13 → 12.
- control-2026100402, frozen_long_challenge / clean-scan: CER 2.5201% → 3.0314%; exact 13 → 10.
- frozen-2026100402, fixed_dev / noisy-scan: CER 2.8490% → 2.9060%; exact 20 → 19.
- frozen-2026100402, fixed_dev / phone-photo: CER 4.8433% → 4.4444%; exact 13 → 12.
- frozen-2026100402, frozen_long_challenge / clean-scan: CER 2.5201% → 2.8488%; exact 13 → 12.
- control-2026100403, fixed_dev / clean-scan: CER 2.8490% → 2.9060%; exact 19 → 19.
- control-2026100403, fixed_dev / noisy-scan: CER 2.8490% → 2.9060%; exact 20 → 20.
- control-2026100403, fixed_dev / phone-photo: CER 4.8433% → 4.4444%; exact 13 → 12.
- control-2026100403, frozen_short_challenge / clean-scan: CER 3.1306% → 3.1008%; exact 99 → 98.
- control-2026100403, frozen_short_challenge / phone-photo: CER 3.7269% → 3.7865%; exact 86 → 88.
- control-2026100403, frozen_long_challenge / clean-scan: CER 2.5201% → 2.9218%; exact 13 → 10.
- frozen-2026100403, fixed_dev / unknown: CER 6.2222% → 5.7778%; exact 5 → 4.
- frozen-2026100403, fixed_dev / clean-scan: CER 2.8490% → 3.0199%; exact 19 → 18.
- frozen-2026100403, fixed_dev / phone-photo: CER 4.8433% → 4.6154%; exact 13 → 12.
- frozen-2026100403, frozen_short_challenge / clean-scan: CER 3.1306% → 3.0411%; exact 99 → 98.
- frozen-2026100403, frozen_long_challenge / clean-scan: CER 2.5201% → 2.8123%; exact 13 → 10.
- frozen-2026100403, frozen_long_challenge / phone-photo: CER 7.8159% → 7.5968%; exact 5 → 4.

## Verification and retained artifacts

All twelve stages have successful receipts and strict own-run epoch-2 resumes.
Independent checks verify actual optimizer counters, seeds, learning rates,
paired shuffle states, source provenance, fixed endpoint weights and selection arithmetic.
Every frozen-policy endpoint preserves the parent BatchNorm buffers bit-for-bit.
Input verification covers 287 protected files and 4,560 image hashes.
Both primary exports pass 24 sampled native/export and CPU/MPS prediction comparisons
with zero mismatches. Secondary endpoints remain resumable states, not selectable exports.

Local software gates: 1,786 Python tests passed, two optional S3 cases skipped;
145 web tests, repository Ruff, web lint and web build passed. Dependency deprecation
warnings remain. These gates do not measure OCR accuracy.

Carry-forward export:

`training/runs/text-coverage-ed769db/expanded/final/recognizer.pt2`

All six resume states remain under `training/runs/batch-norm-d09f644/<arm>/model/`.
Primary exports and metadata are under `<arm>/final/`. Historical artifacts remain intact.
No production model setting was changed. No private text, predictions, images, fonts or weights are published.

Frozen plan SHA256: `6dad1587491ebeffc72778d63d19aaf99d9fd857b022b5c3bcb3fb22cbaf39ed`.
Private execution and review evidence: `benchmarks/private/model-batch-norm-d09f644/`.

| Primary export | SHA256 |
| --- | --- |
| control-2026100401 | `25462124677e0fa4322e35720f4447d8f8d29937b21d9cfc4c5e326914fa829f` |
| frozen-2026100401 | `4dc7faa8382924a4d6cf33df1e5f7e0aa92b45d17279b33bcc50f39622cd2d27` |

## Limits

Three seeds share the same small, previously inspected synthetic development set.
The short guard covers only 30 images and 18 normalized labels. Seed replication
does not establish independent document coverage or real optical accuracy.
Challenges are regression sets and cannot revise the saved selection.
Rights-reviewed real scan/photo validation and a fixed-set Tesseract comparison remain outstanding.
