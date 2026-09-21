# Contributing

Thanks for helping improve open Lao document tooling.

## Development

1. Fork or clone the repository.
2. Install the Python development environment with `make install`.
3. Install the web dependencies with `pnpm install` inside `apps/web`.
4. Add tests for behavior changes.
5. Run `make test` and `make lint`.

## OCR contributions

For datasets, models, or benchmark samples, include clear provenance and a license that allows redistribution and use for the stated purpose.

Do not submit private, confidential, copyrighted, or personally identifying documents unless you have the required rights and the contribution is appropriate for public redistribution.

## Accuracy claims

Include the exact benchmark subset, metric, and model version.

Do not describe a model as having a specific accuracy percentage without reproducible evidence.

## Real scan / phone-photo contributions

Prefer project capture packs over random third-party documents. Capture packs pair approved source text with deterministic printable pages and exact ground truth.

Before contributing a capture:

- use a page from a reviewed capture pack;
- create the scan/photo yourself or otherwise control its rights;
- choose an explicit release license (CC0-1.0 or CC-BY-4.0 are preferred);
- avoid unrelated private/confidential material in the frame;
- register it with `lao-ocr register-capture --confirm-release`;
- review the resulting manifest diff before opening a PR.

See [docs/capture-benchmark-workflow.md](docs/capture-benchmark-workflow.md).
