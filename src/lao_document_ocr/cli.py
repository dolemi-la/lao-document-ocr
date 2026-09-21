from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lao_document_ocr.benchmarking import benchmark_dataset, write_report
from lao_document_ocr.corpus import (
    CorpusFilter,
    iter_jsonl,
    iter_plain_text,
    prepare_corpus,
    write_corpus,
)
from lao_document_ocr.dataset import (
    DatasetManifestError,
    DatasetSplit,
    load_manifest,
    validate_dataset,
)
from lao_document_ocr.ocr import OcrEngineError, TesseractEngine
from lao_document_ocr.synthetic import generate_synthetic_lines, load_corpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lao-ocr",
        description="Lao Document OCR developer and benchmark utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-dataset", help="Validate a benchmark dataset.")
    validate.add_argument("--manifest", required=True, type=Path)
    validate.add_argument("--dataset-root", required=True, type=Path)
    validate.add_argument("--no-hash-check", action="store_true")

    benchmark = subparsers.add_parser("benchmark", help="Run OCR against a dataset split.")
    benchmark.add_argument("--manifest", required=True, type=Path)
    benchmark.add_argument("--dataset-root", required=True, type=Path)
    benchmark.add_argument("--output", required=True, type=Path)
    benchmark.add_argument(
        "--split",
        choices=[split.value for split in DatasetSplit],
        default=DatasetSplit.TEST.value,
    )
    benchmark.add_argument("--languages", default="lao+eng")
    benchmark.add_argument("--psm", type=int, default=3)
    benchmark.add_argument("--no-hash-check", action="store_true")

    corpus = subparsers.add_parser(
        "prepare-corpus",
        help="Normalize/filter Lao text for synthetic OCR training.",
    )
    corpus.add_argument("--input", required=True, type=Path)
    corpus.add_argument("--output", required=True, type=Path)
    corpus.add_argument("--format", choices=["text", "jsonl"], default="text")
    corpus.add_argument("--field", default="text")
    corpus.add_argument("--min-chars", type=int, default=8)
    corpus.add_argument("--max-chars", type=int, default=180)
    corpus.add_argument("--min-lao-ratio", type=float, default=0.5)
    corpus.add_argument("--limit", type=int)
    corpus.add_argument("--keep-duplicates", action="store_true")

    synthetic = subparsers.add_parser(
        "generate-synthetic",
        help="Render labeled text-line images from a prepared corpus.",
    )
    synthetic.add_argument("--corpus", required=True, type=Path)
    synthetic.add_argument("--output", required=True, type=Path)
    synthetic.add_argument("--font", required=True, type=Path, action="append")
    synthetic.add_argument("--variants-per-line", type=int, default=1)
    synthetic.add_argument("--seed", type=int, default=20260921)
    synthetic.add_argument("--min-font-size", type=int, default=40)
    synthetic.add_argument("--max-font-size", type=int, default=56)
    synthetic.add_argument("--max-samples", type=int)

    return parser


def _validate(args: argparse.Namespace) -> int:
    samples = load_manifest(args.manifest)
    errors = validate_dataset(
        samples,
        args.dataset_root,
        verify_hashes=not args.no_hash_check,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    counts: dict[str, int] = {}
    for sample in samples:
        key = f"{sample.split.value}/{sample.subset.value}"
        counts[key] = counts.get(key, 0) + 1

    print(f"Valid dataset: {len(samples)} samples")
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    samples = load_manifest(args.manifest)
    engine = TesseractEngine(languages=args.languages, psm=args.psm)
    if not engine.is_available():
        available = ", ".join(engine.available_languages())
        print(
            f"Required OCR languages '{args.languages}' are unavailable. "
            f"Installed languages: {available or 'none'}",
            file=sys.stderr,
        )
        return 2

    report = benchmark_dataset(
        samples,
        args.dataset_root,
        engine,
        split=DatasetSplit(args.split),
        verify_hashes=not args.no_hash_check,
    )
    output = write_report(report, args.output)
    overall = report["overall"]
    print(f"Report: {output}")
    print(f"Samples: {overall['samples']}")
    print(f"CER: {overall['cer']:.4f}")
    print(f"WER: {overall['wer']:.4f}")
    return 0


def _prepare_corpus(args: argparse.Namespace) -> int:
    if args.format == "jsonl":
        source = iter_jsonl(args.input, field=args.field)
    else:
        source = iter_plain_text(args.input)

    config = CorpusFilter(
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        min_lao_ratio=args.min_lao_ratio,
        deduplicate=not args.keep_duplicates,
    )
    lines = prepare_corpus(source, config, limit=args.limit)
    output = write_corpus(lines, args.output)
    print(f"Corpus: {output}")
    print(f"Lines: {len(lines)}")
    return 0


def _generate_synthetic(args: argparse.Namespace) -> int:
    lines = load_corpus(args.corpus)
    manifest = generate_synthetic_lines(
        lines,
        args.output,
        args.font,
        variants_per_line=args.variants_per_line,
        seed=args.seed,
        min_font_size=args.min_font_size,
        max_font_size=args.max_font_size,
        max_samples=args.max_samples,
    )
    count = sum(1 for line in manifest.read_text(encoding="utf-8").splitlines() if line)
    print(f"Manifest: {manifest}")
    print(f"Samples: {count}")
    return 0


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        if args.command == "validate-dataset":
            return _validate(args)
        if args.command == "benchmark":
            return _benchmark(args)
        if args.command == "prepare-corpus":
            return _prepare_corpus(args)
        if args.command == "generate-synthetic":
            return _generate_synthetic(args)
    except (DatasetManifestError, OcrEngineError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
