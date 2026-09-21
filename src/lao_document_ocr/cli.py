from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lao_document_ocr.benchmarking import benchmark_dataset, write_report
from lao_document_ocr.conversion import convert_document_to_outputs
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
    DatasetSubset,
    load_manifest,
    validate_dataset,
)
from lao_document_ocr.dataset_intake import add_dataset_sample
from lao_document_ocr.ocr import OcrEngineError, TesseractEngine
from lao_document_ocr.synthetic import generate_synthetic_lines, load_corpus
from lao_document_ocr.training_manifest import load_training_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lao-ocr",
        description="Lao Document OCR developer and benchmark utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser(
        "convert-document",
        help="Convert one image/PDF into DOCX, Markdown, TXT, and JSON.",
    )
    convert.add_argument("--input", required=True, type=Path)
    convert.add_argument("--output-dir", required=True, type=Path)
    convert.add_argument("--engine", choices=["tesseract", "owned"], default="tesseract")
    convert.add_argument("--languages", default="lao+eng")
    convert.add_argument("--psm", type=int, default=3)
    convert.add_argument("--model", type=Path)
    convert.add_argument("--calibration", type=Path)
    convert.add_argument("--max-pages", type=int, default=60)
    convert.add_argument("--font", default="Noto Sans Lao")

    validate = subparsers.add_parser("validate-dataset", help="Validate a benchmark dataset.")
    validate.add_argument("--manifest", required=True, type=Path)
    validate.add_argument("--dataset-root", required=True, type=Path)
    validate.add_argument("--no-hash-check", action="store_true")

    intake = subparsers.add_parser(
        "add-dataset-sample",
        help="Copy a rights-cleared benchmark page into the public dataset.",
    )
    intake.add_argument("--dataset-root", required=True, type=Path)
    intake.add_argument("--manifest", required=True, type=Path)
    intake.add_argument("--id", required=True, dest="sample_id")
    intake.add_argument("--document-id", required=True)
    intake.add_argument(
        "--subset",
        required=True,
        choices=[subset.value for subset in DatasetSubset],
    )
    intake.add_argument("--image", required=True, type=Path)
    intake.add_argument("--ground-truth", required=True, type=Path)
    intake.add_argument("--license", required=True)
    intake.add_argument("--provenance", required=True)
    intake.add_argument("--language", default="lo")
    intake.add_argument("--source-url")
    intake.add_argument("--license-url")
    intake.add_argument("--notes")
    intake.add_argument(
        "--split",
        choices=[split.value for split in DatasetSplit],
    )
    intake.add_argument(
        "--confirm-redistributable",
        action="store_true",
        help="Confirm redistribution and OCR/model-evaluation rights for this sample.",
    )

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

    train = subparsers.add_parser(
        "train-recognizer",
        help="Train the CRNN+CTC Lao line recognizer.",
    )
    train.add_argument("--manifest", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--dev-ratio", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=20260921)
    train.add_argument("--image-height", type=int, default=48)
    train.add_argument("--max-width", type=int, default=512)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--no-hash-check", action="store_true")

    export = subparsers.add_parser(
        "export-recognizer",
        help="Export a trained recognizer checkpoint as a torch.export artifact.",
    )
    export.add_argument("--checkpoint", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)

    recognize = subparsers.add_parser(
        "recognize-line",
        help="Run an exported recognizer against one cropped text-line image.",
    )
    recognize.add_argument("--model", required=True, type=Path)
    recognize.add_argument("--image", required=True, type=Path)

    recognizer_benchmark = subparsers.add_parser(
        "benchmark-recognizer",
        help="Measure an exported line recognizer on a labeled manifest.",
    )
    recognizer_benchmark.add_argument("--manifest", required=True, type=Path)
    recognizer_benchmark.add_argument("--model", required=True, type=Path)
    recognizer_benchmark.add_argument("--output", required=True, type=Path)
    recognizer_benchmark.add_argument("--no-hash-check", action="store_true")
    recognizer_benchmark.add_argument("--calibration", type=Path)

    calibrate = subparsers.add_parser(
        "calibrate-recognizer",
        help="Fit confidence calibration from a held-out recognizer benchmark report.",
    )
    calibrate.add_argument("--report", required=True, type=Path)
    calibrate.add_argument("--output", required=True, type=Path)
    calibrate.add_argument("--bins", type=int, default=10)

    recognize.add_argument("--calibration", type=Path)

    return parser


def _convert_document(args: argparse.Namespace) -> int:
    if args.engine == "tesseract":
        engine = TesseractEngine(languages=args.languages, psm=args.psm)
    else:
        if args.model is None:
            raise ValueError("--model is required when --engine=owned")
        from lao_document_ocr.ocr import OwnedRecognizerEngine

        engine = OwnedRecognizerEngine(
            args.model,
            calibration_path=args.calibration,
        )

    outputs = convert_document_to_outputs(
        args.input,
        args.output_dir,
        engine=engine,
        max_pages=args.max_pages,
        font_name=args.font,
    )
    print(json.dumps(outputs.to_dict(), indent=2))
    return 0


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


def _add_dataset_sample(args: argparse.Namespace) -> int:
    sample = add_dataset_sample(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        sample_id=args.sample_id,
        document_id=args.document_id,
        subset=DatasetSubset(args.subset),
        image_path=args.image,
        ground_truth_path=args.ground_truth,
        license=args.license,
        provenance=args.provenance,
        language=args.language,
        source_url=args.source_url,
        license_url=args.license_url,
        notes=args.notes,
        split=DatasetSplit(args.split) if args.split else None,
        rights_confirmed=args.confirm_redistributable,
    )
    print(json.dumps(sample.model_dump(mode="json", exclude_none=True), ensure_ascii=False))
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


def _train_recognizer(args: argparse.Namespace) -> int:
    try:
        from lao_document_ocr.recognizer_training import TrainingConfig, train_recognizer
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    samples = load_training_manifest(
        args.manifest,
        verify_hashes=not args.no_hash_check,
    )
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        image_height=args.image_height,
        max_width=args.max_width,
        num_workers=args.num_workers,
    )
    result = train_recognizer(samples, args.output, training_config=config)
    print(f"Checkpoint: {result['checkpoint']}")
    print(f"Metadata: {result['metadata']}")
    print(f"Best dev CER: {result['best_dev_cer']:.4f}")
    return 0


def _export_recognizer(args: argparse.Namespace) -> int:
    try:
        from lao_document_ocr.recognizer_training import export_recognizer
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    artifact = export_recognizer(args.checkpoint, args.output)
    print(f"Exported recognizer: {artifact}")
    return 0


def _recognize_line(args: argparse.Namespace) -> int:
    try:
        from lao_document_ocr.recognizer_inference import ExportedLineRecognizer
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    recognizer = ExportedLineRecognizer(args.model, calibration_path=args.calibration)
    result = recognizer.recognize(args.image)
    print(result.text)
    print(f"uncalibrated_confidence={result.confidence:.4f}", file=sys.stderr)
    if result.calibrated_confidence is not None:
        print(f"calibrated_confidence={result.calibrated_confidence:.4f}", file=sys.stderr)
    return 0


def _benchmark_recognizer(args: argparse.Namespace) -> int:
    try:
        from lao_document_ocr.recognizer_benchmark import (
            benchmark_recognizer,
            write_recognizer_report,
        )
        from lao_document_ocr.recognizer_inference import ExportedLineRecognizer
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    samples = load_training_manifest(
        args.manifest,
        verify_hashes=not args.no_hash_check,
    )
    recognizer = ExportedLineRecognizer(
        args.model,
        calibration_path=args.calibration,
    )
    report = benchmark_recognizer(samples, recognizer)
    output = write_recognizer_report(report, args.output)
    overall = report["overall"]
    print(f"Report: {output}")
    print(f"Samples: {overall['samples']}")
    print(f"CER: {overall['cer']:.4f}")
    print(f"WER: {overall['wer']:.4f}")
    return 0


def _calibrate_recognizer(args: argparse.Namespace) -> int:
    from lao_document_ocr.confidence_calibration import fit_from_recognizer_report

    report = json.loads(args.report.read_text(encoding="utf-8"))
    calibration = fit_from_recognizer_report(report, max_bins=args.bins)
    output = calibration.save(args.output)
    print(f"Calibration: {output}")
    print(f"Samples: {calibration.sample_count}")
    print(f"Raw MAE: {calibration.raw_mae:.4f}")
    print(f"Calibrated MAE: {calibration.calibrated_mae:.4f}")
    return 0


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        if args.command == "convert-document":
            return _convert_document(args)
        if args.command == "validate-dataset":
            return _validate(args)
        if args.command == "add-dataset-sample":
            return _add_dataset_sample(args)
        if args.command == "benchmark":
            return _benchmark(args)
        if args.command == "prepare-corpus":
            return _prepare_corpus(args)
        if args.command == "generate-synthetic":
            return _generate_synthetic(args)
        if args.command == "train-recognizer":
            return _train_recognizer(args)
        if args.command == "export-recognizer":
            return _export_recognizer(args)
        if args.command == "recognize-line":
            return _recognize_line(args)
        if args.command == "benchmark-recognizer":
            return _benchmark_recognizer(args)
        if args.command == "calibrate-recognizer":
            return _calibrate_recognizer(args)
    except (DatasetManifestError, OcrEngineError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
