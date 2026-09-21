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

    dataset_report = subparsers.add_parser(
        "dataset-report",
        help="Summarize dataset coverage, splits, licenses, and validation status.",
    )
    dataset_report.add_argument("--manifest", required=True, type=Path)
    dataset_report.add_argument("--dataset-root", required=True, type=Path)
    dataset_report.add_argument("--output", required=True, type=Path)
    dataset_report.add_argument("--no-hash-check", action="store_true")

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

    layout_benchmark = subparsers.add_parser(
        "benchmark-layout",
        help="Compare two document AST JSON files for layout/structure quality.",
    )
    layout_benchmark.add_argument("--reference", required=True, type=Path)
    layout_benchmark.add_argument("--prediction", required=True, type=Path)
    layout_benchmark.add_argument("--output", required=True, type=Path)
    layout_benchmark.add_argument("--iou-threshold", type=float, default=0.5)

    docx_benchmark = subparsers.add_parser(
        "benchmark-docx",
        help="Render a DOCX and compare it visually against a reference image/PDF.",
    )
    docx_benchmark.add_argument("--reference", required=True, type=Path)
    docx_benchmark.add_argument("--docx", required=True, type=Path)
    docx_benchmark.add_argument("--output", required=True, type=Path)
    docx_benchmark.add_argument("--dpi", type=int, default=144)
    docx_benchmark.add_argument("--office-binary", type=Path)

    bundle = subparsers.add_parser(
        "bundle-benchmarks",
        help="Bundle benchmark reports with hashes and one source revision.",
    )
    bundle.add_argument("--ocr", type=Path)
    bundle.add_argument("--recognizer", type=Path)
    bundle.add_argument("--layout", type=Path)
    bundle.add_argument("--docx", type=Path)
    bundle.add_argument("--output", required=True, type=Path)
    bundle.add_argument("--revision", required=True)
    bundle.add_argument("--label")

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

    capture_pack = subparsers.add_parser(
        "generate-capture-pack",
        help="Generate printable rights-clear benchmark capture pages from a corpus.",
    )
    capture_pack.add_argument("--corpus", required=True, type=Path)
    capture_pack.add_argument("--output", required=True, type=Path)
    capture_pack.add_argument("--font", required=True, type=Path)
    capture_pack.add_argument("--pack-id", required=True)
    capture_pack.add_argument("--text-license", required=True)
    capture_pack.add_argument("--text-provenance", required=True)
    capture_pack.add_argument("--dpi", type=int, default=150)
    capture_pack.add_argument("--lines-per-page", type=int, default=10)
    capture_pack.add_argument("--max-pages", type=int)

    capture_register = subparsers.add_parser(
        "register-capture",
        help="Register a released flatbed/phone capture from a capture-pack page.",
    )
    capture_register.add_argument("--pack-manifest", required=True, type=Path)
    capture_register.add_argument("--page-id", required=True)
    capture_register.add_argument("--capture-image", required=True, type=Path)
    capture_register.add_argument("--capture-id", required=True)
    capture_register.add_argument(
        "--mode",
        required=True,
        choices=["flatbed-scan", "degraded-scan", "phone-photo"],
    )
    capture_register.add_argument("--contributor", required=True)
    capture_register.add_argument("--release-license", required=True)
    capture_register.add_argument("--dataset-root", required=True, type=Path)
    capture_register.add_argument("--dataset-manifest", required=True, type=Path)
    capture_register.add_argument("--notes")
    capture_register.add_argument("--confirm-release", action="store_true")

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


def _dataset_report(args: argparse.Namespace) -> int:
    from lao_document_ocr.dataset_report import (
        build_dataset_report,
        write_dataset_report,
    )

    samples = load_manifest(args.manifest)
    report = build_dataset_report(
        samples,
        args.dataset_root,
        verify_hashes=not args.no_hash_check,
    )
    output = write_dataset_report(report, args.output)
    print(f"Report: {output}")
    print(f"Samples: {report['sample_count']}")
    print(f"Documents: {report['document_count']}")
    print(f"Validation: {'ok' if report['validation']['ok'] else 'failed'}")
    print(f"Missing subsets: {len(report['missing_subsets'])}")
    return 0 if report["validation"]["ok"] else 1


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


def _benchmark_layout(args: argparse.Namespace) -> int:
    from lao_document_ocr.layout_benchmark import (
        benchmark_layout,
        load_document_ast,
        write_layout_report,
    )

    reference = load_document_ast(args.reference)
    prediction = load_document_ast(args.prediction)
    report = benchmark_layout(
        reference,
        prediction,
        iou_threshold=args.iou_threshold,
    )
    output = write_layout_report(report, args.output)
    metrics = report["metrics"]
    print(f"Report: {output}")
    print(f"Block F1: {metrics['block_f1']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"Type accuracy: {metrics['block_type_accuracy']:.4f}")
    print(f"Reading-order accuracy: {metrics['reading_order_accuracy']:.4f}")
    print(f"Table-cell F1: {metrics['table_cell_f1']:.4f}")
    return 0


def _benchmark_docx(args: argparse.Namespace) -> int:
    from lao_document_ocr.visual_fidelity import (
        benchmark_docx_fidelity,
        write_fidelity_report,
    )

    report = benchmark_docx_fidelity(
        args.reference,
        args.docx,
        dpi=args.dpi,
        office_binary=args.office_binary,
    )
    output = write_fidelity_report(report, args.output)
    metrics = report["metrics"]
    print(f"Report: {output}")
    print(f"Composite: {metrics['composite_score']:.4f}")
    print(f"Pixel similarity: {metrics['pixel_similarity']:.4f}")
    print(f"Foreground IoU: {metrics['foreground_iou']:.4f}")
    print(f"Edge F1: {metrics['edge_f1']:.4f}")
    print(f"Page-count score: {metrics['page_count_score']:.4f}")
    return 0


def _bundle_benchmarks(args: argparse.Namespace) -> int:
    from lao_document_ocr.benchmark_bundle import (
        build_benchmark_bundle,
        write_benchmark_bundle,
    )

    reports = {
        kind: path
        for kind, path in {
            "ocr": args.ocr,
            "recognizer": args.recognizer,
            "layout": args.layout,
            "docx": args.docx,
        }.items()
        if path is not None
    }
    bundle = build_benchmark_bundle(
        reports,
        source_revision=args.revision,
        label=args.label,
    )
    output = write_benchmark_bundle(bundle, args.output)
    print(f"Bundle: {output}")
    print(f"Revision: {args.revision}")
    print(f"Reports: {len(bundle['reports'])}")
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


def _generate_capture_pack(args: argparse.Namespace) -> int:
    from lao_document_ocr.capture_pack import generate_capture_pack

    manifest = generate_capture_pack(
        load_corpus(args.corpus),
        args.output,
        args.font,
        pack_id=args.pack_id,
        text_license=args.text_license,
        text_provenance=args.text_provenance,
        dpi=args.dpi,
        lines_per_page=args.lines_per_page,
        max_pages=args.max_pages,
    )
    print(f"Capture pack: {manifest}")
    print(f"Printable PDF: {manifest.parent / (args.pack_id + '.pdf')}")
    return 0


def _register_capture(args: argparse.Namespace) -> int:
    from lao_document_ocr.capture_registration import (
        CaptureMode,
        register_capture,
    )

    sample = register_capture(
        capture_pack_manifest=args.pack_manifest,
        page_id=args.page_id,
        capture_image=args.capture_image,
        capture_id=args.capture_id,
        capture_mode=CaptureMode(args.mode),
        contributor=args.contributor,
        release_license=args.release_license,
        dataset_root=args.dataset_root,
        dataset_manifest=args.dataset_manifest,
        confirm_release=args.confirm_release,
        notes=args.notes,
    )
    print(json.dumps(sample.model_dump(mode="json", exclude_none=True), ensure_ascii=False))
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
        if args.command == "dataset-report":
            return _dataset_report(args)
        if args.command == "add-dataset-sample":
            return _add_dataset_sample(args)
        if args.command == "benchmark-layout":
            return _benchmark_layout(args)
        if args.command == "benchmark-docx":
            return _benchmark_docx(args)
        if args.command == "bundle-benchmarks":
            return _bundle_benchmarks(args)
        if args.command == "benchmark":
            return _benchmark(args)
        if args.command == "prepare-corpus":
            return _prepare_corpus(args)
        if args.command == "generate-synthetic":
            return _generate_synthetic(args)
        if args.command == "generate-capture-pack":
            return _generate_capture_pack(args)
        if args.command == "register-capture":
            return _register_capture(args)
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
    except (DatasetManifestError, OcrEngineError, ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
