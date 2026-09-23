from __future__ import annotations

from collections import Counter
from pathlib import Path

from lao_document_ocr.benchmarking import benchmark_dataset
from lao_document_ocr.capture_pack import load_capture_pack
from lao_document_ocr.capture_suite import load_capture_suite
from lao_document_ocr.dataset import (
    DatasetSample,
    DatasetSplit,
    DatasetSubset,
)
from lao_document_ocr.ocr.base import OcrEngine
from lao_document_ocr.reading_order import ReadingOrderResolver

_TEMPLATE_SUBSETS = {
    "plain": DatasetSubset.CLEAN_PRINT,
    "two-column": DatasetSubset.MULTI_COLUMN,
    "ruled-table": DatasetSubset.SIMPLE_TABLE,
    "borderless-table": DatasetSubset.COMPLEX_TABLE,
    "receipt": DatasetSubset.RECEIPT,
    "form": DatasetSubset.FORM,
}


def build_capture_suite_qa_samples(
    suite_manifest: str | Path,
) -> tuple[Path, str, list[DatasetSample]]:
    suite_path = Path(suite_manifest).resolve()
    suite = load_capture_suite(suite_path)
    suite_root = suite_path.parent
    samples: list[DatasetSample] = []

    for pack in suite.packs:
        subset = _TEMPLATE_SUBSETS.get(pack.template, DatasetSubset.OTHER)
        pack_manifest = (suite_root / pack.manifest).resolve()
        pack_root, manifest = load_capture_pack(pack_manifest)

        for page in manifest.pages:
            source = (pack_root / page.image).resolve()
            truth = (pack_root / page.ground_truth).resolve()
            samples.append(
                DatasetSample(
                    id=page.id,
                    document_id=page.id,
                    split=DatasetSplit.TEST,
                    subset=subset,
                    source=source.relative_to(suite_root).as_posix(),
                    ground_truth=truth.relative_to(suite_root).as_posix(),
                    language="lo",
                    license=suite.text_license,
                    provenance=(
                        "Digital capture-suite QA source; "
                        f"{suite.text_provenance}"
                    ),
                    sha256=page.sha256,
                    notes=(
                        "Generated digital capture-pack page; "
                        "not real optical benchmark evidence."
                    ),
                    tags=[
                        *page.tags,
                        "qa:digital-capture-suite",
                        "source:digital-capture-suite",
                    ],
                )
            )

    if not samples:
        raise ValueError("Capture suite contains no QA pages")

    return suite_root, suite.suite_id, samples


def benchmark_capture_suite(
    suite_manifest: str | Path,
    engine: OcrEngine,
    *,
    verify_hashes: bool = True,
    reading_order_resolver: ReadingOrderResolver | None = None,
) -> dict:
    suite_root, suite_id, samples = build_capture_suite_qa_samples(
        suite_manifest
    )
    report = benchmark_dataset(
        samples,
        suite_root,
        engine,
        split=DatasetSplit.TEST,
        verify_hashes=verify_hashes,
        reading_order_resolver=reading_order_resolver,
    )
    template_counts = Counter(
        tag.split(":", 1)[1]
        for sample in samples
        for tag in sample.tags
        if tag.startswith("template:")
    )
    report["qa"] = {
        "kind": "digital-capture-suite",
        "suite_id": suite_id,
        "not_real_benchmark": True,
        "sample_count": len(samples),
        "templates": dict(sorted(template_counts.items())),
        "warning": (
            "Generated digital pages measure template/OCR pipeline sanity only. "
            "Do not report these results as real scan/photo OCR accuracy."
        ),
    }
    return report
