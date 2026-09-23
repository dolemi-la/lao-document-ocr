from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from lao_document_ocr.dataset import (
    DatasetSample,
    DatasetSplit,
    DatasetSubset,
    validate_dataset,
)


@dataclass(frozen=True)
class CoverageDimension:
    name: str
    subsets: tuple[DatasetSubset, ...] = ()
    tags: tuple[str, ...] = ()

    def matches(self, sample: DatasetSample) -> bool:
        if sample.subset in self.subsets:
            return True
        sample_tags = set(sample.tags)
        return any(tag in sample_tags for tag in self.tags)


REAL_SOURCE_TAGS = {
    "source:real-capture",
    "source:real-document",
}


def _is_real_source(sample: DatasetSample) -> bool:
    tags = set(sample.tags)
    if "source:real-document" in tags:
        return True
    return (
        "source:real-capture" in tags
        and "capture:optical-evidence" in tags
    )


DEFAULT_COVERAGE_DIMENSIONS = (
    CoverageDimension(
        "clean-print",
        subsets=(DatasetSubset.CLEAN_PRINT,),
        tags=("capture:flatbed-scan",),
    ),
    CoverageDimension(
        "noisy-scan",
        subsets=(DatasetSubset.NOISY_SCAN,),
        tags=("capture:degraded-scan",),
    ),
    CoverageDimension(
        "phone-photo",
        subsets=(DatasetSubset.PHONE_PHOTO,),
        tags=("capture:phone-photo",),
    ),
    CoverageDimension(
        "mixed-lao-english",
        subsets=(DatasetSubset.MIXED_LAO_ENGLISH,),
        tags=("language:mixed",),
    ),
    CoverageDimension(
        "multi-column",
        subsets=(DatasetSubset.MULTI_COLUMN,),
        tags=("layout:multi-column",),
    ),
    CoverageDimension(
        "simple-table",
        subsets=(DatasetSubset.SIMPLE_TABLE,),
        tags=("table:ruled",),
    ),
    CoverageDimension(
        "complex-table",
        subsets=(DatasetSubset.COMPLEX_TABLE,),
        tags=("table:borderless", "table:complex"),
    ),
    CoverageDimension(
        "receipt",
        subsets=(DatasetSubset.RECEIPT,),
        tags=("document:receipt", "layout:receipt"),
    ),
    CoverageDimension(
        "form",
        subsets=(DatasetSubset.FORM,),
        tags=("document:form", "layout:form"),
    ),
)


def build_benchmark_readiness_report(
    samples: list[DatasetSample],
    dataset_root: str | Path,
    *,
    split: DatasetSplit = DatasetSplit.TEST,
    min_documents_per_dimension: int = 1,
    min_total_documents: int = 0,
    min_layout_labeled_documents: int = 0,
    verify_hashes: bool = True,
    require_real_sources: bool = True,
) -> dict:
    if min_documents_per_dimension < 1:
        raise ValueError("min_documents_per_dimension must be at least 1")
    if min_total_documents < 0:
        raise ValueError("min_total_documents must be non-negative")
    if min_layout_labeled_documents < 0:
        raise ValueError("min_layout_labeled_documents must be non-negative")

    validation_errors = validate_dataset(
        samples,
        dataset_root,
        verify_hashes=verify_hashes,
    )
    selected = [sample for sample in samples if sample.split == split]
    eligible = (
        [sample for sample in selected if _is_real_source(sample)]
        if require_real_sources
        else selected
    )
    document_ids = {sample.document_id for sample in eligible}
    layout_document_ids = {
        sample.document_id
        for sample in eligible
        if sample.layout_ground_truth is not None
    }
    unverified_source_samples = [
        sample.id
        for sample in selected
        if not _is_real_source(sample)
    ]

    dimensions: dict[str, dict] = {}
    missing: list[str] = []
    for dimension in DEFAULT_COVERAGE_DIMENSIONS:
        matching = [sample for sample in eligible if dimension.matches(sample)]
        matching_documents = sorted({sample.document_id for sample in matching})
        passed = len(matching_documents) >= min_documents_per_dimension
        dimensions[dimension.name] = {
            "sample_count": len(matching),
            "document_count": len(matching_documents),
            "document_ids": matching_documents,
            "minimum_documents": min_documents_per_dimension,
            "passed": passed,
        }
        if not passed:
            missing.append(dimension.name)

    total_documents_passed = len(document_ids) >= min_total_documents
    layout_passed = len(layout_document_ids) >= min_layout_labeled_documents
    validation_passed = not validation_errors

    return {
        "schema_version": "1",
        "generated_at": datetime.now(UTC).isoformat(),
        "split": split.value,
        "ready": (
            validation_passed
            and not missing
            and total_documents_passed
            and layout_passed
        ),
        "validation": {
            "ok": validation_passed,
            "errors": validation_errors,
        },
        "sample_count": len(selected),
        "eligible_sample_count": len(eligible),
        "document_count": len(document_ids),
        "require_real_sources": require_real_sources,
        "accepted_real_source_tags": sorted(REAL_SOURCE_TAGS),
        "unverified_source_samples": sorted(unverified_source_samples),
        "minimum_total_documents": min_total_documents,
        "total_documents_passed": total_documents_passed,
        "layout_labeled_document_count": len(layout_document_ids),
        "minimum_layout_labeled_documents": min_layout_labeled_documents,
        "layout_labeled_documents_passed": layout_passed,
        "minimum_documents_per_dimension": min_documents_per_dimension,
        "dimensions": dimensions,
        "missing_dimensions": missing,
    }


def write_benchmark_readiness_report(report: dict, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
