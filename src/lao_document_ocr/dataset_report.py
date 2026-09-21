from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from lao_document_ocr.dataset import (
    DatasetSample,
    DatasetSplit,
    DatasetSubset,
    validate_dataset,
)


def build_dataset_report(
    samples: list[DatasetSample],
    dataset_root: str | Path,
    *,
    verify_hashes: bool = True,
) -> dict:
    if not samples:
        raise ValueError("Dataset report requires at least one sample")

    errors = validate_dataset(
        samples,
        dataset_root,
        verify_hashes=verify_hashes,
    )

    by_split = Counter(sample.split.value for sample in samples)
    by_subset = Counter(sample.subset.value for sample in samples)
    by_license = Counter(sample.license for sample in samples)
    by_language = Counter(sample.language for sample in samples)

    document_splits: dict[str, DatasetSplit] = {}
    captures_per_document: Counter[str] = Counter()
    matrix: dict[str, Counter[str]] = defaultdict(Counter)

    for sample in samples:
        document_splits.setdefault(sample.document_id, sample.split)
        captures_per_document[sample.document_id] += 1
        matrix[sample.split.value][sample.subset.value] += 1

    required_subsets = [subset.value for subset in DatasetSubset if subset != DatasetSubset.OTHER]
    missing_subsets = sorted(
        subset
        for subset in required_subsets
        if by_subset.get(subset, 0) == 0
    )

    split_document_counts = Counter(split.value for split in document_splits.values())
    capture_counts = list(captures_per_document.values())

    return {
        "schema_version": "1",
        "generated_at": datetime.now(UTC).isoformat(),
        "sample_count": len(samples),
        "document_count": len(document_splits),
        "validation": {
            "ok": not errors,
            "errors": errors,
        },
        "by_split": dict(sorted(by_split.items())),
        "by_subset": dict(sorted(by_subset.items())),
        "by_license": dict(sorted(by_license.items())),
        "by_language": dict(sorted(by_language.items())),
        "split_document_counts": dict(sorted(split_document_counts.items())),
        "coverage_matrix": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(matrix.items())
        },
        "missing_subsets": missing_subsets,
        "captures_per_document": {
            "min": min(capture_counts),
            "max": max(capture_counts),
            "mean": sum(capture_counts) / len(capture_counts),
        },
    }


def write_dataset_report(report: dict, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
