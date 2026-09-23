from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lao_document_ocr.dataset import (
    DatasetSample,
    DatasetSplit,
    sha256_file,
    validate_dataset,
)
from lao_document_ocr.dataset_review import is_review_approved

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


class BenchmarkFreezeError(ValueError):
    pass


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sample_lock_entry(sample: DatasetSample, root: Path) -> dict[str, Any]:
    source = root / sample.source
    truth = root / sample.ground_truth
    layout = (
        root / sample.layout_ground_truth
        if sample.layout_ground_truth is not None
        else None
    )

    return {
        "id": sample.id,
        "document_id": sample.document_id,
        "subset": sample.subset.value,
        "tags": list(sample.tags),
        "license": sample.license,
        "review": (
            sample.review.model_dump(mode="json")
            if sample.review is not None
            else None
        ),
        "source": sample.source,
        "source_sha256": sha256_file(source),
        "ground_truth": sample.ground_truth,
        "ground_truth_sha256": sha256_file(truth),
        "layout_ground_truth": sample.layout_ground_truth,
        "layout_ground_truth_sha256": (
            sha256_file(layout) if layout is not None else None
        ),
    }


def freeze_benchmark(
    samples: list[DatasetSample],
    dataset_root: str | Path,
    *,
    output_manifest: str | Path,
    output_lock: str | Path,
    split: DatasetSplit = DatasetSplit.TEST,
    source_manifest: str | Path | None = None,
    source_revision: str | None = None,
    require_real_sources: bool = False,
    require_manual_review: bool = False,
) -> tuple[Path, Path]:
    errors = validate_dataset(samples, dataset_root, verify_hashes=True)
    if errors:
        raise BenchmarkFreezeError(
            "Dataset validation failed before freeze:\n"
            + "\n".join(f"- {error}" for error in errors)
        )

    selected = sorted(
        (sample for sample in samples if sample.split == split),
        key=lambda sample: sample.id,
    )
    if not selected:
        raise BenchmarkFreezeError(
            f"No samples found for split '{split.value}'."
        )

    if require_real_sources:
        unverified = [sample.id for sample in selected if not _is_real_source(sample)]
        if unverified:
            raise BenchmarkFreezeError(
                "Frozen public benchmark contains samples without verified real-source "
                "evidence: " + ", ".join(unverified)
            )

    if require_manual_review:
        unapproved = [sample.id for sample in selected if not is_review_approved(sample)]
        if unapproved:
            raise BenchmarkFreezeError(
                "Frozen public benchmark contains samples without approved manual "
                "review: " + ", ".join(unapproved)
            )

    root = Path(dataset_root).resolve()
    manifest_path = Path(output_manifest)
    lock_path = Path(output_lock)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_lines: list[str] = []
    for sample in selected:
        payload = sample.model_dump(mode="json", exclude_none=True)
        payload["sha256"] = sha256_file(root / sample.source)
        manifest_lines.append(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    manifest_bytes = ("\n".join(manifest_lines) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)

    source_manifest_path = (
        Path(source_manifest) if source_manifest is not None else None
    )
    source_manifest_sha256 = (
        sha256_file(source_manifest_path)
        if source_manifest_path is not None
        else None
    )

    subset_counts = Counter(sample.subset.value for sample in selected)
    tag_counts = Counter(tag for sample in selected for tag in sample.tags)
    document_ids = sorted({sample.document_id for sample in selected})

    lock = {
        "schema_version": "1",
        "generated_at": datetime.now(UTC).isoformat(),
        "split": split.value,
        "source_revision": source_revision,
        "source_manifest": (
            source_manifest_path.name if source_manifest_path is not None else None
        ),
        "source_manifest_sha256": source_manifest_sha256,
        "frozen_manifest": manifest_path.name,
        "frozen_manifest_sha256": _sha256_bytes(manifest_bytes),
        "sample_count": len(selected),
        "document_count": len(document_ids),
        "document_ids": document_ids,
        "subsets": dict(sorted(subset_counts.items())),
        "tags": dict(sorted(tag_counts.items())),
        "samples": [
            _sample_lock_entry(sample, root)
            for sample in selected
        ],
    }
    lock_path.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path, lock_path


def benchmark_freeze_identity(lock_path: str | Path) -> dict[str, Any]:
    lock_file = Path(lock_path)
    try:
        data = lock_file.read_bytes()
        lock = json.loads(data.decode("utf-8"))
    except OSError as exc:
        raise BenchmarkFreezeError(
            f"Could not read benchmark lock: {lock_file}"
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkFreezeError(
            f"Invalid benchmark lock JSON: {lock_file}"
        ) from exc

    if lock.get("schema_version") != "1":
        raise BenchmarkFreezeError("Unsupported benchmark lock schema")

    return {
        "lock_file": lock_file.name,
        "lock_sha256": _sha256_bytes(data),
        "frozen_manifest": lock.get("frozen_manifest"),
        "frozen_manifest_sha256": lock.get("frozen_manifest_sha256"),
        "split": lock.get("split"),
        "sample_count": lock.get("sample_count"),
        "source_revision": lock.get("source_revision"),
    }


def verify_benchmark_freeze(
    lock_path: str | Path,
    dataset_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> list[str]:
    lock_file = Path(lock_path)
    try:
        lock = json.loads(lock_file.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BenchmarkFreezeError(
            f"Could not read benchmark lock: {lock_file}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkFreezeError(
            f"Invalid benchmark lock JSON: {lock_file}"
        ) from exc

    if lock.get("schema_version") != "1":
        raise BenchmarkFreezeError("Unsupported benchmark lock schema")

    frozen_manifest = (
        Path(manifest_path)
        if manifest_path is not None
        else lock_file.parent / str(lock.get("frozen_manifest", ""))
    )
    errors: list[str] = []
    if not frozen_manifest.is_file():
        errors.append(f"frozen manifest missing: {frozen_manifest}")
    else:
        actual = sha256_file(frozen_manifest)
        expected = lock.get("frozen_manifest_sha256")
        if actual != expected:
            errors.append(
                "frozen manifest sha256 mismatch "
                f"(expected {expected}, got {actual})"
            )

    root = Path(dataset_root).resolve()
    samples = lock.get("samples")
    if not isinstance(samples, list):
        raise BenchmarkFreezeError("Benchmark lock samples must be a list")

    for item in samples:
        if not isinstance(item, dict):
            errors.append("invalid sample lock entry")
            continue
        sample_id = str(item.get("id", "unknown"))
        for path_key, hash_key in (
            ("source", "source_sha256"),
            ("ground_truth", "ground_truth_sha256"),
            ("layout_ground_truth", "layout_ground_truth_sha256"),
        ):
            relative = item.get(path_key)
            expected = item.get(hash_key)
            if relative is None:
                continue
            path = (root / str(relative)).resolve()
            if root not in path.parents and path != root:
                errors.append(f"{sample_id}: {path_key} escapes dataset root")
                continue
            if not path.is_file():
                errors.append(f"{sample_id}: missing {path_key}: {relative}")
                continue
            actual = sha256_file(path)
            if actual != expected:
                errors.append(
                    f"{sample_id}: {path_key} sha256 mismatch "
                    f"(expected {expected}, got {actual})"
                )

    return errors
