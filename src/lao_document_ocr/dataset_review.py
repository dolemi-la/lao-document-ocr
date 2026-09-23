from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from lao_document_ocr.dataset import (
    DatasetReview,
    DatasetReviewStatus,
    DatasetSample,
    load_manifest,
    validate_dataset,
)


def is_review_approved(sample: DatasetSample) -> bool:
    return (
        sample.review is not None
        and sample.review.status == DatasetReviewStatus.APPROVED
    )


def set_dataset_sample_review(
    manifest_path: str | Path,
    dataset_root: str | Path,
    *,
    sample_id: str,
    status: DatasetReviewStatus,
    reviewer: str,
    notes: str | None = None,
    reviewed_at: datetime | None = None,
) -> DatasetSample:
    manifest = Path(manifest_path)
    samples = load_manifest(manifest)

    target_index = next(
        (
            index
            for index, sample in enumerate(samples)
            if sample.id == sample_id
        ),
        None,
    )
    if target_index is None:
        raise ValueError(f"Dataset sample not found: {sample_id}")

    target = samples[target_index]
    if status == DatasetReviewStatus.APPROVED:
        errors = validate_dataset(
            [target],
            dataset_root,
            verify_hashes=True,
        )
        if errors:
            raise ValueError(
                "Cannot approve invalid dataset sample:\n"
                + "\n".join(f"- {error}" for error in errors)
            )

    timestamp = reviewed_at or datetime.now(UTC)
    review = DatasetReview(
        status=status,
        reviewer=reviewer,
        reviewed_at=timestamp,
        notes=notes.strip() if notes and notes.strip() else None,
    )
    updated = target.model_copy(update={"review": review})
    samples[target_index] = updated

    manifest.parent.mkdir(parents=True, exist_ok=True)
    original_mode = manifest.stat().st_mode & 0o777

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=manifest.parent,
            prefix=f".{manifest.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for sample in samples:
                handle.write(
                    json.dumps(
                        sample.model_dump(
                            mode="json",
                            exclude_none=True,
                        ),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())

        temporary_path.chmod(original_mode)
        os.replace(temporary_path, manifest)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)

    return updated
