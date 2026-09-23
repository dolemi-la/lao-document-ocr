from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from lao_document_ocr.dataset import (
    DatasetReview,
    DatasetReviewStatus,
    DatasetSample,
    load_manifest,
    validate_dataset,
)


@dataclass(frozen=True)
class ReviewDecision:
    sample_id: str
    status: DatasetReviewStatus
    reviewer: str
    notes: str | None = None


def is_review_approved(sample: DatasetSample) -> bool:
    return (
        sample.review is not None
        and sample.review.status == DatasetReviewStatus.APPROVED
    )


def _write_manifest_atomic(
    manifest: Path,
    samples: list[DatasetSample],
) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    original_mode = (
        manifest.stat().st_mode & 0o777
        if manifest.exists()
        else 0o644
    )

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


def _validated_review(
    sample: DatasetSample,
    dataset_root: str | Path,
    *,
    status: DatasetReviewStatus,
    reviewer: str,
    notes: str | None,
    reviewed_at: datetime,
) -> DatasetReview:
    if status == DatasetReviewStatus.APPROVED:
        errors = validate_dataset(
            [sample],
            dataset_root,
            verify_hashes=True,
        )
        if errors:
            raise ValueError(
                f"Cannot approve invalid dataset sample {sample.id}:\n"
                + "\n".join(f"- {error}" for error in errors)
            )

    return DatasetReview(
        status=status,
        reviewer=reviewer,
        reviewed_at=reviewed_at,
        notes=notes.strip() if notes and notes.strip() else None,
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
    review = _validated_review(
        target,
        dataset_root,
        status=status,
        reviewer=reviewer,
        notes=notes,
        reviewed_at=reviewed_at or datetime.now(UTC),
    )
    updated = target.model_copy(update={"review": review})
    samples[target_index] = updated
    _write_manifest_atomic(manifest, samples)
    return updated


def load_review_decisions(path: str | Path) -> list[ReviewDecision]:
    source = Path(path)
    try:
        handle = source.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise ValueError(f"Could not read review decisions: {source}") from exc

    with handle:
        reader = csv.DictReader(handle)
        required = {"id", "status", "reviewer", "notes"}
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise ValueError(
                "Review decision CSV is missing columns: "
                + ", ".join(missing)
            )

        decisions: list[ReviewDecision] = []
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            sample_id = (row.get("id") or "").strip()
            status_value = (row.get("status") or "").strip().lower()
            reviewer = (row.get("reviewer") or "").strip()
            notes = (row.get("notes") or "").strip() or None

            if not status_value:
                continue
            if not sample_id:
                raise ValueError(
                    f"Review decision row {row_number} has no sample id"
                )
            if sample_id in seen:
                raise ValueError(
                    f"Duplicate review decision for sample: {sample_id}"
                )
            seen.add(sample_id)
            try:
                status = DatasetReviewStatus(status_value)
            except ValueError as exc:
                raise ValueError(
                    f"Review decision row {row_number} has invalid status: "
                    f"{status_value}"
                ) from exc
            if not reviewer:
                raise ValueError(
                    f"Review decision row {row_number} requires reviewer"
                )
            decisions.append(
                ReviewDecision(
                    sample_id=sample_id,
                    status=status,
                    reviewer=reviewer,
                    notes=notes,
                )
            )

    if not decisions:
        raise ValueError("Review decision CSV contains no decisions")
    return decisions


def apply_review_decisions(
    manifest_path: str | Path,
    dataset_root: str | Path,
    decisions_path: str | Path,
    *,
    confirm: bool = False,
    reviewed_at: datetime | None = None,
) -> dict:
    manifest = Path(manifest_path)
    samples = load_manifest(manifest)
    decisions = load_review_decisions(decisions_path)
    by_id = {sample.id: index for index, sample in enumerate(samples)}

    unknown = sorted(
        decision.sample_id
        for decision in decisions
        if decision.sample_id not in by_id
    )
    if unknown:
        raise ValueError(
            "Review decisions reference unknown samples: "
            + ", ".join(unknown)
        )

    timestamp = reviewed_at or datetime.now(UTC)
    updated_samples = list(samples)
    approved: list[str] = []
    rejected: list[str] = []

    # Validate and build the complete new manifest in memory first. Nothing is
    # written unless every explicit decision is valid.
    for decision in decisions:
        index = by_id[decision.sample_id]
        sample = samples[index]
        review = _validated_review(
            sample,
            dataset_root,
            status=decision.status,
            reviewer=decision.reviewer,
            notes=decision.notes,
            reviewed_at=timestamp,
        )
        updated_samples[index] = sample.model_copy(
            update={"review": review}
        )
        if decision.status == DatasetReviewStatus.APPROVED:
            approved.append(sample.id)
        else:
            rejected.append(sample.id)

    if confirm:
        _write_manifest_atomic(manifest, updated_samples)

    return {
        "schema_version": "1",
        "dry_run": not confirm,
        "decision_count": len(decisions),
        "approved_count": len(approved),
        "rejected_count": len(rejected),
        "approved_sample_ids": sorted(approved),
        "rejected_sample_ids": sorted(rejected),
        "reviewed_at": timestamp.isoformat(),
    }
