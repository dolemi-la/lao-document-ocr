import csv
import hashlib
from datetime import UTC, datetime

import pytest
from PIL import Image

from lao_document_ocr.dataset import load_manifest
from lao_document_ocr.dataset_review import apply_review_decisions


def _dataset(tmp_path, *, invalid_second_hash: bool = False):
    entries = []
    for index, sample_id in enumerate(("a", "b"), start=1):
        image = tmp_path / f"{sample_id}.png"
        truth = tmp_path / f"{sample_id}.txt"
        Image.new("RGB", (100, 60), (index * 50, 240, 250)).save(image)
        truth.write_text(f"truth {sample_id}", encoding="utf-8")
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if invalid_second_hash and sample_id == "b":
            digest = "0" * 64
        entries.append(
            {
                "id": sample_id,
                "document_id": f"doc-{sample_id}",
                "split": "test",
                "subset": "clean-print",
                "source": image.name,
                "ground_truth": truth.name,
                "license": "CC0-1.0",
                "provenance": "batch review test",
                "sha256": digest,
            }
        )

    manifest = tmp_path / "manifest.jsonl"
    import json

    manifest.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )
    return manifest


def _decisions(tmp_path, rows):
    path = tmp_path / "decisions.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "status", "reviewer", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_dry_run_validates_without_mutating_manifest(tmp_path) -> None:
    manifest = _dataset(tmp_path)
    original = manifest.read_bytes()
    decisions = _decisions(
        tmp_path,
        [
            {
                "id": "a",
                "status": "approved",
                "reviewer": "Reviewer A",
                "notes": "Looks good",
            },
            {
                "id": "b",
                "status": "rejected",
                "reviewer": "Reviewer A",
                "notes": "Needs recapture",
            },
        ],
    )

    report = apply_review_decisions(
        manifest,
        tmp_path,
        decisions,
        confirm=False,
    )

    assert report["dry_run"] is True
    assert report["approved_count"] == 1
    assert report["rejected_count"] == 1
    assert manifest.read_bytes() == original
    assert all(sample.review is None for sample in load_manifest(manifest))


def test_confirm_applies_all_decisions_atomically(tmp_path) -> None:
    manifest = _dataset(tmp_path)
    decisions = _decisions(
        tmp_path,
        [
            {
                "id": "a",
                "status": "approved",
                "reviewer": "Reviewer A",
                "notes": "",
            },
            {
                "id": "b",
                "status": "rejected",
                "reviewer": "Reviewer B",
                "notes": "Blurred",
            },
        ],
    )
    timestamp = datetime(2026, 9, 23, 9, 0, tzinfo=UTC)

    report = apply_review_decisions(
        manifest,
        tmp_path,
        decisions,
        confirm=True,
        reviewed_at=timestamp,
    )

    assert report["dry_run"] is False
    samples = {sample.id: sample for sample in load_manifest(manifest)}
    assert samples["a"].review.status.value == "approved"
    assert samples["a"].review.reviewer == "Reviewer A"
    assert samples["a"].review.reviewed_at == timestamp
    assert samples["b"].review.status.value == "rejected"
    assert samples["b"].review.notes == "Blurred"


def test_invalid_approval_aborts_entire_batch(tmp_path) -> None:
    manifest = _dataset(tmp_path, invalid_second_hash=True)
    original = manifest.read_bytes()
    decisions = _decisions(
        tmp_path,
        [
            {
                "id": "a",
                "status": "approved",
                "reviewer": "Reviewer",
                "notes": "",
            },
            {
                "id": "b",
                "status": "approved",
                "reviewer": "Reviewer",
                "notes": "",
            },
        ],
    )

    with pytest.raises(ValueError, match="Cannot approve invalid"):
        apply_review_decisions(
            manifest,
            tmp_path,
            decisions,
            confirm=True,
        )

    assert manifest.read_bytes() == original


def test_duplicate_decision_is_rejected(tmp_path) -> None:
    manifest = _dataset(tmp_path)
    decisions = _decisions(
        tmp_path,
        [
            {
                "id": "a",
                "status": "approved",
                "reviewer": "Reviewer",
                "notes": "",
            },
            {
                "id": "a",
                "status": "rejected",
                "reviewer": "Reviewer",
                "notes": "",
            },
        ],
    )

    with pytest.raises(ValueError, match="Duplicate review decision"):
        apply_review_decisions(
            manifest,
            tmp_path,
            decisions,
        )


def test_blank_status_rows_are_ignored(tmp_path) -> None:
    manifest = _dataset(tmp_path)
    decisions = _decisions(
        tmp_path,
        [
            {
                "id": "a",
                "status": "",
                "reviewer": "",
                "notes": "",
            },
            {
                "id": "b",
                "status": "rejected",
                "reviewer": "Reviewer",
                "notes": "",
            },
        ],
    )

    report = apply_review_decisions(
        manifest,
        tmp_path,
        decisions,
    )

    assert report["decision_count"] == 1
    assert report["rejected_sample_ids"] == ["b"]
