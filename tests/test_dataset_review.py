import hashlib
import json
from datetime import UTC, datetime

import pytest
from PIL import Image

from lao_document_ocr.dataset import (
    DatasetReviewStatus,
    load_manifest,
)
from lao_document_ocr.dataset_review import (
    is_review_approved,
    set_dataset_sample_review,
)


def _manifest(tmp_path, *, valid_hash: bool = True):
    image = tmp_path / "page.png"
    truth = tmp_path / "page.txt"
    Image.new("RGB", (120, 80), "white").save(image)
    truth.write_text("ສະບາຍດີ", encoding="utf-8")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    if not valid_hash:
        digest = "0" * 64

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "document_id": "doc-001",
                "ground_truth": "page.txt",
                "id": "sample-001",
                "license": "CC0-1.0",
                "provenance": "review test",
                "sha256": digest,
                "source": "page.png",
                "split": "test",
                "subset": "clean-print",
                "tags": [
                    "source:real-capture",
                    "capture:optical-evidence",
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_approve_valid_sample_rewrites_manifest_atomically(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    reviewed_at = datetime(2026, 9, 23, 7, 30, tzinfo=UTC)

    updated = set_dataset_sample_review(
        manifest,
        tmp_path,
        sample_id="sample-001",
        status=DatasetReviewStatus.APPROVED,
        reviewer="Reviewer A",
        notes="Checked orientation and ground truth.",
        reviewed_at=reviewed_at,
    )

    assert is_review_approved(updated) is True
    loaded = load_manifest(manifest)[0]
    assert loaded.review is not None
    assert loaded.review.status == DatasetReviewStatus.APPROVED
    assert loaded.review.reviewer == "Reviewer A"
    assert loaded.review.reviewed_at == reviewed_at
    assert loaded.review.notes == "Checked orientation and ground truth."


def test_approval_refuses_invalid_hash(tmp_path) -> None:
    manifest = _manifest(tmp_path, valid_hash=False)

    with pytest.raises(ValueError, match="Cannot approve invalid dataset sample"):
        set_dataset_sample_review(
            manifest,
            tmp_path,
            sample_id="sample-001",
            status=DatasetReviewStatus.APPROVED,
            reviewer="Reviewer A",
        )

    assert load_manifest(manifest)[0].review is None


def test_rejection_can_record_invalid_sample(tmp_path) -> None:
    manifest = _manifest(tmp_path, valid_hash=False)

    updated = set_dataset_sample_review(
        manifest,
        tmp_path,
        sample_id="sample-001",
        status=DatasetReviewStatus.REJECTED,
        reviewer="Reviewer A",
        notes="Hash mismatch; needs recapture.",
    )

    assert updated.review is not None
    assert updated.review.status == DatasetReviewStatus.REJECTED
    assert is_review_approved(updated) is False


def test_unknown_sample_id_is_rejected(tmp_path) -> None:
    manifest = _manifest(tmp_path)

    with pytest.raises(ValueError, match="sample not found"):
        set_dataset_sample_review(
            manifest,
            tmp_path,
            sample_id="missing",
            status=DatasetReviewStatus.APPROVED,
            reviewer="Reviewer A",
        )
