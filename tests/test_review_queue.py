import csv
import json
from datetime import UTC, datetime

from PIL import Image

from lao_document_ocr.dataset import (
    DatasetReview,
    DatasetReviewStatus,
    DatasetSample,
    DatasetSplit,
    DatasetSubset,
)
from lao_document_ocr.review_queue import (
    build_review_queue,
    review_queue_summary,
)


def _sample(tmp_path, sample_id: str, review=None):
    image = tmp_path / f"{sample_id}.png"
    truth = tmp_path / f"{sample_id}.txt"
    Image.new("RGB", (300, 180), (210, 240, 250)).save(image)
    truth.write_text(f"<script>{sample_id}</script>\nສະບາຍດີ", encoding="utf-8")
    return DatasetSample(
        id=sample_id,
        document_id=f"doc-{sample_id}",
        split=DatasetSplit.TEST,
        subset=DatasetSubset.PHONE_PHOTO,
        source=image.name,
        ground_truth=truth.name,
        license="CC0-1.0",
        provenance="review queue test",
        tags=["capture:phone-photo", "source:real-capture"],
        review=review,
    )


def test_review_queue_defaults_to_unreviewed_and_rejected(tmp_path) -> None:
    approved = DatasetReview(
        status=DatasetReviewStatus.APPROVED,
        reviewer="Reviewer",
        reviewed_at=datetime(2026, 9, 23, 8, 0, tzinfo=UTC),
    )
    rejected = DatasetReview(
        status=DatasetReviewStatus.REJECTED,
        reviewer="Reviewer",
        reviewed_at=datetime(2026, 9, 23, 8, 5, tzinfo=UTC),
        notes="Needs recapture",
    )
    samples = [
        _sample(tmp_path, "unreviewed"),
        _sample(tmp_path, "rejected", rejected),
        _sample(tmp_path, "approved", approved),
    ]

    html_path, json_path = build_review_queue(
        samples,
        tmp_path,
        tmp_path / "queue",
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["sample_count"] == 2
    assert {entry["id"] for entry in payload["entries"]} == {
        "unreviewed",
        "rejected",
    }
    assert all(entry["thumbnail"] for entry in payload["entries"])
    assert html_path.is_file()
    decisions_path = json_path.parent / payload["decision_template"]
    assert decisions_path.is_file()
    with decisions_path.open(encoding="utf-8", newline="") as handle:
        decision_rows = list(csv.DictReader(handle))
    assert {row["id"] for row in decision_rows} == {
        "unreviewed",
        "rejected",
    }
    assert all(row["status"] == "" for row in decision_rows)
    assert review_queue_summary(json_path) == {
        "sample_count": 2,
        "with_problems": 0,
    }


def test_review_queue_html_escapes_ground_truth(tmp_path) -> None:
    sample = _sample(tmp_path, "unsafe")
    html_path, _ = build_review_queue(
        [sample],
        tmp_path,
        tmp_path / "queue",
    )

    html_text = html_path.read_text(encoding="utf-8")
    assert "<script>unsafe</script>" not in html_text
    assert "&lt;script&gt;unsafe&lt;/script&gt;" in html_text


def test_review_queue_can_show_only_approved(tmp_path) -> None:
    approved = DatasetReview(
        status=DatasetReviewStatus.APPROVED,
        reviewer="Reviewer",
        reviewed_at=datetime(2026, 9, 23, 8, 0, tzinfo=UTC),
    )
    samples = [
        _sample(tmp_path, "pending"),
        _sample(tmp_path, "approved", approved),
    ]

    _, json_path = build_review_queue(
        samples,
        tmp_path,
        tmp_path / "queue",
        status="approved",
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert [entry["id"] for entry in payload["entries"]] == ["approved"]


def test_review_queue_reports_missing_files(tmp_path) -> None:
    sample = DatasetSample(
        id="missing",
        document_id="doc-missing",
        split=DatasetSplit.TEST,
        subset=DatasetSubset.CLEAN_PRINT,
        source="missing.png",
        ground_truth="missing.txt",
        license="CC0-1.0",
        provenance="review queue test",
    )

    _, json_path = build_review_queue(
        [sample],
        tmp_path,
        tmp_path / "queue",
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(payload["entries"][0]["problems"]) == 2
    assert review_queue_summary(json_path)["with_problems"] == 1
