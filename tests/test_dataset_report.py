import hashlib
import json

from lao_document_ocr.dataset import load_manifest
from lao_document_ocr.dataset_report import build_dataset_report, write_dataset_report


def _sample_entry(
    sample_id,
    document_id,
    split,
    subset,
    source,
    truth,
    digest,
    tags=None,
):
    return {
        "id": sample_id,
        "document_id": document_id,
        "split": split,
        "subset": subset,
        "source": source,
        "ground_truth": truth,
        "language": "lo",
        "license": "CC0-1.0",
        "provenance": "Dataset report unit test",
        "sha256": digest,
        "tags": tags or [],
    }


def test_dataset_report_summarizes_coverage(tmp_path) -> None:
    entries = []
    specs = [
        ("a", "doc-a", "train", "clean-print"),
        ("b", "doc-a", "train", "phone-photo"),
        ("c", "doc-c", "test", "noisy-scan"),
    ]

    for sample_id, document_id, split, subset in specs:
        image = tmp_path / f"{sample_id}.png"
        truth = tmp_path / f"{sample_id}.txt"
        image.write_bytes(f"image-{sample_id}".encode())
        truth.write_text(f"truth {sample_id}", encoding="utf-8")
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        entries.append(
            _sample_entry(
                sample_id,
                document_id,
                split,
                subset,
                image.name,
                truth.name,
                digest,
                tags=[f"capture:{subset}"],
            )
        )

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )

    report = build_dataset_report(load_manifest(manifest), tmp_path)

    assert report["sample_count"] == 3
    assert report["document_count"] == 2
    assert report["validation"]["ok"] is True
    assert report["by_split"] == {"test": 1, "train": 2}
    assert report["by_subset"]["phone-photo"] == 1
    assert report["coverage_matrix"]["train"]["clean-print"] == 1
    assert report["by_tag"]["capture:clean-print"] == 1
    assert report["captures_per_document"]["max"] == 2
    assert "complex-table" in report["missing_subsets"]


def test_dataset_report_round_trip(tmp_path) -> None:
    image = tmp_path / "sample.png"
    truth = tmp_path / "sample.txt"
    image.write_bytes(b"image")
    truth.write_text("truth", encoding="utf-8")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            _sample_entry(
                "sample",
                "doc",
                "test",
                "clean-print",
                image.name,
                truth.name,
                digest,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_dataset_report(load_manifest(manifest), tmp_path)
    output = write_dataset_report(report, tmp_path / "report.json")

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1"
    assert payload["sample_count"] == 1


def test_dataset_report_counts_layout_labeled_samples(tmp_path) -> None:
    from PIL import Image

    from lao_document_ocr.layout_ground_truth import write_layout_ground_truth
    from lao_document_ocr.models import Block, BlockType, BoundingBox, Document, Page

    entries = []
    for sample_id, split, with_layout in (
        ("layout-a", "train", True),
        ("layout-b", "test", False),
    ):
        image = tmp_path / f"{sample_id}.png"
        truth = tmp_path / f"{sample_id}.txt"
        Image.new("RGB", (120, 60), "white").save(image)
        truth.write_text(sample_id, encoding="utf-8")
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        entry = _sample_entry(
            sample_id,
            f"doc-{sample_id}",
            split,
            "clean-print",
            image.name,
            truth.name,
            digest,
        )
        if with_layout:
            layout = tmp_path / f"{sample_id}.json"
            write_layout_ground_truth(
                Document(
                    pages=[
                        Page(
                            number=1,
                            width=120,
                            height=60,
                            blocks=[
                                Block(
                                    type=BlockType.PARAGRAPH,
                                    bbox=BoundingBox(
                                        x=10,
                                        y=10,
                                        width=80,
                                        height=20,
                                    ),
                                )
                            ],
                        )
                    ]
                ),
                layout,
            )
            entry["layout_ground_truth"] = layout.name
        entries.append(entry)

    manifest = tmp_path / "layout-report-manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )

    report = build_dataset_report(load_manifest(manifest), tmp_path)

    layout = report["layout_ground_truth"]
    assert layout["sample_count"] == 1
    assert layout["document_count"] == 1
    assert layout["coverage_ratio"] == 0.5
    assert layout["by_split"] == {"train": 1}
    assert layout["by_subset"] == {"clean-print": 1}


def test_dataset_report_counts_manual_review_status(tmp_path) -> None:
    from datetime import UTC, datetime

    entries = []
    for sample_id, review in (
        (
            "approved",
            {
                "status": "approved",
                "reviewer": "Reviewer A",
                "reviewed_at": datetime(2026, 9, 23, 8, 0, tzinfo=UTC).isoformat(),
            },
        ),
        (
            "rejected",
            {
                "status": "rejected",
                "reviewer": "Reviewer B",
                "reviewed_at": datetime(2026, 9, 23, 8, 5, tzinfo=UTC).isoformat(),
                "notes": "Needs recapture",
            },
        ),
        ("unreviewed", None),
    ):
        image = tmp_path / f"{sample_id}.png"
        truth = tmp_path / f"{sample_id}.txt"
        image.write_bytes(f"image-{sample_id}".encode())
        truth.write_text(sample_id, encoding="utf-8")
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        entry = _sample_entry(
            sample_id,
            f"doc-{sample_id}",
            "test",
            "clean-print",
            image.name,
            truth.name,
            digest,
        )
        if review is not None:
            entry["review"] = review
        entries.append(entry)

    manifest = tmp_path / "review-report-manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )

    report = build_dataset_report(load_manifest(manifest), tmp_path)
    review = report["review"]

    assert review["approved"] == 1
    assert review["rejected"] == 1
    assert review["unreviewed"] == 1
    assert review["approved_coverage_ratio"] == 1 / 3
    assert review["approved_sample_ids"] == ["approved"]
    assert review["rejected_sample_ids"] == ["rejected"]
    assert review["unreviewed_sample_ids"] == ["unreviewed"]
