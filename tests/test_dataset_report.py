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
