import hashlib
import json

import pytest

from lao_document_ocr.dataset import (
    DatasetManifestError,
    DatasetSplit,
    DatasetSubset,
    load_manifest,
    validate_dataset,
)


def _write_sample_dataset(tmp_path):
    image = tmp_path / "sample.png"
    truth = tmp_path / "sample.txt"
    image.write_bytes(b"fake-image")
    truth.write_text("ສະບາຍດີ", encoding="utf-8")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "sample-001",
                "document_id": "document-001",
                "split": "test",
                "subset": "clean-print",
                "source": "sample.png",
                "ground_truth": "sample.txt",
                "language": "lo",
                "license": "CC0-1.0",
                "provenance": "Synthetic unit-test fixture",
                "sha256": digest,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_load_and_validate_manifest(tmp_path) -> None:
    manifest = _write_sample_dataset(tmp_path)
    samples = load_manifest(manifest)

    assert len(samples) == 1
    assert samples[0].split == DatasetSplit.TEST
    assert samples[0].subset == DatasetSubset.CLEAN_PRINT
    assert validate_dataset(samples, tmp_path) == []


def test_hash_mismatch_is_reported(tmp_path) -> None:
    manifest = _write_sample_dataset(tmp_path)
    samples = load_manifest(manifest)
    (tmp_path / "sample.png").write_bytes(b"changed")

    errors = validate_dataset(samples, tmp_path)

    assert any("sha256 mismatch" in error for error in errors)


def test_duplicate_ids_are_rejected(tmp_path) -> None:
    manifest = _write_sample_dataset(tmp_path)
    line = manifest.read_text(encoding="utf-8")
    manifest.write_text(line + line, encoding="utf-8")

    with pytest.raises(DatasetManifestError, match="Duplicate sample id"):
        load_manifest(manifest)


def test_parent_path_is_rejected(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "bad-001",
                "document_id": "bad-doc",
                "split": "test",
                "subset": "other",
                "source": "../secret.png",
                "ground_truth": "truth.txt",
                "license": "CC0-1.0",
                "provenance": "test",
            }
        )
        + "\n"
    )

    with pytest.raises(DatasetManifestError):
        load_manifest(manifest)


def test_document_cannot_cross_dataset_splits(tmp_path) -> None:
    image = tmp_path / "sample.png"
    truth = tmp_path / "sample.txt"
    image.write_bytes(b"fake-image")
    truth.write_text("ground truth", encoding="utf-8")

    common = {
        "document_id": "same-document",
        "subset": "clean-print",
        "source": "sample.png",
        "ground_truth": "sample.txt",
        "license": "CC0-1.0",
        "provenance": "Synthetic unit-test fixture",
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"id": "train-page", "split": "train", **common}) + "\n"
        + json.dumps({"id": "test-page", "split": "test", **common}) + "\n",
        encoding="utf-8",
    )

    errors = validate_dataset(load_manifest(manifest), tmp_path)

    assert any("multiple splits" in error for error in errors)


def test_duplicate_source_hash_is_reported(tmp_path) -> None:
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    truth_a = tmp_path / "a.txt"
    truth_b = tmp_path / "b.txt"
    image_a.write_bytes(b"same-image")
    image_b.write_bytes(b"same-image")
    truth_a.write_text("A", encoding="utf-8")
    truth_b.write_text("B", encoding="utf-8")
    digest = hashlib.sha256(b"same-image").hexdigest()

    manifest = tmp_path / "manifest.jsonl"
    entries = [
        {
            "id": "a",
            "document_id": "doc-a",
            "split": "train",
            "subset": "clean-print",
            "source": "a.png",
            "ground_truth": "a.txt",
            "license": "CC0-1.0",
            "provenance": "unit test",
            "sha256": digest,
        },
        {
            "id": "b",
            "document_id": "doc-b",
            "split": "test",
            "subset": "phone-photo",
            "source": "b.png",
            "ground_truth": "b.txt",
            "license": "CC0-1.0",
            "provenance": "unit test",
            "sha256": digest,
        },
    ]
    manifest.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )

    errors = validate_dataset(load_manifest(manifest), tmp_path)

    assert any("duplicate source image sha256" in error for error in errors)
    assert any("a, b" in error for error in errors)
