import json

from PIL import Image

from lao_document_ocr.benchmark_freeze import (
    freeze_benchmark,
    verify_benchmark_freeze,
)
from lao_document_ocr.dataset import (
    DatasetSample,
    DatasetSplit,
    DatasetSubset,
)


def _sample(tmp_path, sample_id: str, split: DatasetSplit) -> DatasetSample:
    source = tmp_path / f"{sample_id}.png"
    truth = tmp_path / f"{sample_id}.txt"
    Image.new("RGB", (80, 40), (sum(ord(ch) for ch in sample_id) % 255, 255, 255)).save(source)
    truth.write_text(f"truth {sample_id}", encoding="utf-8")
    return DatasetSample(
        id=sample_id,
        document_id=f"doc-{sample_id}",
        split=split,
        subset=DatasetSubset.CLEAN_PRINT,
        source=source.name,
        ground_truth=truth.name,
        license="CC0-1.0",
        provenance="freeze unit test",
        tags=["language:lao"],
    )


def test_freeze_writes_sorted_test_manifest_and_lock(tmp_path) -> None:
    samples = [
        _sample(tmp_path, "b", DatasetSplit.TEST),
        _sample(tmp_path, "train", DatasetSplit.TRAIN),
        _sample(tmp_path, "a", DatasetSplit.TEST),
    ]
    source_manifest = tmp_path / "source.jsonl"
    source_manifest.write_text("source manifest\n", encoding="utf-8")

    manifest, lock = freeze_benchmark(
        samples,
        tmp_path,
        output_manifest=tmp_path / "frozen" / "manifest.jsonl",
        output_lock=tmp_path / "frozen" / "benchmark.lock.json",
        source_manifest=source_manifest,
        source_revision="abc123",
    )

    entries = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    payload = json.loads(lock.read_text(encoding="utf-8"))

    assert [entry["id"] for entry in entries] == ["a", "b"]
    assert payload["sample_count"] == 2
    assert payload["document_count"] == 2
    assert payload["source_revision"] == "abc123"
    assert len(payload["frozen_manifest_sha256"]) == 64
    assert len(payload["samples"][0]["source_sha256"]) == 64
    assert len(payload["samples"][0]["ground_truth_sha256"]) == 64
    assert verify_benchmark_freeze(lock, tmp_path) == []


def test_verify_detects_ground_truth_change(tmp_path) -> None:
    sample = _sample(tmp_path, "a", DatasetSplit.TEST)
    _, lock = freeze_benchmark(
        [sample],
        tmp_path,
        output_manifest=tmp_path / "manifest.jsonl",
        output_lock=tmp_path / "lock.json",
    )

    (tmp_path / "a.txt").write_text("changed", encoding="utf-8")
    errors = verify_benchmark_freeze(lock, tmp_path)

    assert any("ground_truth sha256 mismatch" in error for error in errors)


def test_verify_detects_frozen_manifest_change(tmp_path) -> None:
    sample = _sample(tmp_path, "a", DatasetSplit.TEST)
    manifest, lock = freeze_benchmark(
        [sample],
        tmp_path,
        output_manifest=tmp_path / "manifest.jsonl",
        output_lock=tmp_path / "lock.json",
    )

    manifest.write_text("# modified\n", encoding="utf-8")
    errors = verify_benchmark_freeze(lock, tmp_path)

    assert any("frozen manifest sha256 mismatch" in error for error in errors)


def test_public_freeze_requires_real_source_and_review(tmp_path) -> None:
    import pytest

    from lao_document_ocr.benchmark_freeze import BenchmarkFreezeError

    sample = _sample(tmp_path, "a", DatasetSplit.TEST)
    with pytest.raises(BenchmarkFreezeError, match="real-source"):
        freeze_benchmark(
            [sample],
            tmp_path,
            output_manifest=tmp_path / "manifest.jsonl",
            output_lock=tmp_path / "lock.json",
            require_real_sources=True,
            require_manual_review=True,
        )


def test_public_freeze_accepts_reviewed_real_capture(tmp_path) -> None:
    from datetime import UTC, datetime

    from lao_document_ocr.dataset import DatasetReview, DatasetReviewStatus

    sample = _sample(tmp_path, "a", DatasetSplit.TEST)
    sample.tags.extend(["source:real-capture", "capture:optical-evidence"])
    sample.review = DatasetReview(
        status=DatasetReviewStatus.APPROVED,
        reviewer="Reviewer",
        reviewed_at=datetime(2026, 9, 23, 8, 0, tzinfo=UTC),
    )
    _, lock = freeze_benchmark(
        [sample],
        tmp_path,
        output_manifest=tmp_path / "manifest.jsonl",
        output_lock=tmp_path / "lock.json",
        require_real_sources=True,
        require_manual_review=True,
    )
    payload = json.loads(lock.read_text(encoding="utf-8"))
    assert payload["samples"][0]["review"]["status"] == "approved"


def test_freeze_identity_hashes_lock_file(tmp_path) -> None:
    import hashlib

    from lao_document_ocr.benchmark_freeze import benchmark_freeze_identity

    sample = _sample(tmp_path, "identity", DatasetSplit.TEST)
    manifest, lock = freeze_benchmark(
        [sample],
        tmp_path,
        output_manifest=tmp_path / "frozen.jsonl",
        output_lock=tmp_path / "freeze.lock.json",
        source_revision="abc123",
    )

    identity = benchmark_freeze_identity(lock)

    assert identity["lock_file"] == "freeze.lock.json"
    assert identity["lock_sha256"] == hashlib.sha256(lock.read_bytes()).hexdigest()
    assert identity["frozen_manifest"] == manifest.name
    assert identity["frozen_manifest_sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    assert identity["sample_count"] == 1
    assert identity["source_revision"] == "abc123"
