import hashlib

from PIL import Image

from lao_document_ocr.benchmark_readiness import build_benchmark_readiness_report
from lao_document_ocr.dataset import DatasetSample, DatasetSplit, DatasetSubset


def _sample(tmp_path, sample_id, subset, tags, *, document_id=None, layout=False):
    source = tmp_path / f"{sample_id}.png"
    truth = tmp_path / f"{sample_id}.txt"
    Image.new(
        "RGB",
        (80, 40),
        (sum(ord(char) for char in sample_id) % 255, 250, 240),
    ).save(source)
    truth.write_text(f"truth {sample_id}", encoding="utf-8")
    layout_path = None
    if layout:
        layout_path = tmp_path / f"{sample_id}.layout.json"
        layout_path.write_text(
            '{"version":"1","pages":[]}',
            encoding="utf-8",
        )
    return DatasetSample(
        id=sample_id,
        document_id=document_id or f"doc-{sample_id}",
        split=DatasetSplit.TEST,
        subset=subset,
        source=source.name,
        ground_truth=truth.name,
        layout_ground_truth=(layout_path.name if layout_path else None),
        license="CC0-1.0",
        provenance="readiness unit test",
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        tags=tags,
    )


def _complete_samples(tmp_path):
    return [
        _sample(tmp_path, "clean", DatasetSubset.CLEAN_PRINT, ["capture:flatbed-scan"]),
        _sample(tmp_path, "noisy", DatasetSubset.NOISY_SCAN, ["capture:degraded-scan"]),
        _sample(tmp_path, "phone", DatasetSubset.PHONE_PHOTO, ["capture:phone-photo"]),
        _sample(tmp_path, "mixed", DatasetSubset.CLEAN_PRINT, ["language:mixed"]),
        _sample(tmp_path, "columns", DatasetSubset.CLEAN_PRINT, ["layout:multi-column"]),
        _sample(tmp_path, "ruled", DatasetSubset.CLEAN_PRINT, ["table:ruled"]),
        _sample(tmp_path, "borderless", DatasetSubset.CLEAN_PRINT, ["table:borderless"]),
        _sample(tmp_path, "receipt", DatasetSubset.CLEAN_PRINT, ["document:receipt"]),
        _sample(tmp_path, "form", DatasetSubset.CLEAN_PRINT, ["document:form"]),
    ]


def test_complete_dimension_coverage_is_ready(tmp_path) -> None:
    report = build_benchmark_readiness_report(
        _complete_samples(tmp_path),
        tmp_path,
        min_documents_per_dimension=1,
        min_total_documents=9,
    )

    assert report["ready"] is True
    assert report["missing_dimensions"] == []
    assert report["dimensions"]["multi-column"]["document_count"] == 1
    assert report["dimensions"]["complex-table"]["document_count"] == 1


def test_tags_satisfy_dimensions_even_when_primary_subset_is_capture_type(tmp_path) -> None:
    sample = _sample(
        tmp_path,
        "phone-form",
        DatasetSubset.PHONE_PHOTO,
        ["capture:phone-photo", "document:form", "language:mixed"],
    )
    report = build_benchmark_readiness_report([sample], tmp_path)

    assert report["dimensions"]["phone-photo"]["passed"] is True
    assert report["dimensions"]["form"]["passed"] is True
    assert report["dimensions"]["mixed-lao-english"]["passed"] is True
    assert report["ready"] is False


def test_missing_dimension_and_minimum_document_target_fail_gate(tmp_path) -> None:
    samples = _complete_samples(tmp_path)
    samples = [
        sample
        for sample in samples
        if "document:receipt" not in sample.tags
    ]
    report = build_benchmark_readiness_report(
        samples,
        tmp_path,
        min_documents_per_dimension=1,
        min_total_documents=20,
    )

    assert report["ready"] is False
    assert "receipt" in report["missing_dimensions"]
    assert report["total_documents_passed"] is False


def test_layout_label_target_can_be_required(tmp_path) -> None:
    samples = _complete_samples(tmp_path)
    samples.append(
        _sample(
            tmp_path,
            "layout-labeled",
            DatasetSubset.CLEAN_PRINT,
            ["layout:multi-column"],
            layout=True,
        )
    )

    # Dataset validation sees this intentionally minimal label as invalid, so the
    # readiness gate must not claim ready simply because a label path exists.
    report = build_benchmark_readiness_report(
        samples,
        tmp_path,
        min_layout_labeled_documents=1,
    )

    assert report["layout_labeled_document_count"] == 1
    assert report["layout_labeled_documents_passed"] is True
    assert report["validation"]["ok"] is False
    assert report["ready"] is False
