import json

import pytest

from lao_document_ocr.benchmark_bundle import (
    build_benchmark_bundle,
    write_benchmark_bundle,
)


def _write(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_builds_bundle_with_summaries_and_hashes(tmp_path) -> None:
    ocr = tmp_path / "ocr.json"
    layout = tmp_path / "layout.json"
    docx = tmp_path / "docx.json"
    _write(
        ocr,
        {
            "schema_version": "1",
            "overall": {"samples": 10, "cer": 0.05, "wer": 0.2},
        },
    )
    _write(
        layout,
        {
            "schema_version": "1",
            "metrics": {
                "block_f1": 0.9,
                "mean_iou": 0.8,
                "block_type_accuracy": 0.85,
                "reading_order_accuracy": 0.95,
                "table_cell_f1": 0.75,
            },
        },
    )
    _write(
        docx,
        {
            "schema_version": "1",
            "metrics": {
                "composite_score": 0.82,
                "pixel_similarity": 0.9,
                "foreground_iou": 0.75,
                "edge_f1": 0.7,
                "page_count_score": 1.0,
            },
        },
    )

    bundle = build_benchmark_bundle(
        {"ocr": ocr, "layout": layout, "docx": docx},
        source_revision="abc123",
        label="baseline-v1",
    )

    assert bundle["source_revision"] == "abc123"
    assert bundle["label"] == "baseline-v1"
    assert [entry["kind"] for entry in bundle["reports"]] == [
        "ocr",
        "layout",
        "docx",
    ]
    assert bundle["reports"][0]["summary"]["cer"] == 0.05
    assert bundle["reports"][1]["summary"]["block_f1"] == 0.9
    assert bundle["reports"][2]["summary"]["composite_score"] == 0.82
    assert all(len(entry["sha256"]) == 64 for entry in bundle["reports"])


def test_recognizer_summary_includes_model_version(tmp_path) -> None:
    report = tmp_path / "recognizer.json"
    _write(
        report,
        {
            "schema_version": "1",
            "model": {"model_version": "crnn-ctc-v2"},
            "overall": {"samples": 4, "cer": 0.3, "wer": 0.5},
        },
    )

    bundle = build_benchmark_bundle({"recognizer": report})

    summary = bundle["reports"][0]["summary"]
    assert summary["model"] == "crnn-ctc-v2"
    assert summary["samples"] == 4


def test_bundle_round_trip(tmp_path) -> None:
    report = tmp_path / "ocr.json"
    _write(
        report,
        {
            "schema_version": "1",
            "overall": {"samples": 1, "cer": 0.0, "wer": 0.0},
        },
    )
    bundle = build_benchmark_bundle({"ocr": report})
    output = write_benchmark_bundle(bundle, tmp_path / "bundle.json")

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1"
    assert payload["reports"][0]["file"] == "ocr.json"


def test_rejects_unknown_kind(tmp_path) -> None:
    report = tmp_path / "x.json"
    _write(report, {"schema_version": "1"})

    with pytest.raises(ValueError, match="Unsupported benchmark report kinds"):
        build_benchmark_bundle({"unknown": report})


def test_rejects_malformed_ocr_report(tmp_path) -> None:
    report = tmp_path / "ocr.json"
    _write(report, {"schema_version": "1"})

    with pytest.raises(ValueError, match="overall"):
        build_benchmark_bundle({"ocr": report})
