"""Whole-line metrics and aggregate-only reports must not disclose document text."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lao_document_ocr.confidence_calibration import fit_from_recognizer_report
from lao_document_ocr.recognizer_benchmark import (
    RecognitionMetricCounts,
    benchmark_recognizer,
    write_recognizer_report,
)
from lao_document_ocr.training_manifest import TrainingSample


def _samples(pairs):
    return [
        TrainingSample(f"PRIVATE-ID-{i}", Path(f"PRIVATE-IMAGE-{i}.png"), ref)
        for i, (ref, _) in enumerate(pairs)
    ]


def _recognizer(pairs):
    outputs = iter(hyp for _, hyp in pairs)
    return SimpleNamespace(
        recognize=lambda _: SimpleNamespace(text=next(outputs), confidence=0.8),
        metadata={"private": "PRIVATE-METADATA", "vocabulary": ["PRIVATE-LABEL"]},
    )


@pytest.mark.parametrize(
    "pairs,exact,empty",
    [
        ([("a b", "  a   b  "), ("é", "e\u0301"), ("ລາວ", "ລາວ")], 3, 0),
        ([("abc", "ab"), ("abc", ""), ("abc", " \t\u00a0")], 0, 2),
        ([("abc", "abc"), ("abc", "abd"), ("abc", "abcdef")], 1, 0),
        ([("", ""), ("  ", "a")], 1, 1),
    ],
)
def test_line_counts_use_normalized_equality(pairs, exact, empty):
    report = benchmark_recognizer(_samples(pairs), _recognizer(pairs))
    counts = report["overall"]
    assert counts["exact_lines"] == exact
    assert counts["empty_predictions"] == empty
    assert counts["exact_line_ratio"] == exact / len(pairs)
    assert counts["empty_prediction_ratio"] == empty / len(pairs)
    assert sum(row["exact_match"] for row in report["samples"]) == exact
    assert sum(row["empty_prediction"] for row in report["samples"]) == empty


def test_empty_accumulator_does_not_invent_line_accuracy():
    counts = RecognitionMetricCounts().to_dict()
    assert counts["exact_lines"] == counts["empty_predictions"] == 0
    assert counts["exact_line_ratio"] is None
    assert counts["empty_prediction_ratio"] is None


def test_nonexact_line_is_not_hidden_by_low_cer():
    reference = "a" * 100
    report = benchmark_recognizer(
        _samples([(reference, reference[:-1])]), _recognizer([(reference, reference[:-1])])
    )
    assert report["overall"]["cer"] == 0.01
    assert report["overall"]["exact_line_ratio"] == 0.0


def test_summary_metrics_match_detailed_and_round_trip(tmp_path):
    pairs = [("exact", "exact"), ("two words", "two word"), ("private", " ")]
    detailed = benchmark_recognizer(_samples(pairs), _recognizer(pairs))
    summary = benchmark_recognizer(_samples(pairs), _recognizer(pairs), summary_only=True)
    assert summary["overall"] == detailed["overall"]
    assert summary["metric_version"] == detailed["metric_version"] == "normalized-line-metrics-v1"
    assert set(summary) == {
        "schema_version",
        "report_type",
        "metric_version",
        "elapsed_seconds",
        "overall",
    }
    assert summary["report_type"] == "recognizer-aggregate-only"
    assert "samples" not in summary and "model" not in summary
    path = write_recognizer_report(summary, tmp_path / "summary.json")
    assert json.loads(path.read_text()) == summary
    payload = json.dumps(summary, allow_nan=False)
    for sensitive in (
        "PRIVATE-ID",
        "PRIVATE-IMAGE",
        "PRIVATE-METADATA",
        "PRIVATE-LABEL",
        '"reference"',
        '"hypothesis"',
        "two words",
    ):
        assert sensitive not in payload


def test_summary_does_not_access_untrusted_metadata_or_confidence():
    class Result:
        text = "PRIVATE-LABEL"

        @property
        def confidence(self):
            raise AssertionError("Summary must not inspect per-sample confidence")

        @property
        def calibrated_confidence(self):
            raise AssertionError("Summary must not inspect calibration")

    class Recognizer:
        @property
        def metadata(self):
            raise AssertionError("Summary must not inspect arbitrary model metadata")

        def recognize(self, _):
            return Result()

    report = benchmark_recognizer(
        _samples([("PRIVATE-LABEL", "PRIVATE-LABEL")]), Recognizer(), summary_only=True
    )
    assert report["overall"]["exact_lines"] == 1
    assert "PRIVATE-LABEL" not in json.dumps(report)


def test_summary_cannot_be_mistaken_for_calibration_data():
    pairs = [("a", "b"), ("a", "a")]
    report = benchmark_recognizer(_samples(pairs), _recognizer(pairs), summary_only=True)
    with pytest.raises(ValueError, match="fewer than two usable calibration samples"):
        fit_from_recognizer_report(report)


def test_detailed_default_keeps_calibration_fields():
    pairs = [("a", "b"), ("a", "a")]
    report = benchmark_recognizer(_samples(pairs), _recognizer(pairs))
    assert report["model"]["private"] == "PRIVATE-METADATA"
    assert report["samples"][0]["reference"] == "a"
    assert report["samples"][0]["hypothesis"] == "b"
    assert report["samples"][0]["uncalibrated_confidence"] == 0.8
    assert fit_from_recognizer_report(report).sample_count == 2


@pytest.mark.parametrize("summary_only", [False, True])
def test_empty_benchmark_still_fails(summary_only):
    with pytest.raises(ValueError, match="No recognizer benchmark samples"):
        benchmark_recognizer([], _recognizer([]), summary_only=summary_only)


@pytest.mark.parametrize("summary_only", [False, True])
def test_cli_line_report_mode(tmp_path, monkeypatch, capsys, summary_only):
    import lao_document_ocr.cli as cli
    import lao_document_ocr.recognizer_inference as inference

    pairs = [("PRIVATE-LABEL", "PRIVATE-LABEL"), ("abc", " ")]
    monkeypatch.setattr(cli, "load_training_manifest", lambda *a, **kw: _samples(pairs))
    monkeypatch.setattr(inference, "ExportedLineRecognizer", lambda *a, **kw: _recognizer(pairs))
    output = tmp_path / "report.json"
    args = [
        "lao-ocr",
        "benchmark-recognizer",
        "--manifest",
        str(tmp_path / "data.jsonl"),
        "--model",
        str(tmp_path / "model.pt2"),
        "--output",
        str(output),
    ]
    if summary_only:
        args.append("--summary-only")
    monkeypatch.setattr(sys, "argv", args)
    assert cli.main() == 0
    report = json.loads(output.read_text())
    assert ("samples" not in report) is summary_only
    assert report["overall"]["exact_lines"] == 1
    assert report["overall"]["empty_predictions"] == 1
    stdout = capsys.readouterr().out
    assert "Exact lines: 1/2" in stdout
    assert "Normalized-empty predictions: 1/2" in stdout
    assert "PRIVATE-LABEL" not in stdout
