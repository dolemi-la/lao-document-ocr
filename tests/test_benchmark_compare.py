import pytest

from lao_document_ocr.benchmark_compare import (
    BenchmarkComparisonError,
    compare_benchmark_reports,
)


def _report(
    *,
    cer: float,
    sample_cers: tuple[float, float] = (0.1, 0.2),
    tag_cer: float | None = None,
):
    if tag_cer is None:
        tag_cer = cer
    return {
        "schema_version": "1",
        "split": "test",
        "engine": {"name": "engine"},
        "reading_order": {"name": "DeterministicReadingOrderResolver"},
        "overall": {"cer": cer, "wer": cer * 2, "samples": 2},
        "subsets": {
            "clean-print": {"cer": cer, "wer": cer * 2, "samples": 2},
        },
        "tags": {
            "language:lao": {"cer": tag_cer, "wer": tag_cer * 2, "samples": 2},
        },
        "samples": [
            {
                "id": "a",
                "subset": "clean-print",
                "tags": ["language:lao"],
                "cer": sample_cers[0],
            },
            {
                "id": "b",
                "subset": "clean-print",
                "tags": ["language:lao"],
                "cer": sample_cers[1],
            },
        ],
    }


def test_candidate_passes_when_overall_improves_without_slice_regression() -> None:
    report = compare_benchmark_reports(
        _report(cer=0.20),
        _report(cer=0.15),
    )

    assert report["passed"] is True
    assert report["overall"]["improvement"] == pytest.approx(0.05)
    assert report["regressions"] == []


def test_equal_overall_does_not_count_as_beating_baseline() -> None:
    report = compare_benchmark_reports(
        _report(cer=0.20),
        _report(cer=0.20),
    )

    assert report["overall"]["passed"] is False
    assert report["passed"] is False


def test_slice_regression_can_fail_otherwise_better_candidate() -> None:
    baseline = _report(cer=0.20, tag_cer=0.10)
    candidate = _report(cer=0.15, tag_cer=0.14)

    report = compare_benchmark_reports(
        baseline,
        candidate,
        max_slice_regression=0.02,
    )

    assert report["overall"]["passed"] is True
    assert report["passed"] is False
    assert report["regressions"][0]["name"] == "language:lao"
    assert report["regressions"][0]["delta"] == pytest.approx(0.04)


def test_minimum_improvement_threshold_is_enforced() -> None:
    report = compare_benchmark_reports(
        _report(cer=0.20),
        _report(cer=0.18),
        min_improvement=0.03,
    )

    assert report["overall"]["passed"] is False


def test_mismatched_sample_sets_are_rejected() -> None:
    baseline = _report(cer=0.20)
    candidate = _report(cer=0.15)
    candidate["samples"][1]["id"] = "other"

    with pytest.raises(BenchmarkComparisonError, match="same fixed sample set"):
        compare_benchmark_reports(baseline, candidate)


def test_relabeled_sample_is_rejected() -> None:
    baseline = _report(cer=0.20)
    candidate = _report(cer=0.15)
    candidate["samples"][0]["tags"] = ["capture:phone-photo"]

    with pytest.raises(BenchmarkComparisonError, match="relabeled samples"):
        compare_benchmark_reports(baseline, candidate)


def test_invalid_thresholds_are_rejected() -> None:
    with pytest.raises(BenchmarkComparisonError, match="min_improvement"):
        compare_benchmark_reports(
            _report(cer=0.20),
            _report(cer=0.15),
            min_improvement=-0.01,
        )


def test_matching_frozen_lock_provenance_is_preserved() -> None:
    baseline = _report(cer=0.20)
    candidate = _report(cer=0.15)
    freeze = {
        "lock_file": "test-v1.lock.json",
        "lock_sha256": "a" * 64,
        "frozen_manifest": "test-v1.jsonl",
        "frozen_manifest_sha256": "b" * 64,
        "sample_count": 2,
    }
    baseline["freeze"] = dict(freeze)
    candidate["freeze"] = dict(freeze)

    report = compare_benchmark_reports(baseline, candidate)

    assert report["passed"] is True
    assert report["freeze"]["lock_sha256"] == "a" * 64
    assert report["freeze"]["frozen_manifest_sha256"] == "b" * 64


def test_one_sided_frozen_lock_provenance_is_rejected() -> None:
    baseline = _report(cer=0.20)
    candidate = _report(cer=0.15)
    baseline["freeze"] = {
        "lock_sha256": "a" * 64,
        "frozen_manifest_sha256": "b" * 64,
    }

    with pytest.raises(BenchmarkComparisonError, match="Both benchmark reports"):
        compare_benchmark_reports(baseline, candidate)


def test_mismatched_frozen_lock_provenance_is_rejected() -> None:
    baseline = _report(cer=0.20)
    candidate = _report(cer=0.15)
    baseline["freeze"] = {
        "lock_sha256": "a" * 64,
        "frozen_manifest_sha256": "b" * 64,
    }
    candidate["freeze"] = {
        "lock_sha256": "c" * 64,
        "frozen_manifest_sha256": "b" * 64,
    }

    with pytest.raises(BenchmarkComparisonError, match="different frozen test sets"):
        compare_benchmark_reports(baseline, candidate)
