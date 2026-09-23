from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class BenchmarkComparisonError(ValueError):
    pass


def load_benchmark_report(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BenchmarkComparisonError(
            f"Could not read benchmark report: {source}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkComparisonError(
            f"Invalid benchmark JSON: {source}"
        ) from exc
    if not isinstance(payload, dict):
        raise BenchmarkComparisonError("Benchmark report must be a JSON object")
    if payload.get("schema_version") != "1":
        raise BenchmarkComparisonError("Unsupported benchmark report schema")
    if not isinstance(payload.get("overall"), dict):
        raise BenchmarkComparisonError("Benchmark report is missing overall metrics")
    if not isinstance(payload.get("samples"), list):
        raise BenchmarkComparisonError("Benchmark report is missing sample results")
    return payload


def _sample_signature(report: dict[str, Any]) -> dict[str, tuple[str, tuple[str, ...]]]:
    signature: dict[str, tuple[str, tuple[str, ...]]] = {}
    for item in report["samples"]:
        if not isinstance(item, dict):
            raise BenchmarkComparisonError("Benchmark sample result must be an object")
        sample_id = item.get("id")
        subset = item.get("subset")
        tags = item.get("tags", [])
        if not isinstance(sample_id, str) or not sample_id:
            raise BenchmarkComparisonError("Benchmark sample is missing id")
        if sample_id in signature:
            raise BenchmarkComparisonError(f"Duplicate benchmark sample id: {sample_id}")
        if not isinstance(subset, str):
            raise BenchmarkComparisonError(
                f"Benchmark sample {sample_id} is missing subset"
            )
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise BenchmarkComparisonError(
                f"Benchmark sample {sample_id} has invalid tags"
            )
        signature[sample_id] = (subset, tuple(sorted(tags)))
    return signature


def _metric(section: dict[str, Any], name: str) -> float:
    value = section.get(name)
    if not isinstance(value, (int, float)):
        raise BenchmarkComparisonError(f"Benchmark metric '{name}' is missing")
    return float(value)


def _slice_comparisons(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    kind: str,
    metric: str,
    max_regression: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_slices = baseline.get(kind, {})
    candidate_slices = candidate.get(kind, {})
    if not isinstance(baseline_slices, dict) or not isinstance(candidate_slices, dict):
        raise BenchmarkComparisonError(f"Benchmark report has invalid {kind}")

    names = sorted(set(baseline_slices) & set(candidate_slices))
    comparisons: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []

    for name in names:
        baseline_metrics = baseline_slices[name]
        candidate_metrics = candidate_slices[name]
        if not isinstance(baseline_metrics, dict) or not isinstance(
            candidate_metrics, dict
        ):
            continue
        baseline_value = _metric(baseline_metrics, metric)
        candidate_value = _metric(candidate_metrics, metric)
        delta = candidate_value - baseline_value
        entry = {
            "kind": kind[:-1] if kind.endswith("s") else kind,
            "name": name,
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": delta,
            "baseline_samples": baseline_metrics.get("samples"),
            "candidate_samples": candidate_metrics.get("samples"),
            "passed": delta <= max_regression,
        }
        comparisons.append(entry)
        if not entry["passed"]:
            regressions.append(entry)

    return comparisons, regressions


def _validate_freeze_binding(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
    baseline_freeze = baseline.get("freeze")
    candidate_freeze = candidate.get("freeze")

    if baseline_freeze is None and candidate_freeze is None:
        return None
    if not isinstance(baseline_freeze, dict) or not isinstance(
        candidate_freeze, dict
    ):
        raise BenchmarkComparisonError(
            "Both benchmark reports must include frozen-lock provenance"
        )

    required = ("lock_sha256", "frozen_manifest_sha256")
    for key in required:
        baseline_value = baseline_freeze.get(key)
        candidate_value = candidate_freeze.get(key)
        if not isinstance(baseline_value, str) or not baseline_value:
            raise BenchmarkComparisonError(
                f"Baseline frozen-lock provenance is missing {key}"
            )
        if not isinstance(candidate_value, str) or not candidate_value:
            raise BenchmarkComparisonError(
                f"Candidate frozen-lock provenance is missing {key}"
            )
        if baseline_value != candidate_value:
            raise BenchmarkComparisonError(
                "Benchmark reports use different frozen test sets "
                f"({key} mismatch)"
            )

    return {
        "lock_sha256": baseline_freeze["lock_sha256"],
        "frozen_manifest_sha256": baseline_freeze["frozen_manifest_sha256"],
        "lock_file": baseline_freeze.get("lock_file"),
        "frozen_manifest": baseline_freeze.get("frozen_manifest"),
        "sample_count": baseline_freeze.get("sample_count"),
    }


def compare_benchmark_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    primary_metric: str = "cer",
    min_improvement: float = 0.0,
    max_slice_regression: float = 0.02,
) -> dict[str, Any]:
    if primary_metric not in {"cer", "wer"}:
        raise BenchmarkComparisonError("primary_metric must be 'cer' or 'wer'")
    if min_improvement < 0:
        raise BenchmarkComparisonError("min_improvement must be non-negative")
    if max_slice_regression < 0:
        raise BenchmarkComparisonError("max_slice_regression must be non-negative")

    if baseline.get("split") != candidate.get("split"):
        raise BenchmarkComparisonError("Benchmark reports use different splits")

    freeze_binding = _validate_freeze_binding(baseline, candidate)

    baseline_signature = _sample_signature(baseline)
    candidate_signature = _sample_signature(candidate)
    if baseline_signature != candidate_signature:
        missing = sorted(set(baseline_signature) - set(candidate_signature))
        extra = sorted(set(candidate_signature) - set(baseline_signature))
        relabeled = sorted(
            sample_id
            for sample_id in set(baseline_signature) & set(candidate_signature)
            if baseline_signature[sample_id] != candidate_signature[sample_id]
        )
        details = []
        if missing:
            details.append(f"missing candidate samples: {', '.join(missing)}")
        if extra:
            details.append(f"extra candidate samples: {', '.join(extra)}")
        if relabeled:
            details.append(f"relabeled samples: {', '.join(relabeled)}")
        raise BenchmarkComparisonError(
            "Benchmark reports do not describe the same fixed sample set"
            + (f" ({'; '.join(details)})" if details else "")
        )

    baseline_value = _metric(baseline["overall"], primary_metric)
    candidate_value = _metric(candidate["overall"], primary_metric)
    improvement = baseline_value - candidate_value
    # "Beat baseline" is intentionally strict when min_improvement is zero.
    overall_passed = (
        improvement >= min_improvement
        if min_improvement > 0
        else candidate_value < baseline_value
    )

    subset_comparisons, subset_regressions = _slice_comparisons(
        baseline,
        candidate,
        kind="subsets",
        metric=primary_metric,
        max_regression=max_slice_regression,
    )
    tag_comparisons, tag_regressions = _slice_comparisons(
        baseline,
        candidate,
        kind="tags",
        metric=primary_metric,
        max_regression=max_slice_regression,
    )
    regressions = [*subset_regressions, *tag_regressions]

    return {
        "schema_version": "1",
        "generated_at": datetime.now(UTC).isoformat(),
        "split": baseline.get("split"),
        "freeze": freeze_binding,
        "primary_metric": primary_metric,
        "thresholds": {
            "min_improvement": min_improvement,
            "max_slice_regression": max_slice_regression,
        },
        "baseline": {
            "engine": baseline.get("engine"),
            "reading_order": baseline.get("reading_order"),
            "overall": baseline_value,
        },
        "candidate": {
            "engine": candidate.get("engine"),
            "reading_order": candidate.get("reading_order"),
            "overall": candidate_value,
        },
        "overall": {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "improvement": improvement,
            "passed": overall_passed,
        },
        "sample_count": len(baseline_signature),
        "subsets": subset_comparisons,
        "tags": tag_comparisons,
        "regressions": regressions,
        "passed": overall_passed and not regressions,
    }


def write_comparison_report(report: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
