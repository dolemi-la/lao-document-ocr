from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SUPPORTED_REPORT_KINDS = ("ocr", "recognizer", "layout", "docx")


@dataclass(frozen=True)
class BenchmarkReportEntry:
    kind: str
    file: str
    sha256: str
    schema_version: str
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read benchmark report {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid benchmark JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark report must be a JSON object: {path}")
    return payload


def _float_value(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None


def summarize_report(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    if kind not in SUPPORTED_REPORT_KINDS:
        raise ValueError(f"Unsupported benchmark report kind: {kind}")

    if kind in {"ocr", "recognizer"}:
        overall = payload.get("overall")
        if not isinstance(overall, dict):
            raise ValueError(f"{kind} report is missing an 'overall' object")
        summary = {
            "samples": overall.get("samples"),
            "cer": _float_value(overall, "cer"),
            "wer": _float_value(overall, "wer"),
        }
        if kind == "recognizer":
            model = payload.get("model")
            if isinstance(model, dict):
                summary["model"] = model.get("model_version") or model.get("model")
        return summary

    if kind == "layout":
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("layout report is missing a 'metrics' object")
        return {
            "block_f1": _float_value(metrics, "block_f1"),
            "mean_iou": _float_value(metrics, "mean_iou"),
            "block_type_accuracy": _float_value(metrics, "block_type_accuracy"),
            "reading_order_accuracy": _float_value(metrics, "reading_order_accuracy"),
            "table_cell_f1": _float_value(metrics, "table_cell_f1"),
        }

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("docx report is missing a 'metrics' object")
    return {
        "composite_score": _float_value(metrics, "composite_score"),
        "pixel_similarity": _float_value(metrics, "pixel_similarity"),
        "foreground_iou": _float_value(metrics, "foreground_iou"),
        "edge_f1": _float_value(metrics, "edge_f1"),
        "page_count_score": _float_value(metrics, "page_count_score"),
    }


def build_benchmark_bundle(
    reports: dict[str, str | Path],
    *,
    source_revision: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    if not reports:
        raise ValueError("At least one benchmark report is required")

    entries: list[BenchmarkReportEntry] = []
    for kind in SUPPORTED_REPORT_KINDS:
        raw_path = reports.get(kind)
        if raw_path is None:
            continue
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Benchmark report not found: {path}")
        payload = _load_json(path)
        schema_version = str(payload.get("schema_version", "unknown"))
        entries.append(
            BenchmarkReportEntry(
                kind=kind,
                file=path.name,
                sha256=_sha256_file(path),
                schema_version=schema_version,
                summary=summarize_report(kind, payload),
            )
        )

    unknown = sorted(set(reports) - set(SUPPORTED_REPORT_KINDS))
    if unknown:
        raise ValueError(f"Unsupported benchmark report kinds: {', '.join(unknown)}")

    return {
        "schema_version": "1",
        "generated_at": datetime.now(UTC).isoformat(),
        "label": label,
        "source_revision": source_revision,
        "reports": [entry.to_dict() for entry in entries],
    }


def write_benchmark_bundle(bundle: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
