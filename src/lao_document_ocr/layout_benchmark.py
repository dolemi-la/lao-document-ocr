from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from lao_document_ocr.layout_metrics import evaluate_layout
from lao_document_ocr.models import Document


def load_document_ast(path: str | Path) -> Document:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read document AST: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid document AST JSON: {exc}") from exc
    try:
        return Document.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"Invalid document AST schema: {exc}") from exc


def benchmark_layout(
    reference: Document,
    prediction: Document,
    *,
    iou_threshold: float = 0.5,
) -> dict:
    metrics = evaluate_layout(
        reference,
        prediction,
        iou_threshold=iou_threshold,
    )
    return {
        "schema_version": "1",
        "generated_at": datetime.now(UTC).isoformat(),
        "iou_threshold": iou_threshold,
        "metrics": metrics.to_dict(),
    }


def write_layout_report(report: dict, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
