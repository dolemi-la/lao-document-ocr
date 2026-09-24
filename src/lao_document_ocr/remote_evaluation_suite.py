from __future__ import annotations

import json
import platform
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lao_document_ocr.ocr.base import OcrEngine
from lao_document_ocr.reading_order import ReadingOrderResolver
from lao_document_ocr.remote_evaluation import (
    RemoteEvaluationError,
    RemoteFetcher,
    download_remote_source,
    evaluate_remote_sources,
    load_remote_registry_sources,
)


@dataclass(frozen=True)
class RemoteSuiteEntry:
    source_id: str
    pages: tuple[int, ...]
    expected_text_layer: str
    rotation_probe: bool = False
    max_source_mb: float | None = None
    note: str | None = None


@dataclass(frozen=True)
class RemoteDiagnosticSuite:
    schema_version: str
    suite_id: str
    description: str
    entries: tuple[RemoteSuiteEntry, ...]
    max_source_mb: float = 25.0
    timeout_seconds: float = 20.0
    max_document_pages: int = 500
    max_page_pixels: int = 40_000_000


def load_remote_diagnostic_suite(path: str | Path) -> RemoteDiagnosticSuite:
    suite_path = Path(path)
    try:
        payload = json.loads(suite_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RemoteEvaluationError(f"Could not read remote diagnostic suite: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RemoteEvaluationError(f"Invalid remote diagnostic suite JSON: {exc}") from exc

    if payload.get("schema_version") != "1":
        raise RemoteEvaluationError("Unsupported remote diagnostic suite schema version.")

    suite_id = str(payload.get("id", "")).strip()
    if not suite_id:
        raise RemoteEvaluationError("Remote diagnostic suite requires a non-empty id.")

    description = str(payload.get("description", "")).strip()
    defaults = payload.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise RemoteEvaluationError("Remote diagnostic suite defaults must be an object.")

    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise RemoteEvaluationError("Remote diagnostic suite contains no sources.")

    entries: list[RemoteSuiteEntry] = []
    seen: set[str] = set()
    for raw in raw_sources:
        if not isinstance(raw, dict):
            raise RemoteEvaluationError("Remote diagnostic suite source entries must be objects.")
        source_id = str(raw.get("id", "")).strip()
        if not source_id:
            raise RemoteEvaluationError("Remote diagnostic suite source requires an id.")
        if source_id in seen:
            raise RemoteEvaluationError(
                f"Remote diagnostic suite contains duplicate source id: {source_id}"
            )
        seen.add(source_id)

        raw_pages = raw.get("pages")
        if not isinstance(raw_pages, list) or not raw_pages:
            raise RemoteEvaluationError(
                f"Remote diagnostic suite source requires explicit pages: {source_id}"
            )
        pages: list[int] = []
        for value in raw_pages:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise RemoteEvaluationError(
                    f"Remote diagnostic suite has invalid page for {source_id}: {value!r}"
                )
            if value not in pages:
                pages.append(value)

        expected_text_layer = str(raw.get("expected_text_layer", "")).strip()
        if not expected_text_layer:
            raise RemoteEvaluationError(
                f"Remote diagnostic suite source requires expected_text_layer: {source_id}"
            )

        rotation_probe = raw.get("rotation_probe", False)
        if not isinstance(rotation_probe, bool):
            raise RemoteEvaluationError(
                f"Remote diagnostic suite rotation_probe must be boolean: {source_id}"
            )

        max_source_mb = raw.get("max_source_mb")
        if max_source_mb is not None:
            if isinstance(max_source_mb, bool) or not isinstance(max_source_mb, (int, float)):
                raise RemoteEvaluationError(
                    f"Remote diagnostic suite max_source_mb must be numeric: {source_id}"
                )
            max_source_mb = float(max_source_mb)
            if max_source_mb <= 0:
                raise RemoteEvaluationError(
                    f"Remote diagnostic suite max_source_mb must be positive: {source_id}"
                )

        note = raw.get("note")
        entries.append(
            RemoteSuiteEntry(
                source_id=source_id,
                pages=tuple(pages),
                expected_text_layer=expected_text_layer,
                rotation_probe=rotation_probe,
                max_source_mb=max_source_mb,
                note=str(note).strip() if note is not None else None,
            )
        )

    max_source_mb = float(defaults.get("max_source_mb", 25.0))
    timeout_seconds = float(defaults.get("timeout_seconds", 20.0))
    max_document_pages = int(defaults.get("max_document_pages", 500))
    max_page_pixels = int(defaults.get("max_page_pixels", 40_000_000))

    if max_source_mb <= 0:
        raise RemoteEvaluationError("Suite default max_source_mb must be positive.")
    if timeout_seconds <= 0:
        raise RemoteEvaluationError("Suite default timeout_seconds must be positive.")
    if max_document_pages < 1:
        raise RemoteEvaluationError("Suite default max_document_pages must be at least 1.")
    if max_page_pixels < 1:
        raise RemoteEvaluationError("Suite default max_page_pixels must be at least 1.")

    return RemoteDiagnosticSuite(
        schema_version="1",
        suite_id=suite_id,
        description=description,
        entries=tuple(entries),
        max_source_mb=max_source_mb,
        timeout_seconds=timeout_seconds,
        max_document_pages=max_document_pages,
        max_page_pixels=max_page_pixels,
    )


def validate_suite_against_registry(
    suite: RemoteDiagnosticSuite,
    registry_path: str | Path,
) -> None:
    for entry in suite.entries:
        sources = load_remote_registry_sources(
            registry_path,
            source_ids=[entry.source_id],
        )
        source = sources[0]
        status = str(source.get("status", ""))
        if status != "remote-evaluation-candidate-not-approved":
            raise RemoteEvaluationError(
                "Default diagnostic suites may only include verified "
                f"remote-evaluation-candidate-not-approved sources: {entry.source_id}"
            )

        evidence = source.get("evidence")
        if not isinstance(evidence, dict):
            raise RemoteEvaluationError(
                f"Suite source has no evidence object: {entry.source_id}"
            )
        text_layer = str(evidence.get("text_layer", ""))
        if text_layer != entry.expected_text_layer:
            raise RemoteEvaluationError(
                "Suite source text-layer classification changed: "
                f"{entry.source_id} expected {entry.expected_text_layer!r}, "
                f"registry has {text_layer!r}"
            )

        page_count = evidence.get("pages")
        if not isinstance(page_count, int) or page_count < 1:
            raise RemoteEvaluationError(
                f"Suite source must have a verified page count: {entry.source_id}"
            )
        invalid = [page for page in entry.pages if page > page_count]
        if invalid:
            raise RemoteEvaluationError(
                f"Suite source page is outside verified range 1..{page_count}: "
                f"{entry.source_id} page {invalid[0]}"
            )


def _aggregate_suite_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    layer_gaps: Counter[str] = Counter()
    registry_layers: Counter[str] = Counter()
    source_statuses: Counter[str] = Counter()
    confidence_bands: Counter[str] = Counter()
    page_media: Counter[str] = Counter()
    rotation_recommendations: Counter[str] = Counter()
    applied_orientations: Counter[str] = Counter()
    auto_orientation_probe_states: Counter[str] = Counter()
    page_count = 0
    ocr_lao_characters = 0
    native_lao_characters = 0
    elapsed_seconds = 0.0

    for item in results:
        source_statuses[str(item.get("status", "unknown"))] += 1
        registry_layers[str(item.get("registry_text_layer") or "unknown")] += 1
        document = item.get("document")
        if not isinstance(document, dict):
            continue
        pages = document.get("pages")
        if not isinstance(pages, list):
            continue
        page_count += len(pages)
        for page in pages:
            if not isinstance(page, dict):
                continue
            layer_gap = page.get("layer_gap")
            if isinstance(layer_gap, dict):
                layer_gaps[str(layer_gap.get("classification", "unknown"))] += 1
            ocr_quality = page.get("ocr_quality")
            if isinstance(ocr_quality, dict):
                confidence_bands[str(ocr_quality.get("band", "unknown"))] += 1
            media = page.get("page_media")
            if isinstance(media, dict):
                page_media[str(media.get("classification", "unknown"))] += 1
            rotation_probe = page.get("rotation_probe")
            if isinstance(rotation_probe, dict):
                recommended = rotation_probe.get("recommended_degrees_clockwise")
                key = "none" if recommended is None else str(int(recommended))
                rotation_recommendations[key] += 1
            auto_orientation = page.get("auto_orientation")
            if isinstance(auto_orientation, dict):
                applied = int(auto_orientation.get("degrees_clockwise", 0))
                applied_orientations[str(applied)] += 1
                diagnostics = auto_orientation.get("diagnostics")
                if isinstance(diagnostics, dict):
                    if diagnostics.get("probe_skipped"):
                        reason = str(
                            diagnostics.get("probe_skip_reason") or "skipped"
                        )
                    else:
                        reason = "probed"
                    auto_orientation_probe_states[reason] += 1
            native = page.get("native_text")
            if isinstance(native, dict):
                native_lao_characters += int(native.get("lao_characters", 0))
            ocr = page.get("ocr")
            if isinstance(ocr, dict):
                text_stats = ocr.get("text")
                if isinstance(text_stats, dict):
                    ocr_lao_characters += int(text_stats.get("lao_characters", 0))
            elapsed_seconds += float(page.get("elapsed_seconds", 0.0))

    ok = source_statuses.get("ok", 0)
    total = len(results)
    return {
        "sources": total,
        "ok": ok,
        "errors": total - ok,
        "sampled_pages": page_count,
        "native_lao_characters": native_lao_characters,
        "ocr_lao_characters": ocr_lao_characters,
        "ocr_minus_native_lao_characters": (
            ocr_lao_characters - native_lao_characters
        ),
        "ocr_elapsed_seconds": elapsed_seconds,
        "layer_gap_classifications": dict(sorted(layer_gaps.items())),
        "ocr_confidence_bands": dict(sorted(confidence_bands.items())),
        "page_media_classifications": dict(sorted(page_media.items())),
        "rotation_recommendations": dict(sorted(rotation_recommendations.items())),
        "applied_auto_orientations": dict(sorted(applied_orientations.items())),
        "auto_orientation_probe_states": dict(
            sorted(auto_orientation_probe_states.items())
        ),
        "registry_text_layers": dict(sorted(registry_layers.items())),
        "source_statuses": dict(sorted(source_statuses.items())),
    }


def evaluate_remote_diagnostic_suite(
    suite_path: str | Path,
    registry_path: str | Path,
    *,
    engine: OcrEngine,
    reading_order_resolver: ReadingOrderResolver | None = None,
    auto_orient_right_angles: bool = False,
    enable_rotation_probes: bool = True,
    fetcher: RemoteFetcher = download_remote_source,
) -> dict[str, Any]:
    suite = load_remote_diagnostic_suite(suite_path)
    validate_suite_against_registry(suite, registry_path)

    source_results: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []

    for entry in suite.entries:
        max_source_mb = (
            entry.max_source_mb
            if entry.max_source_mb is not None
            else suite.max_source_mb
        )
        report = evaluate_remote_sources(
            registry_path,
            engine=engine,
            source_ids=[entry.source_id],
            requested_pages=entry.pages,
            max_pages_per_source=len(entry.pages),
            max_source_bytes=max(1, int(max_source_mb * 1024 * 1024)),
            max_document_pages=suite.max_document_pages,
            max_page_pixels=suite.max_page_pixels,
            timeout_seconds=suite.timeout_seconds,
            reading_order_resolver=reading_order_resolver,
            probe_right_angle_rotations=(
                enable_rotation_probes
                and entry.rotation_probe
                and not auto_orient_right_angles
            ),
            auto_orient_right_angles=auto_orient_right_angles,
            fetcher=fetcher,
        )
        result = report["sources"][0]
        result["suite_pages"] = list(entry.pages)
        if entry.note:
            result["suite_note"] = entry.note
        source_results.append(result)
        selections.append(
            {
                "id": entry.source_id,
                "pages": list(entry.pages),
                "expected_text_layer": entry.expected_text_layer,
                "rotation_probe": entry.rotation_probe,
                "max_source_mb": max_source_mb,
            }
        )

    return {
        "schema_version": "1",
        "report_type": "remote-source-diagnostic-suite",
        "not_benchmark_accuracy": True,
        "generated_at": datetime.now(UTC).isoformat(),
        "suite": {
            "id": suite.suite_id,
            "description": suite.description,
            "path": str(Path(suite_path)),
        },
        "engine": engine.metadata(),
        "reading_order": (
            reading_order_resolver.metadata()
            if reading_order_resolver is not None
            else {"name": "DeterministicReadingOrderResolver"}
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "selection": {
            "sources": selections,
            "timeout_seconds": suite.timeout_seconds,
            "max_document_pages": suite.max_document_pages,
            "max_page_pixels": suite.max_page_pixels,
            "auto_orient_right_angles": auto_orient_right_angles,
            "rotation_probes_enabled": enable_rotation_probes,
        },
        "summary": _aggregate_suite_results(source_results),
        "sources": source_results,
    }
