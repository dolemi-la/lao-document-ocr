from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from lao_document_ocr.capture_pack import load_capture_pack
from lao_document_ocr.capture_registration import CaptureMode
from lao_document_ocr.capture_suite import load_capture_suite
from lao_document_ocr.dataset import DatasetSample

_DEFAULT_REQUIRED_MODES = (
    CaptureMode.FLATBED_SCAN,
    CaptureMode.PHONE_PHOTO,
)


def _sample_capture_modes(sample: DatasetSample) -> set[str]:
    modes = {
        tag.split(":", 1)[1]
        for tag in sample.tags
        if tag.startswith("capture:") and ":" in tag
    }
    if modes:
        return modes

    # Backward-compatible inference for older manifests created before
    # multi-axis tags existed.
    fallback = {
        "clean-print": CaptureMode.FLATBED_SCAN.value,
        "noisy-scan": CaptureMode.DEGRADED_SCAN.value,
        "phone-photo": CaptureMode.PHONE_PHOTO.value,
    }
    mapped = fallback.get(sample.subset.value)
    return {mapped} if mapped else set()


def build_capture_campaign_report(
    suite_manifest: str | Path,
    samples: list[DatasetSample],
    *,
    required_modes: list[CaptureMode] | tuple[CaptureMode, ...] | None = None,
) -> dict:
    suite_path = Path(suite_manifest)
    suite = load_capture_suite(suite_path)
    root = suite_path.parent

    selected_modes = tuple(required_modes or _DEFAULT_REQUIRED_MODES)
    if not selected_modes:
        raise ValueError("At least one required capture mode is needed")
    if len(set(selected_modes)) != len(selected_modes):
        raise ValueError("Required capture modes must be unique")
    required_values = tuple(mode.value for mode in selected_modes)

    expected_pages: dict[str, str] = {}
    for pack in suite.packs:
        pack_path = root / pack.manifest
        _, manifest = load_capture_pack(pack_path)
        for page in manifest.pages:
            if page.id in expected_pages:
                raise ValueError(f"Duplicate page id across capture suite: {page.id}")
            expected_pages[page.id] = pack.template

    sample_modes: dict[str, set[str]] = defaultdict(set)
    registered_samples = 0
    unexpected_samples = 0
    for sample in samples:
        if sample.document_id not in expected_pages:
            unexpected_samples += 1
            continue
        registered_samples += 1
        sample_modes[sample.document_id].update(_sample_capture_modes(sample))

    missing: list[dict] = []
    completed_pairs = 0
    mode_completed: Counter[str] = Counter()
    template_required: Counter[str] = Counter()
    template_completed: Counter[str] = Counter()

    for page_id, template in sorted(expected_pages.items()):
        present = sample_modes.get(page_id, set())
        missing_modes: list[str] = []
        for mode in required_values:
            template_required[template] += 1
            if mode in present:
                completed_pairs += 1
                template_completed[template] += 1
                mode_completed[mode] += 1
            else:
                missing_modes.append(mode)
        if missing_modes:
            missing.append(
                {
                    "page_id": page_id,
                    "template": template,
                    "missing_modes": missing_modes,
                    "present_modes": sorted(present),
                }
            )

    required_pairs = len(expected_pages) * len(required_values)
    completion_ratio = (
        completed_pairs / required_pairs
        if required_pairs
        else 1.0
    )

    template_coverage = {}
    for template in sorted(template_required):
        required = template_required[template]
        completed = template_completed[template]
        template_coverage[template] = {
            "required_captures": required,
            "completed_captures": completed,
            "completion_ratio": completed / required if required else 1.0,
        }

    return {
        "schema_version": "1",
        "generated_at": datetime.now(UTC).isoformat(),
        "suite_id": suite.suite_id,
        "required_modes": list(required_values),
        "expected_pages": len(expected_pages),
        "required_captures": required_pairs,
        "completed_captures": completed_pairs,
        "completion_ratio": completion_ratio,
        "registered_samples_for_suite": registered_samples,
        "unexpected_dataset_samples": unexpected_samples,
        "mode_coverage": {
            mode: {
                "completed_pages": mode_completed[mode],
                "expected_pages": len(expected_pages),
                "completion_ratio": (
                    mode_completed[mode] / len(expected_pages)
                    if expected_pages
                    else 1.0
                ),
            }
            for mode in required_values
        },
        "template_coverage": template_coverage,
        "missing": missing,
    }


def write_capture_campaign_report(report: dict, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
