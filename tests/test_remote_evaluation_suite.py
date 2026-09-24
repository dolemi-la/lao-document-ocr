from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from lao_document_ocr.models import BoundingBox
from lao_document_ocr.ocr.base import OcrEngine, RecognizedLine
from lao_document_ocr.remote_evaluation import DownloadedRemote, RemoteEvaluationError
from lao_document_ocr.remote_evaluation_suite import (
    evaluate_remote_diagnostic_suite,
    load_remote_diagnostic_suite,
    validate_suite_against_registry,
)


class FixedEngine(OcrEngine):
    def is_available(self) -> bool:
        return True

    def metadata(self):
        return {"name": "FixedEngine"}

    def recognize(self, image: Image.Image) -> list[RecognizedLine]:
        return [
            RecognizedLine(
                text="ທົດສອບ ລາວ",
                bbox=BoundingBox(x=10, y=10, width=120, height=20),
                confidence=0.8,
                block_id=1,
                paragraph_id=1,
                line_id=1,
            )
        ]


def _registry(
    tmp_path: Path,
    *,
    second_status: str = "remote-evaluation-candidate-not-approved",
) -> Path:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "sources": [
                    {
                        "id": "scan-a",
                        "name": "Scan A",
                        "url": "https://example.com/a.pdf",
                        "status": "remote-evaluation-candidate-not-approved",
                        "evidence": {"pages": 2, "text_layer": "absent"},
                    },
                    {
                        "id": "scan-b",
                        "name": "Scan B",
                        "url": "https://example.com/b.pdf",
                        "status": second_status,
                        "evidence": {"pages": 3, "text_layer": "scanner-watermark-only"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _suite(tmp_path: Path) -> Path:
    path = tmp_path / "suite.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "id": "suite-v1",
                "description": "unit test suite",
                "defaults": {
                    "max_source_mb": 5,
                    "timeout_seconds": 4,
                    "max_document_pages": 10,
                    "max_page_pixels": 1000000,
                },
                "sources": [
                    {
                        "id": "scan-a",
                        "pages": [1],
                        "expected_text_layer": "absent",
                        "max_source_mb": 2,
                        "note": "empty layer",
                    },
                    {
                        "id": "scan-b",
                        "pages": [1, 3],
                        "expected_text_layer": "scanner-watermark-only",
                        "rotation_probe": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _pdf(path: Path, pages: int) -> Path:
    pdf = pymupdf.open()
    for index in range(pages):
        page = pdf.new_page(width=240, height=160)
        if index == 1:
            page.insert_text((20, 40), "native text", fontsize=12)
    pdf.save(path)
    pdf.close()
    return path


def test_load_remote_diagnostic_suite_parses_defaults_and_entries(tmp_path) -> None:
    suite = load_remote_diagnostic_suite(_suite(tmp_path))

    assert suite.suite_id == "suite-v1"
    assert suite.max_source_mb == 5
    assert suite.timeout_seconds == 4
    assert suite.entries[0].source_id == "scan-a"
    assert suite.entries[0].pages == (1,)
    assert suite.entries[0].expected_text_layer == "absent"
    assert suite.entries[0].max_source_mb == 2
    assert suite.entries[1].pages == (1, 3)
    assert suite.entries[1].rotation_probe is True


def test_remote_suite_rejects_duplicate_source_ids(tmp_path) -> None:
    path = _suite(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["sources"].append(dict(payload["sources"][0]))
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RemoteEvaluationError, match="duplicate source id"):
        load_remote_diagnostic_suite(path)


def test_remote_suite_requires_verified_registry_candidates(tmp_path) -> None:
    suite = load_remote_diagnostic_suite(_suite(tmp_path))
    registry = _registry(
        tmp_path,
        second_status="remote-evaluation-candidate-unverified",
    )

    with pytest.raises(RemoteEvaluationError, match="verified"):
        validate_suite_against_registry(suite, registry)


def test_remote_suite_rejects_pages_outside_verified_range(tmp_path) -> None:
    path = _suite(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["sources"][0]["pages"] = [9]
    path.write_text(json.dumps(payload), encoding="utf-8")

    suite = load_remote_diagnostic_suite(path)
    with pytest.raises(RemoteEvaluationError, match="outside verified range"):
        validate_suite_against_registry(suite, _registry(tmp_path))


def test_remote_suite_runs_per_source_pages_and_aggregates(tmp_path) -> None:
    registry = _registry(tmp_path)
    suite = _suite(tmp_path)
    source_a = _pdf(tmp_path / "a.pdf", 2)
    source_b = _pdf(tmp_path / "b.pdf", 3)
    destinations: list[Path] = []

    def fetcher(url, destination_base, *, max_bytes, timeout_seconds):
        del timeout_seconds
        source = source_a if url.endswith("/a.pdf") else source_b
        destination = Path(destination_base).with_suffix(".pdf")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        destinations.append(destination)
        data = destination.read_bytes()
        assert len(data) <= max_bytes
        return DownloadedRemote(
            path=destination,
            final_url=url,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            content_type="application/pdf",
            etag=None,
            last_modified=None,
            format="pdf",
        )

    report = evaluate_remote_diagnostic_suite(
        suite,
        registry,
        engine=FixedEngine(),
        fetcher=fetcher,
    )

    assert report["report_type"] == "remote-source-diagnostic-suite"
    assert report["not_benchmark_accuracy"] is True
    assert report["suite"]["id"] == "suite-v1"
    assert report["summary"]["sources"] == 2
    assert report["summary"]["ok"] == 2
    assert report["summary"]["errors"] == 0
    assert report["summary"]["sampled_pages"] == 3
    assert report["summary"]["ocr_lao_characters"] > 0
    assert report["summary"]["ocr_confidence_bands"] == {"high": 3}
    assert report["summary"]["page_media_classifications"] == {
        "no-large-raster-layer": 3
    }
    assert report["summary"]["rotation_recommendations"] == {"none": 2}
    assert report["summary"]["applied_auto_orientations"] == {}
    assert report["summary"]["auto_orientation_probe_states"] == {}
    assert report["sources"][0]["suite_pages"] == [1]
    assert report["sources"][1]["suite_pages"] == [1, 3]
    assert report["sources"][0]["suite_note"] == "empty layer"
    assert report["sources"][0]["document"]["selected_pages"] == [1]
    assert report["sources"][1]["document"]["selected_pages"] == [1, 3]
    serialized = json.dumps(report, ensure_ascii=False)
    assert "native text" not in serialized
    assert "ທົດສອບ ລາວ" not in serialized
    assert all(not path.exists() for path in destinations)


def test_remote_suite_keeps_failed_source_and_aggregates_error(tmp_path) -> None:
    registry = _registry(tmp_path)
    suite = _suite(tmp_path)
    source_a = _pdf(tmp_path / "a.pdf", 2)

    def fetcher(url, destination_base, *, max_bytes, timeout_seconds):
        del max_bytes, timeout_seconds
        if url.endswith("/b.pdf"):
            raise RemoteEvaluationError("blocked source")
        destination = Path(destination_base).with_suffix(".pdf")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_a, destination)
        data = destination.read_bytes()
        return DownloadedRemote(
            path=destination,
            final_url=url,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            content_type="application/pdf",
            etag=None,
            last_modified=None,
            format="pdf",
        )

    report = evaluate_remote_diagnostic_suite(
        suite,
        registry,
        engine=FixedEngine(),
        fetcher=fetcher,
    )

    assert report["summary"]["sources"] == 2
    assert report["summary"]["ok"] == 1
    assert report["summary"]["errors"] == 1
    assert report["sources"][1]["status"] == "error"
    assert report["sources"][1]["error_type"] == "RemoteEvaluationError"


def test_repo_remote_diagnostic_suite_validates_current_registry() -> None:
    root = Path(__file__).parents[1]
    suite = load_remote_diagnostic_suite(
        root / "benchmarks" / "remote-diagnostic-suite.json"
    )
    validate_suite_against_registry(
        suite,
        root / "benchmarks" / "source-registry.json",
    )

    assert suite.suite_id == "real-world-scan-diagnostic-v1"
    assert len(suite.entries) == 6
    worldbank = next(
        entry
        for entry in suite.entries
        if entry.source_id == "worldbank-p172774-kpmg-lao-2024-rotated-raster-pages"
    )
    assert worldbank.pages == (8, 15, 21)
    assert worldbank.rotation_probe is True


def test_remote_suite_detects_registry_text_layer_drift(tmp_path) -> None:
    suite = load_remote_diagnostic_suite(_suite(tmp_path))
    registry = _registry(tmp_path)
    payload = json.loads(registry.read_text(encoding="utf-8"))
    payload["sources"][0]["evidence"]["text_layer"] = "present-but-garbled"
    registry.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RemoteEvaluationError, match="classification changed"):
        validate_suite_against_registry(suite, registry)


def test_remote_suite_can_apply_auto_orientation(tmp_path) -> None:
    registry = _registry(tmp_path)
    suite_path = _suite(tmp_path)
    payload = json.loads(suite_path.read_text(encoding="utf-8"))
    payload["sources"] = [payload["sources"][0]]
    suite_path.write_text(json.dumps(payload), encoding="utf-8")

    source = tmp_path / "source.pdf"
    pdf = pymupdf.open()
    pdf.new_page(width=300, height=160)
    pdf.save(source)
    pdf.close()

    def fetcher(url, destination_base, *, max_bytes, timeout_seconds):
        del url, max_bytes, timeout_seconds
        destination = Path(destination_base).with_suffix(".pdf")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        data = destination.read_bytes()
        return DownloadedRemote(
            path=destination,
            final_url="https://example.com/a.pdf",
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            content_type="application/pdf",
            etag=None,
            last_modified=None,
            format="pdf",
        )

    class OrientationEngine(OcrEngine):
        def is_available(self) -> bool:
            return True

        def metadata(self):
            return {"name": "OrientationEngine"}

        def recognize(self, image: Image.Image) -> list[RecognizedLine]:
            confidence = 0.95 if image.height > image.width else 0.30
            return [
                RecognizedLine(
                    text="orientation text with enough characters",
                    bbox=BoundingBox(x=10, y=10, width=100, height=20),
                    confidence=confidence,
                    block_id=1,
                    paragraph_id=1,
                    line_id=1,
                )
            ]

    report = evaluate_remote_diagnostic_suite(
        suite_path,
        registry,
        engine=OrientationEngine(),
        auto_orient_right_angles=True,
        fetcher=fetcher,
    )

    assert report["summary"]["applied_auto_orientations"] == {"90": 1}
    assert report["summary"]["auto_orientation_probe_states"] == {"probed": 1}
    page = report["sources"][0]["document"]["pages"][0]
    assert page["auto_orientation"]["degrees_clockwise"] == 90
    assert "rotation_probe" not in page

