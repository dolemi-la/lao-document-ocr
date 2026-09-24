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
from lao_document_ocr.remote_evaluation import (
    DownloadedRemote,
    RemoteEvaluationError,
    _layer_gap_diagnostics,
    evaluate_remote_sources,
    load_remote_registry_sources,
    validate_remote_url,
    write_remote_evaluation_report,
)


class FixedEngine(OcrEngine):
    def is_available(self) -> bool:
        return True

    def metadata(self):
        return {"name": "FixedEngine"}

    def recognize(self, image: Image.Image) -> list[RecognizedLine]:
        return [
            RecognizedLine(
                text="SECRET_OCR_CONTENT ສະບາຍດີ",
                bbox=BoundingBox(x=10, y=10, width=160, height=24),
                confidence=0.9,
                block_id=1,
                paragraph_id=1,
                line_id=1,
            )
        ]


def _registry(tmp_path: Path, *, status: str = "remote-evaluation-candidate-not-approved") -> Path:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "sources": [
                    {
                        "id": "remote-scan",
                        "name": "Remote scan",
                        "url": "https://example.com/document.pdf",
                        "status": status,
                        "evidence": {"text_layer": "absent"},
                    },
                    {
                        "id": "approved",
                        "name": "Approved source",
                        "url": "https://example.com/approved.pdf",
                        "status": "approved-text-only",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _pdf(path: Path) -> Path:
    pdf = pymupdf.open()
    for index in range(2):
        page = pdf.new_page(width=300, height=180)
        page.insert_text(
            (20, 40),
            f"SECRET_NATIVE_CONTENT page {index + 1}",
            fontsize=12,
        )
    pdf.save(path)
    pdf.close()
    return path


def test_remote_registry_requires_explicit_remote_selection(tmp_path) -> None:
    registry = _registry(tmp_path)

    with pytest.raises(RemoteEvaluationError, match="requires explicit"):
        load_remote_registry_sources(registry)

    selected = load_remote_registry_sources(
        registry,
        source_ids=["remote-scan"],
    )
    assert [source["id"] for source in selected] == ["remote-scan"]

    with pytest.raises(RemoteEvaluationError, match="not a remote-evaluation"):
        load_remote_registry_sources(
            registry,
            source_ids=["approved"],
        )


def test_validate_remote_url_rejects_non_https_and_private_addresses() -> None:
    with pytest.raises(RemoteEvaluationError, match="HTTPS"):
        validate_remote_url("http://example.com/file.pdf")

    with pytest.raises(RemoteEvaluationError, match="non-public"):
        validate_remote_url("https://127.0.0.1/file.pdf")


def test_remote_evaluation_reports_stats_without_text_or_persisted_bytes(tmp_path) -> None:
    registry = _registry(tmp_path)
    source_pdf = _pdf(tmp_path / "source.pdf")
    fetched_paths: list[Path] = []

    def fetcher(url, destination_base, *, max_bytes, timeout_seconds):
        del url, max_bytes, timeout_seconds
        destination = Path(destination_base).with_suffix(".pdf")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_pdf, destination)
        fetched_paths.append(destination)
        data = destination.read_bytes()
        return DownloadedRemote(
            path=destination,
            final_url="https://example.com/document.pdf",
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            content_type="application/pdf",
            etag='"unit-test"',
            last_modified=None,
            format="pdf",
        )

    report = evaluate_remote_sources(
        registry,
        engine=FixedEngine(),
        source_ids=["remote-scan"],
        max_pages_per_source=3,
        fetcher=fetcher,
    )

    assert report["not_benchmark_accuracy"] is True
    assert report["summary"] == {"sources": 1, "ok": 1, "errors": 0}
    result = report["sources"][0]
    assert result["status"] == "ok"
    assert result["download"]["path_persisted"] is False
    assert result["document"]["page_count"] == 2
    assert result["document"]["selected_pages"] == [1, 2]
    assert result["document"]["pages"][0]["native_text"]["characters"] > 0
    assert result["document"]["pages"][0]["ocr"]["text"]["lao_characters"] > 0

    serialized = json.dumps(report, ensure_ascii=False)
    assert "SECRET_NATIVE_CONTENT" not in serialized
    assert "SECRET_OCR_CONTENT" not in serialized
    assert all(not path.exists() for path in fetched_paths)


def test_remote_evaluation_records_source_error_without_aborting_report(tmp_path) -> None:
    registry = _registry(tmp_path)

    def fetcher(url, destination_base, *, max_bytes, timeout_seconds):
        del url, destination_base, max_bytes, timeout_seconds
        raise RemoteEvaluationError("download unavailable")

    report = evaluate_remote_sources(
        registry,
        engine=FixedEngine(),
        source_ids=["remote-scan"],
        fetcher=fetcher,
    )

    assert report["summary"] == {"sources": 1, "ok": 0, "errors": 1}
    assert report["sources"][0]["status"] == "error"
    assert report["sources"][0]["error_type"] == "RemoteEvaluationError"


def test_write_remote_evaluation_report_is_json(tmp_path) -> None:
    output = write_remote_evaluation_report(
        {
            "schema_version": "1",
            "report_type": "remote-source-diagnostic",
            "not_benchmark_accuracy": True,
        },
        tmp_path / "report.json",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["not_benchmark_accuracy"] is True


def test_remote_evaluation_source_id_cannot_escape_temp_directory(tmp_path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "sources": [
                    {
                        "id": "../escape-attempt",
                        "name": "Traversal-shaped ID",
                        "url": "https://example.com/document.pdf",
                        "status": "remote-evaluation-candidate-not-approved",
                        "evidence": {"text_layer": "absent"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    source_pdf = _pdf(tmp_path / "source.pdf")
    destinations: list[Path] = []

    def fetcher(url, destination_base, *, max_bytes, timeout_seconds):
        del url, max_bytes, timeout_seconds
        destination = Path(destination_base).with_suffix(".pdf")
        destinations.append(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_pdf, destination)
        data = destination.read_bytes()
        return DownloadedRemote(
            path=destination,
            final_url="https://example.com/document.pdf",
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            content_type="application/pdf",
            etag=None,
            last_modified=None,
            format="pdf",
        )

    report = evaluate_remote_sources(
        registry,
        engine=FixedEngine(),
        source_ids=["../escape-attempt"],
        requested_pages=[1],
        fetcher=fetcher,
    )

    assert report["summary"]["ok"] == 1
    assert destinations
    assert ".." not in destinations[0].parts
    assert "escape-attempt" not in destinations[0].parts
    assert not destinations[0].exists()


def test_layer_gap_classifies_empty_and_missing_lao_layers() -> None:
    empty = _layer_gap_diagnostics(
        {
            "nonspace_characters": 0,
            "lao_characters": 0,
        },
        {
            "nonspace_characters": 900,
            "lao_characters": 800,
        },
    )
    assert empty["classification"] == "native-layer-empty"
    assert empty["ocr_to_native_nonspace_ratio"] is None

    missing_lao = _layer_gap_diagnostics(
        {
            "nonspace_characters": 1200,
            "lao_characters": 0,
        },
        {
            "nonspace_characters": 1300,
            "lao_characters": 1100,
        },
    )
    assert missing_lao["classification"] == "lao-missing-from-native-layer"
    assert missing_lao["ocr_minus_native_lao_characters"] == 1100

