from __future__ import annotations

import json
import sys

from PIL import Image

from lao_document_ocr.cli import main


def test_prepare_corpus_cli(tmp_path, monkeypatch, capsys) -> None:
    source = tmp_path / "source.txt"
    output = tmp_path / "corpus.txt"
    source.write_text(
        "ສະບາຍດີ ໂລກ\n"
        "English only\n"
        "ລາວ OCR 123\n"
        "ສະບາຍດີ ໂລກ\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "prepare-corpus",
            "--input",
            str(source),
            "--output",
            str(output),
            "--min-lao-ratio",
            "0.25",
        ],
    )

    assert main() == 0
    assert output.read_text(encoding="utf-8").splitlines() == [
        "ສະບາຍດີ ໂລກ",
        "ລາວ OCR 123",
    ]
    captured = capsys.readouterr()
    assert "Lines: 2" in captured.out


def test_add_dataset_sample_cli(tmp_path, monkeypatch, capsys) -> None:
    image = tmp_path / "page.png"
    truth = tmp_path / "page.txt"
    Image.new("RGB", (100, 50), "white").save(image)
    truth.write_text("ສະບາຍດີ\n", encoding="utf-8")
    dataset_root = tmp_path / "dataset"
    manifest = dataset_root / "manifest.jsonl"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "add-dataset-sample",
            "--dataset-root",
            str(dataset_root),
            "--manifest",
            str(manifest),
            "--id",
            "clean-001",
            "--document-id",
            "doc-001",
            "--subset",
            "clean-print",
            "--image",
            str(image),
            "--ground-truth",
            str(truth),
            "--license",
            "CC0-1.0",
            "--provenance",
            "Created for CLI test",
            "--confirm-redistributable",
        ],
    )

    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["id"] == "clean-001"
    assert len(payload["sha256"]) == 64
    assert (dataset_root / payload["source"]).is_file()


def test_calibrate_recognizer_cli(tmp_path, monkeypatch, capsys) -> None:
    report = tmp_path / "report.json"
    output = tmp_path / "calibration.json"
    report.write_text(
        json.dumps(
            {
                "samples": [
                    {"uncalibrated_confidence": 0.2, "cer": 0.9},
                    {"uncalibrated_confidence": 0.8, "cer": 0.1},
                    {"uncalibrated_confidence": 0.9, "cer": 0.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "calibrate-recognizer",
            "--report",
            str(report),
            "--output",
            str(output),
            "--bins",
            "2",
        ],
    )

    assert main() == 0
    assert output.is_file()
    captured = capsys.readouterr()
    assert "Calibration:" in captured.out
    assert "Samples: 3" in captured.out


def test_benchmark_layout_cli(tmp_path, monkeypatch, capsys) -> None:
    from lao_document_ocr.models import Block, BlockType, BoundingBox, Document, Page

    document = Document(
        pages=[
            Page(
                number=1,
                width=600,
                height=800,
                blocks=[
                    Block(
                        type=BlockType.PARAGRAPH,
                        text="Body",
                        bbox=BoundingBox(x=50, y=100, width=500, height=100),
                    )
                ],
            )
        ]
    )
    reference = tmp_path / "reference.json"
    prediction = tmp_path / "prediction.json"
    output = tmp_path / "layout-report.json"
    reference.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    prediction.write_text(document.model_dump_json(indent=2), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "benchmark-layout",
            "--reference",
            str(reference),
            "--prediction",
            str(prediction),
            "--output",
            str(output),
        ],
    )

    assert main() == 0
    assert output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metrics"]["block_f1"] == 1.0
    assert "Block F1: 1.0000" in capsys.readouterr().out


def test_benchmark_docx_cli_uses_fidelity_report(tmp_path, monkeypatch, capsys) -> None:
    import lao_document_ocr.visual_fidelity as visual_fidelity

    reference = tmp_path / "reference.png"
    docx = tmp_path / "result.docx"
    output = tmp_path / "fidelity.json"
    reference.write_bytes(b"reference")
    docx.write_bytes(b"docx")

    def fake_benchmark(*args, **kwargs):
        return {
            "schema_version": "1",
            "dpi": 144,
            "renderer": {"binary": "fake", "version": "fake", "pymupdf_version": "fake"},
            "metrics": {
                "reference_pages": 1,
                "predicted_pages": 1,
                "compared_pages": 1,
                "page_count_score": 1.0,
                "pixel_similarity": 0.9,
                "foreground_iou": 0.8,
                "edge_f1": 0.7,
                "composite_score": 0.81,
                "pages": [],
            },
        }

    monkeypatch.setattr(visual_fidelity, "benchmark_docx_fidelity", fake_benchmark)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "benchmark-docx",
            "--reference",
            str(reference),
            "--docx",
            str(docx),
            "--output",
            str(output),
        ],
    )

    assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metrics"]["composite_score"] == 0.81
    captured = capsys.readouterr()
    assert "Composite: 0.8100" in captured.out
    assert "Edge F1: 0.7000" in captured.out


def test_bundle_benchmarks_cli(tmp_path, monkeypatch, capsys) -> None:
    ocr = tmp_path / "ocr.json"
    output = tmp_path / "bundle.json"
    ocr.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "overall": {"samples": 3, "cer": 0.1, "wer": 0.2},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "bundle-benchmarks",
            "--ocr",
            str(ocr),
            "--output",
            str(output),
            "--revision",
            "abc123",
            "--label",
            "baseline-v1",
        ],
    )

    assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["source_revision"] == "abc123"
    assert payload["label"] == "baseline-v1"
    assert payload["reports"][0]["kind"] == "ocr"
    captured = capsys.readouterr()
    assert "Reports: 1" in captured.out
