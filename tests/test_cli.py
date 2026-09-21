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
