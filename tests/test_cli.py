from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageFont

from lao_document_ocr.cli import main


def _cli_font_path() -> Path:
    candidates = [
        Path("/usr/share/fonts/truetype/noto/NotoSansLao-Regular.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=20)
        path = getattr(font, "path", None)
        if path and Path(path).is_file():
            return Path(path)
    except OSError:
        pass
    import pytest

    pytest.skip("No TrueType font available for CLI capture-pack test")


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
            "--tag",
            "layout:plain",
            "--tag",
            "language:lao",
            "--confirm-redistributable",
        ],
    )

    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["id"] == "clean-001"
    assert len(payload["sha256"]) == 64
    assert payload["tags"] == ["language:lao", "layout:plain"]
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


def test_generate_capture_pack_cli(tmp_path, monkeypatch, capsys) -> None:
    corpus = tmp_path / "corpus.txt"
    output = tmp_path / "capture-pack"
    corpus.write_text(
        "ສະບາຍດີ ໂລກ\nລາຄາ 20,000 ກີບ\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "generate-capture-pack",
            "--corpus",
            str(corpus),
            "--output",
            str(output),
            "--font",
            str(_cli_font_path()),
            "--pack-id",
            "cli-pack",
            "--text-license",
            "CC0-1.0",
            "--text-provenance",
            "CLI unit-test corpus",
            "--dpi",
            "96",
            "--template",
            "two-column",
        ],
    )

    assert main() == 0
    assert (output / "capture-pack.json").is_file()
    assert (output / "cli-pack.pdf").is_file()
    payload = json.loads((output / "capture-pack.json").read_text(encoding="utf-8"))
    assert payload["pages"][0]["template"] == "two-column"
    assert "layout:multi-column" in payload["pages"][0]["tags"]
    captured = capsys.readouterr()
    assert "Capture pack:" in captured.out
    assert "Printable PDF:" in captured.out


def test_register_capture_cli(tmp_path, monkeypatch, capsys) -> None:
    corpus = tmp_path / "corpus.txt"
    pack_dir = tmp_path / "capture-pack"
    corpus.write_text("ສະບາຍດີ ໂລກ\n", encoding="utf-8")

    from lao_document_ocr.capture_pack import generate_capture_pack

    pack_manifest = generate_capture_pack(
        ["ສະບາຍດີ ໂລກ"],
        pack_dir,
        _cli_font_path(),
        pack_id="cli-register",
        text_license="CC0-1.0",
        text_provenance="CLI unit-test corpus",
        dpi=96,
    )
    capture = tmp_path / "phone.jpg"
    Image.new("RGB", (900, 1200), "white").save(capture)
    dataset_root = tmp_path / "dataset"
    dataset_manifest = dataset_root / "manifest.jsonl"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "register-capture",
            "--pack-manifest",
            str(pack_manifest),
            "--page-id",
            "cli-register-p0001",
            "--capture-image",
            str(capture),
            "--capture-id",
            "phone-a",
            "--mode",
            "phone-photo",
            "--contributor",
            "CLI Contributor",
            "--release-license",
            "CC0-1.0",
            "--dataset-root",
            str(dataset_root),
            "--dataset-manifest",
            str(dataset_manifest),
            "--confirm-release",
        ],
    )

    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["id"] == "cli-register-p0001-phone-a"
    assert payload["subset"] == "phone-photo"
    assert dataset_manifest.is_file()


def test_dataset_report_cli(tmp_path, monkeypatch, capsys) -> None:
    import hashlib

    image = tmp_path / "sample.png"
    truth = tmp_path / "sample.txt"
    image.write_bytes(b"sample-image")
    truth.write_text("ສະບາຍດີ", encoding="utf-8")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "dataset-report.json"
    manifest.write_text(
        json.dumps(
            {
                "id": "sample-001",
                "document_id": "doc-001",
                "split": "test",
                "subset": "clean-print",
                "source": image.name,
                "ground_truth": truth.name,
                "language": "lo",
                "license": "CC0-1.0",
                "provenance": "CLI dataset report test",
                "sha256": digest,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "dataset-report",
            "--manifest",
            str(manifest),
            "--dataset-root",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )

    assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["sample_count"] == 1
    assert payload["document_count"] == 1
    assert payload["validation"]["ok"] is True
    captured = capsys.readouterr()
    assert "Validation: ok" in captured.out


def test_generate_capture_suite_cli(tmp_path, monkeypatch, capsys) -> None:
    corpus = tmp_path / "suite-corpus.txt"
    output = tmp_path / "suite"
    corpus.write_text(
        "ສະບາຍດີ ໂລກ\nຂອບໃຈ ຫຼາຍ\nLao OCR 2026\nລາຄາ 20,000 ກີບ\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "generate-capture-suite",
            "--corpus",
            str(corpus),
            "--output",
            str(output),
            "--font",
            str(_cli_font_path()),
            "--suite-id",
            "cli-suite",
            "--text-license",
            "CC0-1.0",
            "--text-provenance",
            "CLI unit-test corpus",
            "--dpi",
            "96",
            "--max-pages-per-template",
            "1",
            "--template",
            "plain",
            "--template",
            "receipt",
        ],
    )

    assert main() == 0
    manifest = output / "capture-suite.json"
    combined = output / "cli-suite.pdf"
    assert manifest.is_file()
    assert combined.is_file()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert [pack["template"] for pack in payload["packs"]] == ["plain", "receipt"]
    assert "Combined PDF:" in capsys.readouterr().out


def test_capture_campaign_report_cli(tmp_path, monkeypatch, capsys) -> None:
    from lao_document_ocr.capture_suite import generate_capture_suite
    from lao_document_ocr.capture_templates import CaptureTemplate

    suite_manifest = generate_capture_suite(
        ["ສະບາຍດີ", "ຂອບໃຈ", "OCR", "20,000 ₭"],
        tmp_path / "campaign-suite",
        _cli_font_path(),
        suite_id="cli-campaign",
        text_license="CC0-1.0",
        text_provenance="CLI campaign test",
        templates=[CaptureTemplate.PLAIN, CaptureTemplate.RECEIPT],
        dpi=96,
        lines_per_page=4,
        max_pages_per_template=1,
    )
    dataset_manifest = tmp_path / "campaign-manifest.jsonl"
    entries = [
        {
            "id": "plain-flatbed",
            "document_id": "cli-campaign-plain-p0001",
            "split": "test",
            "subset": "clean-print",
            "source": "plain-flatbed.jpg",
            "ground_truth": "plain-flatbed.txt",
            "license": "CC0-1.0",
            "provenance": "CLI campaign test",
            "tags": ["capture:flatbed-scan"],
        },
        {
            "id": "plain-phone",
            "document_id": "cli-campaign-plain-p0001",
            "split": "test",
            "subset": "phone-photo",
            "source": "plain-phone.jpg",
            "ground_truth": "plain-phone.txt",
            "license": "CC0-1.0",
            "provenance": "CLI campaign test",
            "tags": ["capture:phone-photo"],
        },
        {
            "id": "receipt-phone",
            "document_id": "cli-campaign-receipt-p0001",
            "split": "test",
            "subset": "phone-photo",
            "source": "receipt-phone.jpg",
            "ground_truth": "receipt-phone.txt",
            "license": "CC0-1.0",
            "provenance": "CLI campaign test",
            "tags": ["capture:phone-photo"],
        },
    ]
    dataset_manifest.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "campaign-report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "capture-campaign-report",
            "--suite-manifest",
            str(suite_manifest),
            "--dataset-manifest",
            str(dataset_manifest),
            "--output",
            str(output),
        ],
    )

    assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["completed_captures"] == 3
    assert payload["required_captures"] == 4
    assert payload["completion_ratio"] == 0.75
    captured = capsys.readouterr()
    assert "Completion: 75.0%" in captured.out


def test_add_dataset_sample_cli_with_layout_ground_truth(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    from lao_document_ocr.layout_ground_truth import write_layout_ground_truth
    from lao_document_ocr.models import Block, BlockType, BoundingBox, Document, Page

    image = tmp_path / "layout-cli.png"
    truth = tmp_path / "layout-cli.txt"
    layout = tmp_path / "layout-cli.json"
    Image.new("RGB", (100, 50), "white").save(image)
    truth.write_text("ສະບາຍດີ", encoding="utf-8")
    write_layout_ground_truth(
        Document(
            pages=[
                Page(
                    number=1,
                    width=100,
                    height=50,
                    blocks=[
                        Block(
                            type=BlockType.PARAGRAPH,
                            text="ສະບາຍດີ",
                            bbox=BoundingBox(
                                x=5,
                                y=5,
                                width=80,
                                height=20,
                            ),
                        )
                    ],
                )
            ]
        ),
        layout,
    )
    dataset_root = tmp_path / "layout-dataset"
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
            "layout-cli",
            "--document-id",
            "layout-cli-doc",
            "--subset",
            "clean-print",
            "--image",
            str(image),
            "--ground-truth",
            str(truth),
            "--layout-ground-truth",
            str(layout),
            "--license",
            "CC0-1.0",
            "--provenance",
            "CLI layout ground-truth test",
            "--confirm-redistributable",
        ],
    )

    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["layout_ground_truth"] == (
        "layout-ground-truth/clean-print/layout-cli.json"
    )
    assert (dataset_root / payload["layout_ground_truth"]).is_file()


def test_prepare_layout_training_manifest_cli(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import hashlib

    from lao_document_ocr.layout_ground_truth import write_layout_ground_truth
    from lao_document_ocr.models import Block, BlockType, BoundingBox, Document, Page

    image = tmp_path / "layout-train.png"
    truth = tmp_path / "layout-train.txt"
    layout = tmp_path / "layout-train.json"
    Image.new("RGB", (120, 80), (230, 255, 255)).save(image)
    truth.write_text("ສະບາຍດີ", encoding="utf-8")
    write_layout_ground_truth(
        Document(
            pages=[
                Page(
                    number=1,
                    width=120,
                    height=80,
                    blocks=[
                        Block(
                            type=BlockType.HEADING,
                            text="ສະບາຍດີ",
                            bbox=BoundingBox(
                                x=10,
                                y=10,
                                width=90,
                                height=25,
                            ),
                        )
                    ],
                )
            ]
        ),
        layout,
    )
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    manifest = tmp_path / "layout-train-manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "layout-train",
                "document_id": "layout-train-doc",
                "split": "train",
                "subset": "clean-print",
                "source": image.name,
                "ground_truth": truth.name,
                "layout_ground_truth": layout.name,
                "license": "CC0-1.0",
                "provenance": "CLI layout training test",
                "sha256": digest,
                "tags": ["layout:plain"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "layout-training.jsonl"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "prepare-layout-training-manifest",
            "--manifest",
            str(manifest),
            "--dataset-root",
            str(tmp_path),
            "--output",
            str(output),
            "--split",
            "train",
        ],
    )

    assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["id"] == "layout-train"
    assert payload["block_counts"] == {"heading": 1}
    captured = capsys.readouterr().out
    assert "Layout training manifest:" in captured
    assert '"samples": 1' in captured


def test_prepare_layout_targets_cli(tmp_path, monkeypatch, capsys) -> None:
    from lao_document_ocr.layout_ground_truth import write_layout_ground_truth
    from lao_document_ocr.models import Block, BlockType, BoundingBox, Document, Page

    image = tmp_path / "data" / "clean-print" / "target-cli.png"
    layout = tmp_path / "layout-ground-truth" / "clean-print" / "target-cli.json"
    image.parent.mkdir(parents=True, exist_ok=True)
    layout.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (120, 80), "white").save(image)
    write_layout_ground_truth(
        Document(
            pages=[
                Page(
                    number=1,
                    width=120,
                    height=80,
                    blocks=[
                        Block(
                            type=BlockType.PARAGRAPH,
                            text="Body",
                            bbox=BoundingBox(
                                x=10,
                                y=10,
                                width=90,
                                height=30,
                            ),
                        )
                    ],
                )
            ]
        ),
        layout,
    )
    training_manifest = tmp_path / "layout-training.jsonl"
    training_manifest.write_text(
        json.dumps(
            {
                "id": "target-cli",
                "document_id": "target-cli-doc",
                "image": "data/clean-print/target-cli.png",
                "layout_ground_truth": (
                    "layout-ground-truth/clean-print/target-cli.json"
                ),
                "split": "train",
                "subset": "clean-print",
                "tags": ["layout:plain"],
                "sha256": None,
                "width": 120,
                "height": 80,
                "block_counts": {"paragraph": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "targets"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "prepare-layout-targets",
            "--training-manifest",
            str(training_manifest),
            "--dataset-root",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )

    assert main() == 0
    payload = json.loads(
        (output / "targets.jsonl").read_text(encoding="utf-8")
    )
    assert payload["id"] == "target-cli"
    assert payload["mask"] == "masks/target-cli.png"
    assert payload["boxes"] == "boxes/target-cli.json"
    assert (output / payload["mask"]).is_file()
    assert (output / payload["boxes"]).is_file()
    assert (output / "classes.json").is_file()
    captured = capsys.readouterr().out
    assert "Layout targets:" in captured
    assert "Samples: 1" in captured


def test_convert_document_learned_reading_order_requires_model(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (100, 60), "white").save(source)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "convert-document",
            "--input",
            str(source),
            "--output-dir",
            str(tmp_path / "output"),
            "--reading-order",
            "learned",
        ],
    )

    assert main() == 1
    assert "--reading-order-model is required" in capsys.readouterr().err


def test_compare_benchmarks_cli_returns_pass_and_writes_report(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    def report(cer):
        return {
            "schema_version": "1",
            "split": "test",
            "engine": {"name": "engine"},
            "reading_order": {"name": "DeterministicReadingOrderResolver"},
            "overall": {"cer": cer, "wer": cer * 2, "samples": 1},
            "subsets": {
                "clean-print": {"cer": cer, "wer": cer * 2, "samples": 1}
            },
            "tags": {},
            "samples": [
                {
                    "id": "sample",
                    "subset": "clean-print",
                    "tags": [],
                    "cer": cer,
                }
            ],
        }

    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "comparison.json"
    baseline.write_text(json.dumps(report(0.20)), encoding="utf-8")
    candidate.write_text(json.dumps(report(0.10)), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "compare-benchmarks",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--output",
            str(output),
        ],
    )

    assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert "Gate: PASS" in capsys.readouterr().out


def test_compare_benchmarks_cli_returns_one_on_regression(
    tmp_path,
    monkeypatch,
) -> None:
    base = {
        "schema_version": "1",
        "split": "test",
        "engine": {"name": "engine"},
        "reading_order": {"name": "DeterministicReadingOrderResolver"},
        "overall": {"cer": 0.10, "wer": 0.20, "samples": 1},
        "subsets": {
            "clean-print": {"cer": 0.10, "wer": 0.20, "samples": 1}
        },
        "tags": {},
        "samples": [
            {
                "id": "sample",
                "subset": "clean-print",
                "tags": [],
                "cer": 0.10,
            }
        ],
    }
    candidate_payload = json.loads(json.dumps(base))
    candidate_payload["overall"]["cer"] = 0.11
    candidate_payload["subsets"]["clean-print"]["cer"] = 0.11

    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "comparison.json"
    baseline.write_text(json.dumps(base), encoding="utf-8")
    candidate.write_text(json.dumps(candidate_payload), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "compare-benchmarks",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--output",
            str(output),
        ],
    )

    assert main() == 1


def test_freeze_and_verify_benchmark_cli(tmp_path, monkeypatch, capsys) -> None:
    import hashlib

    image = tmp_path / "page.png"
    truth = tmp_path / "page.txt"
    Image.new("RGB", (100, 60), (200, 255, 255)).save(image)
    truth.write_text("ສະບາຍດີ", encoding="utf-8")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    source_manifest = tmp_path / "source.jsonl"
    source_manifest.write_text(
        json.dumps(
            {
                "id": "sample-001",
                "document_id": "doc-001",
                "split": "test",
                "subset": "clean-print",
                "source": image.name,
                "ground_truth": truth.name,
                "license": "CC0-1.0",
                "provenance": "CLI freeze test",
                "sha256": digest,
                "tags": ["language:lao"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    frozen = tmp_path / "frozen.jsonl"
    lock = tmp_path / "benchmark.lock.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "freeze-benchmark",
            "--manifest",
            str(source_manifest),
            "--dataset-root",
            str(tmp_path),
            "--output-manifest",
            str(frozen),
            "--output-lock",
            str(lock),
            "--revision",
            "abc123",
        ],
    )
    assert main() == 0
    assert frozen.is_file()
    assert lock.is_file()
    assert "Benchmark lock:" in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "verify-benchmark-freeze",
            "--lock",
            str(lock),
            "--dataset-root",
            str(tmp_path),
        ],
    )
    assert main() == 0
    assert "Frozen benchmark verified" in capsys.readouterr().out


def test_verify_benchmark_freeze_cli_detects_tamper(
    tmp_path,
    monkeypatch,
) -> None:
    import hashlib

    image = tmp_path / "page.png"
    truth = tmp_path / "page.txt"
    Image.new("RGB", (100, 60), (180, 255, 255)).save(image)
    truth.write_text("truth", encoding="utf-8")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    source_manifest = tmp_path / "source.jsonl"
    source_manifest.write_text(
        json.dumps(
            {
                "id": "sample-001",
                "document_id": "doc-001",
                "split": "test",
                "subset": "clean-print",
                "source": image.name,
                "ground_truth": truth.name,
                "license": "CC0-1.0",
                "provenance": "CLI freeze test",
                "sha256": digest,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    frozen = tmp_path / "frozen.jsonl"
    lock = tmp_path / "lock.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "freeze-benchmark",
            "--manifest",
            str(source_manifest),
            "--dataset-root",
            str(tmp_path),
            "--output-manifest",
            str(frozen),
            "--output-lock",
            str(lock),
        ],
    )
    assert main() == 0

    truth.write_text("tampered", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "verify-benchmark-freeze",
            "--lock",
            str(lock),
            "--dataset-root",
            str(tmp_path),
        ],
    )
    assert main() == 1


def test_benchmark_refuses_tampered_frozen_dataset_before_engine(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import hashlib

    image = tmp_path / "page.png"
    truth = tmp_path / "page.txt"
    Image.new("RGB", (100, 60), (160, 255, 255)).save(image)
    truth.write_text("truth", encoding="utf-8")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    source_manifest = tmp_path / "source.jsonl"
    source_manifest.write_text(
        json.dumps(
            {
                "id": "sample-001",
                "document_id": "doc-001",
                "split": "test",
                "subset": "clean-print",
                "source": image.name,
                "ground_truth": truth.name,
                "license": "CC0-1.0",
                "provenance": "CLI frozen benchmark guard test",
                "sha256": digest,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    frozen = tmp_path / "frozen.jsonl"
    lock = tmp_path / "lock.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "freeze-benchmark",
            "--manifest",
            str(source_manifest),
            "--dataset-root",
            str(tmp_path),
            "--output-manifest",
            str(frozen),
            "--output-lock",
            str(lock),
        ],
    )
    assert main() == 0
    capsys.readouterr()

    truth.write_text("tampered", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "benchmark",
            "--manifest",
            str(frozen),
            "--dataset-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "report.json"),
            "--freeze-lock",
            str(lock),
        ],
    )

    assert main() == 1
    assert "Frozen benchmark verification failed" in capsys.readouterr().err


def test_generate_synthetic_cli_balanced_profiles(tmp_path, monkeypatch) -> None:
    corpus = tmp_path / "synthetic-corpus.txt"
    output = tmp_path / "synthetic-balanced"
    corpus.write_text("OCR one\nOCR two\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "generate-synthetic",
            "--corpus",
            str(corpus),
            "--output",
            str(output),
            "--font",
            str(_cli_font_path()),
            "--variants-per-line",
            "3",
            "--min-font-size",
            "24",
            "--max-font-size",
            "24",
            "--augmentation-profile",
            "balanced",
        ],
    )

    assert main() == 0
    entries = [
        json.loads(line)
        for line in (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [entry["augmentation_profile"] for entry in entries] == [
        "clean-scan",
        "noisy-scan",
        "phone-photo",
        "clean-scan",
        "noisy-scan",
        "phone-photo",
    ]
