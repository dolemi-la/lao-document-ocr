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
    capture_image = Image.new("RGB", (900, 1200), (225, 220, 210))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(capture_image)
    for y in range(180, 900, 80):
        draw.rectangle((130, y, 760, y + 20), fill="black")
    capture_image.save(capture, quality=82)
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


def test_convert_document_cli_forwards_auto_orientation(
    tmp_path,
    monkeypatch,
) -> None:
    import lao_document_ocr.cli as cli

    captured = {}

    class Outputs:
        def to_dict(self):
            return {
                "docx": "out.docx",
                "markdown": "out.md",
                "text": "out.txt",
                "json": "out.json",
            }

    monkeypatch.setattr(cli, "_build_ocr_engine", lambda args: object())
    monkeypatch.setattr(
        cli,
        "_build_reading_order_resolver",
        lambda args: None,
    )

    def fake_convert(input_path, output_dir, **kwargs):
        captured["input"] = input_path
        captured["output_dir"] = output_dir
        captured.update(kwargs)
        return Outputs()

    monkeypatch.setattr(cli, "convert_document_to_outputs", fake_convert)

    source = tmp_path / "page.png"
    Image.new("RGB", (100, 60), "white").save(source)
    output_dir = tmp_path / "output"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "convert-document",
            "--input",
            str(source),
            "--output-dir",
            str(output_dir),
            "--auto-orient-right-angles",
        ],
    )

    assert cli.main() == 0
    assert captured["input"] == source
    assert captured["output_dir"] == output_dir
    assert captured["auto_orient_right_angles"] is True


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
            "--allow-unverified-sources",
            "--allow-unreviewed",
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
            "--allow-unverified-sources",
            "--allow-unreviewed",
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
            "--allow-unverified-sources",
            "--allow-unreviewed",
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


def test_train_char_lm_cli_writes_vocab_bound_artifact(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    from lao_document_ocr.vocabulary import CharacterVocabulary

    corpus = tmp_path / "lm-corpus.txt"
    vocab_path = tmp_path / "vocab.json"
    output = tmp_path / "char-lm.json"
    corpus.write_text("ສະບາຍດີ\nສະບາຍດີ\nຂອບໃຈ\n", encoding="utf-8")
    vocab = CharacterVocabulary.from_texts(["ສະບາຍດີ", "ຂອບໃຈ"])
    vocab.save(vocab_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "train-char-lm",
            "--corpus",
            str(corpus),
            "--vocabulary",
            str(vocab_path),
            "--output",
            str(output),
            "--order",
            "3",
            "--alpha",
            "0.2",
        ],
    )

    assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["type"] == "character-ngram"
    assert payload["order"] == 3
    assert payload["alpha"] == 0.2
    assert payload["vocabulary_checksum"] == vocab.checksum()
    assert payload["training_stats"]["used_lines"] == 3
    assert "Language model:" in capsys.readouterr().out


def test_recognize_line_cli_forwards_language_model_options(
    tmp_path,
    monkeypatch,
) -> None:
    import lao_document_ocr.recognizer_inference as inference

    captured = {}

    class FakeResult:
        text = "ok"
        confidence = 0.8
        calibrated_confidence = None

    class FakeRecognizer:
        def __init__(self, model, **kwargs):
            captured["model"] = model
            captured.update(kwargs)

        def recognize(self, image):
            captured["image"] = image
            return FakeResult()

    monkeypatch.setattr(inference, "ExportedLineRecognizer", FakeRecognizer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "recognize-line",
            "--model",
            str(tmp_path / "model.pt2"),
            "--image",
            str(tmp_path / "line.png"),
            "--decoder",
            "beam",
            "--beam-width",
            "12",
            "--language-model",
            str(tmp_path / "lm.json"),
            "--language-model-weight",
            "0.35",
            "--language-model-token-bonus",
            "0.08",
        ],
    )

    assert main() == 0
    assert captured["decoder"] == "beam"
    assert captured["beam_width"] == 12
    assert captured["language_model_path"] == tmp_path / "lm.json"
    assert captured["language_model_weight"] == 0.35
    assert captured["language_model_token_bonus"] == 0.08


def test_benchmark_readiness_cli_reports_ready_for_complete_dimensions(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import hashlib

    specs = [
        ("clean", "clean-print", ["capture:flatbed-scan"]),
        ("noisy", "noisy-scan", ["capture:degraded-scan"]),
        ("phone", "phone-photo", ["capture:phone-photo"]),
        ("mixed", "clean-print", ["language:mixed"]),
        ("columns", "clean-print", ["layout:multi-column"]),
        ("ruled", "clean-print", ["table:ruled"]),
        ("borderless", "clean-print", ["table:borderless"]),
        ("receipt", "clean-print", ["document:receipt"]),
        ("form", "clean-print", ["document:form"]),
    ]
    entries = []
    for index, (sample_id, subset, tags) in enumerate(specs, start=1):
        image = tmp_path / f"{sample_id}.png"
        truth = tmp_path / f"{sample_id}.txt"
        Image.new("RGB", (100, 60), (index * 20, 240, 250)).save(image)
        truth.write_text(f"truth {sample_id}", encoding="utf-8")
        entries.append(
            {
                "id": sample_id,
                "document_id": f"doc-{sample_id}",
                "split": "test",
                "subset": subset,
                "source": image.name,
                "ground_truth": truth.name,
                "license": "CC0-1.0",
                "provenance": "CLI readiness test",
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "tags": [
                    *tags,
                    "capture:optical-evidence",
                    "source:real-capture",
                ],
                "review": {
                    "status": "approved",
                    "reviewer": "CLI Reviewer",
                    "reviewed_at": "2026-09-23T08:00:00+00:00",
                },
            }
        )

    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "readiness.json"
    manifest.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "benchmark-readiness",
            "--manifest",
            str(manifest),
            "--dataset-root",
            str(tmp_path),
            "--output",
            str(output),
            "--min-total-documents",
            "9",
        ],
    )

    assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ready"] is True
    assert payload["missing_dimensions"] == []
    assert "Readiness: READY" in capsys.readouterr().out


def test_benchmark_readiness_cli_requires_real_source_tags(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import hashlib

    image = tmp_path / "synthetic.png"
    truth = tmp_path / "synthetic.txt"
    Image.new("RGB", (100, 60), (120, 240, 250)).save(image)
    truth.write_text("synthetic", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "readiness.json"
    manifest.write_text(
        json.dumps(
            {
                "id": "synthetic",
                "document_id": "doc-synthetic",
                "split": "test",
                "subset": "clean-print",
                "source": image.name,
                "ground_truth": truth.name,
                "license": "CC0-1.0",
                "provenance": "CLI readiness source test",
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "tags": [
                    "capture:flatbed-scan",
                    "source:capture-pack",
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "benchmark-readiness",
            "--manifest",
            str(manifest),
            "--dataset-root",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )

    assert main() == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["eligible_sample_count"] == 0
    assert payload["unverified_source_samples"] == ["synthetic"]
    assert "Readiness: NOT READY" in capsys.readouterr().out


def test_register_capture_directory_cli_forwards_bulk_options(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import lao_document_ocr.capture_batch as capture_batch
    from lao_document_ocr.capture_batch import CaptureBatchReport

    captured = {}

    def fake_register_capture_directory(**kwargs):
        captured.update(kwargs)
        return CaptureBatchReport(
            suite_id="campaign-v1",
            capture_id="phone-a",
            capture_mode="phone-photo",
            expected_pages=60,
            discovered_captures=60,
            planned_captures=60,
            registered_captures=0,
            missing_page_ids=(),
            ignored_files=(".DS_Store",),
            registered_sample_ids=(),
            dry_run=True,
        )

    monkeypatch.setattr(
        capture_batch,
        "register_capture_directory",
        fake_register_capture_directory,
    )
    report = tmp_path / "bulk-report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "register-capture-directory",
            "--suite-manifest",
            str(tmp_path / "capture-suite.json"),
            "--capture-dir",
            str(tmp_path / "phone"),
            "--capture-id",
            "phone-a",
            "--mode",
            "phone-photo",
            "--contributor",
            "CLI Contributor",
            "--release-license",
            "CC0-1.0",
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--dataset-manifest",
            str(tmp_path / "dataset" / "manifest.jsonl"),
            "--require-complete",
            "--dry-run",
            "--report",
            str(report),
        ],
    )

    assert main() == 0
    assert captured["capture_id"] == "phone-a"
    assert captured["capture_mode"].value == "phone-photo"
    assert captured["require_complete"] is True
    assert captured["dry_run"] is True
    assert captured["confirm_release"] is False
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["planned_captures"] == 60
    assert payload["dry_run"] is True
    rendered = capsys.readouterr().out
    assert "Planned captures: 60" in rendered
    assert "Dry run: yes" in rendered


def test_benchmark_capture_suite_cli_marks_report_as_digital_qa(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import lao_document_ocr.cli as cli_module
    from lao_document_ocr.capture_suite import generate_capture_suite
    from lao_document_ocr.capture_templates import CaptureTemplate
    from lao_document_ocr.models import BoundingBox
    from lao_document_ocr.ocr.base import OcrEngine, RecognizedLine

    class FixedQaEngine(OcrEngine):
        def is_available(self) -> bool:
            return True

        def recognize(self, image: Image.Image) -> list[RecognizedLine]:
            return [
                RecognizedLine(
                    text="digital qa",
                    bbox=BoundingBox(x=10, y=10, width=120, height=24),
                    confidence=1.0,
                    block_id=1,
                    paragraph_id=1,
                    line_id=1,
                )
            ]

    suite_manifest = generate_capture_suite(
        ["ສະບາຍດີ", "ຂອບໃຈ", "Lao OCR", "20,000 ₭"],
        tmp_path / "qa-suite",
        _cli_font_path(),
        suite_id="cli-qa",
        text_license="Apache-2.0",
        text_provenance="CLI digital QA corpus",
        templates=[CaptureTemplate.PLAIN, CaptureTemplate.RECEIPT],
        dpi=96,
        lines_per_page=4,
        max_pages_per_template=1,
    )
    output = tmp_path / "qa-report.json"

    monkeypatch.setattr(cli_module, "_build_ocr_engine", lambda args: FixedQaEngine())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "benchmark-capture-suite",
            "--suite-manifest",
            str(suite_manifest),
            "--output",
            str(output),
        ],
    )

    assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["qa"]["not_real_benchmark"] is True
    assert payload["qa"]["sample_count"] == 2
    assert set(payload["qa"]["templates"]) == {"plain", "receipt"}
    captured = capsys.readouterr()
    assert "not real benchmark accuracy" in captured.out


def test_review_dataset_sample_cli_enables_strict_public_freeze(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import hashlib

    image = tmp_path / "real-page.png"
    truth = tmp_path / "real-page.txt"
    Image.new("RGB", (120, 80), (210, 250, 255)).save(image)
    truth.write_text("ສະບາຍດີ", encoding="utf-8")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "real-001",
                "document_id": "real-doc-001",
                "split": "test",
                "subset": "clean-print",
                "source": image.name,
                "ground_truth": truth.name,
                "license": "CC0-1.0",
                "provenance": "CLI manual-review test",
                "sha256": digest,
                "tags": [
                    "capture:flatbed-scan",
                    "capture:optical-evidence",
                    "source:real-capture",
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    frozen = tmp_path / "strict-frozen.jsonl"
    lock = tmp_path / "strict.lock.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "freeze-benchmark",
            "--manifest",
            str(manifest),
            "--dataset-root",
            str(tmp_path),
            "--output-manifest",
            str(frozen),
            "--output-lock",
            str(lock),
        ],
    )
    assert main() == 1
    assert "approved manual review" in capsys.readouterr().err

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "review-dataset-sample",
            "--manifest",
            str(manifest),
            "--dataset-root",
            str(tmp_path),
            "--id",
            "real-001",
            "--status",
            "approved",
            "--reviewer",
            "Reviewer A",
            "--notes",
            "Checked page identity, orientation, and ground truth.",
        ],
    )
    assert main() == 0
    reviewed_payload = json.loads(capsys.readouterr().out)
    assert reviewed_payload["review"]["status"] == "approved"
    assert reviewed_payload["review"]["reviewer"] == "Reviewer A"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "freeze-benchmark",
            "--manifest",
            str(manifest),
            "--dataset-root",
            str(tmp_path),
            "--output-manifest",
            str(frozen),
            "--output-lock",
            str(lock),
        ],
    )
    assert main() == 0
    assert lock.is_file()
    lock_payload = json.loads(lock.read_text(encoding="utf-8"))
    assert lock_payload["samples"][0]["review"]["status"] == "approved"


def test_build_review_queue_cli_generates_local_bundle(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import hashlib

    image = tmp_path / "review.png"
    truth = tmp_path / "review.txt"
    Image.new("RGB", (240, 140), (220, 245, 255)).save(image)
    truth.write_text("ສະບາຍດີ review", encoding="utf-8")
    manifest = tmp_path / "review-manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "review-001",
                "document_id": "review-doc-001",
                "split": "test",
                "subset": "phone-photo",
                "source": image.name,
                "ground_truth": truth.name,
                "license": "CC0-1.0",
                "provenance": "CLI review queue test",
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "tags": ["capture:phone-photo", "source:real-capture"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "review-queue"
    original_manifest = manifest.read_bytes()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "build-review-queue",
            "--manifest",
            str(manifest),
            "--dataset-root",
            str(tmp_path),
            "--output-dir",
            str(output),
        ],
    )

    assert main() == 0
    assert manifest.read_bytes() == original_manifest
    assert (output / "index.html").is_file()
    payload = json.loads((output / "review-queue.json").read_text(encoding="utf-8"))
    assert payload["sample_count"] == 1
    assert payload["entries"][0]["id"] == "review-001"
    assert (output / payload["entries"][0]["thumbnail"]).is_file()
    captured = capsys.readouterr()
    assert "Samples: 1" in captured.out
    assert "With problems: 0" in captured.out


def test_apply_review_decisions_cli_dry_run_then_confirm(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import csv
    import hashlib

    from lao_document_ocr.dataset import load_manifest

    image = tmp_path / "batch-review.png"
    truth = tmp_path / "batch-review.txt"
    Image.new("RGB", (160, 100), (205, 245, 255)).save(image)
    truth.write_text("ສະບາຍດີ", encoding="utf-8")
    manifest = tmp_path / "batch-review-manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "batch-review-001",
                "document_id": "batch-review-doc-001",
                "split": "test",
                "subset": "clean-print",
                "source": image.name,
                "ground_truth": truth.name,
                "license": "CC0-1.0",
                "provenance": "CLI batch review test",
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    decisions = tmp_path / "review-decisions.csv"
    with decisions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "status", "reviewer", "notes"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "batch-review-001",
                "status": "approved",
                "reviewer": "Reviewer A",
                "notes": "Checked capture and truth",
            }
        )
    report = tmp_path / "review-report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "apply-review-decisions",
            "--manifest",
            str(manifest),
            "--dataset-root",
            str(tmp_path),
            "--decisions",
            str(decisions),
            "--report",
            str(report),
        ],
    )
    assert main() == 0
    assert load_manifest(manifest)[0].review is None
    assert json.loads(report.read_text(encoding="utf-8"))["dry_run"] is True
    assert "Mode: DRY RUN" in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "apply-review-decisions",
            "--manifest",
            str(manifest),
            "--dataset-root",
            str(tmp_path),
            "--decisions",
            str(decisions),
            "--confirm",
        ],
    )
    assert main() == 0
    reviewed = load_manifest(manifest)[0]
    assert reviewed.review is not None
    assert reviewed.review.status.value == "approved"
    assert reviewed.review.reviewer == "Reviewer A"
    assert "Mode: APPLIED" in capsys.readouterr().out


def test_build_capture_kit_cli(tmp_path, monkeypatch, capsys) -> None:
    import zipfile

    from lao_document_ocr.capture_suite import generate_capture_suite
    from lao_document_ocr.capture_templates import CaptureTemplate

    suite_manifest = generate_capture_suite(
        ["ສະບາຍດີ", "ຂອບໃຈ", "OCR", "20,000 ₭"],
        tmp_path / "collector-suite",
        _cli_font_path(),
        suite_id="collector-cli",
        text_license="Apache-2.0",
        text_provenance="CLI collector kit test",
        templates=[CaptureTemplate.PLAIN, CaptureTemplate.RECEIPT],
        dpi=96,
        lines_per_page=4,
        max_pages_per_template=1,
    )
    output = tmp_path / "collector-kit.zip"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "build-capture-kit",
            "--suite-manifest",
            str(suite_manifest),
            "--output",
            str(output),
            "--revision",
            "abc123",
        ],
    )

    assert main() == 0
    assert output.is_file()
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {
            "CAPTURE-INSTRUCTIONS.md",
            "SHA256SUMS",
            "capture-kit.json",
            "capture-worksheet.csv",
            "collector-cli.pdf",
        }
        worksheet = archive.read("capture-worksheet.csv").decode("utf-8")
        assert "ground_truth" not in worksheet
        assert "digital_page" not in worksheet
        manifest = json.loads(archive.read("capture-kit.json"))
        assert manifest["source_revision"] == "abc123"
        assert manifest["excludes_internal_suite_paths"] is True

    assert "Capture kit:" in capsys.readouterr().out


def test_benchmark_report_embeds_verified_freeze_identity(
    tmp_path,
    monkeypatch,
) -> None:
    import hashlib

    import lao_document_ocr.cli as cli_module
    from lao_document_ocr.benchmark_freeze import freeze_benchmark
    from lao_document_ocr.dataset import DatasetSample, DatasetSplit, DatasetSubset
    from lao_document_ocr.models import BoundingBox
    from lao_document_ocr.ocr.base import OcrEngine, RecognizedLine

    class FixedEngine(OcrEngine):
        def is_available(self) -> bool:
            return True

        def recognize(self, image: Image.Image) -> list[RecognizedLine]:
            return [
                RecognizedLine(
                    text="hello world",
                    bbox=BoundingBox(x=10, y=10, width=120, height=24),
                    confidence=1.0,
                    block_id=1,
                    paragraph_id=1,
                    line_id=1,
                )
            ]

    image = tmp_path / "page.png"
    truth = tmp_path / "page.txt"
    Image.new("RGB", (300, 120), (210, 255, 255)).save(image)
    truth.write_text("hello world", encoding="utf-8")
    sample = DatasetSample(
        id="sample-001",
        document_id="doc-001",
        split=DatasetSplit.TEST,
        subset=DatasetSubset.CLEAN_PRINT,
        source=image.name,
        ground_truth=truth.name,
        license="CC0-1.0",
        provenance="CLI freeze provenance test",
        sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        tags=["language:mixed"],
    )
    frozen = tmp_path / "frozen.jsonl"
    lock = tmp_path / "test-v1.lock.json"
    freeze_benchmark(
        [sample],
        tmp_path,
        output_manifest=frozen,
        output_lock=lock,
        source_revision="freeze-rev",
    )
    output = tmp_path / "benchmark.json"

    monkeypatch.setattr(cli_module, "_build_ocr_engine", lambda args: FixedEngine())
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
            str(output),
            "--freeze-lock",
            str(lock),
        ],
    )

    assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["freeze"]["lock_file"] == lock.name
    assert payload["freeze"]["lock_sha256"] == hashlib.sha256(
        lock.read_bytes()
    ).hexdigest()
    assert payload["freeze"]["frozen_manifest_sha256"] == hashlib.sha256(
        frozen.read_bytes()
    ).hexdigest()
    assert payload["freeze"]["source_revision"] == "freeze-rev"


def test_serve_capture_kit_cli_builds_app_and_runs_uvicorn(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import uvicorn

    import lao_document_ocr.capture_collector as collector

    calls = {}
    sentinel_app = object()

    def fake_create(*args, **kwargs):
        calls["create_args"] = args
        calls["create_kwargs"] = kwargs
        return sentinel_app

    def fake_run(app, **kwargs):
        calls["app"] = app
        calls["run_kwargs"] = kwargs

    monkeypatch.setattr(collector, "create_collector_app", fake_create)
    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "serve-capture-kit",
            "--kit",
            str(tmp_path / "collector.zip"),
            "--output-dir",
            str(tmp_path / "captures"),
            "--capture-id",
            "phone-a",
            "--mode",
            "phone-photo",
            "--port",
            "8099",
        ],
    )

    assert main() == 0
    assert calls["app"] is sentinel_app
    assert calls["run_kwargs"] == {
        "host": "127.0.0.1",
        "port": 8099,
        "log_level": "info",
    }
    assert calls["create_kwargs"]["capture_id"] == "phone-a"
    assert calls["create_kwargs"]["mode"].value == "phone-photo"
    assert calls["create_kwargs"]["require_qr"] is True
    assert len(calls["create_kwargs"]["access_token"]) >= 16
    output = capsys.readouterr().out
    assert "Collector access token:" in output
    assert "Local collector: http://127.0.0.1:8099/?token=" in output


def test_serve_capture_kit_cli_warns_when_exposed_on_network(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import uvicorn

    import lao_document_ocr.capture_collector as collector

    calls = {}

    def fake_create(*args, **kwargs):
        calls["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(collector, "create_collector_app", fake_create)
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "serve-capture-kit",
            "--kit",
            str(tmp_path / "collector.zip"),
            "--output-dir",
            str(tmp_path / "captures"),
            "--capture-id",
            "phone-a",
            "--mode",
            "phone-photo",
            "--host",
            "0.0.0.0",
            "--allow-unreadable-qr",
        ],
    )

    assert main() == 0
    assert calls["kwargs"]["require_qr"] is False
    assert len(calls["kwargs"]["access_token"]) >= 16
    assert "trusted network" in capsys.readouterr().err


def test_serve_capture_kit_cli_uses_explicit_access_token(
    tmp_path,
    monkeypatch,
) -> None:
    import uvicorn

    import lao_document_ocr.capture_collector as collector

    calls = {}

    def fake_create(*args, **kwargs):
        calls["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(collector, "create_collector_app", fake_create)
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "serve-capture-kit",
            "--kit",
            str(tmp_path / "collector.zip"),
            "--output-dir",
            str(tmp_path / "captures"),
            "--capture-id",
            "phone-a",
            "--mode",
            "phone-photo",
            "--access-token",
            "collector-token-explicit-1234",
        ],
    )

    assert main() == 0
    assert calls["kwargs"]["access_token"] == "collector-token-explicit-1234"


def test_capture_submission_cli_pack_verify_extract(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    capture_dir = tmp_path / "collector-session"
    capture_dir.mkdir()
    (capture_dir / ".collector-session.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "suite_id": "suite-v1",
                "source_revision": "abc123",
                "kit_sha256": "a" * 64,
                "capture_id": "phone-a",
                "mode": "phone-photo",
                "require_qr": True,
                "expected_pages": 1,
            }
        ),
        encoding="utf-8",
    )
    Image.new("RGB", (120, 80), (120, 160, 200)).save(
        capture_dir / "suite-v1-p0001.jpg"
    )
    submission = tmp_path / "submission.zip"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "pack-capture-submission",
            "--capture-dir",
            str(capture_dir),
            "--output",
            str(submission),
            "--require-complete",
        ],
    )
    assert main() == 0
    assert submission.is_file()
    assert "Capture submission:" in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "verify-capture-submission",
            "--submission",
            str(submission),
        ],
    )
    assert main() == 0
    output = capsys.readouterr().out
    assert "Captured: 1/1" in output
    assert "Complete: True" in output

    extracted = tmp_path / "extracted-submission"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "extract-capture-submission",
            "--submission",
            str(submission),
            "--output-dir",
            str(extracted),
        ],
    )
    assert main() == 0
    assert (extracted / "suite-v1-p0001.jpg").is_file()
    assert (extracted / "capture-submission.json").is_file()


def test_register_capture_submission_cli_uses_submission_metadata(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import lao_document_ocr.capture_submission_import as submission_import
    from lao_document_ocr.capture_batch import CaptureBatchReport
    from lao_document_ocr.capture_submission_import import (
        CaptureSubmissionImportReport,
    )

    captured = {}

    def fake_register_capture_submission(**kwargs):
        captured.update(kwargs)
        return CaptureSubmissionImportReport(
            suite_id="campaign-v1",
            source_revision="abc123",
            kit_sha256="a" * 64,
            capture_id="phone-a",
            capture_mode="phone-photo",
            captured_pages=60,
            expected_pages=60,
            submission_complete=True,
            batch=CaptureBatchReport(
                suite_id="campaign-v1",
                capture_id="phone-a",
                capture_mode="phone-photo",
                expected_pages=60,
                discovered_captures=60,
                planned_captures=60,
                registered_captures=0,
                missing_page_ids=(),
                ignored_files=("capture-submission.json",),
                registered_sample_ids=(),
                dry_run=True,
            ),
        )

    monkeypatch.setattr(
        submission_import,
        "register_capture_submission",
        fake_register_capture_submission,
    )
    report = tmp_path / "submission-import-report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "register-capture-submission",
            "--submission",
            str(tmp_path / "phone-a.submission.zip"),
            "--suite-manifest",
            str(tmp_path / "capture-suite.json"),
            "--contributor",
            "CLI Contributor",
            "--release-license",
            "CC0-1.0",
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--dataset-manifest",
            str(tmp_path / "dataset" / "manifest.jsonl"),
            "--require-complete",
            "--dry-run",
            "--expected-kit-sha256",
            "a" * 64,
            "--expected-source-revision",
            "abc123",
            "--report",
            str(report),
        ],
    )

    assert main() == 0
    assert captured["require_complete"] is True
    assert captured["dry_run"] is True
    assert captured["confirm_release"] is False
    assert captured["expected_kit_sha256"] == "a" * 64
    assert captured["expected_source_revision"] == "abc123"
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["capture_id"] == "phone-a"
    assert payload["batch"]["planned_captures"] == 60
    rendered = capsys.readouterr().out
    assert "Capture ID: phone-a" in rendered
    assert "Planned captures: 60" in rendered
    assert "Dry run: yes" in rendered



def test_evaluate_remote_sources_cli_forwards_limits_and_writes_report(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import lao_document_ocr.cli as cli
    import lao_document_ocr.remote_evaluation as remote_evaluation

    captured = {}

    class Engine:
        def metadata(self):
            return {"name": "test-engine"}

    class Resolver:
        def metadata(self):
            return {"name": "test-resolver"}

    def fake_evaluate_remote_sources(registry, **kwargs):
        captured["registry"] = registry
        captured.update(kwargs)
        return {
            "schema_version": "1",
            "report_type": "remote-source-diagnostic",
            "not_benchmark_accuracy": True,
            "summary": {"sources": 1, "ok": 1, "errors": 0},
            "sources": [],
        }

    monkeypatch.setattr(cli, "_build_ocr_engine", lambda args: Engine())
    monkeypatch.setattr(
        cli,
        "_build_reading_order_resolver",
        lambda args: Resolver(),
    )
    monkeypatch.setattr(
        remote_evaluation,
        "evaluate_remote_sources",
        fake_evaluate_remote_sources,
    )

    registry = tmp_path / "registry.json"
    registry.write_text('{"schema_version":"1","sources":[]}', encoding="utf-8")
    output = tmp_path / "remote-report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "evaluate-remote-sources",
            "--registry",
            str(registry),
            "--source-id",
            "candidate-a",
            "--source-id",
            "candidate-b",
            "--page",
            "2",
            "--page",
            "5",
            "--max-pages-per-source",
            "4",
            "--max-source-mb",
            "7.5",
            "--timeout-seconds",
            "8",
            "--max-document-pages",
            "90",
            "--max-page-pixels",
            "123456",
            "--output",
            str(output),
        ],
    )

    assert cli.main() == 0
    assert captured["registry"] == registry
    assert captured["source_ids"] == ["candidate-a", "candidate-b"]
    assert captured["all_remote"] is False
    assert captured["requested_pages"] == [2, 5]
    assert captured["max_pages_per_source"] == 4
    assert captured["max_source_bytes"] == int(7.5 * 1024 * 1024)
    assert captured["timeout_seconds"] == 8.0
    assert captured["max_document_pages"] == 90
    assert captured["max_page_pixels"] == 123456
    assert captured["reading_order_resolver"].metadata()["name"] == "test-resolver"

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["not_benchmark_accuracy"] is True
    rendered = capsys.readouterr().out
    assert "diagnostic only" in rendered
    assert "Errors: 0" in rendered


def test_evaluate_remote_sources_cli_returns_nonzero_on_source_error(
    tmp_path,
    monkeypatch,
) -> None:
    import lao_document_ocr.cli as cli
    import lao_document_ocr.remote_evaluation as remote_evaluation

    monkeypatch.setattr(cli, "_build_ocr_engine", lambda args: object())
    monkeypatch.setattr(cli, "_build_reading_order_resolver", lambda args: None)
    monkeypatch.setattr(
        remote_evaluation,
        "evaluate_remote_sources",
        lambda *args, **kwargs: {
            "schema_version": "1",
            "report_type": "remote-source-diagnostic",
            "not_benchmark_accuracy": True,
            "summary": {"sources": 1, "ok": 0, "errors": 1},
            "sources": [],
        },
    )

    output = tmp_path / "remote-report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "evaluate-remote-sources",
            "--source-id",
            "candidate-a",
            "--output",
            str(output),
        ],
    )

    assert cli.main() == 1



def test_evaluate_remote_suite_cli_forwards_paths_and_writes_report(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import lao_document_ocr.cli as cli
    import lao_document_ocr.remote_evaluation_suite as remote_suite

    captured = {}

    class Engine:
        def metadata(self):
            return {"name": "test-engine"}

    class Resolver:
        def metadata(self):
            return {"name": "test-resolver"}

    def fake_evaluate_remote_diagnostic_suite(
        suite_path,
        registry_path,
        **kwargs,
    ):
        captured["suite_path"] = suite_path
        captured["registry_path"] = registry_path
        captured.update(kwargs)
        return {
            "schema_version": "1",
            "report_type": "remote-source-diagnostic-suite",
            "not_benchmark_accuracy": True,
            "suite": {"id": "suite-v1", "description": "test", "path": str(suite_path)},
            "summary": {
                "sources": 2,
                "ok": 2,
                "errors": 0,
                "sampled_pages": 3,
                "native_lao_characters": 0,
                "ocr_lao_characters": 100,
                "ocr_minus_native_lao_characters": 100,
                "ocr_elapsed_seconds": 1.0,
                "layer_gap_classifications": {"native-layer-empty": 3},
                "registry_text_layers": {"absent": 2},
                "source_statuses": {"ok": 2},
            },
            "sources": [],
        }

    monkeypatch.setattr(cli, "_build_ocr_engine", lambda args: Engine())
    monkeypatch.setattr(
        cli,
        "_build_reading_order_resolver",
        lambda args: Resolver(),
    )
    monkeypatch.setattr(
        remote_suite,
        "evaluate_remote_diagnostic_suite",
        fake_evaluate_remote_diagnostic_suite,
    )

    suite = tmp_path / "suite.json"
    suite.write_text("{}", encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text("{}", encoding="utf-8")
    output = tmp_path / "suite-report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "evaluate-remote-suite",
            "--suite",
            str(suite),
            "--registry",
            str(registry),
            "--auto-orient-right-angles",
            "--no-rotation-probes",
            "--output",
            str(output),
        ],
    )

    assert cli.main() == 0
    assert captured["suite_path"] == suite
    assert captured["registry_path"] == registry
    assert captured["engine"].metadata()["name"] == "test-engine"
    assert captured["reading_order_resolver"].metadata()["name"] == "test-resolver"
    assert captured["auto_orient_right_angles"] is True
    assert captured["enable_rotation_probes"] is False

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["report_type"] == "remote-source-diagnostic-suite"
    rendered = capsys.readouterr().out
    assert "Suite: suite-v1" in rendered
    assert "Sampled pages: 3" in rendered
    assert "OCR - native Lao chars: 100" in rendered


def test_evaluate_remote_suite_cli_returns_nonzero_on_partial_failure(
    tmp_path,
    monkeypatch,
) -> None:
    import lao_document_ocr.cli as cli
    import lao_document_ocr.remote_evaluation_suite as remote_suite

    monkeypatch.setattr(cli, "_build_ocr_engine", lambda args: object())
    monkeypatch.setattr(cli, "_build_reading_order_resolver", lambda args: None)
    monkeypatch.setattr(
        remote_suite,
        "evaluate_remote_diagnostic_suite",
        lambda *args, **kwargs: {
            "schema_version": "1",
            "report_type": "remote-source-diagnostic-suite",
            "not_benchmark_accuracy": True,
            "suite": {"id": "suite-v1", "description": "", "path": "suite.json"},
            "summary": {
                "sources": 2,
                "ok": 1,
                "errors": 1,
                "sampled_pages": 1,
                "native_lao_characters": 0,
                "ocr_lao_characters": 0,
                "ocr_minus_native_lao_characters": 0,
                "ocr_elapsed_seconds": 0.0,
                "layer_gap_classifications": {},
                "registry_text_layers": {},
                "source_statuses": {"ok": 1, "error": 1},
            },
            "sources": [],
        },
    )

    output = tmp_path / "suite-report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "evaluate-remote-suite",
            "--output",
            str(output),
        ],
    )

    assert cli.main() == 1
