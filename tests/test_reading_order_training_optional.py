from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from lao_document_ocr.models import (  # noqa: E402
    Block,
    BlockType,
    BoundingBox,
    Document,
    Page,
)
from lao_document_ocr.reading_order_training import (  # noqa: E402
    MODEL_VERSION,
    ReadingOrderTrainingConfig,
    build_pair_examples,
    export_reading_order_model,
    train_reading_order_model,
)


def _write_fixture(tmp_path: Path):
    dataset = tmp_path / "dataset"
    layout_dir = dataset / "layout"
    data_dir = dataset / "data"
    layout_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    entries = []
    for index, split in enumerate(("train", "dev"), start=1):
        image = data_dir / f"sample-{index}.png"
        image.write_bytes(b"placeholder")
        document = Document(
            pages=[
                Page(
                    number=1,
                    width=600,
                    height=800,
                    blocks=[
                        Block(
                            type=BlockType.HEADING,
                            text="Heading",
                            bbox=BoundingBox(
                                x=50,
                                y=30,
                                width=500,
                                height=50,
                            ),
                        ),
                        Block(
                            type=BlockType.PARAGRAPH,
                            text="Left",
                            bbox=BoundingBox(
                                x=50,
                                y=130,
                                width=220,
                                height=80,
                            ),
                        ),
                        Block(
                            type=BlockType.PARAGRAPH,
                            text="Right",
                            bbox=BoundingBox(
                                x=330,
                                y=130,
                                width=220,
                                height=80,
                            ),
                        ),
                    ],
                )
            ]
        )
        layout_path = layout_dir / f"sample-{index}.json"
        layout_path.write_text(
            document.model_dump_json(indent=2),
            encoding="utf-8",
        )
        entries.append(
            {
                "id": f"sample-{index}",
                "document_id": f"doc-{index}",
                "image": f"data/sample-{index}.png",
                "layout_ground_truth": f"layout/sample-{index}.json",
                "split": split,
                "subset": "clean-print",
                "tags": ["layout:multi-column"],
                "sha256": None,
                "width": 600,
                "height": 800,
                "block_counts": {
                    "heading": 1,
                    "paragraph": 2,
                },
            }
        )

    manifest = tmp_path / "layout-training.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )
    return dataset, manifest


def test_pair_examples_are_balanced_by_direction(tmp_path) -> None:
    dataset, manifest = _write_fixture(tmp_path)

    from lao_document_ocr.layout_training_manifest import (
        load_layout_training_manifest,
    )

    entries = load_layout_training_manifest(manifest)
    examples = build_pair_examples(
        [entries[0]],
        dataset_root=dataset,
    )

    labels = [example.label for example in examples]
    assert len(examples) == 6
    assert labels.count(1.0) == 3
    assert labels.count(0.0) == 3


def test_one_epoch_training_and_export_smoke(tmp_path) -> None:
    dataset, manifest = _write_fixture(tmp_path)
    output = tmp_path / "run"

    result = train_reading_order_model(
        manifest,
        dataset,
        output,
        training_config=ReadingOrderTrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            hidden_size=16,
            device="cpu",
        ),
    )

    assert result["checkpoint"].is_file()
    assert result["metadata"].is_file()
    metadata = json.loads(
        result["metadata"].read_text(encoding="utf-8")
    )
    assert metadata["model_version"] == MODEL_VERSION
    assert metadata["train_pages"] == 1
    assert metadata["dev_pages"] == 1
    assert metadata["train_pairs"] == 6
    assert metadata["dev_pairs"] == 6
    assert 0 <= metadata["best_dev_pair_accuracy"] <= 1
    assert len(metadata["checkpoint_sha256"]) == 64

    artifact = export_reading_order_model(
        result["checkpoint"],
        output / "reading-order.pt2",
    )
    assert artifact.is_file()
    sidecar = json.loads(
        artifact.with_suffix(".pt2.json").read_text(
            encoding="utf-8"
        )
    )
    assert sidecar["model_version"] == MODEL_VERSION
    assert len(sidecar["artifact_sha256"]) == 64


def test_exported_training_artifact_supports_batched_inference(tmp_path) -> None:
    from lao_document_ocr.layout_ground_truth import load_layout_ground_truth
    from lao_document_ocr.reading_order_inference import (
        ExportedReadingOrderResolver,
    )

    dataset, manifest = _write_fixture(tmp_path)
    output = tmp_path / "run-batched"
    result = train_reading_order_model(
        manifest,
        dataset,
        output,
        training_config=ReadingOrderTrainingConfig(
            epochs=1,
            batch_size=2,
            hidden_size=16,
            device="cpu",
        ),
    )
    artifact = export_reading_order_model(
        result["checkpoint"],
        output / "reading-order.pt2",
    )
    resolver = ExportedReadingOrderResolver(artifact, device="cpu")
    page = load_layout_ground_truth(
        dataset / "layout" / "sample-2.json"
    ).pages[0]

    ordered = resolver.order(
        page.blocks,
        page_width=page.width,
        page_height=page.height,
    )

    assert len(ordered) == len(page.blocks)
    assert {block.text for block in ordered} == {block.text for block in page.blocks}
