from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")

import torch  # noqa: E402

from lao_document_ocr.layout_segmentation_training import (  # noqa: E402
    LayoutTrainingConfig,
    compute_layout_class_weights,
    export_layout_detector,
    load_layout_target_samples,
    metrics_from_confusion,
    prepare_layout_image_mask,
    train_layout_detector,
)


def _write_target_fixture(tmp_path):
    dataset = tmp_path / "dataset"
    targets = tmp_path / "targets"
    (dataset / "data").mkdir(parents=True)
    (targets / "masks").mkdir(parents=True)

    entries = []
    for index, split in enumerate(("train", "dev"), start=1):
        image = dataset / "data" / f"sample-{index}.png"
        mask = targets / "masks" / f"sample-{index}.png"

        canvas = Image.new("RGB", (96, 64), "white")
        pixels = np.zeros((64, 96), dtype=np.uint8)
        pixels[12:32, 10:86] = 1 if index == 1 else 2

        canvas.save(image)
        Image.fromarray(pixels).save(mask)

        entries.append(
            {
                "id": f"sample-{index}",
                "image": f"data/sample-{index}.png",
                "layout_ground_truth": f"layout/sample-{index}.json",
                "mask": f"masks/sample-{index}.png",
                "boxes": f"boxes/sample-{index}.json",
                "split": split,
                "subset": "clean-print",
                "tags": ["layout:plain"],
                "width": 96,
                "height": 64,
            }
        )

    manifest = targets / "targets.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )
    return dataset, targets, manifest


def test_prepare_layout_image_mask_letterboxes_without_changing_classes(tmp_path) -> None:
    image = tmp_path / "image.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (100, 50), "white").save(image)
    mask_array = np.zeros((50, 100), dtype=np.uint8)
    mask_array[:, 20:80] = 4
    Image.fromarray(mask_array).save(mask)

    prepared_image, prepared_mask = prepare_layout_image_mask(
        image,
        mask,
        image_height=64,
        image_width=64,
    )

    assert prepared_image.shape == (3, 64, 64)
    assert prepared_mask.shape == (64, 64)
    assert set(np.unique(prepared_mask).tolist()) == {0, 4}


def test_metrics_from_confusion_reports_per_class_iou() -> None:
    confusion = torch.tensor(
        [
            [8, 2, 0],
            [1, 9, 0],
            [0, 0, 0],
        ],
        dtype=torch.int64,
    )

    metrics = metrics_from_confusion(confusion)

    assert metrics["pixel_accuracy"] == pytest.approx(17 / 20)
    assert metrics["per_class_iou"]["background"] == pytest.approx(8 / 11)
    assert metrics["per_class_iou"]["heading"] == pytest.approx(9 / 12)
    assert metrics["per_class_iou"]["paragraph"] is None
    assert metrics["foreground_mean_iou"] == pytest.approx(9 / 12)


def test_layout_target_manifest_loader_resolves_image_and_mask_paths(tmp_path) -> None:
    dataset, _, manifest = _write_target_fixture(tmp_path)

    samples = load_layout_target_samples(
        manifest,
        dataset_root=dataset,
    )

    assert [sample.split for sample in samples] == ["train", "dev"]
    assert all(sample.image.is_file() for sample in samples)
    assert all(sample.mask.is_file() for sample in samples)


def test_layout_detector_one_epoch_smoke_writes_checkpoint(tmp_path) -> None:
    dataset, _, manifest = _write_target_fixture(tmp_path)
    samples = load_layout_target_samples(
        manifest,
        dataset_root=dataset,
    )

    result = train_layout_detector(
        samples,
        tmp_path / "run",
        training_config=LayoutTrainingConfig(
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            image_height=64,
            image_width=64,
            base_channels=8,
            device="cpu",
        ),
    )

    assert result["checkpoint"].is_file()
    assert result["metadata"].is_file()
    metadata = json.loads(
        result["metadata"].read_text(encoding="utf-8")
    )
    assert metadata["model_version"] == "tiny-layout-unet-v1"
    assert metadata["train_samples"] == 1
    assert metadata["dev_samples"] == 1
    assert metadata["device"] == "cpu"
    assert len(metadata["checkpoint_sha256"]) == 64
    assert len(metadata["history"]) == 1
    assert 0 <= metadata["best_dev_foreground_mean_iou"] <= 1

    artifact = export_layout_detector(
        result["checkpoint"],
        tmp_path / "run" / "layout-detector.pt2",
    )
    assert artifact.is_file()
    sidecar = json.loads(
        artifact.with_suffix(".pt2.json").read_text(encoding="utf-8")
    )
    assert sidecar["model_version"] == "tiny-layout-unet-v1"
    assert len(sidecar["artifact_sha256"]) == 64


def test_class_weights_downweight_background(tmp_path) -> None:
    dataset, _, manifest = _write_target_fixture(tmp_path)
    samples = load_layout_target_samples(
        manifest,
        dataset_root=dataset,
    )
    train_samples = [sample for sample in samples if sample.split == "train"]

    weights = compute_layout_class_weights(
        train_samples,
        num_classes=6,
    )

    assert weights.shape == (6,)
    assert weights[0] < weights[1]
    assert weights[2] == 0
    assert weights[3] == 0
    assert weights[4] == 0
    assert weights[5] == 0
