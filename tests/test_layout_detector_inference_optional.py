from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")

import torch  # noqa: E402

from lao_document_ocr.layout_detector_inference import (  # noqa: E402
    ExportedLayoutRegionDetector,
    prepare_layout_inference_image,
    restore_layout_mask,
)
from lao_document_ocr.layout_model import (  # noqa: E402
    LayoutSegmentationConfig,
    TinyLayoutUNet,
)
from lao_document_ocr.layout_targets import LAYOUT_CLASS_IDS  # noqa: E402


def _artifact(tmp_path):
    config = LayoutSegmentationConfig(
        image_height=64,
        image_width=64,
        base_channels=8,
        num_classes=6,
    )
    model = TinyLayoutUNet(config)
    model.eval()
    example = torch.zeros((1, 3, 64, 64))
    exported = torch.export.export(model, (example,))
    artifact = tmp_path / "layout.pt2"
    torch.export.save(exported, artifact)
    metadata = {
        "schema_version": "1",
        "format": "torch-export",
        "model": "TinyLayoutUNet",
        "model_version": "tiny-layout-unet-v1",
        "artifact": artifact.name,
        "artifact_sha256": hashlib.sha256(
            artifact.read_bytes()
        ).hexdigest(),
        "model_config": config.to_dict(),
        "class_ids": LAYOUT_CLASS_IDS,
    }
    artifact.with_suffix(".pt2.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    return artifact


def test_layout_inference_letterbox_round_trip_mask() -> None:
    image = Image.new("RGB", (100, 50), "white")
    _, transform = prepare_layout_inference_image(
        image,
        image_height=64,
        image_width=64,
    )

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[
        transform.offset_y : transform.offset_y + transform.resized_height,
        transform.offset_x : transform.offset_x + transform.resized_width,
    ] = LAYOUT_CLASS_IDS["paragraph"]

    restored = restore_layout_mask(mask, transform)

    assert restored.shape == (50, 100)
    assert np.all(
        restored == LAYOUT_CLASS_IDS["paragraph"]
    )


def test_exported_layout_detector_loads_on_cpu(tmp_path) -> None:
    detector = ExportedLayoutRegionDetector(
        _artifact(tmp_path),
        device="cpu",
        confidence_threshold=0.0,
    )

    assert detector.device.type == "cpu"
    assert detector.metadata()["version"] == "tiny-layout-unet-v1"
    assert detector.metadata()["device"] == "cpu"


def test_layout_detector_rejects_tampered_artifact(tmp_path) -> None:
    artifact = _artifact(tmp_path)
    artifact.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        ExportedLayoutRegionDetector(artifact)


def test_layout_detector_preserves_semantic_region_classes(tmp_path) -> None:
    from types import MethodType

    from lao_document_ocr.models import BlockType

    image = Image.new("RGB", (64, 64), "white")
    _, transform = prepare_layout_inference_image(
        image,
        image_height=64,
        image_width=64,
    )
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[4:14, 4:28] = LAYOUT_CLASS_IDS["heading"]
    mask[18:30, 4:40] = LAYOUT_CLASS_IDS["paragraph"]
    mask[34:44, 4:36] = LAYOUT_CLASS_IDS["list"]
    mask[48:60, 4:52] = LAYOUT_CLASS_IDS["table"]

    detector = object.__new__(ExportedLayoutRegionDetector)
    detector.min_region_area_ratio = 0.0001

    def fake_predict(self, source):
        return mask, transform

    detector._predict_mask = MethodType(fake_predict, detector)
    regions = detector.detect(image)

    assert [region.semantic_type for region in regions] == [
        BlockType.HEADING,
        BlockType.PARAGRAPH,
        BlockType.LIST,
        BlockType.TABLE,
    ]
    assert all(region.detector == "tiny-layout-unet-v1" for region in regions)
