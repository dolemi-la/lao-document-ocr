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


def test_layout_detector_extracts_image_class_as_visual_block(tmp_path) -> None:
    from types import MethodType

    image = Image.new("RGB", (100, 80), "white")
    pixels = np.asarray(image).copy()
    pixels[25:60, 30:80] = [80, 140, 210]
    image = Image.fromarray(pixels)
    _, transform = prepare_layout_inference_image(
        image,
        image_height=64,
        image_width=64,
    )
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:48, 18:52] = LAYOUT_CLASS_IDS["image"]

    detector = object.__new__(ExportedLayoutRegionDetector)
    detector.min_region_area_ratio = 0.0001
    detector._cached_image = None
    detector._cached_restored_mask = None

    def fake_predict(self, source):
        return mask, transform

    detector._predict_mask = MethodType(fake_predict, detector)
    blocks = detector.detect_visual_blocks(
        image,
        source_image=image,
    )

    assert len(blocks) == 1
    block = blocks[0]
    assert block.type.value == "image"
    assert block.metadata["source"] == "learned-layout-image"
    assert block.metadata["semantic_class"] == "image"
    assert block.metadata["image_base64"]
    assert 0 < block.metadata["area_ratio"] < 0.75


def test_layout_detector_reuses_cached_mask_for_text_and_visual_regions() -> None:
    from types import MethodType

    image = Image.new("RGB", (64, 64), "white")
    _, transform = prepare_layout_inference_image(
        image,
        image_height=64,
        image_width=64,
    )
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[4:18, 4:40] = LAYOUT_CLASS_IDS["paragraph"]
    mask[30:55, 10:50] = LAYOUT_CLASS_IDS["image"]
    calls = {"count": 0}

    detector = object.__new__(ExportedLayoutRegionDetector)
    detector.min_region_area_ratio = 0.0001
    detector._cached_image = None
    detector._cached_restored_mask = None

    def fake_predict(self, source):
        calls["count"] += 1
        return mask, transform

    detector._predict_mask = MethodType(fake_predict, detector)

    text_regions = detector.detect(image)
    visual_blocks = detector.detect_visual_blocks(image)

    assert len(text_regions) == 1
    assert len(visual_blocks) == 1
    assert calls["count"] == 1


def test_layout_visual_block_is_rejected_on_large_excluded_overlap() -> None:
    from types import MethodType

    from lao_document_ocr.models import BoundingBox

    image = Image.new("RGB", (64, 64), "white")
    _, transform = prepare_layout_inference_image(
        image,
        image_height=64,
        image_width=64,
    )
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[12:48, 12:52] = LAYOUT_CLASS_IDS["image"]

    detector = object.__new__(ExportedLayoutRegionDetector)
    detector.min_region_area_ratio = 0.0001
    detector._cached_image = None
    detector._cached_restored_mask = None

    def fake_predict(self, source):
        return mask, transform

    detector._predict_mask = MethodType(fake_predict, detector)
    blocks = detector.detect_visual_blocks(
        image,
        exclude_boxes=[BoundingBox(x=8, y=8, width=50, height=50)],
    )

    assert blocks == []
