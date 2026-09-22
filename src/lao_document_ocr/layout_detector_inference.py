from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from lao_document_ocr.layout_targets import LAYOUT_CLASS_IDS
from lao_document_ocr.models import Block, BlockType, BoundingBox
from lao_document_ocr.recognizer_inference import resolve_torch_device
from lao_document_ocr.text_regions import TextRegion

_SEMANTIC_CLASS_TYPES = {
    LAYOUT_CLASS_IDS["heading"]: BlockType.HEADING,
    LAYOUT_CLASS_IDS["paragraph"]: BlockType.PARAGRAPH,
    LAYOUT_CLASS_IDS["list"]: BlockType.LIST,
    LAYOUT_CLASS_IDS["table"]: BlockType.TABLE,
}


@dataclass(frozen=True)
class LayoutLetterboxTransform:
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    resized_width: int
    resized_height: int
    offset_x: int
    offset_y: int


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_layout_inference_image(
    image: Image.Image,
    *,
    image_height: int,
    image_width: int,
) -> tuple[np.ndarray, LayoutLetterboxTransform]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if image.width < 1 or image.height < 1:
        raise ValueError("Invalid layout inference image dimensions")

    scale = min(
        image_width / image.width,
        image_height / image.height,
    )
    resized_width = max(1, round(image.width * scale))
    resized_height = max(1, round(image.height * scale))
    resized = image.resize(
        (resized_width, resized_height),
        Image.Resampling.BILINEAR,
    )

    canvas = Image.new(
        "RGB",
        (image_width, image_height),
        "white",
    )
    offset_x = (image_width - resized_width) // 2
    offset_y = (image_height - resized_height) // 2
    canvas.paste(resized, (offset_x, offset_y))

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return array, LayoutLetterboxTransform(
        source_width=image.width,
        source_height=image.height,
        target_width=image_width,
        target_height=image_height,
        resized_width=resized_width,
        resized_height=resized_height,
        offset_x=offset_x,
        offset_y=offset_y,
    )


def restore_layout_mask(
    mask: np.ndarray,
    transform: LayoutLetterboxTransform,
) -> np.ndarray:
    if mask.shape != (
        transform.target_height,
        transform.target_width,
    ):
        raise ValueError(
            "Predicted layout mask does not match model canvas dimensions"
        )

    cropped = mask[
        transform.offset_y : transform.offset_y + transform.resized_height,
        transform.offset_x : transform.offset_x + transform.resized_width,
    ]
    restored = Image.fromarray(cropped.astype(np.uint8)).resize(
        (
            transform.source_width,
            transform.source_height,
        ),
        Image.Resampling.NEAREST,
    )
    return np.asarray(restored, dtype=np.uint8)


class ExportedLayoutRegionDetector:
    def __init__(
        self,
        artifact_path: str | Path,
        *,
        device: str = "cpu",
        confidence_threshold: float = 0.55,
        min_region_area_ratio: float = 0.0005,
    ) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0 < min_region_area_ratio < 1:
            raise ValueError("min_region_area_ratio must be in (0, 1)")

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Learned layout inference requires the optional 'train' dependencies. "
                "Install with: pip install -e '.[train]'"
            ) from exc

        self.artifact_path = Path(artifact_path)
        metadata_path = self.artifact_path.with_suffix(
            self.artifact_path.suffix + ".json"
        )
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Layout detector metadata not found: {metadata_path}"
            )

        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        if metadata.get("format") != "torch-export":
            raise ValueError("Unsupported layout detector artifact format")
        expected_sha256 = metadata.get("artifact_sha256")
        if (
            expected_sha256
            and _sha256_file(self.artifact_path) != expected_sha256
        ):
            raise ValueError(
                "Layout detector artifact SHA-256 does not match metadata"
            )

        configured_classes = {
            str(name): int(class_id)
            for name, class_id in dict(
                metadata.get("class_ids", {})
            ).items()
        }
        if configured_classes != LAYOUT_CLASS_IDS:
            raise ValueError(
                "Layout detector class map does not match this runtime"
            )

        model_config = metadata["model_config"]
        self.image_height = int(model_config["image_height"])
        self.image_width = int(model_config["image_width"])
        self.device = resolve_torch_device(device)
        self.confidence_threshold = confidence_threshold
        self.min_region_area_ratio = min_region_area_ratio
        self.metadata_payload = {
            **metadata,
            "runtime_device": str(self.device),
            "confidence_threshold": confidence_threshold,
        }

        exported = torch.export.load(str(self.artifact_path))
        self.model = exported.module().to(self.device)
        self._cached_image: Image.Image | None = None
        self._cached_restored_mask: np.ndarray | None = None

    def metadata(self) -> dict:
        return {
            "name": self.__class__.__name__,
            "version": self.metadata_payload.get("model_version"),
            "device": str(self.device),
            "confidence_threshold": self.confidence_threshold,
        }

    def _predict_mask(
        self,
        image: Image.Image,
    ) -> tuple[np.ndarray, LayoutLetterboxTransform]:
        import torch

        array, transform = prepare_layout_inference_image(
            image,
            image_height=self.image_height,
            image_width=self.image_width,
        )
        tensor = (
            torch.from_numpy(array)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = logits.softmax(dim=1)
            confidence, classes = probabilities.max(dim=1)

        predicted = classes[0].detach().cpu().numpy().astype(np.uint8)
        confidence_map = confidence[0].detach().cpu().numpy()
        predicted[confidence_map < self.confidence_threshold] = 0
        return predicted, transform

    def _restored_mask(self, image: Image.Image) -> np.ndarray:
        if (
            getattr(self, "_cached_image", None) is image
            and getattr(self, "_cached_restored_mask", None) is not None
        ):
            return self._cached_restored_mask

        predicted, transform = self._predict_mask(image)
        restored = restore_layout_mask(
            predicted,
            transform,
        )
        self._cached_image = image
        self._cached_restored_mask = restored
        return restored

    def detect(self, image: Image.Image) -> list[TextRegion]:
        restored = self._restored_mask(image)

        kernel_size = max(
            1,
            min(7, min(image.width, image.height) // 250),
        )
        kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (kernel_size, kernel_size),
            )
            if kernel_size > 1
            else None
        )
        page_area = max(1, image.width * image.height)
        min_area = max(
            16,
            round(page_area * self.min_region_area_ratio),
        )

        regions: list[TextRegion] = []
        for class_id, semantic_type in _SEMANTIC_CLASS_TYPES.items():
            binary = (restored == class_id).astype(np.uint8) * 255
            if not np.any(binary):
                continue
            if kernel is not None:
                binary = cv2.morphologyEx(
                    binary,
                    cv2.MORPH_CLOSE,
                    kernel,
                )

            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for contour in contours:
                x, y, width, height = cv2.boundingRect(contour)
                if width * height < min_area:
                    continue
                margin = max(
                    2,
                    round(min(width, height) * 0.03),
                )
                left = max(0, x - margin)
                top = max(0, y - margin)
                right = min(image.width, x + width + margin)
                bottom = min(image.height, y + height + margin)
                regions.append(
                    TextRegion(
                        bbox=BoundingBox(
                            x=left,
                            y=top,
                            width=right - left,
                            height=bottom - top,
                        ),
                        detector="tiny-layout-unet-v1",
                        semantic_type=semantic_type,
                    )
                )

        return sorted(
            regions,
            key=lambda region: (
                region.bbox.y,
                region.bbox.x,
            ),
        )


    def detect_visual_blocks(
        self,
        image: Image.Image,
        *,
        source_image: Image.Image | None = None,
        exclude_boxes: list[BoundingBox] | None = None,
        max_area_ratio: float = 0.75,
    ) -> list[Block]:
        if not 0 < max_area_ratio <= 1:
            raise ValueError("max_area_ratio must be in (0, 1]")

        restored = self._restored_mask(image)
        image_class = LAYOUT_CLASS_IDS["image"]
        binary = (restored == image_class).astype(np.uint8) * 255
        if not np.any(binary):
            return []

        kernel_size = max(
            1,
            min(9, min(image.width, image.height) // 220),
        )
        if kernel_size > 1:
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(
                    cv2.MORPH_RECT,
                    (kernel_size, kernel_size),
                ),
            )

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        page_area = max(1, image.width * image.height)
        min_area = max(
            16,
            round(page_area * self.min_region_area_ratio),
        )
        crop_source = (source_image or image).convert("RGB")
        if crop_source.size != image.size:
            crop_source = image.convert("RGB")

        def overlap_ratio(box: BoundingBox) -> float:
            area = max(1, box.width * box.height)
            total_overlap = 0
            for excluded in exclude_boxes or []:
                left = max(box.x, excluded.x)
                top = max(box.y, excluded.y)
                right = min(
                    box.x + box.width,
                    excluded.x + excluded.width,
                )
                bottom = min(
                    box.y + box.height,
                    excluded.y + excluded.height,
                )
                total_overlap += (
                    max(0, right - left)
                    * max(0, bottom - top)
                )
            return min(1.0, total_overlap / area)

        blocks: list[Block] = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if width * height < min_area:
                continue
            margin = max(
                2,
                round(min(width, height) * 0.03),
            )
            left = max(0, x - margin)
            top = max(0, y - margin)
            right = min(image.width, x + width + margin)
            bottom = min(image.height, y + height + margin)
            box = BoundingBox(
                x=left,
                y=top,
                width=right - left,
                height=bottom - top,
            )
            area_ratio = (box.width * box.height) / page_area
            if area_ratio >= max_area_ratio:
                continue
            if overlap_ratio(box) >= 0.35:
                continue

            crop = crop_source.crop(
                (
                    box.x,
                    box.y,
                    box.x + box.width,
                    box.y + box.height,
                )
            )
            buffer = io.BytesIO()
            crop.save(buffer, format="PNG", optimize=True)
            blocks.append(
                Block(
                    type=BlockType.IMAGE,
                    bbox=box,
                    metadata={
                        "source": "learned-layout-image",
                        "detector": "tiny-layout-unet-v1",
                        "semantic_class": "image",
                        "media_type": "image/png",
                        "image_base64": base64.b64encode(
                            buffer.getvalue()
                        ).decode("ascii"),
                        "width_ratio": box.width / image.width,
                        "area_ratio": area_ratio,
                    },
                )
            )

        return sorted(
            blocks,
            key=lambda block: (
                block.bbox.y if block.bbox else 0,
                block.bbox.x if block.bbox else 0,
            ),
        )
