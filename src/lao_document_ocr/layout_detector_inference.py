from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from lao_document_ocr.layout_targets import LAYOUT_CLASS_IDS
from lao_document_ocr.models import BoundingBox
from lao_document_ocr.recognizer_inference import resolve_torch_device
from lao_document_ocr.text_regions import TextRegion


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

    def detect(self, image: Image.Image) -> list[TextRegion]:
        predicted, transform = self._predict_mask(image)
        restored = restore_layout_mask(
            predicted,
            transform,
        )

        text_class_ids = {
            LAYOUT_CLASS_IDS["heading"],
            LAYOUT_CLASS_IDS["paragraph"],
            LAYOUT_CLASS_IDS["list"],
            LAYOUT_CLASS_IDS["table"],
        }
        binary = np.isin(
            restored,
            list(text_class_ids),
        ).astype(np.uint8) * 255

        kernel_size = max(
            1,
            min(7, min(image.width, image.height) // 250),
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

        regions: list[TextRegion] = []
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
                )
            )

        return sorted(
            regions,
            key=lambda region: (
                region.bbox.y,
                region.bbox.x,
            ),
        )
