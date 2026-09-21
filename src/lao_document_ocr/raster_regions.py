from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image

from lao_document_ocr.models import Block, BlockType, BoundingBox
from lao_document_ocr.ocr.base import RecognizedLine


def _mask_box(mask: np.ndarray, box: BoundingBox, margin: int = 4) -> None:
    height, width = mask.shape
    left = max(0, box.x - margin)
    top = max(0, box.y - margin)
    right = min(width, box.x + box.width + margin)
    bottom = min(height, box.y + box.height + margin)
    if right > left and bottom > top:
        mask[top:bottom, left:right] = 0


def _encode_crop(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def detect_raster_regions(
    image: Image.Image,
    lines: list[RecognizedLine],
    *,
    source_image: Image.Image | None = None,
    exclude_boxes: list[BoundingBox] | None = None,
    min_area_ratio: float = 0.008,
    max_area_ratio: float = 0.55,
) -> list[Block]:
    """Detect dense photo/logo-like regions after masking recognized text.

    This is intentionally conservative. It is not a general image/diagram
    segmentation model and avoids near-full-page regions.
    """
    detection_image = image.convert("RGB")
    crop_source = (source_image or image).convert("RGB")
    if crop_source.size != detection_image.size:
        crop_source = detection_image
    rgb = np.asarray(detection_image)
    height, width, _ = rgb.shape
    page_area = max(1, height * width)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    color_range = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)

    content = ((gray < 242) | (color_range > 20)).astype(np.uint8)

    for line in lines:
        _mask_box(content, line.bbox, margin=max(4, line.bbox.height // 4))
    for box in exclude_boxes or []:
        _mask_box(content, box, margin=4)

    kernel_size = max(15, min(61, min(width, height) // 18))
    if kernel_size % 2 == 0:
        kernel_size += 1

    density = cv2.blur(content.astype(np.float32), (kernel_size, kernel_size))
    dense = (density >= 0.30).astype(np.uint8) * 255
    dense = cv2.morphologyEx(
        dense,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)),
    )

    contours, _ = cv2.findContours(dense, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blocks: list[Block] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        margin = max(4, kernel_size // 2)
        left = max(0, x - margin)
        top = max(0, y - margin)
        right = min(width, x + box_width + margin)
        bottom = min(height, y + box_height + margin)
        box_width = right - left
        box_height = bottom - top

        if box_width < max(50, width // 20) or box_height < max(40, height // 25):
            continue

        area_ratio = (box_width * box_height) / page_area
        if area_ratio < min_area_ratio or area_ratio >= max_area_ratio:
            continue

        crop_array = rgb[top:bottom, left:right]
        crop_gray = gray[top:bottom, left:right]
        if crop_array.size == 0:
            continue

        visual_density = float(
            np.mean(
                (crop_gray < 245)
                | (
                    crop_array.max(axis=2).astype(np.int16)
                    - crop_array.min(axis=2).astype(np.int16)
                    > 20
                )
            )
        )
        visual_variation = float(np.std(crop_array.astype(np.float32)))

        if visual_density < 0.22 or visual_variation < 10:
            continue

        crop = crop_source.crop((left, top, right, bottom))
        blocks.append(
            Block(
                type=BlockType.IMAGE,
                bbox=BoundingBox(
                    x=left,
                    y=top,
                    width=box_width,
                    height=box_height,
                ),
                metadata={
                    "source": "raster-region",
                    "detector": "dense-visual-v1",
                    "media_type": "image/png",
                    "image_base64": _encode_crop(crop),
                    "width_ratio": box_width / width,
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
