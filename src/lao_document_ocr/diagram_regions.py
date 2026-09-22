from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image

from lao_document_ocr.models import Block, BlockType, BoundingBox
from lao_document_ocr.ocr.base import RecognizedLine


def _mask_box(mask: np.ndarray, box: BoundingBox, *, margin: int = 4) -> None:
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




def _union_box(left: BoundingBox, right: BoundingBox) -> BoundingBox:
    x0 = min(left.x, right.x)
    y0 = min(left.y, right.y)
    x1 = max(left.x + left.width, right.x + right.width)
    y1 = max(left.y + left.height, right.y + right.height)
    return BoundingBox(x=x0, y=y0, width=x1 - x0, height=y1 - y0)


def _axis_overlap(start_a: int, length_a: int, start_b: int, length_b: int) -> int:
    return max(
        0,
        min(start_a + length_a, start_b + length_b) - max(start_a, start_b),
    )


def _axis_gap(start_a: int, length_a: int, start_b: int, length_b: int) -> int:
    return max(
        0,
        max(start_a, start_b) - min(start_a + length_a, start_b + length_b),
    )


def _merge_component_boxes(
    boxes: list[BoundingBox],
    *,
    page_width: int,
    page_height: int,
) -> list[BoundingBox]:
    if not boxes:
        return []

    horizontal_gap_limit = max(10, int(page_width * 0.02))
    vertical_gap_limit = max(10, int(page_height * 0.03))
    merged = sorted(boxes, key=lambda box: (box.y, box.x))

    changed = True
    while changed:
        changed = False
        output: list[BoundingBox] = []
        consumed = [False] * len(merged)
        for index, current in enumerate(merged):
            if consumed[index]:
                continue
            aggregate = current
            consumed[index] = True

            for other_index in range(index + 1, len(merged)):
                if consumed[other_index]:
                    continue
                other = merged[other_index]

                horizontal_overlap = _axis_overlap(
                    aggregate.x,
                    aggregate.width,
                    other.x,
                    other.width,
                )
                vertical_overlap = _axis_overlap(
                    aggregate.y,
                    aggregate.height,
                    other.y,
                    other.height,
                )
                horizontal_ratio = horizontal_overlap / max(
                    1, min(aggregate.width, other.width)
                )
                vertical_ratio = vertical_overlap / max(
                    1, min(aggregate.height, other.height)
                )
                horizontal_gap = _axis_gap(
                    aggregate.x,
                    aggregate.width,
                    other.x,
                    other.width,
                )
                vertical_gap = _axis_gap(
                    aggregate.y,
                    aggregate.height,
                    other.y,
                    other.height,
                )

                stacked = horizontal_ratio >= 0.60 and vertical_gap <= vertical_gap_limit
                side_by_side = vertical_ratio >= 0.60 and horizontal_gap <= horizontal_gap_limit
                if stacked or side_by_side:
                    aggregate = _union_box(aggregate, other)
                    consumed[other_index] = True
                    changed = True

            output.append(aggregate)

        merged = sorted(output, key=lambda box: (box.y, box.x))

    return merged

def detect_diagram_regions(
    image: Image.Image,
    lines: list[RecognizedLine],
    *,
    source_image: Image.Image | None = None,
    exclude_boxes: list[BoundingBox] | None = None,
    min_area_ratio: float = 0.006,
    max_area_ratio: float = 0.50,
) -> list[Block]:
    """Detect conservative line-art/diagram regions after masking text.

    This baseline targets compact edge-heavy regions such as simple diagrams,
    boxed illustrations, and connector-heavy line art. Editable tables and
    already-detected visual regions should be supplied through exclude_boxes.
    """
    detection_image = image.convert("RGB")
    crop_source = (source_image or image).convert("RGB")
    if crop_source.size != detection_image.size:
        crop_source = detection_image

    rgb = np.asarray(detection_image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    page_area = max(1, height * width)

    edges = cv2.Canny(gray, 55, 150)

    for line in lines:
        _mask_box(
            edges,
            line.bbox,
            margin=max(5, line.bbox.height // 3),
        )
    for box in exclude_boxes or []:
        _mask_box(edges, box, margin=6)

    connector_kernel = max(3, min(9, min(width, height) // 120))
    if connector_kernel % 2 == 0:
        connector_kernel += 1
    connected = cv2.dilate(
        edges,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (connector_kernel, connector_kernel),
        ),
        iterations=1,
    )

    close_width = max(11, min(41, width // 28))
    close_height = max(11, min(41, height // 28))
    grouped = cv2.morphologyEx(
        connected,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (close_width, close_height),
        ),
        iterations=1,
    )

    contours, _ = cv2.findContours(
        grouped,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidate_boxes: list[BoundingBox] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        margin = max(4, connector_kernel)
        left = max(0, x - margin)
        top = max(0, y - margin)
        right = min(width, x + box_width + margin)
        bottom = min(height, y + box_height + margin)
        candidate_boxes.append(
            BoundingBox(
                x=left,
                y=top,
                width=right - left,
                height=bottom - top,
            )
        )

    candidate_boxes = _merge_component_boxes(
        candidate_boxes,
        page_width=width,
        page_height=height,
    )

    blocks: list[Block] = []
    for candidate in candidate_boxes:
        left = candidate.x
        top = candidate.y
        box_width = candidate.width
        box_height = candidate.height
        right = left + box_width
        bottom = top + box_height

        if box_width < max(60, width // 18):
            continue
        if box_height < max(45, height // 22):
            continue

        area_ratio = (box_width * box_height) / page_area
        if area_ratio < min_area_ratio or area_ratio >= max_area_ratio:
            continue

        crop_edges = edges[top:bottom, left:right]
        crop_gray = gray[top:bottom, left:right]
        if crop_edges.size == 0:
            continue

        edge_density = float(np.count_nonzero(crop_edges) / crop_edges.size)
        dark_density = float(np.mean(crop_gray < 235))

        if edge_density < 0.012:
            continue
        if dark_density > 0.38:
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
                    "source": "diagram-region",
                    "detector": "edge-line-art-v1",
                    "media_type": "image/png",
                    "image_base64": _encode_crop(crop),
                    "width_ratio": box_width / width,
                    "area_ratio": area_ratio,
                    "edge_density": edge_density,
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
