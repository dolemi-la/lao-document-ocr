from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from lao_document_ocr.models import BoundingBox


def _merge_line_boxes(
    boxes: list[BoundingBox],
    *,
    max_horizontal_gap: int,
) -> list[BoundingBox]:
    if not boxes:
        return []

    ordered = sorted(boxes, key=lambda box: (box.y, box.x))
    merged: list[BoundingBox] = []

    for box in ordered:
        candidate_index: int | None = None
        best_overlap = 0.0

        for index in range(len(merged) - 1, -1, -1):
            current = merged[index]
            current_bottom = current.y + current.height
            box_bottom = box.y + box.height
            overlap = max(0, min(current_bottom, box_bottom) - max(current.y, box.y))
            min_height = max(1, min(current.height, box.height))
            overlap_ratio = overlap / min_height

            current_right = current.x + current.width
            box_right = box.x + box.width
            horizontal_gap = max(0, max(current.x, box.x) - min(current_right, box_right))

            if overlap_ratio >= 0.55 and horizontal_gap <= max_horizontal_gap:
                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    candidate_index = index

        if candidate_index is None:
            merged.append(box)
            continue

        current = merged[candidate_index]
        left = min(current.x, box.x)
        top = min(current.y, box.y)
        right = max(current.x + current.width, box.x + box.width)
        bottom = max(current.y + current.height, box.y + box.height)
        merged[candidate_index] = BoundingBox(
            x=left,
            y=top,
            width=right - left,
            height=bottom - top,
        )

    return sorted(merged, key=lambda box: (box.y, box.x))


def detect_text_lines(image: Image.Image) -> list[BoundingBox]:
    """Detect likely printed text lines using deterministic morphology.

    This baseline intentionally favors clean scans. It is not a replacement for
    a learned layout detector and does not attempt multi-column reading order.
    """
    gray = np.asarray(image.convert("L"))
    if gray.size == 0:
        return []

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    height, width = binary.shape
    horizontal_kernel = max(15, min(80, width // 28))
    vertical_kernel = max(2, min(5, height // 400 + 2))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_kernel, vertical_kernel),
    )
    joined = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_width = max(12, width // 100)
    min_height = max(6, height // 300)
    max_height = max(min_height + 1, int(height * 0.2))

    boxes: list[BoundingBox] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width < min_width:
            continue
        if box_height < min_height or box_height > max_height:
            continue

        margin_x = max(2, horizontal_kernel // 4)
        margin_y = max(2, box_height // 6)
        left = max(0, x - margin_x)
        top = max(0, y - margin_y)
        right = min(width, x + box_width + margin_x)
        bottom = min(height, y + box_height + margin_y)

        boxes.append(
            BoundingBox(
                x=left,
                y=top,
                width=right - left,
                height=bottom - top,
            )
        )

    return _merge_line_boxes(boxes, max_horizontal_gap=horizontal_kernel * 2)
