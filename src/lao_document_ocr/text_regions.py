from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from lao_document_ocr.models import BoundingBox


@dataclass(frozen=True)
class TextRegion:
    bbox: BoundingBox
    detector: str = "morphology-region-v1"


class TextRegionDetector(Protocol):
    def detect(self, image: Image.Image) -> list[TextRegion]: ...

    def metadata(self) -> dict: ...


def _union_box(left: BoundingBox, right: BoundingBox) -> BoundingBox:
    x0 = min(left.x, right.x)
    y0 = min(left.y, right.y)
    x1 = max(left.x + left.width, right.x + right.width)
    y1 = max(left.y + left.height, right.y + right.height)
    return BoundingBox(x=x0, y=y0, width=x1 - x0, height=y1 - y0)


def _vertical_overlap_ratio(left: BoundingBox, right: BoundingBox) -> float:
    top = max(left.y, right.y)
    bottom = min(left.y + left.height, right.y + right.height)
    overlap = max(0, bottom - top)
    return overlap / max(1, min(left.height, right.height))


def _horizontal_overlap_ratio(left: BoundingBox, right: BoundingBox) -> float:
    start = max(left.x, right.x)
    end = min(left.x + left.width, right.x + right.width)
    overlap = max(0, end - start)
    return overlap / max(1, min(left.width, right.width))


def _merge_nearby_regions(
    boxes: list[BoundingBox],
    *,
    page_width: int,
    page_height: int,
) -> list[BoundingBox]:
    if not boxes:
        return []

    horizontal_gap_limit = max(12, int(page_width * 0.02))
    vertical_gap_limit = max(10, int(page_height * 0.018))

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

                horizontal_gap = max(
                    0,
                    max(aggregate.x, other.x)
                    - min(
                        aggregate.x + aggregate.width,
                        other.x + other.width,
                    ),
                )
                vertical_gap = max(
                    0,
                    max(aggregate.y, other.y)
                    - min(
                        aggregate.y + aggregate.height,
                        other.y + other.height,
                    ),
                )

                same_row_band = (
                    _vertical_overlap_ratio(aggregate, other) >= 0.60
                    and horizontal_gap <= horizontal_gap_limit
                )
                same_column_band = (
                    _horizontal_overlap_ratio(aggregate, other) >= 0.55
                    and vertical_gap <= vertical_gap_limit
                )

                if same_row_band or same_column_band:
                    aggregate = _union_box(aggregate, other)
                    consumed[other_index] = True
                    changed = True

            output.append(aggregate)

        merged = sorted(output, key=lambda box: (box.y, box.x))

    return merged


class MorphologyTextRegionDetector:
    def __init__(
        self,
        *,
        min_region_width_ratio: float = 0.05,
        min_region_height_ratio: float = 0.025,
        max_region_area_ratio: float = 0.92,
    ) -> None:
        self.min_region_width_ratio = min_region_width_ratio
        self.min_region_height_ratio = min_region_height_ratio
        self.max_region_area_ratio = max_region_area_ratio

    def metadata(self) -> dict:
        return {
            "name": self.__class__.__name__,
            "version": "morphology-region-v1",
        }

    def detect(self, image: Image.Image) -> list[TextRegion]:
        gray = np.asarray(image.convert("L"))
        if gray.size == 0:
            return []

        height, width = gray.shape
        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        line_kernel_width = max(18, min(70, width // 18))
        line_kernel_height = max(2, min(4, height // 500 + 2))
        line_mask = cv2.dilate(
            binary,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (line_kernel_width, line_kernel_height),
            ),
            iterations=1,
        )

        region_kernel_width = max(5, min(18, width // 110))
        region_kernel_height = max(18, min(80, height // 15))
        region_mask = cv2.morphologyEx(
            line_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (region_kernel_width, region_kernel_height),
            ),
            iterations=1,
        )

        contours, _ = cv2.findContours(
            region_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        page_area = max(1, width * height)
        min_width = max(18, int(width * self.min_region_width_ratio))
        min_height = max(12, int(height * self.min_region_height_ratio))

        boxes: list[BoundingBox] = []
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            area_ratio = (box_width * box_height) / page_area
            if box_width < min_width or box_height < min_height:
                continue
            if area_ratio >= self.max_region_area_ratio:
                continue

            margin_x = max(3, line_kernel_width // 2)
            margin_y = max(3, line_kernel_height * 2)
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

        boxes = _merge_nearby_regions(
            boxes,
            page_width=width,
            page_height=height,
        )
        return [TextRegion(bbox=box) for box in boxes]


def detect_text_regions(image: Image.Image) -> list[TextRegion]:
    return MorphologyTextRegionDetector().detect(image)
