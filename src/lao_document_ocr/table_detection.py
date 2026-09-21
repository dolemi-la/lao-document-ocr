from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from lao_document_ocr.models import Block, BlockType, BoundingBox, TableCell
from lao_document_ocr.ocr.base import RecognizedLine


@dataclass(frozen=True)
class RuledTable:
    bbox: BoundingBox
    x_lines: tuple[int, ...]
    y_lines: tuple[int, ...]


def _collapse_positions(positions: np.ndarray, max_gap: int = 2) -> tuple[int, ...]:
    if positions.size == 0:
        return ()
    groups: list[list[int]] = [[int(positions[0])]]
    for value in positions[1:]:
        value = int(value)
        if value - groups[-1][-1] <= max_gap:
            groups[-1].append(value)
        else:
            groups.append([value])
    return tuple(round(sum(group) / len(group)) for group in groups)


def detect_ruled_tables(image: Image.Image) -> list[RuledTable]:
    gray = np.asarray(image.convert("L"))
    if gray.size == 0:
        return []

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    height, width = binary.shape

    horizontal_kernel_width = max(24, width // 18)
    vertical_kernel_height = max(24, height // 18)

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_width, 1)),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_height)),
    )
    grid = cv2.bitwise_or(horizontal, vertical)
    grid = cv2.dilate(
        grid,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tables: list[RuledTable] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width < max(80, width // 10) or box_height < max(50, height // 20):
            continue

        h_region = horizontal[y : y + box_height, x : x + box_width]
        v_region = vertical[y : y + box_height, x : x + box_width]

        horizontal_density = np.count_nonzero(h_region, axis=1)
        vertical_density = np.count_nonzero(v_region, axis=0)

        y_positions = np.flatnonzero(horizontal_density >= max(10, int(box_width * 0.45)))
        x_positions = np.flatnonzero(vertical_density >= max(10, int(box_height * 0.45)))

        local_y_lines = _collapse_positions(y_positions)
        local_x_lines = _collapse_positions(x_positions)
        if len(local_x_lines) < 2 or len(local_y_lines) < 2:
            continue

        absolute_x = tuple(x + value for value in local_x_lines)
        absolute_y = tuple(y + value for value in local_y_lines)

        tables.append(
            RuledTable(
                bbox=BoundingBox(
                    x=min(absolute_x),
                    y=min(absolute_y),
                    width=max(absolute_x) - min(absolute_x),
                    height=max(absolute_y) - min(absolute_y),
                ),
                x_lines=absolute_x,
                y_lines=absolute_y,
            )
        )

    return sorted(tables, key=lambda table: (table.bbox.y, table.bbox.x))


def _inside(box: BoundingBox, table: RuledTable) -> bool:
    center_x = box.x + box.width / 2
    center_y = box.y + box.height / 2
    return (
        table.bbox.x <= center_x <= table.bbox.x + table.bbox.width
        and table.bbox.y <= center_y <= table.bbox.y + table.bbox.height
    )


def _cell_index(value: float, boundaries: tuple[int, ...]) -> int | None:
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:], strict=True)):
        if start <= value <= end:
            return index
    return None


def build_table_block(table: RuledTable, lines: list[RecognizedLine]) -> Block:
    cell_lines: dict[tuple[int, int], list[RecognizedLine]] = {}

    for line in lines:
        center_x = line.bbox.x + line.bbox.width / 2
        center_y = line.bbox.y + line.bbox.height / 2
        column = _cell_index(center_x, table.x_lines)
        row = _cell_index(center_y, table.y_lines)
        if row is None or column is None:
            continue
        cell_lines.setdefault((row, column), []).append(line)

    cells: list[TableCell] = []
    row_text: list[str] = []
    row_count = len(table.y_lines) - 1
    column_count = len(table.x_lines) - 1

    for row in range(row_count):
        values: list[str] = []
        for column in range(column_count):
            grouped = sorted(
                cell_lines.get((row, column), []),
                key=lambda line: (line.bbox.y, line.bbox.x),
            )
            text = " ".join(line.text.strip() for line in grouped if line.text.strip()).strip()
            values.append(text)
            cells.append(TableCell(row=row, column=column, text=text))
        row_text.append("\t".join(values))

    confidences = [line.confidence for line in lines if _inside(line.bbox, table)]
    confidence = sum(confidences) / len(confidences) if confidences else None

    return Block(
        type=BlockType.TABLE,
        text="\n".join(row_text),
        bbox=table.bbox,
        confidence=confidence,
        cells=cells,
        metadata={
            "detector": "ruled-grid-v1",
            "rows": row_count,
            "columns": column_count,
        },
    )


def split_table_lines(
    lines: list[RecognizedLine],
    tables: list[RuledTable],
) -> tuple[list[RecognizedLine], list[tuple[RuledTable, list[RecognizedLine]]]]:
    remaining: list[RecognizedLine] = []
    assignments: list[tuple[RuledTable, list[RecognizedLine]]] = [
        (table, []) for table in tables
    ]

    for line in lines:
        assigned = False
        for index, (table, table_lines) in enumerate(assignments):
            if _inside(line.bbox, table):
                table_lines.append(line)
                assignments[index] = (table, table_lines)
                assigned = True
                break
        if not assigned:
            remaining.append(line)

    return remaining, assignments
