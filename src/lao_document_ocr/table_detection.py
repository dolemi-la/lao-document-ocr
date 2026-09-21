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
    # (row, interior vertical boundary index)
    missing_vertical_borders: tuple[tuple[int, int], ...] = ()
    # (interior horizontal boundary index, column)
    missing_horizontal_borders: tuple[tuple[int, int], ...] = ()


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


def _vertical_border_coverage(
    vertical_mask: np.ndarray,
    *,
    x: int,
    y0: int,
    y1: int,
) -> float:
    if y1 <= y0:
        return 1.0
    height, width = vertical_mask.shape
    left = max(0, x - 2)
    right = min(width, x + 3)
    top = max(0, y0 + 3)
    bottom = min(height, y1 - 3)
    if bottom <= top or right <= left:
        return 1.0
    segment = vertical_mask[top:bottom, left:right]
    return float(np.mean(np.any(segment > 0, axis=1)))


def _horizontal_border_coverage(
    horizontal_mask: np.ndarray,
    *,
    y: int,
    x0: int,
    x1: int,
) -> float:
    if x1 <= x0:
        return 1.0
    height, width = horizontal_mask.shape
    top = max(0, y - 2)
    bottom = min(height, y + 3)
    left = max(0, x0 + 3)
    right = min(width, x1 - 3)
    if bottom <= top or right <= left:
        return 1.0
    segment = horizontal_mask[top:bottom, left:right]
    return float(np.mean(np.any(segment > 0, axis=0)))


def _missing_borders(
    horizontal_mask: np.ndarray,
    vertical_mask: np.ndarray,
    x_lines: tuple[int, ...],
    y_lines: tuple[int, ...],
    *,
    min_coverage: float = 0.55,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    missing_vertical: list[tuple[int, int]] = []
    missing_horizontal: list[tuple[int, int]] = []

    row_count = len(y_lines) - 1
    column_count = len(x_lines) - 1

    for boundary_index in range(1, len(x_lines) - 1):
        x = x_lines[boundary_index]
        for row in range(row_count):
            coverage = _vertical_border_coverage(
                vertical_mask,
                x=x,
                y0=y_lines[row],
                y1=y_lines[row + 1],
            )
            if coverage < min_coverage:
                missing_vertical.append((row, boundary_index))

    for boundary_index in range(1, len(y_lines) - 1):
        y = y_lines[boundary_index]
        for column in range(column_count):
            coverage = _horizontal_border_coverage(
                horizontal_mask,
                y=y,
                x0=x_lines[column],
                x1=x_lines[column + 1],
            )
            if coverage < min_coverage:
                missing_horizontal.append((boundary_index, column))

    return tuple(missing_vertical), tuple(missing_horizontal)


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
        missing_vertical, missing_horizontal = _missing_borders(
            horizontal,
            vertical,
            absolute_x,
            absolute_y,
        )

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
                missing_vertical_borders=missing_vertical,
                missing_horizontal_borders=missing_horizontal,
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


def _merged_components(table: RuledTable) -> list[set[tuple[int, int]]]:
    row_count = len(table.y_lines) - 1
    column_count = len(table.x_lines) - 1
    parent = {
        (row, column): (row, column)
        for row in range(row_count)
        for column in range(column_count)
    }

    def find(cell: tuple[int, int]) -> tuple[int, int]:
        root = cell
        while parent[root] != root:
            root = parent[root]
        while parent[cell] != cell:
            next_cell = parent[cell]
            parent[cell] = root
            cell = next_cell
        return root

    def union(left: tuple[int, int], right: tuple[int, int]) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row, boundary_index in table.missing_vertical_borders:
        left_column = boundary_index - 1
        right_column = boundary_index
        if 0 <= row < row_count and 0 <= left_column < right_column < column_count:
            union((row, left_column), (row, right_column))

    for boundary_index, column in table.missing_horizontal_borders:
        top_row = boundary_index - 1
        bottom_row = boundary_index
        if 0 <= column < column_count and 0 <= top_row < bottom_row < row_count:
            union((top_row, column), (bottom_row, column))

    grouped: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for cell in parent:
        grouped.setdefault(find(cell), set()).add(cell)

    components: list[set[tuple[int, int]]] = []
    for component in grouped.values():
        if len(component) == 1:
            components.append(component)
            continue
        rows = [cell[0] for cell in component]
        columns = [cell[1] for cell in component]
        expected = {
            (row, column)
            for row in range(min(rows), max(rows) + 1)
            for column in range(min(columns), max(columns) + 1)
        }
        if component == expected:
            components.append(component)
        else:
            components.extend({cell} for cell in sorted(component))

    return sorted(
        components,
        key=lambda component: min(component),
    )


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

    row_count = len(table.y_lines) - 1
    column_count = len(table.x_lines) - 1
    cells: list[TableCell] = []
    display_grid = [["" for _ in range(column_count)] for _ in range(row_count)]

    for component in _merged_components(table):
        rows = [cell[0] for cell in component]
        columns = [cell[1] for cell in component]
        anchor_row = min(rows)
        anchor_column = min(columns)
        row_span = max(rows) - anchor_row + 1
        column_span = max(columns) - anchor_column + 1

        grouped_lines: list[RecognizedLine] = []
        for coordinate in sorted(component):
            grouped_lines.extend(cell_lines.get(coordinate, []))
        grouped_lines.sort(key=lambda line: (line.bbox.y, line.bbox.x))
        text = " ".join(
            line.text.strip() for line in grouped_lines if line.text.strip()
        ).strip()

        cells.append(
            TableCell(
                row=anchor_row,
                column=anchor_column,
                text=text,
                row_span=row_span,
                column_span=column_span,
            )
        )
        display_grid[anchor_row][anchor_column] = text

    confidences = [line.confidence for line in lines if _inside(line.bbox, table)]
    confidence = sum(confidences) / len(confidences) if confidences else None

    return Block(
        type=BlockType.TABLE,
        text="\n".join("\t".join(row) for row in display_grid),
        bbox=table.bbox,
        confidence=confidence,
        cells=cells,
        metadata={
            "detector": "ruled-grid-v2",
            "rows": row_count,
            "columns": column_count,
            "merged_cells": sum(
                1 for cell in cells if cell.row_span > 1 or cell.column_span > 1
            ),
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
