from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median

from lao_document_ocr.models import Block, BlockType, BoundingBox, TableCell
from lao_document_ocr.ocr.base import RecognizedLine

_CURRENCY_LIKE = re.compile(
    r"(?:[$€£¥₭]|\b(?:USD|LAK|THB|EUR)\b)",
    re.IGNORECASE,
)
_NUMERIC_LIKE = re.compile(r"^[\d\s.,:/()%+\-]+$")


@dataclass(frozen=True)
class TextRow:
    lines: tuple[RecognizedLine, ...]

    @property
    def center_y(self) -> float:
        return median(
            line.bbox.y + line.bbox.height / 2
            for line in self.lines
        )

    @property
    def height(self) -> float:
        return median(line.bbox.height for line in self.lines)


@dataclass(frozen=True)
class RowCellAssignment:
    line: RecognizedLine
    column: int
    column_span: int


@dataclass(frozen=True)
class BorderlessSchema:
    column_count: int
    anchors: tuple[float, ...]
    rows: tuple[tuple[RowCellAssignment, ...], ...]


def _cluster_rows(lines: list[RecognizedLine]) -> list[TextRow]:
    if not lines:
        return []

    typical_height = median(line.bbox.height for line in lines)
    tolerance = max(8.0, typical_height * 0.65)
    rows: list[list[RecognizedLine]] = []

    for line in sorted(lines, key=lambda item: (item.bbox.y, item.bbox.x)):
        center_y = line.bbox.y + line.bbox.height / 2
        if not rows:
            rows.append([line])
            continue

        previous_centers = [
            item.bbox.y + item.bbox.height / 2
            for item in rows[-1]
        ]
        previous_center = sum(previous_centers) / len(previous_centers)
        if abs(center_y - previous_center) <= tolerance:
            rows[-1].append(line)
        else:
            rows.append([line])

    return [
        TextRow(tuple(sorted(row, key=lambda item: item.bbox.x)))
        for row in rows
    ]


def _looks_value_like(text: str) -> bool:
    value = text.strip()
    if not value:
        return False
    if _CURRENCY_LIKE.search(value):
        return True
    return bool(
        _NUMERIC_LIKE.fullmatch(value)
        and any(char.isdigit() for char in value)
    )


def _union_bbox(lines: list[RecognizedLine]) -> BoundingBox:
    left = min(line.bbox.x for line in lines)
    top = min(line.bbox.y for line in lines)
    right = max(line.bbox.x + line.bbox.width for line in lines)
    bottom = max(line.bbox.y + line.bbox.height for line in lines)
    return BoundingBox(
        x=left,
        y=top,
        width=right - left,
        height=bottom - top,
    )


def _validate_row_spacing(rows: list[TextRow]) -> bool:
    row_centers = [row.center_y for row in rows]
    row_gaps = [
        right - left
        for left, right in zip(
            row_centers,
            row_centers[1:],
            strict=False,
        )
    ]
    typical_height = median(row.height for row in rows)
    return not row_gaps or max(row_gaps) <= max(
        90,
        typical_height * 3.2,
    )


def _infer_anchors(
    rows: list[TextRow],
    column_count: int,
    page_width: int,
) -> tuple[float, ...] | None:
    full_rows = [
        row
        for row in rows
        if len(row.lines) == column_count
    ]
    if len(full_rows) < 2:
        return None

    anchor_tolerance = max(14, int(page_width * 0.025))
    anchors = tuple(
        median(row.lines[column].bbox.x for row in full_rows)
        for column in range(column_count)
    )

    for row in full_rows:
        for column, line in enumerate(row.lines):
            if abs(line.bbox.x - anchors[column]) > anchor_tolerance:
                return None

    min_gap = max(16, int(page_width * 0.02))
    for row in full_rows:
        for left, right in zip(
            row.lines,
            row.lines[1:],
            strict=False,
        ):
            gap = right.bbox.x - (
                left.bbox.x + left.bbox.width
            )
            if gap < min_gap:
                return None

    return anchors


def _assign_line_to_columns(
    line: RecognizedLine,
    anchors: tuple[float, ...],
    *,
    tolerance: int,
) -> RowCellAssignment | None:
    left = line.bbox.x - tolerance
    right = line.bbox.x + line.bbox.width + tolerance
    covered = [
        index
        for index, anchor in enumerate(anchors)
        if left <= anchor <= right
    ]

    if not covered:
        nearest = min(
            range(len(anchors)),
            key=lambda index: abs(anchors[index] - line.bbox.x),
        )
        if abs(anchors[nearest] - line.bbox.x) > tolerance:
            return None
        covered = [nearest]

    first = min(covered)
    last = max(covered)
    if covered != list(range(first, last + 1)):
        return None

    return RowCellAssignment(
        line=line,
        column=first,
        column_span=last - first + 1,
    )


def _assign_rows(
    rows: list[TextRow],
    anchors: tuple[float, ...],
    *,
    page_width: int,
) -> tuple[tuple[RowCellAssignment, ...], ...] | None:
    column_count = len(anchors)
    tolerance = max(14, int(page_width * 0.025))
    assigned_rows: list[tuple[RowCellAssignment, ...]] = []

    for row in rows:
        assignments: list[RowCellAssignment] = []
        for line in row.lines:
            assignment = _assign_line_to_columns(
                line,
                anchors,
                tolerance=tolerance,
            )
            if assignment is None:
                return None
            assignments.append(assignment)

        assignments.sort(key=lambda item: item.column)
        covered_columns: set[int] = set()
        has_span = False
        for assignment in assignments:
            columns = set(
                range(
                    assignment.column,
                    assignment.column + assignment.column_span,
                )
            )
            if covered_columns & columns:
                return None
            covered_columns.update(columns)
            has_span = has_span or assignment.column_span > 1

        if covered_columns != set(range(column_count)):
            return None

        if len(assignments) < column_count and not has_span:
            return None

        assigned_rows.append(tuple(assignments))

    return tuple(assigned_rows)


def _valid_two_column_value_pattern(
    rows: list[TextRow],
    column_count: int,
) -> bool:
    if column_count != 2:
        return True

    full_rows = [
        row
        for row in rows
        if len(row.lines) == column_count
    ]
    if len(full_rows) < 2:
        return False

    value_counts = [
        sum(
            _looks_value_like(row.lines[column].text)
            for row in full_rows
        )
        for column in range(column_count)
    ]
    required = max(2, int(len(full_rows) * 0.6))
    return max(value_counts) >= required


def _infer_schema(
    rows: list[TextRow],
    page_width: int,
) -> BorderlessSchema | None:
    if len(rows) < 3:
        return None

    column_count = max(len(row.lines) for row in rows)
    if column_count < 2 or column_count > 6:
        return None
    if any(
        len(row.lines) < 1 or len(row.lines) > column_count
        for row in rows
    ):
        return None

    anchors = _infer_anchors(
        rows,
        column_count,
        page_width,
    )
    if anchors is None:
        return None

    if not _validate_row_spacing(rows):
        return None

    text_lengths = [
        len(line.text.strip())
        for row in rows
        for line in row.lines
        if line.text.strip()
    ]
    if not text_lengths or median(text_lengths) > 34:
        return None

    if not _valid_two_column_value_pattern(
        rows,
        column_count,
    ):
        return None

    assignments = _assign_rows(
        rows,
        anchors,
        page_width=page_width,
    )
    if assignments is None:
        return None

    return BorderlessSchema(
        column_count=column_count,
        anchors=anchors,
        rows=assignments,
    )


def _block_from_schema(
    rows: list[TextRow],
    schema: BorderlessSchema,
) -> Block:
    cells: list[TableCell] = []
    all_lines: list[RecognizedLine] = []
    grid = [
        ["" for _ in range(schema.column_count)]
        for _ in rows
    ]

    merged_cells = 0
    for row_index, assignments in enumerate(schema.rows):
        for assignment in assignments:
            line = assignment.line
            text = line.text.strip()
            all_lines.append(line)
            cells.append(
                TableCell(
                    row=row_index,
                    column=assignment.column,
                    text=text,
                    column_span=assignment.column_span,
                )
            )
            grid[row_index][assignment.column] = text
            if assignment.column_span > 1:
                merged_cells += 1

    confidence = (
        sum(line.confidence for line in all_lines)
        / len(all_lines)
        if all_lines
        else None
    )

    return Block(
        type=BlockType.TABLE,
        text="\n".join(
            "\t".join(row)
            for row in grid
        ),
        bbox=_union_bbox(all_lines),
        confidence=confidence,
        cells=cells,
        metadata={
            "detector": "aligned-text-v2",
            "rows": len(rows),
            "columns": schema.column_count,
            "merged_cells": merged_cells,
            "merge_support": "horizontal-only",
        },
    )


def _longest_valid_run(
    rows: list[TextRow],
    start: int,
    page_width: int,
) -> tuple[int, BorderlessSchema] | None:
    for end in range(len(rows), start + 2, -1):
        run = rows[start:end]
        schema = _infer_schema(
            run,
            page_width,
        )
        if schema is not None:
            return end, schema
    return None


def detect_borderless_tables(
    lines: list[RecognizedLine],
    *,
    page_width: int,
) -> tuple[list[RecognizedLine], list[Block]]:
    rows = _cluster_rows(lines)
    if len(rows) < 3:
        return lines, []

    table_blocks: list[Block] = []
    consumed_ids: set[int] = set()
    index = 0

    while index < len(rows):
        match = _longest_valid_run(
            rows,
            index,
            page_width,
        )
        if match is None:
            index += 1
            continue

        end, schema = match
        run = rows[index:end]
        table_blocks.append(
            _block_from_schema(
                run,
                schema,
            )
        )
        consumed_ids.update(
            id(line)
            for row in run
            for line in row.lines
        )
        index = end

    remaining = [
        line
        for line in lines
        if id(line) not in consumed_ids
    ]
    return remaining, table_blocks
