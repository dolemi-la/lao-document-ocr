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
    return bool(_NUMERIC_LIKE.fullmatch(value) and any(char.isdigit() for char in value))


def _union_bbox(lines: list[RecognizedLine]) -> BoundingBox:
    left = min(line.bbox.x for line in lines)
    top = min(line.bbox.y for line in lines)
    right = max(line.bbox.x + line.bbox.width for line in lines)
    bottom = max(line.bbox.y + line.bbox.height for line in lines)
    return BoundingBox(x=left, y=top, width=right - left, height=bottom - top)


def _valid_run(rows: list[TextRow], page_width: int) -> bool:
    if len(rows) < 3:
        return False
    column_count = len(rows[0].lines)
    if column_count < 2 or column_count > 6:
        return False
    if any(len(row.lines) != column_count for row in rows):
        return False

    anchor_tolerance = max(14, int(page_width * 0.025))
    anchors = [
        median(row.lines[column].bbox.x for row in rows)
        for column in range(column_count)
    ]

    for row in rows:
        for column, line in enumerate(row.lines):
            if abs(line.bbox.x - anchors[column]) > anchor_tolerance:
                return False

    min_gap = max(16, int(page_width * 0.02))
    for row in rows:
        for left, right in zip(row.lines, row.lines[1:], strict=False):
            gap = right.bbox.x - (left.bbox.x + left.bbox.width)
            if gap < min_gap:
                return False

    row_centers = [row.center_y for row in rows]
    row_gaps = [
        right - left
        for left, right in zip(row_centers, row_centers[1:], strict=False)
    ]
    typical_height = median(row.height for row in rows)
    if row_gaps and max(row_gaps) > max(90, typical_height * 3.2):
        return False

    text_lengths = [
        len(line.text.strip())
        for row in rows
        for line in row.lines
        if line.text.strip()
    ]
    if not text_lengths or median(text_lengths) > 34:
        return False

    # Two columns are easy to confuse with ordinary page columns. Require one
    # column to look value-like in most non-header rows.
    if column_count == 2:
        value_counts = [
            sum(_looks_value_like(row.lines[column].text) for row in rows)
            for column in range(column_count)
        ]
        required = max(2, int(len(rows) * 0.6))
        if max(value_counts) < required:
            return False

    return True


def _block_from_rows(rows: list[TextRow]) -> Block:
    column_count = len(rows[0].lines)
    cells: list[TableCell] = []
    all_lines: list[RecognizedLine] = []

    for row_index, row in enumerate(rows):
        for column_index, line in enumerate(row.lines):
            all_lines.append(line)
            cells.append(
                TableCell(
                    row=row_index,
                    column=column_index,
                    text=line.text.strip(),
                )
            )

    confidence = (
        sum(line.confidence for line in all_lines) / len(all_lines)
        if all_lines
        else None
    )
    grid = [
        [row.lines[column].text.strip() for column in range(column_count)]
        for row in rows
    ]

    return Block(
        type=BlockType.TABLE,
        text="\n".join("\t".join(row) for row in grid),
        bbox=_union_bbox(all_lines),
        confidence=confidence,
        cells=cells,
        metadata={
            "detector": "aligned-text-v1",
            "rows": len(rows),
            "columns": column_count,
            "merged_cells": 0,
        },
    )


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
        column_count = len(rows[index].lines)
        if column_count < 2:
            index += 1
            continue

        end = index + 1
        while end < len(rows) and len(rows[end].lines) == column_count:
            end += 1

        run = rows[index:end]
        if _valid_run(run, page_width):
            table_blocks.append(_block_from_rows(run))
            consumed_ids.update(id(line) for row in run for line in row.lines)

        index = end

    remaining = [line for line in lines if id(line) not in consumed_ids]
    return remaining, table_blocks
