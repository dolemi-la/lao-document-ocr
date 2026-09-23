from __future__ import annotations

from collections import defaultdict
from statistics import median

from lao_document_ocr.models import Block, BlockType, BoundingBox, TableCell
from lao_document_ocr.ocr.base import RecognizedLine


def _union_bbox(lines: list[RecognizedLine]) -> BoundingBox:
    left = min(line.bbox.x for line in lines)
    top = min(line.bbox.y for line in lines)
    right = max(line.bbox.x + line.bbox.width for line in lines)
    bottom = max(line.bbox.y + line.bbox.height for line in lines)
    return BoundingBox(x=left, y=top, width=right - left, height=bottom - top)


def _cluster_rows(lines: list[RecognizedLine]) -> list[list[RecognizedLine]]:
    ordered = sorted(lines, key=lambda line: (line.bbox.y, line.bbox.x))
    typical_height = median(line.bbox.height for line in ordered)
    tolerance = max(8.0, typical_height * 0.7)
    rows: list[list[RecognizedLine]] = []
    for line in ordered:
        center = line.bbox.y + line.bbox.height / 2
        if not rows:
            rows.append([line])
            continue
        previous = median(
            item.bbox.y + item.bbox.height / 2
            for item in rows[-1]
        )
        if abs(center - previous) <= tolerance:
            rows[-1].append(line)
        else:
            rows.append([line])
    for row in rows:
        row.sort(key=lambda line: line.bbox.x)
    return rows


def _cluster_anchors(rows: list[list[RecognizedLine]], page_width: int) -> list[float]:
    xs = sorted(line.bbox.x for row in rows for line in row)
    if not xs:
        return []
    tolerance = max(18, int(page_width * 0.04))
    clusters: list[list[float]] = [[float(xs[0])]]
    for x in xs[1:]:
        current_center = sum(clusters[-1]) / len(clusters[-1])
        if abs(x - current_center) <= tolerance:
            clusters[-1].append(float(x))
        else:
            clusters.append([float(x)])
    return [median(cluster) for cluster in clusters]


def _assign_column(line: RecognizedLine, anchors: list[float]) -> int:
    return min(
        range(len(anchors)),
        key=lambda index: abs(line.bbox.x - anchors[index]),
    )


def reconstruct_learned_table_region(
    lines: list[RecognizedLine],
    *,
    page_width: int,
) -> Block | None:
    if len(lines) < 4:
        return None

    rows = _cluster_rows(lines)
    if len(rows) < 2:
        return None

    anchors = _cluster_anchors(rows, page_width)
    if len(anchors) < 2 or len(anchors) > 8:
        return None

    cells: list[TableCell] = []
    used: set[tuple[int, int]] = set()
    for row_index, row in enumerate(rows):
        for line in row:
            column = _assign_column(line, anchors)
            coordinate = (row_index, column)
            if coordinate in used:
                # Ambiguous duplicate assignment in one inferred cell: keep the
                # region as normal text rather than inventing structure.
                return None
            used.add(coordinate)
            cells.append(
                TableCell(
                    row=row_index,
                    column=column,
                    text=line.text.strip(),
                )
            )

    # Require at least two rows with multiple populated columns so a learned
    # false-positive paragraph region does not become a table.
    populated_per_row: defaultdict[int, int] = defaultdict(int)
    for cell in cells:
        populated_per_row[cell.row] += 1
    if sum(count >= 2 for count in populated_per_row.values()) < 2:
        return None

    grid = [["" for _ in anchors] for _ in rows]
    for cell in cells:
        grid[cell.row][cell.column] = cell.text

    confidence = sum(line.confidence for line in lines) / len(lines)
    return Block(
        type=BlockType.TABLE,
        text="\n".join("\t".join(row) for row in grid),
        bbox=_union_bbox(lines),
        confidence=max(0.0, min(1.0, confidence)),
        cells=cells,
        metadata={
            "detector": "learned-layout-table-v1",
            "rows": len(rows),
            "columns": len(anchors),
            "merged_cells": 0,
            "sparse_cells": sum(1 for row in grid for value in row if not value),
        },
    )


def detect_learned_tables(
    lines: list[RecognizedLine],
    *,
    page_width: int,
) -> tuple[list[RecognizedLine], list[Block]]:
    grouped: dict[int, list[RecognizedLine]] = defaultdict(list)
    remaining: list[RecognizedLine] = []

    for line in lines:
        if line.semantic_type == BlockType.TABLE:
            grouped[line.block_id].append(line)
        else:
            remaining.append(line)

    blocks: list[Block] = []
    for region_lines in grouped.values():
        block = reconstruct_learned_table_region(
            region_lines,
            page_width=page_width,
        )
        if block is None:
            remaining.extend(region_lines)
        else:
            blocks.append(block)

    return remaining, sorted(
        blocks,
        key=lambda block: (
            block.bbox.y if block.bbox else 0,
            block.bbox.x if block.bbox else 0,
        ),
    )
