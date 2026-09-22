from __future__ import annotations

from dataclasses import dataclass

from lao_document_ocr.models import Block, BoundingBox


@dataclass(frozen=True)
class ColumnBand:
    left: int
    right: int
    center: float
    top: int
    bottom: int


@dataclass(frozen=True)
class MultiColumnLayout:
    columns: tuple[ColumnBand, ...]
    body_top: int
    body_bottom: int


@dataclass(frozen=True)
class TwoColumnLayout:
    gutter_left: int
    gutter_right: int
    body_top: int
    body_bottom: int


def _center_x(box: BoundingBox) -> float:
    return box.x + box.width / 2


def _right(box: BoundingBox) -> int:
    return box.x + box.width


def _bottom(box: BoundingBox) -> int:
    return box.y + box.height


def _cluster_by_largest_center_gaps(
    blocks: list[Block],
    column_count: int,
) -> list[list[Block]]:
    ordered = sorted(blocks, key=lambda block: _center_x(block.bbox))
    if len(ordered) < column_count:
        return []

    gaps = [
        (
            _center_x(ordered[index + 1].bbox)
            - _center_x(ordered[index].bbox),
            index,
        )
        for index in range(len(ordered) - 1)
    ]
    split_after = {
        index
        for _, index in sorted(
            gaps,
            key=lambda item: item[0],
            reverse=True,
        )[: column_count - 1]
    }

    groups: list[list[Block]] = [[]]
    for index, block in enumerate(ordered):
        groups[-1].append(block)
        if index in split_after:
            groups.append([])

    return groups if len(groups) == column_count else []


def _column_band(blocks: list[Block]) -> ColumnBand:
    boxes = [block.bbox for block in blocks if block.bbox is not None]
    left = min(box.x for box in boxes)
    right = max(_right(box) for box in boxes)
    top = min(box.y for box in boxes)
    bottom = max(_bottom(box) for box in boxes)
    center = sum(_center_x(box) for box in boxes) / len(boxes)
    return ColumnBand(
        left=left,
        right=right,
        center=center,
        top=top,
        bottom=bottom,
    )


def _body_overlaps(box: BoundingBox, body_top: int, body_bottom: int) -> bool:
    return box.y < body_bottom and _bottom(box) > body_top


def _horizontal_intersection(box: BoundingBox, column: ColumnBand) -> int:
    return max(
        0,
        min(_right(box), column.right) - max(box.x, column.left),
    )


def detect_multi_column_layout(
    blocks: list[Block],
    page_width: int,
    *,
    max_columns: int = 4,
    min_blocks_per_column: int = 2,
) -> MultiColumnLayout | None:
    positioned = [block for block in blocks if block.bbox is not None]
    if len(positioned) < min_blocks_per_column * 2 or page_width <= 0:
        return None
    if max_columns < 2:
        raise ValueError("max_columns must be at least 2")
    if min_blocks_per_column < 2:
        raise ValueError("min_blocks_per_column must be at least 2")

    candidates = [
        block
        for block in positioned
        if block.bbox.width < page_width * 0.58
    ]
    if len(candidates) < min_blocks_per_column * 2:
        return None

    max_candidate_columns = min(
        max_columns,
        len(candidates) // min_blocks_per_column,
    )
    min_gutter = max(20, int(page_width * 0.045))
    max_center_spread = max(30, int(page_width * 0.10))

    for column_count in range(max_candidate_columns, 1, -1):
        groups = _cluster_by_largest_center_gaps(candidates, column_count)
        if not groups:
            continue
        if any(len(group) < min_blocks_per_column for group in groups):
            continue

        center_spreads = []
        for group in groups:
            centers = [_center_x(block.bbox) for block in group]
            center_spreads.append(max(centers) - min(centers))
        if any(spread > max_center_spread for spread in center_spreads):
            continue

        columns = tuple(
            sorted(
                (_column_band(group) for group in groups),
                key=lambda column: column.center,
            )
        )

        gutters = [
            right.left - left.right
            for left, right in zip(columns, columns[1:], strict=False)
        ]
        if any(gutter < min_gutter for gutter in gutters):
            continue

        spans = [column.bottom - column.top for column in columns]
        common_vertical_overlap = max(
            0,
            min(column.bottom for column in columns)
            - max(column.top for column in columns),
        )
        if common_vertical_overlap < max(24, min(spans) * 0.25):
            continue

        body_top = min(column.top for column in columns)
        body_bottom = max(column.bottom for column in columns)

        ambiguous = False
        for block in positioned:
            box = block.bbox
            if not _body_overlaps(box, body_top, body_bottom):
                continue

            intersections = [
                _horizontal_intersection(box, column)
                for column in columns
            ]
            meaningful = sum(
                intersection >= max(8, min(box.width, 24) // 2)
                for intersection in intersections
            )
            if meaningful > 1:
                ambiguous = True
                break

            center = _center_x(box)
            belongs_to_column = any(
                column.left <= center <= column.right
                for column in columns
            )
            if not belongs_to_column:
                ambiguous = True
                break

        if ambiguous:
            continue

        return MultiColumnLayout(
            columns=columns,
            body_top=body_top,
            body_bottom=body_bottom,
        )

    return None


def detect_two_column_layout(
    blocks: list[Block],
    page_width: int,
) -> TwoColumnLayout | None:
    layout = detect_multi_column_layout(
        blocks,
        page_width,
        max_columns=2,
    )
    if layout is None or len(layout.columns) != 2:
        return None
    left, right = layout.columns
    return TwoColumnLayout(
        gutter_left=left.right,
        gutter_right=right.left,
        body_top=layout.body_top,
        body_bottom=layout.body_bottom,
    )


def order_blocks(
    blocks: list[Block],
    *,
    page_width: int,
) -> list[Block]:
    normal = sorted(
        blocks,
        key=lambda block: (
            block.bbox.y if block.bbox else 0,
            block.bbox.x if block.bbox else 0,
        ),
    )
    layout = detect_multi_column_layout(normal, page_width)
    if layout is None:
        return normal

    top: list[Block] = []
    columns: list[list[Block]] = [[] for _ in layout.columns]
    bottom: list[Block] = []

    for block in normal:
        if block.bbox is None:
            top.append(block)
            continue

        box = block.bbox
        if _bottom(box) <= layout.body_top:
            top.append(block)
            continue
        if box.y >= layout.body_bottom:
            bottom.append(block)
            continue

        center = _center_x(box)
        matches = [
            index
            for index, column in enumerate(layout.columns)
            if column.left <= center <= column.right
        ]
        if len(matches) != 1:
            return normal
        columns[matches[0]].append(block)

    def by_position(block: Block) -> tuple[int, int]:
        return (
            block.bbox.y if block.bbox else 0,
            block.bbox.x if block.bbox else 0,
        )

    ordered_columns: list[Block] = []
    for column_blocks in columns:
        ordered_columns.extend(sorted(column_blocks, key=by_position))

    return (
        sorted(top, key=by_position)
        + ordered_columns
        + sorted(bottom, key=by_position)
    )
