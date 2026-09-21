from __future__ import annotations

from dataclasses import dataclass

from lao_document_ocr.models import Block


@dataclass(frozen=True)
class TwoColumnLayout:
    gutter_left: int
    gutter_right: int
    body_top: int
    body_bottom: int


def detect_two_column_layout(
    blocks: list[Block],
    page_width: int,
) -> TwoColumnLayout | None:
    positioned = [block for block in blocks if block.bbox is not None]
    if len(positioned) < 4 or page_width <= 0:
        return None

    midpoint = page_width / 2
    left = [
        block
        for block in positioned
        if block.bbox.x + block.bbox.width / 2 < midpoint
        and block.bbox.width < page_width * 0.65
    ]
    right = [
        block
        for block in positioned
        if block.bbox.x + block.bbox.width / 2 >= midpoint
        and block.bbox.width < page_width * 0.65
    ]

    if len(left) < 2 or len(right) < 2:
        return None

    gutter_left = max(block.bbox.x + block.bbox.width for block in left)
    gutter_right = min(block.bbox.x for block in right)
    gutter_width = gutter_right - gutter_left
    if gutter_width < max(20, int(page_width * 0.05)):
        return None

    body_top = min(block.bbox.y for block in left + right)
    body_bottom = max(block.bbox.y + block.bbox.height for block in left + right)

    # If a block crosses the gutter inside the column body, the layout is more
    # complex than the conservative two-column model. Keep normal reading order.
    for block in positioned:
        box = block.bbox
        crosses_gutter = box.x < gutter_right and box.x + box.width > gutter_left
        overlaps_body = box.y < body_bottom and box.y + box.height > body_top
        is_column_block = block in left or block in right
        if crosses_gutter and overlaps_body and not is_column_block:
            # A full-width heading is safe only if it ends before the body starts.
            if box.y + box.height > body_top:
                return None

    return TwoColumnLayout(
        gutter_left=gutter_left,
        gutter_right=gutter_right,
        body_top=body_top,
        body_bottom=body_bottom,
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
    layout = detect_two_column_layout(normal, page_width)
    if layout is None:
        return normal

    top: list[Block] = []
    left: list[Block] = []
    right: list[Block] = []
    bottom: list[Block] = []

    for block in normal:
        if block.bbox is None:
            top.append(block)
            continue
        box = block.bbox
        center_x = box.x + box.width / 2

        if box.y + box.height <= layout.body_top:
            top.append(block)
        elif box.y >= layout.body_bottom:
            bottom.append(block)
        elif center_x < layout.gutter_left:
            left.append(block)
        elif center_x > layout.gutter_right:
            right.append(block)
        else:
            # Should be unreachable after conservative layout detection, but
            # preserving the original order is safer than guessing.
            return normal

    def by_position(block: Block) -> tuple[int, int]:
        return (
            block.bbox.y if block.bbox else 0,
            block.bbox.x if block.bbox else 0,
        )
    return (
        sorted(top, key=by_position)
        + sorted(left, key=by_position)
        + sorted(right, key=by_position)
        + sorted(bottom, key=by_position)
    )
