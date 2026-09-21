from __future__ import annotations

import re
from collections import defaultdict
from statistics import median

from PIL import Image

from lao_document_ocr.models import Block, BlockType, BoundingBox
from lao_document_ocr.normalization import normalize_lao_text
from lao_document_ocr.ocr.base import RecognizedLine
from lao_document_ocr.reading_order import order_blocks
from lao_document_ocr.table_detection import (
    build_table_block,
    detect_ruled_tables,
    split_table_lines,
)

_LIST_PREFIX = re.compile(r"^(?:[•▪◦●*-]|\d{1,3}[.)])\s+")


def _union_bbox(lines: list[RecognizedLine]) -> BoundingBox:
    left = min(line.bbox.x for line in lines)
    top = min(line.bbox.y for line in lines)
    right = max(line.bbox.x + line.bbox.width for line in lines)
    bottom = max(line.bbox.y + line.bbox.height for line in lines)
    return BoundingBox(x=left, y=top, width=right - left, height=bottom - top)


def build_blocks(lines: list[RecognizedLine]) -> list[Block]:
    if not lines:
        return []

    typical_height = median(line.bbox.height for line in lines)
    grouped: dict[tuple[int, int], list[RecognizedLine]] = defaultdict(list)
    for line in lines:
        grouped[(line.block_id, line.paragraph_id)].append(line)

    blocks: list[Block] = []
    for paragraph_lines in grouped.values():
        paragraph_lines.sort(key=lambda item: item.line_id)
        text = normalize_lao_text("\n".join(line.text for line in paragraph_lines))
        if not text:
            continue

        heights = [line.bbox.height for line in paragraph_lines]
        confidence = sum(line.confidence for line in paragraph_lines) / len(paragraph_lines)
        first_line = paragraph_lines[0].text
        is_list = bool(_LIST_PREFIX.match(first_line))
        is_heading = (
            len(paragraph_lines) <= 2
            and len(text) <= 140
            and median(heights) >= typical_height * 1.35
        )

        if is_list:
            block_type = BlockType.LIST
            level = None
        elif is_heading:
            block_type = BlockType.HEADING
            level = 1
        else:
            block_type = BlockType.PARAGRAPH
            level = None

        blocks.append(
            Block(
                type=block_type,
                text=text,
                bbox=_union_bbox(paragraph_lines),
                confidence=max(0.0, min(1.0, confidence)),
                level=level,
            )
        )

    return sorted(
        blocks,
        key=lambda block: (
            block.bbox.y if block.bbox else 0,
            block.bbox.x if block.bbox else 0,
        ),
    )


def build_page_blocks(lines: list[RecognizedLine], image: Image.Image) -> list[Block]:
    tables = detect_ruled_tables(image)
    if not tables:
        return order_blocks(build_blocks(lines), page_width=image.width)

    remaining, assignments = split_table_lines(lines, tables)
    blocks = build_blocks(remaining)
    blocks.extend(build_table_block(table, table_lines) for table, table_lines in assignments)
    return order_blocks(blocks, page_width=image.width)
