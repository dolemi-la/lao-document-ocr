from __future__ import annotations

from collections import defaultdict
from statistics import median

from PIL import Image

from lao_document_ocr.borderless_tables import detect_borderless_tables
from lao_document_ocr.models import Block, BoundingBox
from lao_document_ocr.normalization import normalize_lao_text
from lao_document_ocr.ocr.base import RecognizedLine
from lao_document_ocr.reading_order import order_blocks
from lao_document_ocr.semantic import apply_semantic_hint, classify_paragraph
from lao_document_ocr.table_detection import (
    build_table_block,
    detect_ruled_tables,
    split_table_lines,
)


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

        confidence = sum(line.confidence for line in paragraph_lines) / len(paragraph_lines)
        classification = classify_paragraph(
            paragraph_lines,
            typical_height=typical_height,
        )
        classification = apply_semantic_hint(
            classification,
            paragraph_lines,
            typical_height=typical_height,
        )

        blocks.append(
            Block(
                type=classification.block_type,
                text=text,
                bbox=_union_bbox(paragraph_lines),
                confidence=max(0.0, min(1.0, confidence)),
                level=classification.level,
                metadata=classification.metadata or {},
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
    ruled_tables = detect_ruled_tables(image)
    remaining = lines
    ruled_blocks: list[Block] = []

    if ruled_tables:
        remaining, assignments = split_table_lines(lines, ruled_tables)
        ruled_blocks = [
            build_table_block(table, table_lines)
            for table, table_lines in assignments
        ]

    remaining, borderless_blocks = detect_borderless_tables(
        remaining,
        page_width=image.width,
    )

    blocks = build_blocks(remaining)
    blocks.extend(ruled_blocks)
    blocks.extend(borderless_blocks)
    return order_blocks(blocks, page_width=image.width)
