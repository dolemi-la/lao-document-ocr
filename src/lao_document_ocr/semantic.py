from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median

from lao_document_ocr.models import BlockType
from lao_document_ocr.normalization import normalize_lao_text
from lao_document_ocr.ocr.base import RecognizedLine

_BULLET_PREFIX = re.compile(r"^[•▪◦●*\-]\s+")
_ORDERED_PREFIX = re.compile(r"^\d{1,3}[.)]\s+")


@dataclass(frozen=True)
class SemanticClassification:
    block_type: BlockType
    level: int | None = None
    metadata: dict | None = None


def _strip_list_prefix(text: str, style: str) -> str:
    pattern = _ORDERED_PREFIX if style == "ordered" else _BULLET_PREFIX
    return pattern.sub("", text, count=1).strip()


def _prefix_style(text: str) -> str | None:
    if _ORDERED_PREFIX.match(text):
        return "ordered"
    if _BULLET_PREFIX.match(text):
        return "bullet"
    return None


def _list_metadata(lines: list[str]) -> dict | None:
    if not lines:
        return None

    first_style = _prefix_style(lines[0])
    if first_style is None:
        return None

    items: list[str] = []
    current = _strip_list_prefix(lines[0], first_style)

    for line in lines[1:]:
        style = _prefix_style(line)
        if style is None:
            continuation = line.strip()
            if continuation:
                current = f"{current} {continuation}".strip()
            continue
        if style != first_style:
            return None
        if current:
            items.append(current)
        current = _strip_list_prefix(line, first_style)

    if current:
        items.append(current)
    if not items:
        return None

    return {
        "classifier": "heuristic-v2",
        "list_style": first_style,
        "items": items,
    }


def _heading_level(height_ratio: float) -> int:
    if height_ratio >= 1.75:
        return 1
    if height_ratio >= 1.50:
        return 2
    return 3


def classify_paragraph(
    paragraph_lines: list[RecognizedLine],
    *,
    typical_height: float,
) -> SemanticClassification:
    if not paragraph_lines:
        return SemanticClassification(BlockType.PARAGRAPH)

    normalized_lines = [
        normalize_lao_text(line.text)
        for line in paragraph_lines
        if normalize_lao_text(line.text)
    ]
    if not normalized_lines:
        return SemanticClassification(BlockType.PARAGRAPH)

    list_metadata = _list_metadata(normalized_lines)
    if list_metadata is not None:
        return SemanticClassification(
            BlockType.LIST,
            metadata=list_metadata,
        )

    text = normalize_lao_text("\n".join(normalized_lines))
    paragraph_height = median(line.bbox.height for line in paragraph_lines)
    height_ratio = paragraph_height / max(1.0, typical_height)

    is_heading = (
        len(paragraph_lines) <= 2
        and len(text) <= 160
        and height_ratio >= 1.30
    )
    if is_heading:
        level = _heading_level(height_ratio)
        return SemanticClassification(
            BlockType.HEADING,
            level=level,
            metadata={
                "classifier": "heuristic-v2",
                "height_ratio": height_ratio,
            },
        )

    return SemanticClassification(
        BlockType.PARAGRAPH,
        metadata={"classifier": "heuristic-v2"},
    )
