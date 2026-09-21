from __future__ import annotations

from collections import Counter

from lao_document_ocr.models import Block, BlockType, Page
from lao_document_ocr.normalization import normalize_lao_text


def _candidate_text(block: Block) -> str:
    if block.type in {BlockType.TABLE, BlockType.IMAGE}:
        return ""
    return normalize_lao_text(block.text)


def _top_candidate(block: Block, page: Page, ratio: float) -> bool:
    if block.bbox is None:
        return False
    return block.bbox.y <= page.height * ratio


def _bottom_candidate(block: Block, page: Page, ratio: float) -> bool:
    if block.bbox is None:
        return False
    bottom = block.bbox.y + block.bbox.height
    return bottom >= page.height * (1 - ratio)


def mark_repeated_headers_footers(
    pages: list[Page],
    *,
    margin_ratio: float = 0.12,
    min_page_ratio: float = 0.6,
) -> None:
    if len(pages) < 2:
        return
    if not 0 < margin_ratio < 0.5:
        raise ValueError("margin_ratio must be between 0 and 0.5")
    if not 0 < min_page_ratio <= 1:
        raise ValueError("min_page_ratio must be between 0 and 1")

    header_pages: Counter[str] = Counter()
    footer_pages: Counter[str] = Counter()

    for page in pages:
        header_seen: set[str] = set()
        footer_seen: set[str] = set()
        for block in page.blocks:
            text = _candidate_text(block)
            if not text:
                continue
            if _top_candidate(block, page, margin_ratio):
                header_seen.add(text)
            if _bottom_candidate(block, page, margin_ratio):
                footer_seen.add(text)
        header_pages.update(header_seen)
        footer_pages.update(footer_seen)

    minimum_pages = max(2, int(len(pages) * min_page_ratio + 0.999999))
    repeated_headers = {
        text for text, count in header_pages.items() if count >= minimum_pages
    }
    repeated_footers = {
        text for text, count in footer_pages.items() if count >= minimum_pages
    }

    for page in pages:
        for block in page.blocks:
            text = _candidate_text(block)
            if not text:
                continue
            if text in repeated_headers and _top_candidate(block, page, margin_ratio):
                block.metadata["role"] = "header"
            elif text in repeated_footers and _bottom_candidate(block, page, margin_ratio):
                block.metadata["role"] = "footer"


def repeated_role_texts(pages: list[Page], role: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for page in pages:
        for block in page.blocks:
            if block.metadata.get("role") != role:
                continue
            text = normalize_lao_text(block.text)
            if text and text not in seen:
                seen.add(text)
                values.append(text)
    return values
