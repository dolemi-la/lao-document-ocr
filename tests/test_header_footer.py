from lao_document_ocr.header_footer import (
    mark_repeated_headers_footers,
    repeated_role_texts,
)
from lao_document_ocr.models import Block, BlockType, BoundingBox, Page


def _block(text: str, y: int) -> Block:
    return Block(
        type=BlockType.PARAGRAPH,
        text=text,
        bbox=BoundingBox(x=50, y=y, width=300, height=24),
    )


def _page(number: int, header: str, footer: str) -> Page:
    return Page(
        number=number,
        width=600,
        height=800,
        blocks=[
            _block(header, 20),
            _block(f"Body {number}", 250),
            _block(footer, 750),
        ],
    )


def test_repeated_header_and_footer_are_marked() -> None:
    pages = [
        _page(1, "Lao OCR Project", "Confidential"),
        _page(2, "Lao OCR Project", "Confidential"),
        _page(3, "Lao OCR Project", "Confidential"),
    ]

    mark_repeated_headers_footers(pages)

    assert pages[0].blocks[0].metadata["role"] == "header"
    assert pages[0].blocks[2].metadata["role"] == "footer"
    assert repeated_role_texts(pages, "header") == ["Lao OCR Project"]
    assert repeated_role_texts(pages, "footer") == ["Confidential"]


def test_non_repeated_margin_text_is_not_marked() -> None:
    pages = [
        _page(1, "Header A", "Footer A"),
        _page(2, "Header B", "Footer B"),
        _page(3, "Header C", "Footer C"),
    ]

    mark_repeated_headers_footers(pages)

    assert all("role" not in block.metadata for page in pages for block in page.blocks)


def test_single_page_is_unchanged() -> None:
    page = _page(1, "Header", "Footer")
    mark_repeated_headers_footers([page])
    assert all("role" not in block.metadata for block in page.blocks)
