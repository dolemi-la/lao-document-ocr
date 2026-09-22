from lao_document_ocr.models import Block, BlockType, BoundingBox
from lao_document_ocr.reading_order import detect_two_column_layout, order_blocks


def _block(text: str, x: int, y: int, width: int = 180, height: int = 30) -> Block:
    return Block(
        type=BlockType.PARAGRAPH,
        text=text,
        bbox=BoundingBox(x=x, y=y, width=width, height=height),
    )


def test_two_columns_are_read_left_then_right() -> None:
    blocks = [
        _block("L1", 60, 100),
        _block("R1", 360, 100),
        _block("L2", 60, 160),
        _block("R2", 360, 160),
        _block("L3", 60, 220),
        _block("R3", 360, 220),
    ]

    ordered = order_blocks(blocks, page_width=600)

    assert [block.text for block in ordered] == ["L1", "L2", "L3", "R1", "R2", "R3"]


def test_full_width_heading_above_columns_stays_first() -> None:
    blocks = [
        _block("Heading", 50, 20, width=500, height=40),
        _block("L1", 60, 100),
        _block("R1", 360, 100),
        _block("L2", 60, 160),
        _block("R2", 360, 160),
    ]

    ordered = order_blocks(blocks, page_width=600)

    assert [block.text for block in ordered] == ["Heading", "L1", "L2", "R1", "R2"]


def test_ambiguous_spanning_block_inside_body_disables_column_reorder() -> None:
    blocks = [
        _block("L1", 60, 100),
        _block("R1", 360, 100),
        _block("Span", 120, 145, width=360, height=35),
        _block("L2", 60, 200),
        _block("R2", 360, 200),
    ]

    assert detect_two_column_layout(blocks, 600) is None
    ordered = order_blocks(blocks, page_width=600)
    assert [block.text for block in ordered] == ["L1", "R1", "Span", "L2", "R2"]


def test_single_column_keeps_top_to_bottom_order() -> None:
    blocks = [
        _block("B", 60, 160, width=480),
        _block("A", 60, 100, width=480),
        _block("C", 60, 220, width=480),
    ]

    ordered = order_blocks(blocks, page_width=600)

    assert [block.text for block in ordered] == ["A", "B", "C"]


def test_three_columns_are_read_left_to_right_by_column() -> None:
    from lao_document_ocr.reading_order import detect_multi_column_layout

    blocks = [
        _block("L1", 40, 100, width=130),
        _block("M1", 235, 100, width=130),
        _block("R1", 430, 100, width=130),
        _block("L2", 40, 160, width=130),
        _block("M2", 235, 160, width=130),
        _block("R2", 430, 160, width=130),
        _block("L3", 40, 220, width=130),
        _block("M3", 235, 220, width=130),
        _block("R3", 430, 220, width=130),
    ]

    layout = detect_multi_column_layout(blocks, page_width=600)
    ordered = order_blocks(blocks, page_width=600)

    assert layout is not None
    assert len(layout.columns) == 3
    assert [block.text for block in ordered] == [
        "L1",
        "L2",
        "L3",
        "M1",
        "M2",
        "M3",
        "R1",
        "R2",
        "R3",
    ]


def test_three_columns_preserve_full_width_heading_and_footer() -> None:
    blocks = [
        _block("Heading", 30, 20, width=540, height=40),
        _block("L1", 40, 100, width=130),
        _block("M1", 235, 100, width=130),
        _block("R1", 430, 100, width=130),
        _block("L2", 40, 160, width=130),
        _block("M2", 235, 160, width=130),
        _block("R2", 430, 160, width=130),
        _block("Footer", 30, 300, width=540, height=35),
    ]

    ordered = order_blocks(blocks, page_width=600)

    assert [block.text for block in ordered] == [
        "Heading",
        "L1",
        "L2",
        "M1",
        "M2",
        "R1",
        "R2",
        "Footer",
    ]


def test_three_column_spanning_body_block_disables_reorder() -> None:
    blocks = [
        _block("L1", 40, 100, width=130),
        _block("M1", 235, 100, width=130),
        _block("R1", 430, 100, width=130),
        _block("Span", 120, 145, width=360, height=35),
        _block("L2", 40, 200, width=130),
        _block("M2", 235, 200, width=130),
        _block("R2", 430, 200, width=130),
    ]

    ordered = order_blocks(blocks, page_width=600)

    assert [block.text for block in ordered] == [
        "L1",
        "M1",
        "R1",
        "Span",
        "L2",
        "M2",
        "R2",
    ]
