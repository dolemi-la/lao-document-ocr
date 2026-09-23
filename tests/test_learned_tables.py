from lao_document_ocr.learned_tables import (
    detect_learned_tables,
    reconstruct_learned_table_region,
)
from lao_document_ocr.models import BlockType, BoundingBox
from lao_document_ocr.ocr.base import RecognizedLine


def _line(text: str, x: int, y: int, block_id: int = 7) -> RecognizedLine:
    return RecognizedLine(
        text=text,
        bbox=BoundingBox(x=x, y=y, width=110, height=24),
        confidence=0.9,
        block_id=block_id,
        paragraph_id=block_id,
        line_id=y,
        semantic_type=BlockType.TABLE,
    )


def test_learned_region_reconstructs_sparse_borderless_table() -> None:
    lines = [
        _line("Item", 60, 80),
        _line("Qty", 250, 80),
        _line("Amount", 420, 80),
        _line("Coffee", 60, 125),
        _line("2", 250, 125),
        _line("20,000", 420, 125),
        _line("Tea", 60, 170),
        _line("10,000", 420, 170),
    ]

    block = reconstruct_learned_table_region(lines, page_width=600)

    assert block is not None
    assert block.type == BlockType.TABLE
    assert block.metadata["detector"] == "learned-layout-table-v1"
    assert block.metadata["rows"] == 3
    assert block.metadata["columns"] == 3
    assert block.metadata["sparse_cells"] == 1
    assert any(cell.text == "Tea" and cell.column == 0 for cell in block.cells)
    assert any(cell.text == "10,000" and cell.column == 2 for cell in block.cells)


def test_learned_table_lines_are_consumed_before_other_structure() -> None:
    table_lines = [
        _line("A", 60, 80),
        _line("1", 300, 80),
        _line("B", 60, 125),
        _line("2", 300, 125),
    ]
    body = RecognizedLine(
        text="Body paragraph",
        bbox=BoundingBox(x=50, y=240, width=350, height=24),
        confidence=0.95,
        block_id=9,
        paragraph_id=9,
        line_id=1,
        semantic_type=BlockType.PARAGRAPH,
    )

    remaining, blocks = detect_learned_tables(
        [*table_lines, body],
        page_width=600,
    )

    assert remaining == [body]
    assert len(blocks) == 1
    assert blocks[0].metadata["columns"] == 2


def test_ambiguous_learned_table_falls_back_to_lines() -> None:
    lines = [
        _line("one", 60, 80),
        _line("two", 70, 125),
        _line("three", 65, 170),
        _line("four", 68, 215),
    ]

    remaining, blocks = detect_learned_tables(lines, page_width=600)

    assert blocks == []
    assert remaining == lines
