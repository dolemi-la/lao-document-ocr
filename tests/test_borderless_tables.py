from lao_document_ocr.borderless_tables import detect_borderless_tables
from lao_document_ocr.models import BlockType, BoundingBox
from lao_document_ocr.ocr.base import RecognizedLine


def _line(text: str, x: int, y: int, width: int = 110, line_id: int = 1) -> RecognizedLine:
    return RecognizedLine(
        text=text,
        bbox=BoundingBox(x=x, y=y, width=width, height=24),
        confidence=0.9,
        block_id=line_id,
        paragraph_id=line_id,
        line_id=line_id,
    )


def test_detects_two_column_name_value_table() -> None:
    lines = [
        _line("Item", 70, 80, line_id=1),
        _line("Amount", 330, 80, line_id=2),
        _line("Coffee", 70, 125, line_id=3),
        _line("20,000 ₭", 330, 125, line_id=4),
        _line("Tea", 70, 170, line_id=5),
        _line("10,000 ₭", 330, 170, line_id=6),
        _line("Water", 70, 215, line_id=7),
        _line("5,000 ₭", 330, 215, line_id=8),
    ]

    remaining, blocks = detect_borderless_tables(lines, page_width=600)

    assert remaining == []
    assert len(blocks) == 1
    block = blocks[0]
    assert block.type == BlockType.TABLE
    assert block.metadata["detector"] == "aligned-text-v1"
    assert block.metadata["rows"] == 4
    assert block.metadata["columns"] == 2
    assert block.cells[2].text == "Coffee"
    assert block.cells[3].text == "20,000 ₭"


def test_does_not_turn_two_column_prose_into_table() -> None:
    lines = [
        _line("This is a long sentence in the left column.", 50, 80, width=230, line_id=1),
        _line("Another long sentence in the right column.", 340, 80, width=220, line_id=2),
        _line("More prose continues down the left side.", 50, 125, width=230, line_id=3),
        _line("More prose continues down the right side.", 340, 125, width=220, line_id=4),
        _line("Still ordinary paragraph content over here.", 50, 170, width=230, line_id=5),
        _line("Still ordinary paragraph content over there.", 340, 170, width=220, line_id=6),
    ]

    remaining, blocks = detect_borderless_tables(lines, page_width=600)

    assert blocks == []
    assert remaining == lines


def test_detects_three_column_short_text_table_without_numbers() -> None:
    lines = [
        _line("Name", 40, 80, width=70, line_id=1),
        _line("Role", 220, 80, width=70, line_id=2),
        _line("City", 400, 80, width=70, line_id=3),
        _line("Ana", 40, 125, width=70, line_id=4),
        _line("Dev", 220, 125, width=70, line_id=5),
        _line("VTE", 400, 125, width=70, line_id=6),
        _line("Kai", 40, 170, width=70, line_id=7),
        _line("QA", 220, 170, width=70, line_id=8),
        _line("PKZ", 400, 170, width=70, line_id=9),
    ]

    remaining, blocks = detect_borderless_tables(lines, page_width=600)

    assert remaining == []
    assert len(blocks) == 1
    assert blocks[0].metadata["columns"] == 3


def test_alignment_break_prevents_detection() -> None:
    lines = [
        _line("Item", 70, 80, line_id=1),
        _line("Amount", 330, 80, line_id=2),
        _line("Coffee", 70, 125, line_id=3),
        _line("20,000", 330, 125, line_id=4),
        _line("Tea", 150, 170, line_id=5),
        _line("10,000", 330, 170, line_id=6),
    ]

    remaining, blocks = detect_borderless_tables(lines, page_width=600)

    assert blocks == []
    assert remaining == lines
