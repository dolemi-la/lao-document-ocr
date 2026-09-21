from lao_document_ocr.models import BlockType, BoundingBox
from lao_document_ocr.ocr.base import RecognizedLine
from lao_document_ocr.semantic import classify_paragraph


def _line(text: str, height: int, line_id: int = 1) -> RecognizedLine:
    return RecognizedLine(
        text=text,
        bbox=BoundingBox(x=40, y=line_id * 40, width=300, height=height),
        confidence=0.95,
        block_id=1,
        paragraph_id=1,
        line_id=line_id,
    )


def test_heading_levels_follow_relative_height() -> None:
    h1 = classify_paragraph([_line("Main title", 40)], typical_height=20)
    h2 = classify_paragraph([_line("Section", 32)], typical_height=20)
    h3 = classify_paragraph([_line("Subsection", 27)], typical_height=20)

    assert (h1.block_type, h1.level) == (BlockType.HEADING, 1)
    assert (h2.block_type, h2.level) == (BlockType.HEADING, 2)
    assert (h3.block_type, h3.level) == (BlockType.HEADING, 3)


def test_ordered_list_extracts_items_and_continuation() -> None:
    classification = classify_paragraph(
        [
            _line("1. ກາເຟ", 20, 1),
            _line("ແລະ ຊາ", 20, 2),
            _line("2. ນ້ຳ", 20, 3),
        ],
        typical_height=20,
    )

    assert classification.block_type == BlockType.LIST
    assert classification.metadata["list_style"] == "ordered"
    assert classification.metadata["items"] == [
        "ກາເຟ ແລະ ຊາ",
        "ນ້ຳ",
    ]


def test_bullet_list_extracts_items_without_markers() -> None:
    classification = classify_paragraph(
        [
            _line("• First", 20, 1),
            _line("- Second", 20, 2),
        ],
        typical_height=20,
    )

    assert classification.block_type == BlockType.LIST
    assert classification.metadata["list_style"] == "bullet"
    assert classification.metadata["items"] == ["First", "Second"]


def test_mixed_list_marker_styles_fall_back_to_paragraph() -> None:
    classification = classify_paragraph(
        [
            _line("1. First", 20, 1),
            _line("• Second", 20, 2),
        ],
        typical_height=20,
    )

    assert classification.block_type == BlockType.PARAGRAPH


def test_regular_body_is_paragraph() -> None:
    classification = classify_paragraph(
        [_line("Normal body paragraph", 20)],
        typical_height=20,
    )

    assert classification.block_type == BlockType.PARAGRAPH
    assert classification.metadata["classifier"] == "heuristic-v2"
