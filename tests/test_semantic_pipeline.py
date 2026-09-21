from docx import Document as WordDocument

from lao_document_ocr.exporters import export_docx
from lao_document_ocr.exporters.markdown import document_to_markdown
from lao_document_ocr.models import Block, BlockType, BoundingBox, Document, Page
from lao_document_ocr.ocr.base import RecognizedLine
from lao_document_ocr.structure import build_blocks


def _line(
    text: str,
    *,
    block_id: int,
    paragraph_id: int,
    line_id: int,
    y: int,
    height: int,
) -> RecognizedLine:
    return RecognizedLine(
        text=text,
        bbox=BoundingBox(x=50, y=y, width=320, height=height),
        confidence=0.96,
        block_id=block_id,
        paragraph_id=paragraph_id,
        line_id=line_id,
    )


def test_structure_assigns_heading_level_from_relative_height() -> None:
    lines = [
        _line(
            "Main title",
            block_id=1,
            paragraph_id=1,
            line_id=1,
            y=20,
            height=40,
        ),
        _line(
            "Body one",
            block_id=2,
            paragraph_id=1,
            line_id=1,
            y=100,
            height=20,
        ),
        _line(
            "Body two",
            block_id=3,
            paragraph_id=1,
            line_id=1,
            y=150,
            height=20,
        ),
        _line(
            "Body three",
            block_id=4,
            paragraph_id=1,
            line_id=1,
            y=200,
            height=20,
        ),
    ]

    blocks = build_blocks(lines)

    assert blocks[0].type == BlockType.HEADING
    assert blocks[0].level == 1
    assert blocks[0].metadata["classifier"] == "heuristic-v2"
    assert all(block.type == BlockType.PARAGRAPH for block in blocks[1:])


def test_structure_preserves_original_list_text_but_stores_clean_items() -> None:
    lines = [
        _line(
            "1. ກາເຟ",
            block_id=1,
            paragraph_id=1,
            line_id=1,
            y=50,
            height=20,
        ),
        _line(
            "ແລະ ຊາ",
            block_id=1,
            paragraph_id=1,
            line_id=2,
            y=80,
            height=20,
        ),
        _line(
            "2. ນ້ຳ",
            block_id=1,
            paragraph_id=1,
            line_id=3,
            y=110,
            height=20,
        ),
    ]

    block = build_blocks(lines)[0]

    assert block.type == BlockType.LIST
    assert block.text == "1. ກາເຟ\nແລະ ຊາ\n2. ນ້ຳ"
    assert block.metadata["list_style"] == "ordered"
    assert block.metadata["items"] == ["ກາເຟ ແລະ ຊາ", "ນ້ຳ"]


def test_ordered_list_exports_as_word_numbered_list_without_markers(tmp_path) -> None:
    block = Block(
        type=BlockType.LIST,
        text="1. First\n2. Second",
        metadata={
            "list_style": "ordered",
            "items": ["First", "Second"],
        },
    )
    document = Document(
        pages=[
            Page(
                number=1,
                width=600,
                height=800,
                blocks=[block],
            )
        ]
    )

    path = export_docx(document, tmp_path / "numbered.docx")
    word = WordDocument(path)

    paragraphs = [paragraph for paragraph in word.paragraphs if paragraph.text]
    assert [paragraph.text for paragraph in paragraphs] == ["First", "Second"]
    assert [paragraph.style.name for paragraph in paragraphs] == [
        "List Number",
        "List Number",
    ]


def test_markdown_list_style_uses_clean_items() -> None:
    ordered = Block(
        type=BlockType.LIST,
        text="1. First\n2. Second",
        metadata={
            "list_style": "ordered",
            "items": ["First", "Second"],
        },
    )
    bullets = Block(
        type=BlockType.LIST,
        text="• Alpha\n- Beta",
        metadata={
            "list_style": "bullet",
            "items": ["Alpha", "Beta"],
        },
    )
    document = Document(
        pages=[
            Page(
                number=1,
                width=600,
                height=800,
                blocks=[ordered, bullets],
            )
        ]
    )

    markdown = document_to_markdown(document)

    assert "1. First" in markdown
    assert "2. Second" in markdown
    assert "- Alpha" in markdown
    assert "- Beta" in markdown
    assert "- 1. First" not in markdown
