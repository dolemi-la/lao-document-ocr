from lao_document_ocr.layout_ground_truth import (
    validate_layout_ground_truth,
)
from lao_document_ocr.models import (
    Block,
    BlockType,
    BoundingBox,
    Document,
    Page,
    TableCell,
)


def _document(blocks, *, width=600, height=800):
    return Document(
        pages=[
            Page(
                number=1,
                width=width,
                height=height,
                blocks=blocks,
            )
        ]
    )


def test_valid_layout_ground_truth() -> None:
    document = _document(
        [
            Block(
                type=BlockType.PARAGRAPH,
                text="Body",
                bbox=BoundingBox(
                    x=50,
                    y=100,
                    width=300,
                    height=40,
                ),
            )
        ]
    )

    assert validate_layout_ground_truth(
        document,
        image_size=(600, 800),
    ) == []


def test_layout_requires_one_page_and_matching_dimensions() -> None:
    document = Document(
        pages=[
            Page(number=1, width=600, height=800),
            Page(number=2, width=600, height=800),
        ]
    )

    errors = validate_layout_ground_truth(
        document,
        image_size=(600, 800),
    )
    assert "exactly one page" in errors[0]

    single = _document([])
    errors = validate_layout_ground_truth(
        single,
        image_size=(601, 800),
    )
    assert any("do not match" in error for error in errors)


def test_layout_rejects_out_of_bounds_block() -> None:
    document = _document(
        [
            Block(
                type=BlockType.PARAGRAPH,
                bbox=BoundingBox(
                    x=550,
                    y=760,
                    width=100,
                    height=60,
                ),
            )
        ]
    )

    errors = validate_layout_ground_truth(document)
    assert any("exceeds page bounds" in error for error in errors)


def test_table_layout_validates_spans_and_overlap() -> None:
    table = Block(
        type=BlockType.TABLE,
        bbox=BoundingBox(
            x=50,
            y=100,
            width=500,
            height=300,
        ),
        metadata={"rows": 2, "columns": 2},
        cells=[
            TableCell(
                row=0,
                column=0,
                text="Header",
                column_span=2,
            ),
            TableCell(
                row=1,
                column=0,
                text="A",
            ),
            TableCell(
                row=1,
                column=0,
                text="Duplicate",
            ),
        ],
    )

    errors = validate_layout_ground_truth(
        _document([table])
    )

    assert any("overlaps another cell" in error for error in errors)
