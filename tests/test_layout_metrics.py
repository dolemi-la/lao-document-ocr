import pytest

from lao_document_ocr.layout_metrics import block_iou, evaluate_layout
from lao_document_ocr.models import (
    Block,
    BlockType,
    BoundingBox,
    Document,
    Page,
    TableCell,
)


def _block(
    text: str,
    block_type: BlockType,
    x: int,
    y: int,
    width: int = 180,
    height: int = 40,
) -> Block:
    return Block(
        type=block_type,
        text=text,
        bbox=BoundingBox(x=x, y=y, width=width, height=height),
    )


def _document(
    blocks: list[Block],
    *,
    width: int = 600,
    height: int = 800,
) -> Document:
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


def test_iou_is_scale_invariant_across_page_sizes() -> None:
    reference = _block("A", BlockType.PARAGRAPH, 60, 80, 180, 80)
    prediction = _block("A", BlockType.PARAGRAPH, 120, 160, 360, 160)
    reference_page = Page(
        number=1,
        width=600,
        height=800,
        blocks=[reference],
    )
    prediction_page = Page(
        number=1,
        width=1200,
        height=1600,
        blocks=[prediction],
    )

    assert block_iou(
        reference,
        reference_page,
        prediction,
        prediction_page,
    ) == pytest.approx(1.0)


def test_perfect_layout_scores_one() -> None:
    blocks = [
        _block("Heading", BlockType.HEADING, 50, 40, 500, 50),
        _block("Body", BlockType.PARAGRAPH, 60, 130, 480, 100),
    ]
    reference = _document(blocks)
    prediction = _document([block.model_copy(deep=True) for block in blocks])

    metrics = evaluate_layout(reference, prediction)

    assert metrics.block_f1 == 1.0
    assert metrics.mean_iou == 1.0
    assert metrics.block_type_accuracy == 1.0
    assert metrics.reading_order_accuracy == 1.0


def test_type_and_reading_order_errors_are_separate() -> None:
    reference = _document(
        [
            _block("A", BlockType.HEADING, 50, 50),
            _block("B", BlockType.PARAGRAPH, 50, 150),
            _block("C", BlockType.PARAGRAPH, 50, 250),
        ]
    )
    prediction = _document(
        [
            _block("C", BlockType.PARAGRAPH, 50, 250),
            _block("A", BlockType.PARAGRAPH, 50, 50),
            _block("B", BlockType.PARAGRAPH, 50, 150),
        ]
    )

    metrics = evaluate_layout(reference, prediction)

    assert metrics.block_f1 == 1.0
    assert metrics.block_type_accuracy == pytest.approx(2 / 3)
    assert metrics.reading_order_accuracy == pytest.approx(1 / 3)


def test_table_structure_scores_spans() -> None:
    reference_table = Block(
        type=BlockType.TABLE,
        bbox=BoundingBox(x=50, y=100, width=500, height=300),
        cells=[
            TableCell(row=0, column=0, text="Header", column_span=2),
            TableCell(row=1, column=0, text="A"),
            TableCell(row=1, column=1, text="B"),
        ],
        metadata={"rows": 2, "columns": 2},
    )
    prediction_table = Block(
        type=BlockType.TABLE,
        bbox=BoundingBox(x=50, y=100, width=500, height=300),
        cells=[
            TableCell(row=0, column=0, text="Header"),
            TableCell(row=0, column=1, text=""),
            TableCell(row=1, column=0, text="A"),
            TableCell(row=1, column=1, text="B"),
        ],
        metadata={"rows": 2, "columns": 2},
    )

    metrics = evaluate_layout(
        _document([reference_table]),
        _document([prediction_table]),
    )

    assert metrics.table_shape_accuracy == 1.0
    assert metrics.table_cell_recall == pytest.approx(2 / 3)
    assert metrics.table_cell_precision == pytest.approx(0.5)
    assert metrics.table_cell_f1 == pytest.approx(4 / 7)


def test_missing_prediction_reduces_block_recall() -> None:
    reference = _document(
        [
            _block("A", BlockType.PARAGRAPH, 50, 50),
            _block("B", BlockType.PARAGRAPH, 50, 150),
        ]
    )
    prediction = _document(
        [_block("A", BlockType.PARAGRAPH, 50, 50)]
    )

    metrics = evaluate_layout(reference, prediction)

    assert metrics.block_precision == 1.0
    assert metrics.block_recall == 0.5
    assert metrics.block_f1 == pytest.approx(2 / 3)
