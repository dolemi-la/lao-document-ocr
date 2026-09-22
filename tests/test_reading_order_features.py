import numpy as np
import pytest

from lao_document_ocr.models import Block, BlockType, BoundingBox
from lao_document_ocr.reading_order_features import (
    BLOCK_FEATURE_DIM,
    FEATURE_VERSION,
    PAIR_FEATURE_DIM,
    block_feature_vector,
    pair_feature_vector,
)


def _block(
    text: str,
    block_type: BlockType,
    x: int,
    y: int,
    width: int,
    height: int,
) -> Block:
    return Block(
        type=block_type,
        text=text,
        bbox=BoundingBox(
            x=x,
            y=y,
            width=width,
            height=height,
        ),
    )


def test_block_feature_vector_has_stable_dimension() -> None:
    block = _block(
        "Heading",
        BlockType.HEADING,
        100,
        80,
        400,
        60,
    )
    features = block_feature_vector(
        block,
        page_width=1000,
        page_height=1500,
    )

    assert features.shape == (BLOCK_FEATURE_DIM,)
    assert features.dtype == np.float32
    assert FEATURE_VERSION == "reading-order-pair-v1"
    assert features[0] == 0.1
    assert features[1] == pytest.approx(80 / 1500)


def test_pair_features_are_directional() -> None:
    first = _block(
        "A",
        BlockType.PARAGRAPH,
        50,
        100,
        200,
        40,
    )
    second = _block(
        "B",
        BlockType.PARAGRAPH,
        50,
        200,
        200,
        40,
    )

    forward = pair_feature_vector(
        first,
        second,
        page_width=600,
        page_height=800,
    )
    reverse = pair_feature_vector(
        second,
        first,
        page_width=600,
        page_height=800,
    )

    assert forward.shape == (PAIR_FEATURE_DIM,)
    assert reverse.shape == (PAIR_FEATURE_DIM,)
    assert not np.array_equal(forward, reverse)
    assert forward[-5] > 0
    assert reverse[-5] < 0


def test_feature_extraction_requires_bbox() -> None:
    block = Block(
        type=BlockType.PARAGRAPH,
        text="No box",
    )
    with pytest.raises(ValueError, match="bounding boxes"):
        block_feature_vector(
            block,
            page_width=600,
            page_height=800,
        )
