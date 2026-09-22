from __future__ import annotations

import numpy as np

from lao_document_ocr.models import Block, BlockType

_BLOCK_TYPES = (
    BlockType.HEADING,
    BlockType.PARAGRAPH,
    BlockType.LIST,
    BlockType.TABLE,
    BlockType.IMAGE,
)
BLOCK_FEATURE_DIM = 16
PAIR_FEATURE_DIM = BLOCK_FEATURE_DIM * 2 + 6
FEATURE_VERSION = "reading-order-pair-v1"


def _overlap_ratio(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    overlap = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    denominator = max(1e-6, min(end_a - start_a, end_b - start_b))
    return overlap / denominator


def block_feature_vector(
    block: Block,
    *,
    page_width: int,
    page_height: int,
) -> np.ndarray:
    if block.bbox is None:
        raise ValueError("reading-order features require block bounding boxes")
    if page_width < 1 or page_height < 1:
        raise ValueError("page dimensions must be positive")

    box = block.bbox
    x0 = box.x / page_width
    y0 = box.y / page_height
    width = box.width / page_width
    height = box.height / page_height
    x1 = (box.x + box.width) / page_width
    y1 = (box.y + box.height) / page_height
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    area = width * height

    type_features = [
        1.0 if block.type == block_type else 0.0
        for block_type in _BLOCK_TYPES
    ]
    role = str(block.metadata.get("role", ""))
    role_features = [
        1.0 if role == "header" else 0.0,
        1.0 if role == "footer" else 0.0,
    ]

    values = [
        x0,
        y0,
        x1,
        y1,
        center_x,
        center_y,
        width,
        height,
        area,
        *type_features,
        *role_features,
    ]
    assert len(values) == BLOCK_FEATURE_DIM
    return np.asarray(values, dtype=np.float32)


def pair_feature_vector(
    first: Block,
    second: Block,
    *,
    page_width: int,
    page_height: int,
) -> np.ndarray:
    first_features = block_feature_vector(
        first,
        page_width=page_width,
        page_height=page_height,
    )
    second_features = block_feature_vector(
        second,
        page_width=page_width,
        page_height=page_height,
    )

    first_box = first.bbox
    second_box = second.bbox
    assert first_box is not None
    assert second_box is not None

    first_center_x = first_features[4]
    first_center_y = first_features[5]
    second_center_x = second_features[4]
    second_center_y = second_features[5]

    horizontal_overlap = _overlap_ratio(
        first_features[0],
        first_features[2],
        second_features[0],
        second_features[2],
    )
    vertical_overlap = _overlap_ratio(
        first_features[1],
        first_features[3],
        second_features[1],
        second_features[3],
    )

    relative = np.asarray(
        [
            second_center_x - first_center_x,
            second_center_y - first_center_y,
            second_features[6] - first_features[6],
            second_features[7] - first_features[7],
            horizontal_overlap,
            vertical_overlap,
        ],
        dtype=np.float32,
    )
    result = np.concatenate(
        [first_features, second_features, relative],
    )
    assert result.shape == (PAIR_FEATURE_DIM,)
    return result
