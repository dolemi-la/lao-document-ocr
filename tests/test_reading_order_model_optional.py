import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from lao_document_ocr.reading_order_features import PAIR_FEATURE_DIM  # noqa: E402
from lao_document_ocr.reading_order_model import (  # noqa: E402
    PairwiseReadingOrderMLP,
    ReadingOrderModelConfig,
)


def test_pairwise_model_forward_shape() -> None:
    model = PairwiseReadingOrderMLP(
        ReadingOrderModelConfig(hidden_size=16)
    )
    features = torch.zeros((4, PAIR_FEATURE_DIM))

    logits = model(features)

    assert logits.shape == (4,)


def test_model_config_rejects_wrong_feature_size() -> None:
    with pytest.raises(ValueError, match="feature size"):
        ReadingOrderModelConfig(
            input_dim=PAIR_FEATURE_DIM + 1,
        )
