from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from lao_document_ocr.layout_model import (  # noqa: E402
    LayoutSegmentationConfig,
    TinyLayoutUNet,
)


def test_layout_unet_preserves_spatial_dimensions() -> None:
    config = LayoutSegmentationConfig(
        image_height=128,
        image_width=160,
        base_channels=16,
        num_classes=6,
    )
    model = TinyLayoutUNet(config)

    output = model(
        torch.zeros(
            (2, 3, config.image_height, config.image_width),
            dtype=torch.float32,
        )
    )

    assert tuple(output.shape) == (2, 6, 128, 160)


def test_layout_config_validates_dimensions() -> None:
    with pytest.raises(ValueError, match="divisible by 4"):
        LayoutSegmentationConfig(
            image_height=127,
            image_width=128,
        )
    with pytest.raises(ValueError, match="at least 64"):
        LayoutSegmentationConfig(
            image_height=60,
            image_width=128,
        )
