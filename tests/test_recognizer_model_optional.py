import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from lao_document_ocr.recognizer_model import LaoCrnnRecognizer, RecognizerConfig  # noqa: E402


def test_model_forward_shape() -> None:
    config = RecognizerConfig(image_height=48, max_width=256, hidden_size=64, lstm_layers=1)
    model = LaoCrnnRecognizer(num_classes=20, config=config)
    images = torch.zeros((2, 1, 48, 200), dtype=torch.float32)
    output = model(images)

    assert output.shape[1] == 2
    assert output.shape[2] == 20
    assert output.shape[0] == 50


def test_output_lengths() -> None:
    widths = torch.tensor([200, 101], dtype=torch.long)
    assert LaoCrnnRecognizer.output_lengths(widths).tolist() == [50, 25]


def test_blank_bias_initialization() -> None:
    config = RecognizerConfig(hidden_size=32, lstm_layers=1, cnn_channels=64)
    model = LaoCrnnRecognizer(num_classes=10, config=config)
    assert model.classifier.bias[0].item() == pytest.approx(config.blank_logit_bias)
