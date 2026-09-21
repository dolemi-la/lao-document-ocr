from __future__ import annotations

from dataclasses import asdict, dataclass

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised through lazy CLI error path
    raise RuntimeError(
        "Recognizer training requires the optional 'train' dependencies. "
        "Install with: pip install -e '.[train]'"
    ) from exc


@dataclass(frozen=True)
class RecognizerConfig:
    image_height: int = 48
    max_width: int = 512
    cnn_channels: int = 256
    hidden_size: int = 192
    lstm_layers: int = 2
    blank_logit_bias: float = -2.0

    def to_dict(self) -> dict:
        return asdict(self)


class LaoCrnnRecognizer(nn.Module):
    width_downsample_factor = 4

    def __init__(self, num_classes: int, config: RecognizerConfig | None = None) -> None:
        super().__init__()
        self.config = config or RecognizerConfig()
        c = self.config.cnn_channels

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        self.sequence = nn.LSTM(
            input_size=c,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.lstm_layers,
            bidirectional=True,
            dropout=0.1 if self.config.lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(self.config.hidden_size * 2, num_classes)
        with torch.no_grad():
            self.classifier.bias[0] = self.config.blank_logit_bias

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = features.mean(dim=2)
        sequence = features.permute(2, 0, 1)
        sequence, _ = self.sequence(sequence)
        logits = self.classifier(sequence)
        return logits.log_softmax(dim=-1)

    @classmethod
    def output_lengths(cls, input_widths: torch.Tensor) -> torch.Tensor:
        return torch.clamp(input_widths // cls.width_downsample_factor, min=1)
