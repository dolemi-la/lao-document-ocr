from __future__ import annotations

from dataclasses import asdict, dataclass

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - lazy CLI error path
    raise RuntimeError(
        "Layout detector training requires the optional 'train' dependencies. "
        "Install with: pip install -e '.[train]'"
    ) from exc


@dataclass(frozen=True)
class LayoutSegmentationConfig:
    image_height: int = 256
    image_width: int = 256
    base_channels: int = 32
    num_classes: int = 6

    def __post_init__(self) -> None:
        if self.image_height < 64 or self.image_width < 64:
            raise ValueError("layout image dimensions must be at least 64")
        if self.image_height % 4 or self.image_width % 4:
            raise ValueError("layout image dimensions must be divisible by 4")
        if self.base_channels < 8:
            raise ValueError("base_channels must be at least 8")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")

    def to_dict(self) -> dict:
        return asdict(self)


class _ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


class TinyLayoutUNet(nn.Module):
    """Small semantic-layout segmentation model for project-owned training."""

    def __init__(
        self,
        config: LayoutSegmentationConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or LayoutSegmentationConfig()
        base = self.config.base_channels

        self.encoder1 = _ConvBlock(3, base)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = _ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = _ConvBlock(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(
            base * 4,
            base * 2,
            kernel_size=2,
            stride=2,
        )
        self.decoder2 = _ConvBlock(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(
            base * 2,
            base,
            kernel_size=2,
            stride=2,
        )
        self.decoder1 = _ConvBlock(base * 2, base)

        self.classifier = nn.Conv2d(
            base,
            self.config.num_classes,
            kernel_size=1,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        encoded1 = self.encoder1(images)
        encoded2 = self.encoder2(self.pool1(encoded1))
        bottleneck = self.bottleneck(self.pool2(encoded2))

        decoded2 = self.up2(bottleneck)
        decoded2 = self.decoder2(
            torch.cat([decoded2, encoded2], dim=1)
        )

        decoded1 = self.up1(decoded2)
        decoded1 = self.decoder1(
            torch.cat([decoded1, encoded1], dim=1)
        )
        return self.classifier(decoded1)
