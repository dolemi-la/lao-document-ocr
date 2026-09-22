from __future__ import annotations

from dataclasses import asdict, dataclass

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - lazy CLI error path
    raise RuntimeError(
        "Reading-order training requires the optional 'train' dependencies. "
        "Install with: pip install -e '.[train]'"
    ) from exc

from lao_document_ocr.reading_order_features import PAIR_FEATURE_DIM


@dataclass(frozen=True)
class ReadingOrderModelConfig:
    input_dim: int = PAIR_FEATURE_DIM
    hidden_size: int = 64

    def __post_init__(self) -> None:
        if self.input_dim != PAIR_FEATURE_DIM:
            raise ValueError(
                f"input_dim must match runtime feature size {PAIR_FEATURE_DIM}"
            )
        if self.hidden_size < 8:
            raise ValueError("hidden_size must be at least 8")

    def to_dict(self) -> dict:
        return asdict(self)


class PairwiseReadingOrderMLP(nn.Module):
    def __init__(
        self,
        config: ReadingOrderModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ReadingOrderModelConfig()
        hidden = self.config.hidden_size
        self.network = nn.Sequential(
            nn.Linear(self.config.input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, max(8, hidden // 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, hidden // 2), 1),
        )

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        return self.network(pair_features).squeeze(-1)
