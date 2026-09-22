from __future__ import annotations

import hashlib
import json
from types import MethodType

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from lao_document_ocr.models import (  # noqa: E402
    Block,
    BlockType,
    BoundingBox,
)
from lao_document_ocr.reading_order_features import (  # noqa: E402
    FEATURE_VERSION,
    PAIR_FEATURE_DIM,
)
from lao_document_ocr.reading_order_inference import (  # noqa: E402
    ExportedReadingOrderResolver,
)
from lao_document_ocr.reading_order_model import (  # noqa: E402
    PairwiseReadingOrderMLP,
    ReadingOrderModelConfig,
)


def _artifact(tmp_path):
    config = ReadingOrderModelConfig(hidden_size=16)
    model = PairwiseReadingOrderMLP(config)
    model.eval()
    example = torch.zeros((1, PAIR_FEATURE_DIM))
    exported = torch.export.export(model, (example,))
    artifact = tmp_path / "reading-order.pt2"
    torch.export.save(exported, artifact)
    metadata = {
        "schema_version": "1",
        "format": "torch-export",
        "model": "PairwiseReadingOrderMLP",
        "model_version": "pairwise-reading-order-v1",
        "feature_version": FEATURE_VERSION,
        "artifact": artifact.name,
        "artifact_sha256": hashlib.sha256(
            artifact.read_bytes()
        ).hexdigest(),
        "model_config": config.to_dict(),
    }
    artifact.with_suffix(".pt2.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    return artifact


def _block(text: str, x: int, y: int) -> Block:
    return Block(
        type=BlockType.PARAGRAPH,
        text=text,
        bbox=BoundingBox(
            x=x,
            y=y,
            width=180,
            height=40,
        ),
    )


def test_exported_reading_order_model_loads_on_cpu(tmp_path) -> None:
    resolver = ExportedReadingOrderResolver(
        _artifact(tmp_path),
        device="cpu",
    )

    assert resolver.device.type == "cpu"
    assert resolver.metadata()["version"] == "pairwise-reading-order-v1"
    assert resolver.metadata()["feature_version"] == FEATURE_VERSION


def test_reading_order_rejects_tampered_artifact(tmp_path) -> None:
    artifact = _artifact(tmp_path)
    artifact.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        ExportedReadingOrderResolver(artifact)


def test_pairwise_scores_produce_deterministic_order(tmp_path) -> None:
    resolver = ExportedReadingOrderResolver(
        _artifact(tmp_path),
        device="cpu",
    )
    blocks = [
        _block("C", 50, 300),
        _block("A", 50, 100),
        _block("B", 50, 200),
    ]

    rank = {
        "A": 0,
        "B": 1,
        "C": 2,
    }

    def fake_probabilities(
        self,
        pairs,
        *,
        page_width,
        page_height,
    ):
        return [
            0.95 if rank[first.text] < rank[second.text] else 0.05
            for first, second in pairs
        ]

    resolver._pair_probabilities = MethodType(
        fake_probabilities,
        resolver,
    )
    ordered = resolver.order(
        blocks,
        page_width=600,
        page_height=800,
    )

    assert [block.text for block in ordered] == ["A", "B", "C"]


def test_missing_bbox_falls_back_to_deterministic_order(tmp_path) -> None:
    resolver = ExportedReadingOrderResolver(
        _artifact(tmp_path),
        device="cpu",
    )
    blocks = [
        Block(type=BlockType.PARAGRAPH, text="No box"),
        _block("Positioned", 50, 100),
    ]

    ordered = resolver.order(
        blocks,
        page_width=600,
        page_height=800,
    )

    assert [block.text for block in ordered] == [
        "No box",
        "Positioned",
    ]


def test_max_block_cap_falls_back_to_deterministic_order(tmp_path) -> None:
    resolver = ExportedReadingOrderResolver(
        _artifact(tmp_path),
        device="cpu",
        max_blocks=2,
    )
    blocks = [
        _block("B", 50, 200),
        _block("A", 50, 100),
        _block("C", 50, 300),
    ]

    ordered = resolver.order(
        blocks,
        page_width=600,
        page_height=800,
    )

    assert [block.text for block in ordered] == ["A", "B", "C"]
