from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lao_document_ocr.models import Block
from lao_document_ocr.reading_order import order_blocks
from lao_document_ocr.reading_order_features import (
    FEATURE_VERSION,
    PAIR_FEATURE_DIM,
    pair_feature_vector,
)
from lao_document_ocr.recognizer_inference import resolve_torch_device


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExportedReadingOrderResolver:
    def __init__(
        self,
        artifact_path: str | Path,
        *,
        device: str = "cpu",
        max_blocks: int = 256,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Learned reading-order inference requires the optional "
                "'train' dependencies. Install with: pip install -e '.[train]'"
            ) from exc

        if max_blocks < 2:
            raise ValueError("max_blocks must be at least 2")

        self.artifact_path = Path(artifact_path)
        metadata_path = self.artifact_path.with_suffix(
            self.artifact_path.suffix + ".json"
        )
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Reading-order metadata not found: {metadata_path}"
            )

        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        if metadata.get("format") != "torch-export":
            raise ValueError("Unsupported reading-order artifact format")
        if metadata.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Reading-order feature version mismatch")

        expected_sha256 = metadata.get("artifact_sha256")
        if (
            expected_sha256
            and _sha256_file(self.artifact_path) != expected_sha256
        ):
            raise ValueError(
                "Reading-order artifact SHA-256 does not match metadata"
            )

        self.device = resolve_torch_device(device)
        self.max_blocks = max_blocks
        self.inference_batch_size = int(
            metadata.get("inference_batch_size", 1)
        )
        if self.inference_batch_size < 1:
            raise ValueError(
                "Reading-order inference_batch_size must be positive"
            )
        self.metadata_payload = {
            **metadata,
            "runtime_device": str(self.device),
        }
        exported = torch.export.load(str(self.artifact_path))
        self.model = exported.module().to(self.device)

    def metadata(self) -> dict:
        return {
            "name": self.__class__.__name__,
            "version": self.metadata_payload.get("model_version"),
            "feature_version": FEATURE_VERSION,
            "device": str(self.device),
            "max_blocks": self.max_blocks,
            "inference_batch_size": self.inference_batch_size,
        }

    def _pair_probabilities(
        self,
        pairs: list[tuple[Block, Block]],
        *,
        page_width: int,
        page_height: int,
    ) -> list[float]:
        import numpy as np
        import torch

        if not pairs:
            return []
        results: list[float] = []
        for start in range(0, len(pairs), self.inference_batch_size):
            chunk = pairs[start : start + self.inference_batch_size]
            batch = np.zeros(
                (self.inference_batch_size, PAIR_FEATURE_DIM),
                dtype=np.float32,
            )
            for index, (first, second) in enumerate(chunk):
                batch[index] = pair_feature_vector(
                    first,
                    second,
                    page_width=page_width,
                    page_height=page_height,
                )
            tensor = torch.from_numpy(batch).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probabilities = torch.sigmoid(logits)
            results.extend(
                float(value)
                for value in probabilities[: len(chunk)]
                .detach()
                .cpu()
                .tolist()
            )
        return results

    def _probability_before(
        self,
        first: Block,
        second: Block,
        *,
        page_width: int,
        page_height: int,
    ) -> float:
        return self._pair_probabilities(
            [(first, second)],
            page_width=page_width,
            page_height=page_height,
        )[0]

    def order(
        self,
        blocks: list[Block],
        *,
        page_width: int,
        page_height: int,
    ) -> list[Block]:
        if len(blocks) < 2:
            return list(blocks)
        if len(blocks) > self.max_blocks:
            return order_blocks(
                blocks,
                page_width=page_width,
            )
        if any(block.bbox is None for block in blocks):
            return order_blocks(
                blocks,
                page_width=page_width,
            )

        scores = [0.0 for _ in blocks]
        pair_indices: list[tuple[int, int]] = []
        directed_pairs: list[tuple[Block, Block]] = []
        for first_index in range(len(blocks)):
            for second_index in range(first_index + 1, len(blocks)):
                pair_indices.append((first_index, second_index))
                directed_pairs.append(
                    (blocks[first_index], blocks[second_index])
                )
                directed_pairs.append(
                    (blocks[second_index], blocks[first_index])
                )

        probabilities = self._pair_probabilities(
            directed_pairs,
            page_width=page_width,
            page_height=page_height,
        )
        for pair_index, (first_index, second_index) in enumerate(pair_indices):
            forward = probabilities[pair_index * 2]
            reverse = probabilities[pair_index * 2 + 1]
            probability = (forward + (1.0 - reverse)) / 2.0
            scores[first_index] += probability
            scores[second_index] += 1.0 - probability

        ordered_indices = sorted(
            range(len(blocks)),
            key=lambda index: (
                -scores[index],
                index,
            ),
        )
        return [blocks[index] for index in ordered_indices]
