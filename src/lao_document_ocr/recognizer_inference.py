from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lao_document_ocr.recognizer_training import prepare_line_image
from lao_document_ocr.vocabulary import CharacterVocabulary


@dataclass(frozen=True)
class RecognitionResult:
    text: str
    confidence: float
    valid_timesteps: int


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Recognizer inference requires the optional 'train' dependencies. "
            "Install with: pip install -e '.[train]'"
        ) from exc
    return torch


class ExportedLineRecognizer:
    def __init__(self, artifact_path: str | Path) -> None:
        torch = _require_torch()
        self.artifact_path = Path(artifact_path)
        metadata_path = self.artifact_path.with_suffix(self.artifact_path.suffix + ".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Recognizer metadata not found: {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") != "torch-export":
            raise ValueError("Unsupported recognizer artifact format")
        expected_sha256 = metadata.get("artifact_sha256")
        if expected_sha256 and _sha256_file(self.artifact_path) != expected_sha256:
            raise ValueError("Recognizer artifact SHA-256 does not match metadata")
        self.metadata = metadata

        model_config = metadata["model_config"]
        self.image_height = int(model_config["image_height"])
        self.max_width = int(model_config["max_width"])
        self.width_downsample_factor = int(metadata["width_downsample_factor"])
        self.vocabulary = CharacterVocabulary(tuple(metadata["vocabulary"]["characters"]))

        exported = torch.export.load(str(self.artifact_path))
        self.model = exported.module()

    def recognize(self, image_path: str | Path) -> RecognitionResult:
        torch = _require_torch()
        array, valid_width = prepare_line_image(
            image_path,
            image_height=self.image_height,
            max_width=self.max_width,
        )

        padded = np.zeros((1, self.image_height, self.max_width), dtype=np.float32)
        padded[:, :, :valid_width] = array

        tensor = torch.from_numpy(padded).unsqueeze(0)
        with torch.no_grad():
            log_probs = self.model(tensor)

        valid_timesteps = max(1, valid_width // self.width_downsample_factor)
        valid_timesteps = min(valid_timesteps, int(log_probs.shape[0]))
        valid = log_probs[:valid_timesteps, 0, :]

        probabilities = valid.exp()
        max_probabilities, token_ids = probabilities.max(dim=-1)
        text = self.vocabulary.decode_ctc(token_ids.tolist())
        confidence = float(max_probabilities.mean().item()) if valid_timesteps else 0.0
        return RecognitionResult(
            text=text,
            confidence=confidence,
            valid_timesteps=valid_timesteps,
        )
