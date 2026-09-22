from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from lao_document_ocr.confidence_calibration import ConfidenceCalibration
from lao_document_ocr.recognizer_training import prepare_line_image, prepare_line_pil_image
from lao_document_ocr.vocabulary import CharacterVocabulary


@dataclass(frozen=True)
class RecognitionResult:
    text: str
    confidence: float
    calibrated_confidence: float | None
    valid_timesteps: int


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_torch_device(requested: str = "cpu"):
    torch = _require_torch()
    device = requested.strip().lower()
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = "mps"
        else:
            device = "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device == "mps":
        available = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
        if not available:
            raise RuntimeError("MPS was requested but is not available.")
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: cpu, cuda, mps, auto")
    return torch.device(device)


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
    def __init__(
        self,
        artifact_path: str | Path,
        *,
        calibration_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
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
        self.device = resolve_torch_device(device)
        self.metadata = {
            **metadata,
            "runtime_device": str(self.device),
        }

        model_config = metadata["model_config"]
        self.image_height = int(model_config["image_height"])
        self.max_width = int(model_config["max_width"])
        self.width_downsample_factor = int(metadata["width_downsample_factor"])
        self.vocabulary = CharacterVocabulary(tuple(metadata["vocabulary"]["characters"]))

        self.calibration = (
            ConfidenceCalibration.load(calibration_path) if calibration_path is not None else None
        )

        exported = torch.export.load(str(self.artifact_path))
        self.model = exported.module().to(self.device)

    def _recognize_array(self, array: np.ndarray, valid_width: int) -> RecognitionResult:
        torch = _require_torch()
        padded = np.zeros((1, self.image_height, self.max_width), dtype=np.float32)
        padded[:, :, :valid_width] = array

        tensor = torch.from_numpy(padded).unsqueeze(0).to(self.device)
        with torch.no_grad():
            log_probs = self.model(tensor)

        valid_timesteps = max(1, valid_width // self.width_downsample_factor)
        valid_timesteps = min(valid_timesteps, int(log_probs.shape[0]))
        valid = log_probs[:valid_timesteps, 0, :].detach().cpu()

        probabilities = valid.exp()
        max_probabilities, token_ids = probabilities.max(dim=-1)
        text = self.vocabulary.decode_ctc(token_ids.tolist())
        confidence = float(max_probabilities.mean().item()) if valid_timesteps else 0.0
        calibrated_confidence = (
            self.calibration.calibrate(confidence) if self.calibration is not None else None
        )
        return RecognitionResult(
            text=text,
            confidence=confidence,
            calibrated_confidence=calibrated_confidence,
            valid_timesteps=valid_timesteps,
        )

    def recognize(self, image_path: str | Path) -> RecognitionResult:
        array, valid_width = prepare_line_image(
            image_path,
            image_height=self.image_height,
            max_width=self.max_width,
        )
        return self._recognize_array(array, valid_width)

    def recognize_image(self, image: Image.Image) -> RecognitionResult:
        array, valid_width = prepare_line_pil_image(
            image,
            image_height=self.image_height,
            max_width=self.max_width,
        )
        return self._recognize_array(array, valid_width)
