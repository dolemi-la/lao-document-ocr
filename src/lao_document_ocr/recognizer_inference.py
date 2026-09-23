from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from lao_document_ocr.confidence_calibration import ConfidenceCalibration
from lao_document_ocr.ctc_decoding import ctc_prefix_beam_search
from lao_document_ocr.language_model import CharacterNgramLanguageModel
from lao_document_ocr.recognizer_training import prepare_line_image, prepare_line_pil_image
from lao_document_ocr.vocabulary import CharacterVocabulary


@dataclass(frozen=True)
class RecognitionResult:
    text: str
    confidence: float
    calibrated_confidence: float | None
    valid_timesteps: int
    decoder_ranking_score: float | None = None
    language_model_log_probability: float | None = None


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
        decoder: str = "greedy",
        beam_width: int = 10,
        language_model_path: str | Path | None = None,
        language_model_weight: float = 0.0,
        language_model_token_bonus: float = 0.0,
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
        decoder_name = decoder.strip().lower()
        if decoder_name not in {"greedy", "beam"}:
            raise ValueError("decoder must be one of: greedy, beam")
        if beam_width < 1:
            raise ValueError("beam_width must be at least 1")
        self.decoder = decoder_name
        self.beam_width = beam_width

        model_config = metadata["model_config"]
        self.image_height = int(model_config["image_height"])
        self.max_width = int(model_config["max_width"])
        self.width_downsample_factor = int(metadata["width_downsample_factor"])
        self.vocabulary = CharacterVocabulary(tuple(metadata["vocabulary"]["characters"]))

        if not np.isfinite(language_model_weight) or language_model_weight < 0:
            raise ValueError("language_model_weight must be finite and non-negative")
        if not np.isfinite(language_model_token_bonus):
            raise ValueError("language_model_token_bonus must be finite")
        if language_model_path is not None and self.decoder != "beam":
            raise ValueError("language model requires decoder='beam'")
        if language_model_path is None and (
            language_model_weight != 0 or language_model_token_bonus != 0
        ):
            raise ValueError(
                "language-model fusion parameters require language_model_path"
            )
        if language_model_path is not None and language_model_weight <= 0:
            raise ValueError(
                "language_model_weight must be positive when a language model is used"
            )

        self.language_model = (
            CharacterNgramLanguageModel.load(language_model_path)
            if language_model_path is not None
            else None
        )
        if self.language_model is not None:
            if self.language_model.vocabulary_checksum != self.vocabulary.checksum():
                raise ValueError("Language model vocabulary checksum mismatch")
            if self.language_model.vocabulary_size != self.vocabulary.size - 1:
                raise ValueError("Language model vocabulary size mismatch")

        self.language_model_weight = float(language_model_weight)
        self.language_model_token_bonus = float(language_model_token_bonus)
        self.metadata = {
            **metadata,
            "runtime_device": str(self.device),
            "decoder": self.decoder,
            "beam_width": self.beam_width if self.decoder == "beam" else None,
            "language_model": (
                self.language_model.metadata()
                if self.language_model is not None
                else None
            ),
            "language_model_weight": (
                self.language_model_weight
                if self.language_model is not None
                else 0.0
            ),
            "language_model_token_bonus": (
                self.language_model_token_bonus
                if self.language_model is not None
                else 0.0
            ),
        }

        self.calibration = (
            ConfidenceCalibration.load(calibration_path)
            if calibration_path is not None
            else None
        )
        if (
            self.calibration is not None
            and self.calibration.decoder != self.decoder
        ):
            raise ValueError(
                "Confidence calibration decoder mismatch "
                f"({self.calibration.decoder} != {self.decoder})"
            )
        if self.calibration is not None:
            expected_lm_sha256 = (
                self.language_model.artifact_sha256
                if self.language_model is not None
                else None
            )
            if self.calibration.language_model_sha256 != expected_lm_sha256:
                raise ValueError(
                    "Confidence calibration language model mismatch"
                )
            if not np.isclose(
                self.calibration.language_model_weight,
                self.language_model_weight,
            ):
                raise ValueError(
                    "Confidence calibration language model weight mismatch"
                )
            if not np.isclose(
                self.calibration.language_model_token_bonus,
                self.language_model_token_bonus,
            ):
                raise ValueError(
                    "Confidence calibration language model token bonus mismatch"
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
        if self.decoder == "beam":
            decoded = ctc_prefix_beam_search(
                valid.numpy(),
                beam_width=self.beam_width,
                extension_scorer=(
                    self.language_model.score_extension
                    if self.language_model is not None
                    else None
                ),
                scorer_weight=self.language_model_weight,
                token_bonus=self.language_model_token_bonus,
            )
            mapping = self.vocabulary.id_to_char
            try:
                text = "".join(mapping[token_id] for token_id in decoded.token_ids)
            except KeyError as exc:
                raise ValueError(
                    f"Beam decoder returned token outside vocabulary: {exc.args[0]}"
                ) from exc
            confidence = float(
                np.exp(decoded.log_probability / max(1, valid_timesteps))
            )
            confidence = max(0.0, min(1.0, confidence))
            calibrated_confidence = (
                self.calibration.calibrate(confidence)
                if self.calibration is not None
                else None
            )
            decoder_ranking_score = decoded.ranking_score
            language_model_log_probability = (
                decoded.language_model_log_probability
                if self.language_model is not None
                else None
            )
        else:
            max_probabilities, token_ids = probabilities.max(dim=-1)
            text = self.vocabulary.decode_ctc(token_ids.tolist())
            confidence = (
                float(max_probabilities.mean().item())
                if valid_timesteps
                else 0.0
            )
            calibrated_confidence = (
                self.calibration.calibrate(confidence)
                if self.calibration is not None
                else None
            )
            decoder_ranking_score = None
            language_model_log_probability = None
        return RecognitionResult(
            text=text,
            confidence=confidence,
            calibrated_confidence=calibrated_confidence,
            valid_timesteps=valid_timesteps,
            decoder_ranking_score=decoder_ranking_score,
            language_model_log_probability=language_model_log_probability,
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
