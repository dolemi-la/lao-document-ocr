from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PIL import Image

from lao_document_ocr.line_detection import detect_text_lines

from .base import OcrEngine, OcrEngineError, RecognizedLine


class ImageLineRecognizer(Protocol):
    metadata: dict

    def recognize_image(self, image: Image.Image): ...


class OwnedRecognizerEngine(OcrEngine):
    """Full-page OCR using our line detector plus project-owned recognizer."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        calibration_path: str | Path | None = None,
        recognizer: ImageLineRecognizer | None = None,
    ) -> None:
        if recognizer is not None:
            self.recognizer = recognizer
            self.model_path = Path(model_path) if model_path is not None else None
            return

        if model_path is None:
            raise OcrEngineError("Owned recognizer engine requires a model path")

        try:
            from lao_document_ocr.recognizer_inference import ExportedLineRecognizer

            self.recognizer = ExportedLineRecognizer(
                model_path,
                calibration_path=calibration_path,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise OcrEngineError(f"Could not load owned recognizer: {exc}") from exc

        self.model_path = Path(model_path)

    def is_available(self) -> bool:
        return self.recognizer is not None

    def metadata(self) -> dict:
        metadata = getattr(self.recognizer, "metadata", None)
        return {
            "name": self.__class__.__name__,
            "model": metadata,
        }

    def recognize(self, image: Image.Image) -> list[RecognizedLine]:
        boxes = detect_text_lines(image)
        lines: list[RecognizedLine] = []

        for line_index, box in enumerate(boxes, start=1):
            crop = image.crop(
                (
                    box.x,
                    box.y,
                    box.x + box.width,
                    box.y + box.height,
                )
            )
            result = self.recognizer.recognize_image(crop)
            text = result.text.strip()
            if not text:
                continue

            confidence = (
                result.calibrated_confidence
                if result.calibrated_confidence is not None
                else result.confidence
            )
            lines.append(
                RecognizedLine(
                    text=text,
                    bbox=box,
                    confidence=max(0.0, min(1.0, float(confidence))),
                    block_id=line_index,
                    paragraph_id=line_index,
                    line_id=line_index,
                )
            )

        return lines
