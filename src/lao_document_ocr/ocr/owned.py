from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PIL import Image

from lao_document_ocr.line_detection import detect_region_aware_lines
from lao_document_ocr.text_regions import (
    MorphologyTextRegionDetector,
    TextRegionDetector,
)

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
        device: str = "cpu",
        recognizer: ImageLineRecognizer | None = None,
        region_detector: TextRegionDetector | None = None,
        region_detector_name: str = "morphology",
        layout_model_path: str | Path | None = None,
        layout_confidence_threshold: float = 0.55,
    ) -> None:
        if region_detector is not None:
            self.region_detector = region_detector
        elif region_detector_name == "morphology":
            self.region_detector = MorphologyTextRegionDetector()
        elif region_detector_name == "learned":
            if layout_model_path is None:
                raise OcrEngineError(
                    "Learned text-region detection requires a layout model path"
                )
            try:
                from lao_document_ocr.layout_detector_inference import (
                    ExportedLayoutRegionDetector,
                )

                self.region_detector = ExportedLayoutRegionDetector(
                    layout_model_path,
                    device=device,
                    confidence_threshold=layout_confidence_threshold,
                )
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                raise OcrEngineError(
                    f"Could not load learned text-region detector: {exc}"
                ) from exc
        else:
            raise OcrEngineError(
                "region_detector_name must be 'morphology' or 'learned'"
            )

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
                device=device,
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
            "text_region_detector": self.region_detector.metadata(),
        }

    def recognize(self, image: Image.Image) -> list[RecognizedLine]:
        boxes = detect_region_aware_lines(
            image,
            region_detector=self.region_detector,
        )
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
