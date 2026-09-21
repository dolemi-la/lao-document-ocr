from __future__ import annotations

from collections import defaultdict
from typing import Any

import pytesseract
from PIL import Image
from pytesseract import Output, TesseractNotFoundError

from lao_document_ocr.models import BoundingBox
from lao_document_ocr.normalization import normalize_lao_text

from .base import OcrEngine, OcrEngineError, RecognizedLine


class TesseractEngine(OcrEngine):
    def __init__(self, languages: str = "lao+eng", psm: int = 3) -> None:
        self.languages = languages
        self.psm = psm

    def available_languages(self) -> list[str]:
        try:
            return sorted(pytesseract.get_languages(config=""))
        except TesseractNotFoundError as exc:
            raise OcrEngineError(
                "Tesseract is not installed. Install Tesseract and Lao language data."
            ) from exc

    def is_available(self) -> bool:
        try:
            available = set(self.available_languages())
        except OcrEngineError:
            return False
        required = set(self.languages.split("+"))
        return required.issubset(available)

    def metadata(self) -> dict[str, Any]:
        try:
            version = str(pytesseract.get_tesseract_version())
        except TesseractNotFoundError:
            version = "unavailable"
        return {
            "name": self.__class__.__name__,
            "tesseract_version": version,
            "languages": self.languages,
            "psm": self.psm,
        }

    def _validate(self) -> None:
        available = set(self.available_languages())
        missing = sorted(set(self.languages.split("+")) - available)
        if missing:
            joined = ", ".join(missing)
            raise OcrEngineError(
                f"Missing Tesseract language data: {joined}. "
                "Install Lao and English traineddata before processing documents."
            )

    def recognize(self, image: Image.Image) -> list[RecognizedLine]:
        self._validate()
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.languages,
                config=f"--psm {self.psm}",
                output_type=Output.DICT,
            )
        except TesseractNotFoundError as exc:
            raise OcrEngineError("Tesseract executable was not found.") from exc
        except pytesseract.TesseractError as exc:
            raise OcrEngineError(f"Tesseract failed: {exc}") from exc

        groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for index, raw_text in enumerate(data["text"]):
            if raw_text and raw_text.strip():
                key = (
                    int(data["block_num"][index]),
                    int(data["par_num"][index]),
                    int(data["line_num"][index]),
                )
                groups[key].append(index)

        lines: list[RecognizedLine] = []
        for (block_id, paragraph_id, line_id), indexes in groups.items():
            words: list[str] = []
            confidences: list[float] = []
            lefts: list[int] = []
            tops: list[int] = []
            rights: list[int] = []
            bottoms: list[int] = []

            for index in indexes:
                word = normalize_lao_text(data["text"][index])
                if not word:
                    continue
                words.append(word)
                try:
                    confidence = float(data["conf"][index])
                except (TypeError, ValueError):
                    confidence = -1
                if confidence >= 0:
                    confidences.append(confidence / 100.0)

                left = int(data["left"][index])
                top = int(data["top"][index])
                width = int(data["width"][index])
                height = int(data["height"][index])
                lefts.append(left)
                tops.append(top)
                rights.append(left + width)
                bottoms.append(top + height)

            text = normalize_lao_text(" ".join(words))
            if not text or not lefts:
                continue

            bbox = BoundingBox(
                x=min(lefts),
                y=min(tops),
                width=max(rights) - min(lefts),
                height=max(bottoms) - min(tops),
            )
            confidence = sum(confidences) / len(confidences) if confidences else 0.0
            lines.append(
                RecognizedLine(
                    text=text,
                    bbox=bbox,
                    confidence=confidence,
                    block_id=block_id,
                    paragraph_id=paragraph_id,
                    line_id=line_id,
                )
            )

        return sorted(lines, key=lambda line: (line.bbox.y, line.bbox.x))
