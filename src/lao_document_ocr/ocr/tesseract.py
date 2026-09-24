from __future__ import annotations

import hashlib
import shlex
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytesseract
from PIL import Image
from pytesseract import Output, TesseractNotFoundError

from lao_document_ocr.models import BoundingBox
from lao_document_ocr.normalization import normalize_lao_text

from .base import OcrEngine, OcrEngineError, RecognizedLine


class TesseractEngine(OcrEngine):
    def __init__(
        self,
        languages: str = "lao+eng",
        psm: int = 3,
        tessdata_dir: str | Path | None = None,
    ) -> None:
        self.languages = languages
        self.psm = psm
        self.tessdata_dir = (
            Path(tessdata_dir).expanduser().resolve()
            if tessdata_dir is not None
            else None
        )
        if self.tessdata_dir is not None and not self.tessdata_dir.is_dir():
            raise ValueError(
                f"Tesseract tessdata directory not found: {self.tessdata_dir}"
            )
        self._traineddata_sha256 = self._compute_traineddata_hashes()

    def _tessdata_config(self) -> str:
        if self.tessdata_dir is None:
            return ""
        return f"--tessdata-dir {shlex.quote(str(self.tessdata_dir))}"

    def _ocr_config(self) -> str:
        parts = [f"--psm {self.psm}"]
        tessdata = self._tessdata_config()
        if tessdata:
            parts.append(tessdata)
        return " ".join(parts)

    def _compute_traineddata_hashes(self) -> dict[str, str]:
        if self.tessdata_dir is None:
            return {}
        hashes: dict[str, str] = {}
        for language in self.languages.split("+"):
            path = self.tessdata_dir / f"{language}.traineddata"
            if not path.is_file():
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[language] = digest.hexdigest()
        return hashes

    def available_languages(self) -> list[str]:
        try:
            return sorted(
                pytesseract.get_languages(config=self._tessdata_config())
            )
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
            "tessdata": {
                "source": (
                    "custom"
                    if self.tessdata_dir is not None
                    else "system-default"
                ),
                "traineddata_sha256": dict(self._traineddata_sha256),
            },
        }

    def orientation_hint(self, image: Image.Image) -> dict[str, Any] | None:
        try:
            data = pytesseract.image_to_osd(
                image,
                config=self._tessdata_config(),
                output_type=Output.DICT,
            )
        except (TesseractNotFoundError, pytesseract.TesseractError):
            return None

        try:
            rotate = int(data.get("rotate", 0)) % 360
        except (TypeError, ValueError):
            rotate = 0
        try:
            orientation_confidence = float(data.get("orientation_conf", 0.0))
        except (TypeError, ValueError):
            orientation_confidence = 0.0
        try:
            script_confidence = float(data.get("script_conf", 0.0))
        except (TypeError, ValueError):
            script_confidence = 0.0

        script = str(data.get("script", "")).strip() or None
        return {
            "degrees_clockwise": rotate,
            "orientation_confidence": orientation_confidence,
            "script": script,
            "script_confidence": script_confidence,
            "source": "tesseract-osd",
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
                config=self._ocr_config(),
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
