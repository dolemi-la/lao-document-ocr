from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image

from lao_document_ocr.models import BoundingBox


class OcrEngineError(RuntimeError):
    pass


@dataclass(frozen=True)
class RecognizedLine:
    text: str
    bbox: BoundingBox
    confidence: float
    block_id: int
    paragraph_id: int
    line_id: int


class OcrEngine(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def recognize(self, image: Image.Image) -> list[RecognizedLine]:
        raise NotImplementedError
