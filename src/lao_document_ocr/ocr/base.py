from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from PIL import Image

from lao_document_ocr.models import Block, BlockType, BoundingBox


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
    semantic_type: BlockType | None = None


class OcrEngine(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def recognize(self, image: Image.Image) -> list[RecognizedLine]:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return {"name": self.__class__.__name__}

    def visual_blocks(
        self,
        image: Image.Image,
        *,
        source_image: Image.Image | None = None,
        exclude_boxes: list[BoundingBox] | None = None,
    ) -> list[Block]:
        del image, source_image, exclude_boxes
        return []
