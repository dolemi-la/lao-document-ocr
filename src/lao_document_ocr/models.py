from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class BlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    IMAGE = "image"


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class TableCell(BaseModel):
    row: int
    column: int
    text: str
    row_span: int = 1
    column_span: int = 1


class Block(BaseModel):
    type: BlockType
    text: str = ""
    bbox: BoundingBox | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    level: int | None = None
    cells: list[TableCell] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Page(BaseModel):
    number: int
    width: int
    height: int
    blocks: list[Block] = Field(default_factory=list)


class Document(BaseModel):
    version: str = "0.1"
    language: str = "lo"
    source_name: str | None = None
    pages: list[Page] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def plain_text(self) -> str:
        page_texts: list[str] = []
        for page in self.pages:
            page_texts.append("\n".join(block.text for block in page.blocks if block.text))
        return "\n\n".join(page_texts)
