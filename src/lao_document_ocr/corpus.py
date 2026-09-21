from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from lao_document_ocr.normalization import normalize_lao_text

_LAO_CHAR = re.compile(r"[\u0E80-\u0EFF]")
_LETTER_OR_DIGIT = re.compile(r"[\w\u0E80-\u0EFF]", re.UNICODE)


@dataclass(frozen=True)
class CorpusFilter:
    min_chars: int = 8
    max_chars: int = 180
    min_lao_ratio: float = 0.5
    deduplicate: bool = True

    def __post_init__(self) -> None:
        if self.min_chars < 1:
            raise ValueError("min_chars must be at least 1")
        if self.max_chars < self.min_chars:
            raise ValueError("max_chars must be >= min_chars")
        if not 0 <= self.min_lao_ratio <= 1:
            raise ValueError("min_lao_ratio must be between 0 and 1")


def lao_ratio(text: str) -> float:
    meaningful = _LETTER_OR_DIGIT.findall(text)
    if not meaningful:
        return 0.0
    lao = sum(1 for char in meaningful if _LAO_CHAR.fullmatch(char))
    return lao / len(meaningful)


def normalize_corpus_line(text: str) -> str:
    return normalize_lao_text(text.replace("\n", " "))


def is_eligible_line(text: str, config: CorpusFilter) -> bool:
    normalized = normalize_corpus_line(text)
    length = len(normalized)
    if length < config.min_chars or length > config.max_chars:
        return False
    return lao_ratio(normalized) >= config.min_lao_ratio


def iter_plain_text(path: str | Path) -> Iterator[str]:
    with Path(path).open("r", encoding="utf-8") as source:
        for line in source:
            yield line.rstrip("\n")


def iter_jsonl(path: str | Path, field: str = "text") -> Iterator[str]:
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} must be an object")
            text = value.get(field)
            if not isinstance(text, str):
                raise ValueError(f"JSONL line {line_number} has no string field '{field}'")
            yield text


def prepare_corpus(
    lines: Iterable[str],
    config: CorpusFilter | None = None,
    *,
    limit: int | None = None,
) -> list[str]:
    config = config or CorpusFilter()
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")

    output: list[str] = []
    seen: set[str] = set()

    for raw in lines:
        normalized = normalize_corpus_line(raw)
        if not is_eligible_line(normalized, config):
            continue

        if config.deduplicate:
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)

        output.append(normalized)
        if limit is not None and len(output) >= limit:
            break

    return output


def write_corpus(lines: Iterable[str], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(lines)
    payload = "\n".join(materialized)
    if payload:
        payload += "\n"
    destination.write_text(payload, encoding="utf-8")
    return destination
