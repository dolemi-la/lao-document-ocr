from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pymupdf

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




def filter_font_compatible_lines(
    lines: Iterable[str],
    font_path: str | Path,
) -> tuple[list[str], dict]:
    font_file = Path(font_path)
    if not font_file.is_file():
        raise FileNotFoundError(f"Font not found: {font_file}")

    font = pymupdf.Font(fontfile=str(font_file))
    compatible: list[str] = []
    missing_line_counts: Counter[int] = Counter()
    total = 0

    for text in lines:
        total += 1
        missing = {
            ord(char)
            for char in text
            if char not in {"\n", "\r", "\t"}
            and font.has_glyph(ord(char)) == 0
        }
        if missing:
            missing_line_counts.update(missing)
            continue
        compatible.append(text)

    digest = hashlib.sha256(font_file.read_bytes()).hexdigest()
    report = {
        "schema_version": "1",
        "font": font_file.name,
        "font_sha256": digest,
        "input_lines": total,
        "compatible_lines": len(compatible),
        "excluded_lines": total - len(compatible),
        "missing_codepoints": [
            {
                "codepoint": f"U+{codepoint:04X}",
                "character": chr(codepoint),
                "affected_lines": missing_line_counts[codepoint],
            }
            for codepoint in sorted(missing_line_counts)
        ],
    }
    return compatible, report


def write_font_coverage_report(report: dict, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def write_corpus(lines: Iterable[str], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(lines)
    payload = "\n".join(materialized)
    if payload:
        payload += "\n"
    destination.write_text(payload, encoding="utf-8")
    return destination
