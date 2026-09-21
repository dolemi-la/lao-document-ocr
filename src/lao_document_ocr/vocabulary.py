from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from lao_document_ocr.normalization import normalize_lao_text

BLANK_ID = 0


@dataclass(frozen=True)
class CharacterVocabulary:
    characters: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.characters)) != len(self.characters):
            raise ValueError("Vocabulary contains duplicate characters")
        if any(len(char) != 1 for char in self.characters):
            raise ValueError("Vocabulary entries must be single Unicode characters")

    @property
    def size(self) -> int:
        return len(self.characters) + 1

    @property
    def char_to_id(self) -> dict[str, int]:
        return {char: index + 1 for index, char in enumerate(self.characters)}

    @property
    def id_to_char(self) -> dict[int, str]:
        return {index + 1: char for index, char in enumerate(self.characters)}

    @classmethod
    def from_texts(cls, texts: list[str]) -> CharacterVocabulary:
        chars: set[str] = set()
        for text in texts:
            chars.update(normalize_lao_text(text))
        chars.discard("\n")
        chars.discard("\r")
        return cls(tuple(sorted(chars)))

    def encode(self, text: str) -> list[int]:
        mapping = self.char_to_id
        normalized = normalize_lao_text(text)
        unknown = sorted({char for char in normalized if char not in mapping})
        if unknown:
            rendered = " ".join(f"U+{ord(char):04X}" for char in unknown)
            raise ValueError(f"Text contains characters missing from vocabulary: {rendered}")
        return [mapping[char] for char in normalized]

    def decode_ctc(self, token_ids: list[int]) -> str:
        mapping = self.id_to_char
        output: list[str] = []
        previous = BLANK_ID
        for token_id in token_ids:
            if token_id == BLANK_ID:
                previous = BLANK_ID
                continue
            if token_id == previous:
                continue
            char = mapping.get(token_id)
            if char is None:
                raise ValueError(f"Token id {token_id} is outside the vocabulary")
            output.append(char)
            previous = token_id
        return "".join(output)

    def to_dict(self) -> dict:
        return {
            "schema_version": "1",
            "blank_id": BLANK_ID,
            "characters": list(self.characters),
            "size": self.size,
        }

    def checksum(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> CharacterVocabulary:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != "1":
            raise ValueError("Unsupported vocabulary schema version")
        if payload.get("blank_id") != BLANK_ID:
            raise ValueError("Vocabulary blank_id must be 0")
        characters = payload.get("characters")
        if not isinstance(characters, list) or not all(
            isinstance(item, str) for item in characters
        ):
            raise ValueError("Vocabulary characters must be a list of strings")
        return cls(tuple(characters))
