import hashlib
import json
from pathlib import Path

from lao_document_ocr.corpus import lao_ratio

ROOT = Path(__file__).parents[1]
CORPUS = ROOT / "resources" / "corpora" / "project-authored-lao-v1.txt"
METADATA = (
    ROOT / "resources" / "corpora" / "project-authored-lao-v1.meta.json"
)


def _lines() -> list[str]:
    return [
        line
        for line in CORPUS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_project_authored_capture_corpus_metadata_matches_file() -> None:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    lines = _lines()

    assert metadata["schema_version"] == "1"
    assert metadata["id"] == "project-authored-lao-v1"
    assert metadata["source"] == "project-authored"
    assert "Created specifically for this project" in metadata["rights_statement"]
    assert metadata["license"] == "CC0-1.0"
    assert metadata["not_real_ocr_evidence"] is True
    assert metadata["line_count"] == len(lines)
    assert metadata["sha256"] == hashlib.sha256(CORPUS.read_bytes()).hexdigest()


def test_project_authored_capture_corpus_has_useful_coverage() -> None:
    lines = _lines()

    assert len(lines) >= 80
    assert len(lines) == len(set(lines))
    assert max(len(line) for line in lines) <= 180

    lao_heavy = sum(lao_ratio(line) >= 0.5 for line in lines)
    latin_mixed = sum(
        any(("A" <= char <= "Z") or ("a" <= char <= "z") for char in line)
        for line in lines
    )
    numeric = sum(any(char.isdigit() for char in line) for line in lines)

    assert lao_heavy >= 70
    assert latin_mixed >= 15
    assert numeric >= 30


def test_project_authored_capture_corpus_intended_use_is_public_capture_safe() -> None:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))

    assert "capture-pack-source-text" in metadata["intended_use"]
    assert "public-benchmark-ground-truth-source" in metadata["intended_use"]
    assert "synthetic-rendering" in metadata["intended_use"]
    assert "OCR regression fixtures" in metadata["intended_use"]
    assert metadata["not_real_ocr_evidence"] is True
