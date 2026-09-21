import json

import pytest

from lao_document_ocr.corpus import (
    CorpusFilter,
    iter_jsonl,
    lao_ratio,
    normalize_corpus_line,
    prepare_corpus,
)


def test_lao_ratio_counts_meaningful_characters() -> None:
    assert lao_ratio("ສະບາຍດີ") == 1.0
    assert lao_ratio("abc") == 0.0
    assert 0 < lao_ratio("ລາວ OCR") < 1


def test_prepare_corpus_filters_normalizes_and_deduplicates() -> None:
    config = CorpusFilter(min_chars=4, max_chars=50, min_lao_ratio=0.25)
    lines = [
        "  ສະບາຍດີ   ໂລກ  ",
        "ສະບາຍດີ ໂລກ",
        "English only line",
        "ລາວ OCR 123",
        "ກ",
    ]

    result = prepare_corpus(lines, config)

    assert result == ["ສະບາຍດີ ໂລກ", "ລາວ OCR 123"]


def test_prepare_corpus_limit() -> None:
    config = CorpusFilter(min_chars=1, max_chars=50, min_lao_ratio=1)
    result = prepare_corpus(["ລາວ", "ສະບາຍດີ"], config, limit=1)
    assert result == ["ລາວ"]


def test_iter_jsonl_reads_selected_field(tmp_path) -> None:
    path = tmp_path / "sample.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"content": "ສະບາຍດີ"}, ensure_ascii=False),
                json.dumps({"content": "ຂອບໃຈ"}, ensure_ascii=False),
            ]
        ),
        encoding="utf-8",
    )
    assert list(iter_jsonl(path, field="content")) == ["ສະບາຍດີ", "ຂອບໃຈ"]


def test_iter_jsonl_rejects_missing_field(tmp_path) -> None:
    path = tmp_path / "sample.jsonl"
    path.write_text('{"other":"x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no string field"):
        list(iter_jsonl(path, field="text"))


def test_normalize_corpus_line_flattens_newlines() -> None:
    assert normalize_corpus_line("ສະບາຍດີ\nໂລກ") == "ສະບາຍດີ ໂລກ"
