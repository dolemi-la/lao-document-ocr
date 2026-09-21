from __future__ import annotations

import sys

from lao_document_ocr.cli import main


def test_prepare_corpus_cli(tmp_path, monkeypatch, capsys) -> None:
    source = tmp_path / "source.txt"
    output = tmp_path / "corpus.txt"
    source.write_text(
        "ສະບາຍດີ ໂລກ\n"
        "English only\n"
        "ລາວ OCR 123\n"
        "ສະບາຍດີ ໂລກ\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "prepare-corpus",
            "--input",
            str(source),
            "--output",
            str(output),
            "--min-lao-ratio",
            "0.25",
        ],
    )

    assert main() == 0
    assert output.read_text(encoding="utf-8").splitlines() == [
        "ສະບາຍດີ ໂລກ",
        "ລາວ OCR 123",
    ]
    captured = capsys.readouterr()
    assert "Lines: 2" in captured.out
