from __future__ import annotations

import json
import sys

from lao_document_ocr.cli import main


def test_sample_hplt_cli_forwards_filter_and_writes_outputs(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import lao_document_ocr.hplt_sampler as sampler

    output = tmp_path / "lao-lines.txt"
    metadata = tmp_path / "lao-lines.meta.json"
    captured = {}

    def fake_sample_hplt_lao(**kwargs):
        captured.update(kwargs)
        output.write_text("ສະບາຍດີ ໂລກ\n", encoding="utf-8")
        payload = {
            "sampling": {"accepted_lines": 1},
            "corpus": {"path": output.name, "sha256": "a" * 64},
        }
        metadata.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        return output, metadata, payload

    monkeypatch.setattr(sampler, "sample_hplt_lao", fake_sample_hplt_lao)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "sample-hplt-lao",
            "--output",
            str(output),
            "--metadata",
            str(metadata),
            "--limit",
            "25",
            "--map-url",
            "https://example.test/lao_map.txt",
            "--timeout",
            "12",
            "--max-lines-per-document",
            "3",
            "--max-documents",
            "500",
            "--min-chars",
            "6",
            "--max-chars",
            "120",
            "--min-lao-ratio",
            "0.7",
        ],
    )

    assert main() == 0
    assert captured["output_path"] == output
    assert captured["metadata_path"] == metadata
    assert captured["limit"] == 25
    assert captured["map_url"] == "https://example.test/lao_map.txt"
    assert captured["timeout"] == 12.0
    assert captured["max_lines_per_document"] == 3
    assert captured["max_documents"] == 500
    config = captured["config"]
    assert config.min_chars == 6
    assert config.max_chars == 120
    assert config.min_lao_ratio == 0.7
    assert config.deduplicate is True

    rendered = capsys.readouterr().out
    assert f"Corpus: {output}" in rendered
    assert f"Metadata: {metadata}" in rendered
    assert "Accepted lines: 1" in rendered


def test_sample_hplt_cli_keep_duplicates_disables_dedup(
    tmp_path,
    monkeypatch,
) -> None:
    import lao_document_ocr.hplt_sampler as sampler

    output = tmp_path / "lao-lines.txt"
    metadata = tmp_path / "lao-lines.meta.json"
    captured = {}

    def fake_sample_hplt_lao(**kwargs):
        captured.update(kwargs)
        output.write_text("ສະບາຍດີ\n", encoding="utf-8")
        payload = {"sampling": {"accepted_lines": 1}}
        metadata.write_text(json.dumps(payload), encoding="utf-8")
        return output, metadata, payload

    monkeypatch.setattr(sampler, "sample_hplt_lao", fake_sample_hplt_lao)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "sample-hplt-lao",
            "--output",
            str(output),
            "--metadata",
            str(metadata),
            "--limit",
            "1",
            "--keep-duplicates",
        ],
    )

    assert main() == 0
    assert captured["config"].deduplicate is False
