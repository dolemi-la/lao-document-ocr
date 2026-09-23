import json

import pytest

from lao_document_ocr.corpus import CorpusFilter
from lao_document_ocr.hplt_sampler import (
    HpltSamplingError,
    order_shards_by_quality,
    parse_map_urls,
    sample_hplt_lao,
)


def test_parse_map_urls_requires_https_and_nonempty() -> None:
    assert parse_map_urls("https://example.test/a.zst\nhttps://example.test/b.zst\n") == [
        "https://example.test/a.zst",
        "https://example.test/b.zst",
    ]
    with pytest.raises(HpltSamplingError, match="no shard"):
        parse_map_urls("\n")
    with pytest.raises(HpltSamplingError, match="non-HTTPS"):
        parse_map_urls("http://example.test/a.zst\n")


def test_bounded_sampler_filters_deduplicates_and_records_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    import lao_document_ocr.hplt_sampler as sampler

    map_url = "https://example.test/lao_map.txt"
    shard_url = "https://example.test/1.jsonl.zst"

    monkeypatch.setattr(
        sampler,
        "fetch_url_text",
        lambda url, timeout=30.0: (
            f"{shard_url}\n"
            if url == map_url
            else "0123456789abcdef0123456789abcdef  1.jsonl.zst\n"
        ),
    )
    monkeypatch.setattr(
        sampler,
        "iter_zstd_jsonl_url",
        lambda url, timeout=60.0: iter(
            [
                {"text": "ສະບາຍດີ ໂລກ\nEnglish only"},
                {"text": "ສະບາຍດີ ໂລກ\nຂອບໃຈ ຫຼາຍ"},
            ]
        ),
    )

    corpus, metadata_path, metadata = sample_hplt_lao(
        output_path=tmp_path / "lao-lines.txt",
        metadata_path=tmp_path / "lao-lines.meta.json",
        limit=2,
        config=CorpusFilter(min_chars=4, max_chars=100, min_lao_ratio=0.5),
        map_url=map_url,
    )

    assert corpus.read_text(encoding="utf-8").splitlines() == [
        "ສະບາຍດີ ໂລກ",
        "ຂອບໃຈ ຫຼາຍ",
    ]
    assert metadata["source"] == "HPLT Monolingual Datasets v3.0"
    assert metadata["variant"] == "sorted"
    assert metadata["shard_order"] == "highest-wds-bin-first"
    assert metadata["sampling"]["accepted_lines"] == 2
    assert metadata["sampling"]["bounded_stream"] is True
    assert metadata["shards"][0]["full_shard_downloaded"] is False
    assert metadata["shards"][0]["expected_full_shard_md5"] == (
        "0123456789abcdef0123456789abcdef"
    )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["corpus"]["sha256"] == metadata["corpus"]["sha256"]
    assert "underlying extracted text" in payload["rights_note"]


def test_sampler_requires_positive_limit(tmp_path) -> None:
    with pytest.raises(ValueError, match="limit"):
        sample_hplt_lao(
            output_path=tmp_path / "x.txt",
            metadata_path=tmp_path / "x.json",
            limit=0,
        )


def test_quality_order_prefers_highest_wds_bin() -> None:
    urls = [
        "https://example.test/lao_Laoo/5_1.jsonl.zst",
        "https://example.test/lao_Laoo/8_1.jsonl.zst",
        "https://example.test/lao_Laoo/7_1.jsonl.zst",
        "https://example.test/lao_Laoo/6_1.jsonl.zst",
    ]

    assert order_shards_by_quality(urls) == [
        "https://example.test/lao_Laoo/8_1.jsonl.zst",
        "https://example.test/lao_Laoo/7_1.jsonl.zst",
        "https://example.test/lao_Laoo/6_1.jsonl.zst",
        "https://example.test/lao_Laoo/5_1.jsonl.zst",
    ]


def test_sampler_caps_lines_per_document(tmp_path, monkeypatch) -> None:
    import lao_document_ocr.hplt_sampler as sampler

    map_url = "https://example.test/lao.map"
    shard_url = "https://example.test/8_1.jsonl.zst"
    monkeypatch.setattr(
        sampler,
        "fetch_url_text",
        lambda url, timeout=30.0: (
            f"{shard_url}\n"
            if url == map_url
            else "0123456789abcdef0123456789abcdef shard\n"
        ),
    )
    monkeypatch.setattr(
        sampler,
        "iter_zstd_jsonl_url",
        lambda url, timeout=60.0: iter(
            [
                {"text": "ລາວ ແຖວ 1\nລາວ ແຖວ 2\nລາວ ແຖວ 3"},
                {"text": "ລາວ ແຖວ 4\nລາວ ແຖວ 5\nລາວ ແຖວ 6"},
            ]
        ),
    )

    corpus, _, metadata = sample_hplt_lao(
        output_path=tmp_path / "lines.txt",
        metadata_path=tmp_path / "meta.json",
        limit=4,
        config=CorpusFilter(min_chars=4, max_chars=100, min_lao_ratio=0.5),
        map_url=map_url,
        max_lines_per_document=2,
    )

    assert corpus.read_text(encoding="utf-8").splitlines() == [
        "ລາວ ແຖວ 1",
        "ລາວ ແຖວ 2",
        "ລາວ ແຖວ 4",
        "ລາວ ແຖວ 5",
    ]
    assert metadata["sampling"]["documents_seen"] == 2
    assert metadata["sampling"]["max_lines_per_document"] == 2


def test_sampler_fails_when_document_bound_prevents_requested_limit(
    tmp_path,
    monkeypatch,
) -> None:
    import lao_document_ocr.hplt_sampler as sampler

    map_url = "https://example.test/lao.map"
    shard_url = "https://example.test/8_1.jsonl.zst"
    monkeypatch.setattr(
        sampler,
        "fetch_url_text",
        lambda url, timeout=30.0: (
            f"{shard_url}\n"
            if url == map_url
            else "0123456789abcdef0123456789abcdef shard\n"
        ),
    )
    monkeypatch.setattr(
        sampler,
        "iter_zstd_jsonl_url",
        lambda url, timeout=60.0: iter(
            [
                {"text": "ລາວ ແຖວ 1\nລາວ ແຖວ 2"},
                {"text": "ລາວ ແຖວ 3\nລາວ ແຖວ 4"},
            ]
        ),
    )

    with pytest.raises(HpltSamplingError, match="1/3 accepted lines"):
        sample_hplt_lao(
            output_path=tmp_path / "lines.txt",
            metadata_path=tmp_path / "meta.json",
            limit=3,
            config=CorpusFilter(
                min_chars=4,
                max_chars=100,
                min_lao_ratio=0.5,
            ),
            map_url=map_url,
            max_lines_per_document=1,
            max_documents=1,
        )
