from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

from lao_document_ocr.corpus import (
    CorpusFilter,
    is_eligible_line,
    normalize_corpus_line,
    write_corpus,
)

HPLT_V3_LAO_SORTED_MAP_URL = (
    "https://data.hplt-project.org/three/sorted/lao_Laoo.map"
)
HPLT_V3_TERMS_URL = "https://hplt-project.org/datasets/v3.0"
_USER_AGENT = "lao-document-ocr/0.1 HPLT bounded sampler"


class HpltSamplingError(RuntimeError):
    pass


def _request(url: str) -> Request:
    return Request(url, headers={"User-Agent": _USER_AGENT})


def fetch_url_text(url: str, *, timeout: float = 30.0) -> str:
    try:
        with urlopen(_request(url), timeout=timeout) as response:  # noqa: S310
            return response.read().decode("utf-8")
    except Exception as exc:
        raise HpltSamplingError(f"Could not fetch {url}: {exc}") from exc


def parse_map_urls(text: str) -> list[str]:
    urls = [line.strip() for line in text.splitlines() if line.strip()]
    if not urls:
        raise HpltSamplingError("HPLT language map contains no shard URLs")
    invalid = [url for url in urls if not url.startswith("https://")]
    if invalid:
        raise HpltSamplingError("HPLT language map contains a non-HTTPS URL")
    return urls


def _quality_bin(url: str) -> int:
    name = url.rsplit("/", 1)[-1]
    prefix = name.split("_", 1)[0]
    return int(prefix) if prefix.isdigit() else -1


def order_shards_by_quality(urls: list[str]) -> list[str]:
    return sorted(
        urls,
        key=lambda url: (_quality_bin(url), url),
        reverse=True,
    )


def _checksum_url(shard_url: str) -> str:
    prefix, name = shard_url.rsplit("/", 1)
    if name.endswith(".jsonl.zst"):
        stem = name[: -len(".zst")]
        return f"{prefix}/.{stem}.md5"
    return f"{shard_url}.md5"


def _zstandard_module():
    try:
        import zstandard
    except ImportError as exc:
        raise RuntimeError(
            "HPLT streaming requires the optional 'data' dependencies. "
            "Install with: pip install -e '.[data]'"
        ) from exc
    return zstandard


def iter_zstd_jsonl_url(
    url: str,
    *,
    timeout: float = 60.0,
):
    zstandard = _zstandard_module()
    try:
        response = urlopen(_request(url), timeout=timeout)  # noqa: S310
    except Exception as exc:
        raise HpltSamplingError(f"Could not open HPLT shard {url}: {exc}") from exc

    try:
        decompressor = zstandard.ZstdDecompressor()
        with decompressor.stream_reader(response) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8") as text_stream:
                for line_number, raw_line in enumerate(text_stream, start=1):
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    try:
                        payload = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        raise HpltSamplingError(
                            f"Invalid HPLT JSONL in {url} at line {line_number}: {exc}"
                        ) from exc
                    if not isinstance(payload, dict):
                        raise HpltSamplingError(
                            f"HPLT JSONL entry at line {line_number} is not an object"
                        )
                    yield payload
    finally:
        response.close()


def _expected_md5(shard_url: str, *, timeout: float) -> str | None:
    try:
        text = fetch_url_text(_checksum_url(shard_url), timeout=timeout)
    except HpltSamplingError:
        return None
    first = text.strip().split()[0] if text.strip() else ""
    if len(first) == 32 and all(char in "0123456789abcdefABCDEF" for char in first):
        return first.lower()
    return None


def sample_hplt_lao(
    *,
    output_path: str | Path,
    metadata_path: str | Path,
    limit: int,
    config: CorpusFilter | None = None,
    map_url: str = HPLT_V3_LAO_SORTED_MAP_URL,
    timeout: float = 60.0,
    max_lines_per_document: int = 4,
    max_documents: int = 10_000,
) -> tuple[Path, Path, dict]:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if max_lines_per_document < 1:
        raise ValueError("max_lines_per_document must be at least 1")
    if max_documents < 1:
        raise ValueError("max_documents must be at least 1")
    config = config or CorpusFilter()

    shard_urls = order_shards_by_quality(
        parse_map_urls(fetch_url_text(map_url, timeout=timeout))
    )
    selected: list[str] = []
    seen_hashes: set[str] = set()
    documents_seen = 0
    segments_seen = 0
    used_shards: list[dict] = []

    for shard_url in shard_urls:
        shard_documents = 0
        for document in iter_zstd_jsonl_url(shard_url, timeout=timeout):
            if documents_seen >= max_documents:
                break
            documents_seen += 1
            shard_documents += 1
            text = document.get("text")
            if not isinstance(text, str):
                continue
            accepted_from_document = 0
            for segment in text.splitlines():
                segments_seen += 1
                normalized = normalize_corpus_line(segment)
                if not is_eligible_line(normalized, config):
                    continue
                digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                if config.deduplicate and digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                selected.append(normalized)
                accepted_from_document += 1
                if (
                    len(selected) >= limit
                    or accepted_from_document >= max_lines_per_document
                ):
                    break
            if len(selected) >= limit:
                break

        used_shards.append(
            {
                "url": shard_url,
                "expected_full_shard_md5": _expected_md5(
                    shard_url,
                    timeout=timeout,
                ),
                "documents_streamed": shard_documents,
                "full_shard_downloaded": False,
                "full_shard_checksum_verified": False,
            }
        )
        if len(selected) >= limit or documents_seen >= max_documents:
            break

    if len(selected) < limit:
        raise HpltSamplingError(
            "HPLT bounded sample ended before the requested line limit "
            f"({len(selected)}/{limit} accepted lines after "
            f"{documents_seen}/{max_documents} documents)."
        )

    corpus_path = write_corpus(selected, output_path)
    metadata = {
        "schema_version": "1",
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "HPLT Monolingual Datasets v3.0",
        "language": "lao_Laoo",
        "variant": "sorted",
        "map_url": map_url,
        "terms_url": HPLT_V3_TERMS_URL,
        "shard_order": "highest-wds-bin-first",
        "rights_note": (
            "HPLT licenses its dataset packaging under CC0 and states that it does "
            "not own the underlying extracted text; downstream users remain "
            "responsible for applicable rights and legal obligations."
        ),
        "sampling": {
            "bounded_stream": True,
            "requested_lines": limit,
            "accepted_lines": len(selected),
            "documents_seen": documents_seen,
            "segments_seen": segments_seen,
            "filter": asdict(config),
            "max_lines_per_document": max_lines_per_document,
            "max_documents": max_documents,
        },
        "shards": used_shards,
        "corpus": {
            "path": corpus_path.name,
            "sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        },
    }
    destination = Path(metadata_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return corpus_path, destination, metadata
