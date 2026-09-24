import json
from pathlib import Path

REGISTRY = Path(__file__).parents[1] / "benchmarks" / "source-registry.json"


def test_source_registry_has_reviewed_entries() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1"
    assert payload["reviewed_at"]
    assert payload["sources"]

    ids = [source["id"] for source in payload["sources"]]
    assert len(ids) == len(set(ids))

    for source in payload["sources"]:
        assert source["name"]
        assert source["url"].startswith("https://")
        assert source["license"]
        assert source["status"]
        assert isinstance(source["allowed_uses"], list)
        assert isinstance(source["disallowed_uses"], list)
        assert source["notes"]


def test_hplt_is_only_approved_as_text_input() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    hplt = next(source for source in payload["sources"] if source["id"] == "hplt-v3-lao")

    assert hplt["license"].startswith("CC0-1.0-packaging")
    assert hplt["status"] == "approved-text-only"
    assert hplt["map_url"] == (
        "https://data.hplt-project.org/three/sorted/lao_Laoo.map"
    )
    assert "underlying extracted text" in hplt["notes"]
    assert "provenance-recorded-training-text" in hplt["allowed_uses"]
    assert (
        "public-capture-pack-source-text-without-source-clearance"
        in hplt["disallowed_uses"]
    )
    assert "claiming-as-real-scan-benchmark" in hplt["disallowed_uses"]


def test_unclear_sources_are_not_approved() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    by_id = {source["id"]: source for source in payload["sources"]}

    assert by_id["khamlao-moe-derived-corpus"]["status"] == "not-approved"
    assert by_id["lao-sabaidee"]["status"] == "pending-unavailable"


def test_project_authored_corpus_is_approved_for_public_capture_source() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    source = next(
        item
        for item in payload["sources"]
        if item["id"] == "project-authored-lao-v1"
    )

    assert source["license"] == "Apache-2.0"
    assert source["status"] == "approved-public-capture-source"
    assert "public-capture-pack-source-text" in source["allowed_uses"]
    assert "public-benchmark-ground-truth-source" in source["allowed_uses"]
    assert (
        "claiming-digital-pages-as-real-optical-evidence"
        in source["disallowed_uses"]
    )


def test_remote_scan_candidates_stay_non_ingestable() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    candidates = [
        source
        for source in payload["sources"]
        if source["status"].startswith("remote-evaluation-")
    ]

    assert len(candidates) >= 12
    for source in candidates:
        assert source["allowed_uses"] == []
        assert "public-benchmark-redistribution" in source["disallowed_uses"]
        assert "training-data-ingestion" in source["disallowed_uses"]
        assert "public-ground-truth" in source["disallowed_uses"]
        pages = source["evidence"]["pages"]
        assert pages is None or pages >= 1
        assert source["evidence"]["text_layer"] in {
            "absent",
            "effectively-absent",
            "present-but-garbled",
            "scanner-watermark-only",
            "unverified",
            "image-form-source-collection",
        }
