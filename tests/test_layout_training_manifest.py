import hashlib
import json

from PIL import Image

from lao_document_ocr.dataset import load_manifest
from lao_document_ocr.layout_ground_truth import write_layout_ground_truth
from lao_document_ocr.layout_training_manifest import (
    build_layout_training_entries,
    load_layout_training_manifest,
    summarize_layout_training_entries,
    write_layout_training_manifest,
)
from lao_document_ocr.models import (
    Block,
    BlockType,
    BoundingBox,
    Document,
    Page,
)


def _add_sample(
    tmp_path,
    *,
    sample_id: str,
    split: str,
    with_layout: bool,
    block_type: BlockType = BlockType.PARAGRAPH,
):
    image = tmp_path / f"{sample_id}.png"
    truth = tmp_path / f"{sample_id}.txt"
    shade = 220 + (ord(sample_id[0]) % 20)
    Image.new("RGB", (120, 80), (shade, 255, 255)).save(image)
    truth.write_text(sample_id, encoding="utf-8")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()

    entry = {
        "id": sample_id,
        "document_id": f"doc-{sample_id}",
        "split": split,
        "subset": "clean-print",
        "source": image.name,
        "ground_truth": truth.name,
        "language": "lo",
        "license": "CC0-1.0",
        "provenance": "layout training unit test",
        "sha256": digest,
        "tags": ["layout:plain"],
    }

    if with_layout:
        layout = tmp_path / f"{sample_id}.json"
        write_layout_ground_truth(
            Document(
                pages=[
                    Page(
                        number=1,
                        width=120,
                        height=80,
                        blocks=[
                            Block(
                                type=block_type,
                                text=sample_id,
                                bbox=BoundingBox(
                                    x=10,
                                    y=10,
                                    width=80,
                                    height=25,
                                ),
                            )
                        ],
                    )
                ]
            ),
            layout,
        )
        entry["layout_ground_truth"] = layout.name

    return entry


def _manifest(tmp_path):
    entries = [
        _add_sample(
            tmp_path,
            sample_id="a",
            split="train",
            with_layout=True,
            block_type=BlockType.HEADING,
        ),
        _add_sample(
            tmp_path,
            sample_id="b",
            split="dev",
            with_layout=True,
        ),
        _add_sample(
            tmp_path,
            sample_id="c",
            split="test",
            with_layout=False,
        ),
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )
    return manifest


def test_build_layout_training_manifest_selects_only_labeled_samples(tmp_path) -> None:
    samples = load_manifest(_manifest(tmp_path))

    entries = build_layout_training_entries(samples, tmp_path)

    assert [entry.id for entry in entries] == ["a", "b"]
    assert entries[0].width == 120
    assert entries[0].height == 80
    assert entries[0].block_counts == {"heading": 1}
    assert entries[1].block_counts == {"paragraph": 1}


def test_layout_training_manifest_can_filter_splits(tmp_path) -> None:
    from lao_document_ocr.dataset import DatasetSplit

    samples = load_manifest(_manifest(tmp_path))
    entries = build_layout_training_entries(
        samples,
        tmp_path,
        splits={DatasetSplit.DEV},
    )

    assert [entry.id for entry in entries] == ["b"]


def test_layout_training_manifest_writes_jsonl_and_summary(tmp_path) -> None:
    samples = load_manifest(_manifest(tmp_path))
    entries = build_layout_training_entries(samples, tmp_path)
    output = write_layout_training_manifest(
        entries,
        tmp_path / "layout-training.jsonl",
    )

    payloads = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [payload["id"] for payload in payloads] == ["a", "b"]

    summary = summarize_layout_training_entries(entries)
    assert summary["samples"] == 2
    assert summary["by_split"] == {"dev": 1, "train": 1}
    assert summary["by_block_type"] == {"heading": 1, "paragraph": 1}
    assert summary["by_tag"] == {"layout:plain": 2}


def test_layout_training_manifest_loader_round_trip(tmp_path) -> None:
    samples = load_manifest(_manifest(tmp_path))
    entries = build_layout_training_entries(samples, tmp_path)
    output = write_layout_training_manifest(
        entries,
        tmp_path / "loader.jsonl",
    )

    loaded = load_layout_training_manifest(output)

    assert loaded == entries


def test_layout_training_manifest_loader_rejects_unsafe_paths(tmp_path) -> None:
    path = tmp_path / "unsafe.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "unsafe",
                "document_id": "unsafe-doc",
                "image": "../escape.png",
                "layout_ground_truth": "layout.json",
                "split": "train",
                "subset": "clean-print",
                "tags": [],
                "sha256": None,
                "width": 100,
                "height": 50,
                "block_counts": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        load_layout_training_manifest(path)
    except ValueError as exc:
        assert "safe relative path" in str(exc)
    else:
        raise AssertionError("Expected unsafe layout training path to fail")
