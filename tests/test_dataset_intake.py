from __future__ import annotations

import json

import pytest
from PIL import Image

from lao_document_ocr.dataset import DatasetSplit, DatasetSubset, load_manifest
from lao_document_ocr.dataset_intake import (
    SplitRatios,
    add_dataset_sample,
    stable_document_split,
)


def _source_files(tmp_path):
    image = tmp_path / "source.png"
    truth = tmp_path / "truth.txt"
    Image.new("RGB", (100, 50), "white").save(image)
    truth.write_text("ສະບາຍດີ ໂລກ\n", encoding="utf-8")
    return image, truth


def test_stable_document_split_is_deterministic() -> None:
    first = stable_document_split("document-123")
    second = stable_document_split("document-123")
    assert first == second
    assert first in {DatasetSplit.TRAIN, DatasetSplit.DEV, DatasetSplit.TEST}


def test_split_ratios_validate_sum() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        SplitRatios(train=0.8, dev=0.2, test=0.2)


def test_add_dataset_sample_copies_and_appends_manifest(tmp_path) -> None:
    image, truth = _source_files(tmp_path)
    root = tmp_path / "public"
    manifest = root / "manifest.jsonl"

    sample = add_dataset_sample(
        dataset_root=root,
        manifest_path=manifest,
        sample_id="clean-001",
        document_id="doc-001",
        subset=DatasetSubset.CLEAN_PRINT,
        image_path=image,
        ground_truth_path=truth,
        license="CC0-1.0",
        provenance="Created for unit test",
        rights_confirmed=True,
    )

    assert (root / sample.source).is_file()
    assert (root / sample.ground_truth).read_text(encoding="utf-8").strip() == "ສະບາຍດີ ໂລກ"
    assert len(sample.sha256 or "") == 64

    loaded = load_manifest(manifest)
    assert loaded == [sample]


def test_same_document_reuses_existing_split(tmp_path) -> None:
    root = tmp_path / "public"
    manifest = root / "manifest.jsonl"

    for index in range(2):
        image = tmp_path / f"source-{index}.png"
        truth = tmp_path / f"truth-{index}.txt"
        Image.new("RGB", (100, 50), "white").save(image)
        truth.write_text(f"page {index}\n", encoding="utf-8")
        add_dataset_sample(
            dataset_root=root,
            manifest_path=manifest,
            sample_id=f"sample-{index}",
            document_id="same-doc",
            subset=DatasetSubset.CLEAN_PRINT,
            image_path=image,
            ground_truth_path=truth,
            license="CC0-1.0",
            provenance="Created for unit test",
            rights_confirmed=True,
        )

    loaded = load_manifest(manifest)
    assert loaded[0].split == loaded[1].split


def test_intake_requires_explicit_rights_confirmation(tmp_path) -> None:
    image, truth = _source_files(tmp_path)
    with pytest.raises(ValueError, match="explicit confirmation"):
        add_dataset_sample(
            dataset_root=tmp_path / "public",
            manifest_path=tmp_path / "public" / "manifest.jsonl",
            sample_id="sample-1",
            document_id="doc-1",
            subset=DatasetSubset.CLEAN_PRINT,
            image_path=image,
            ground_truth_path=truth,
            license="CC0-1.0",
            provenance="Created for unit test",
        )


def test_intake_refuses_duplicate_sample_id(tmp_path) -> None:
    image, truth = _source_files(tmp_path)
    root = tmp_path / "public"
    manifest = root / "manifest.jsonl"
    kwargs = dict(
        dataset_root=root,
        manifest_path=manifest,
        sample_id="sample-1",
        document_id="doc-1",
        subset=DatasetSubset.CLEAN_PRINT,
        image_path=image,
        ground_truth_path=truth,
        license="CC0-1.0",
        provenance="Created for unit test",
        rights_confirmed=True,
    )
    add_dataset_sample(**kwargs)
    with pytest.raises(ValueError, match="already exists"):
        add_dataset_sample(**kwargs)


def test_manifest_entry_is_reviewable_json(tmp_path) -> None:
    image, truth = _source_files(tmp_path)
    root = tmp_path / "public"
    manifest = root / "manifest.jsonl"
    add_dataset_sample(
        dataset_root=root,
        manifest_path=manifest,
        sample_id="sample-1",
        document_id="doc-1",
        subset=DatasetSubset.CLEAN_PRINT,
        image_path=image,
        ground_truth_path=truth,
        license="CC-BY-4.0",
        provenance="Contributor supplied scan",
        source_url="https://example.org/document",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        rights_confirmed=True,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["license"] == "CC-BY-4.0"
    assert payload["source_url"] == "https://example.org/document"


def _layout_file(tmp_path, *, width: int = 100, height: int = 50):
    from lao_document_ocr.layout_ground_truth import write_layout_ground_truth
    from lao_document_ocr.models import Block, BlockType, BoundingBox, Document, Page

    path = tmp_path / f"layout-{width}x{height}.json"
    document = Document(
        pages=[
            Page(
                number=1,
                width=width,
                height=height,
                blocks=[
                    Block(
                        type=BlockType.PARAGRAPH,
                        text="ສະບາຍດີ ໂລກ",
                        bbox=BoundingBox(x=5, y=5, width=80, height=20),
                    )
                ],
            )
        ]
    )
    write_layout_ground_truth(document, path)
    return path


def test_add_dataset_sample_copies_optional_layout_ground_truth(tmp_path) -> None:
    image, truth = _source_files(tmp_path)
    layout = _layout_file(tmp_path)
    root = tmp_path / "public-layout"
    manifest = root / "manifest.jsonl"

    sample = add_dataset_sample(
        dataset_root=root,
        manifest_path=manifest,
        sample_id="layout-001",
        document_id="layout-doc-001",
        subset=DatasetSubset.CLEAN_PRINT,
        image_path=image,
        ground_truth_path=truth,
        layout_ground_truth_path=layout,
        license="CC0-1.0",
        provenance="Created for layout unit test",
        rights_confirmed=True,
    )

    assert sample.layout_ground_truth == "layout-ground-truth/clean-print/layout-001.json"
    assert (root / sample.layout_ground_truth).is_file()
    loaded = load_manifest(manifest)
    assert loaded[0].layout_ground_truth == sample.layout_ground_truth


def test_layout_dimension_mismatch_is_rejected_before_dataset_mutation(tmp_path) -> None:
    image, truth = _source_files(tmp_path)
    layout = _layout_file(tmp_path, width=101, height=50)
    root = tmp_path / "public-mismatch"
    manifest = root / "manifest.jsonl"

    with pytest.raises(ValueError, match="do not match"):
        add_dataset_sample(
            dataset_root=root,
            manifest_path=manifest,
            sample_id="layout-bad",
            document_id="layout-bad-doc",
            subset=DatasetSubset.CLEAN_PRINT,
            image_path=image,
            ground_truth_path=truth,
            layout_ground_truth_path=layout,
            license="CC0-1.0",
            provenance="Created for layout unit test",
            rights_confirmed=True,
        )

    assert not manifest.exists()
    assert not (root / "data").exists()
