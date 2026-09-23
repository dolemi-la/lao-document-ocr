import hashlib
import json

import pytest
from PIL import Image

from lao_document_ocr.training_manifest import deterministic_split, load_training_manifest


def _write_sample(root, sample_id: str, text: str) -> dict:
    image_path = root / f"{sample_id}.png"
    Image.new("L", (120, 32), 255).save(image_path)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    return {
        "id": sample_id,
        "image": image_path.name,
        "text": text,
        "sha256": digest,
    }


def test_load_training_manifest_and_split(tmp_path) -> None:
    entries = [_write_sample(tmp_path, f"s-{index}", f"text {index}") for index in range(20)]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )

    samples = load_training_manifest(manifest)
    train_a, dev_a = deterministic_split(samples, dev_ratio=0.2)
    train_b, dev_b = deterministic_split(samples, dev_ratio=0.2)

    assert len(samples) == 20
    assert [sample.id for sample in train_a] == [sample.id for sample in train_b]
    assert [sample.id for sample in dev_a] == [sample.id for sample in dev_b]
    assert train_a
    assert dev_a
    assert set(item.id for item in train_a).isdisjoint(item.id for item in dev_a)


def test_load_training_manifest_detects_hash_mismatch(tmp_path) -> None:
    entry = _write_sample(tmp_path, "s-1", "text")
    entry["sha256"] = "0" * 64
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_training_manifest(manifest)


def test_split_rejects_invalid_ratio(tmp_path) -> None:
    entries = [_write_sample(tmp_path, f"s-{index}", "text") for index in range(2)]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )
    samples = load_training_manifest(manifest)
    with pytest.raises(ValueError, match="dev_ratio"):
        deterministic_split(samples, dev_ratio=1.0)


def test_split_keeps_augmented_variants_of_same_text_together(tmp_path) -> None:
    entries = [
        _write_sample(tmp_path, "hello-a", "  ສະບາຍດີ   ໂລກ  "),
        _write_sample(tmp_path, "hello-b", "ສະບາຍດີ ໂລກ"),
        _write_sample(tmp_path, "thanks-a", "ຂອບໃຈ ຫຼາຍ"),
        _write_sample(tmp_path, "thanks-b", "ຂອບໃຈ ຫຼາຍ"),
        _write_sample(tmp_path, "ocr-a", "Lao OCR"),
        _write_sample(tmp_path, "ocr-b", "Lao OCR"),
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
        encoding="utf-8",
    )
    samples = load_training_manifest(manifest)

    train, dev = deterministic_split(samples, dev_ratio=0.34)

    train_texts = {" ".join(sample.text.split()) for sample in train}
    dev_texts = {" ".join(sample.text.split()) for sample in dev}
    assert train_texts.isdisjoint(dev_texts)

    placement = {}
    for split_name, split_samples in (("train", train), ("dev", dev)):
        for sample in split_samples:
            normalized = " ".join(sample.text.split())
            placement.setdefault(normalized, set()).add(split_name)
    assert all(len(sides) == 1 for sides in placement.values())


def test_split_rejects_single_unique_text_group(tmp_path) -> None:
    entries = [
        _write_sample(tmp_path, "variant-a", "same text"),
        _write_sample(tmp_path, "variant-b", "same   text"),
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )
    samples = load_training_manifest(manifest)

    with pytest.raises(ValueError, match="unique normalized text groups"):
        deterministic_split(samples, dev_ratio=0.2)
