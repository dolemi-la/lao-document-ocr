from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lao_document_ocr.layout_ground_truth import load_layout_ground_truth
from lao_document_ocr.layout_training_manifest import (
    LayoutTrainingEntry,
    load_layout_training_manifest,
)
from lao_document_ocr.reading_order_features import (
    FEATURE_VERSION,
    PAIR_FEATURE_DIM,
    pair_feature_vector,
)
from lao_document_ocr.recognizer_inference import resolve_torch_device

MODEL_VERSION = "pairwise-reading-order-v1"


@dataclass(frozen=True)
class ReadingOrderTrainingConfig:
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    seed: int = 20260923
    hidden_size: int = 64
    num_workers: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if self.hidden_size < 8:
            raise ValueError("hidden_size must be at least 8")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.device not in {"cpu", "cuda", "mps", "auto"}:
            raise ValueError("device must be one of: cpu, cuda, mps, auto")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PairExample:
    sample_id: str
    features: np.ndarray
    label: float


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError(
            "Reading-order training requires the optional 'train' dependencies. "
            "Install with: pip install -e '.[train]'"
        ) from exc
    return torch, nn, DataLoader, Dataset


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _page_for_entry(
    entry: LayoutTrainingEntry,
    *,
    dataset_root: Path,
):
    document = load_layout_ground_truth(
        dataset_root / entry.layout_ground_truth
    )
    if len(document.pages) != 1:
        raise ValueError(
            f"Reading-order sample {entry.id} must contain exactly one page"
        )
    page = document.pages[0]
    if page.width != entry.width or page.height != entry.height:
        raise ValueError(
            f"Reading-order sample {entry.id} dimensions do not match manifest"
        )
    return page


def build_pair_examples(
    entries: list[LayoutTrainingEntry],
    *,
    dataset_root: str | Path,
) -> list[PairExample]:
    root = Path(dataset_root)
    examples: list[PairExample] = []

    for entry in entries:
        page = _page_for_entry(entry, dataset_root=root)
        positioned = [
            block
            for block in page.blocks
            if block.bbox is not None
        ]
        for first_index in range(len(positioned)):
            for second_index in range(first_index + 1, len(positioned)):
                first = positioned[first_index]
                second = positioned[second_index]
                examples.append(
                    PairExample(
                        sample_id=entry.id,
                        features=pair_feature_vector(
                            first,
                            second,
                            page_width=page.width,
                            page_height=page.height,
                        ),
                        label=1.0,
                    )
                )
                examples.append(
                    PairExample(
                        sample_id=entry.id,
                        features=pair_feature_vector(
                            second,
                            first,
                            page_width=page.width,
                            page_height=page.height,
                        ),
                        label=0.0,
                    )
                )

    if not examples:
        raise ValueError(
            "Reading-order training requires pages with at least two positioned blocks"
        )
    return examples


def _build_dataset_type():
    torch, _, _, Dataset = _require_torch()

    class PairDataset(Dataset):
        def __init__(self, examples: list[PairExample]) -> None:
            self.examples = examples

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> dict[str, Any]:
            example = self.examples[index]
            return {
                "features": torch.from_numpy(example.features),
                "label": torch.tensor(
                    example.label,
                    dtype=torch.float32,
                ),
            }

    return PairDataset


def evaluate_pair_model(model, loader, device) -> dict[str, float]:
    torch, _, _, _ = _require_torch()
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            logits = model(features)
            loss_sum += float(loss_fn(logits, labels).item())
            predictions = (logits >= 0).to(labels.dtype)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())

    return {
        "pair_accuracy": correct / total if total else 0.0,
        "loss": loss_sum / total if total else 0.0,
        "pairs": float(total),
    }


def train_reading_order_model(
    training_manifest: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    training_config: ReadingOrderTrainingConfig | None = None,
) -> dict:
    config = training_config or ReadingOrderTrainingConfig()
    torch, nn, DataLoader, _ = _require_torch()

    from lao_document_ocr.reading_order_model import (
        PairwiseReadingOrderMLP,
        ReadingOrderModelConfig,
    )

    entries = load_layout_training_manifest(training_manifest)
    train_entries = [entry for entry in entries if entry.split == "train"]
    dev_entries = [entry for entry in entries if entry.split == "dev"]
    if not train_entries:
        raise ValueError(
            "Reading-order training requires at least one train sample"
        )
    if not dev_entries:
        raise ValueError(
            "Reading-order training requires at least one dev sample"
        )

    train_examples = build_pair_examples(
        train_entries,
        dataset_root=dataset_root,
    )
    dev_examples = build_pair_examples(
        dev_entries,
        dataset_root=dataset_root,
    )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    model_config = ReadingOrderModelConfig(
        hidden_size=config.hidden_size,
    )
    PairDataset = _build_dataset_type()
    train_dataset = PairDataset(train_examples)
    dev_dataset = PairDataset(dev_examples)

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    device = resolve_torch_device(config.device)
    model = PairwiseReadingOrderMLP(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    history: list[dict[str, float | int]] = []
    best_accuracy = -1.0
    best_state = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        pairs = 0

        for batch in train_loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )
            optimizer.step()

            batch_size = int(labels.numel())
            total_loss += float(loss.item()) * batch_size
            pairs += batch_size

        dev_metrics = evaluate_pair_model(
            model,
            dev_loader,
            device,
        )
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, pairs),
            "dev_pair_accuracy": dev_metrics["pair_accuracy"],
            "dev_loss": dev_metrics["loss"],
        }
        history.append(record)

        if dev_metrics["pair_accuracy"] > best_accuracy:
            best_accuracy = dev_metrics["pair_accuracy"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "reading-order.pt"
    checkpoint = {
        "schema_version": "1",
        "model": "PairwiseReadingOrderMLP",
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "model_config": model_config.to_dict(),
        "training_config": config.to_dict(),
        "state_dict": model.state_dict(),
        "history": history,
        "best_dev_pair_accuracy": best_accuracy,
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = _sha256_file(checkpoint_path)

    metadata = {
        "schema_version": "1",
        "model": "PairwiseReadingOrderMLP",
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "model_config": model_config.to_dict(),
        "training_config": config.to_dict(),
        "train_pages": len(train_entries),
        "dev_pages": len(dev_entries),
        "train_pairs": len(train_examples),
        "dev_pairs": len(dev_examples),
        "best_dev_pair_accuracy": best_accuracy,
        "device": str(device),
        "history": history,
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "checkpoint": checkpoint_path,
        "metadata": metadata_path,
        "best_dev_pair_accuracy": best_accuracy,
    }


def export_reading_order_model(
    checkpoint_path: str | Path,
    output_path: str | Path,
) -> Path:
    torch, _, _, _ = _require_torch()
    from lao_document_ocr.reading_order_model import (
        PairwiseReadingOrderMLP,
        ReadingOrderModelConfig,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Reading-order checkpoint feature version mismatch")

    model_config = ReadingOrderModelConfig(
        **checkpoint["model_config"],
    )
    model = PairwiseReadingOrderMLP(model_config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    inference_batch_size = 512
    example = torch.zeros(
        (inference_batch_size, PAIR_FEATURE_DIM),
        dtype=torch.float32,
    )
    exported = torch.export.export(model, (example,))

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported, str(destination))
    artifact_sha256 = _sha256_file(destination)

    metadata = {
        "schema_version": "1",
        "format": "torch-export",
        "model": "PairwiseReadingOrderMLP",
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "artifact": destination.name,
        "artifact_sha256": artifact_sha256,
        "model_config": checkpoint["model_config"],
        "inference_batch_size": inference_batch_size,
    }
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
