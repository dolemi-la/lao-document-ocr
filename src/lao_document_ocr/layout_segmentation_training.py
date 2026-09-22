from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from lao_document_ocr.layout_targets import LAYOUT_CLASS_IDS
from lao_document_ocr.recognizer_inference import resolve_torch_device

MODEL_VERSION = "tiny-layout-unet-v1"


@dataclass(frozen=True)
class LayoutTargetSample:
    id: str
    image: Path
    mask: Path
    split: str
    subset: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class LayoutTrainingConfig:
    epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 1e-3
    seed: int = 20260922
    image_height: int = 256
    image_width: int = 256
    base_channels: int = 32
    num_workers: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if self.image_height < 64 or self.image_width < 64:
            raise ValueError("layout training dimensions must be at least 64")
        if self.image_height % 4 or self.image_width % 4:
            raise ValueError("layout training dimensions must be divisible by 4")
        if self.base_channels < 8:
            raise ValueError("base_channels must be at least 8")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.device not in {"cpu", "cuda", "mps", "auto"}:
            raise ValueError("device must be one of: cpu, cuda, mps, auto")

    def to_dict(self) -> dict:
        return asdict(self)


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError(
            "Layout detector training requires the optional 'train' dependencies. "
            "Install with: pip install -e '.[train]'"
        ) from exc
    return torch, nn, DataLoader, Dataset


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str, field: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"layout target {field} must be a safe relative path")
    return path


def load_layout_target_samples(
    manifest_path: str | Path,
    *,
    dataset_root: str | Path,
) -> list[LayoutTargetSample]:
    manifest = Path(manifest_path)
    target_root = manifest.parent
    dataset = Path(dataset_root)

    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Could not read layout targets manifest: {exc}") from exc

    samples: list[LayoutTargetSample] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            payload = json.loads(stripped)
            sample_id = str(payload["id"])
            image_relative = _safe_relative_path(
                str(payload["image"]),
                "image",
            )
            mask_relative = _safe_relative_path(
                str(payload["mask"]),
                "mask",
            )
            split = str(payload["split"])
            subset = str(payload["subset"])
            tags = tuple(str(tag) for tag in payload.get("tags", []))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid layout target entry at line {line_number}: {exc}"
            ) from exc

        if split not in {"train", "dev", "test"}:
            raise ValueError(
                f"Invalid layout target split at line {line_number}: {split}"
            )
        if not sample_id:
            raise ValueError(
                f"Invalid layout target id at line {line_number}"
            )
        if sample_id in seen:
            raise ValueError(f"Duplicate layout target sample id: {sample_id}")
        seen.add(sample_id)

        image = dataset / image_relative
        mask = target_root / mask_relative
        if not image.is_file():
            raise FileNotFoundError(f"Layout target image not found: {image}")
        if not mask.is_file():
            raise FileNotFoundError(f"Layout target mask not found: {mask}")

        samples.append(
            LayoutTargetSample(
                id=sample_id,
                image=image,
                mask=mask,
                split=split,
                subset=subset,
                tags=tags,
            )
        )

    if not samples:
        raise ValueError("Layout targets manifest contains no samples")
    return samples


def prepare_layout_image_mask(
    image_path: str | Path,
    mask_path: str | Path,
    *,
    image_height: int,
    image_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    with Image.open(mask_path) as source_mask:
        mask = source_mask.convert("L")

    if image.size != mask.size:
        raise ValueError(
            "Layout image and mask dimensions must match "
            f"({image.size} != {mask.size})"
        )

    scale = min(
        image_width / image.width,
        image_height / image.height,
    )
    resized_width = max(1, round(image.width * scale))
    resized_height = max(1, round(image.height * scale))

    image = image.resize(
        (resized_width, resized_height),
        Image.Resampling.BILINEAR,
    )
    mask = mask.resize(
        (resized_width, resized_height),
        Image.Resampling.NEAREST,
    )

    image_canvas = Image.new(
        "RGB",
        (image_width, image_height),
        "white",
    )
    mask_canvas = Image.new(
        "L",
        (image_width, image_height),
        0,
    )
    offset_x = (image_width - resized_width) // 2
    offset_y = (image_height - resized_height) // 2
    image_canvas.paste(image, (offset_x, offset_y))
    mask_canvas.paste(mask, (offset_x, offset_y))

    image_array = np.asarray(
        image_canvas,
        dtype=np.float32,
    ) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))
    mask_array = np.asarray(mask_canvas, dtype=np.int64)

    max_class = max(LAYOUT_CLASS_IDS.values())
    if mask_array.size and int(mask_array.max()) > max_class:
        raise ValueError(
            f"Layout mask contains class id above {max_class}"
        )
    return image_array, mask_array




def compute_layout_class_weights(
    samples: list[LayoutTargetSample],
    *,
    num_classes: int,
    max_weight: float = 8.0,
) -> np.ndarray:
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if max_weight <= 0:
        raise ValueError("max_weight must be greater than 0")

    counts = np.zeros(num_classes, dtype=np.float64)
    for sample in samples:
        with Image.open(sample.mask) as source:
            mask = np.asarray(source.convert("L"), dtype=np.int64)
        if mask.size and int(mask.max()) >= num_classes:
            raise ValueError(
                f"Layout mask for {sample.id} contains an unknown class id"
            )
        counts += np.bincount(
            mask.reshape(-1),
            minlength=num_classes,
        )[:num_classes]

    present = counts > 0
    if not np.any(present):
        raise ValueError("Layout training masks contain no labeled pixels")

    weights = np.zeros(num_classes, dtype=np.float64)
    present_count = int(np.count_nonzero(present))
    total = float(counts[present].sum())
    weights[present] = total / (present_count * counts[present])

    mean_present = float(weights[present].mean())
    if mean_present > 0:
        weights[present] /= mean_present
    weights[present] = np.clip(weights[present], 0.1, max_weight)
    return weights.astype(np.float32)


def _build_dataset_type():
    torch, _, _, Dataset = _require_torch()

    class LayoutDataset(Dataset):
        def __init__(
            self,
            samples: list[LayoutTargetSample],
            *,
            image_height: int,
            image_width: int,
        ) -> None:
            self.samples = samples
            self.image_height = image_height
            self.image_width = image_width

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> dict[str, Any]:
            sample = self.samples[index]
            image, mask = prepare_layout_image_mask(
                sample.image,
                sample.mask,
                image_height=self.image_height,
                image_width=self.image_width,
            )
            return {
                "id": sample.id,
                "image": torch.from_numpy(image),
                "mask": torch.from_numpy(mask),
            }

    return LayoutDataset


def _confusion_matrix(
    predictions,
    targets,
    *,
    num_classes: int,
):
    torch, _, _, _ = _require_torch()
    valid = (
        (targets >= 0)
        & (targets < num_classes)
    )
    indices = (
        targets[valid] * num_classes
        + predictions[valid]
    )
    return torch.bincount(
        indices,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def metrics_from_confusion(
    confusion,
) -> dict:
    torch, _, _, _ = _require_torch()
    matrix = confusion.to(torch.float64)
    true_positive = torch.diag(matrix)
    reference = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    union = reference + predicted - true_positive

    per_class: dict[str, float | None] = {}
    class_names = {
        class_id: name
        for name, class_id in LAYOUT_CLASS_IDS.items()
    }
    valid_ious: list[float] = []
    foreground_ious: list[float] = []

    for class_id in range(matrix.shape[0]):
        name = class_names.get(class_id, str(class_id))
        if union[class_id] <= 0:
            per_class[name] = None
            continue
        iou = float((true_positive[class_id] / union[class_id]).item())
        per_class[name] = iou
        valid_ious.append(iou)
        if class_id != LAYOUT_CLASS_IDS["background"]:
            foreground_ious.append(iou)

    total = matrix.sum()
    pixel_accuracy = (
        float((true_positive.sum() / total).item())
        if total > 0
        else 0.0
    )
    mean_iou = (
        sum(valid_ious) / len(valid_ious)
        if valid_ious
        else 0.0
    )
    foreground_mean_iou = (
        sum(foreground_ious) / len(foreground_ious)
        if foreground_ious
        else 0.0
    )

    return {
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": mean_iou,
        "foreground_mean_iou": foreground_mean_iou,
        "per_class_iou": per_class,
    }


def evaluate_layout_model(
    model,
    loader,
    device,
    *,
    num_classes: int,
) -> dict:
    torch, _, _, _ = _require_torch()
    model.eval()
    confusion = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
    )

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            masks = batch["masks"].to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            confusion += _confusion_matrix(
                predictions.detach().cpu(),
                masks.detach().cpu(),
                num_classes=num_classes,
            )

    return metrics_from_confusion(confusion)


def _collate_layout(batch: list[dict[str, Any]]) -> dict[str, Any]:
    torch, _, _, _ = _require_torch()
    return {
        "ids": [item["id"] for item in batch],
        "images": torch.stack(
            [item["image"] for item in batch]
        ),
        "masks": torch.stack(
            [item["mask"] for item in batch]
        ),
    }


def train_layout_detector(
    samples: list[LayoutTargetSample],
    output_dir: str | Path,
    *,
    training_config: LayoutTrainingConfig | None = None,
) -> dict:
    config = training_config or LayoutTrainingConfig()
    torch, nn, DataLoader, _ = _require_torch()

    from lao_document_ocr.layout_model import (
        LayoutSegmentationConfig,
        TinyLayoutUNet,
    )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    train_samples = [
        sample
        for sample in samples
        if sample.split == "train"
    ]
    dev_samples = [
        sample
        for sample in samples
        if sample.split == "dev"
    ]
    if not train_samples:
        raise ValueError(
            "Layout detector training requires at least one train sample"
        )
    if not dev_samples:
        raise ValueError(
            "Layout detector training requires at least one dev sample"
        )

    model_config = LayoutSegmentationConfig(
        image_height=config.image_height,
        image_width=config.image_width,
        base_channels=config.base_channels,
        num_classes=len(LAYOUT_CLASS_IDS),
    )
    LayoutDataset = _build_dataset_type()
    train_dataset = LayoutDataset(
        train_samples,
        image_height=config.image_height,
        image_width=config.image_width,
    )
    dev_dataset = LayoutDataset(
        dev_samples,
        image_height=config.image_height,
        image_width=config.image_width,
    )

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=_collate_layout,
        generator=generator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=_collate_layout,
    )

    device = resolve_torch_device(config.device)
    model = TinyLayoutUNet(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )
    class_weights_array = compute_layout_class_weights(
        train_samples,
        num_classes=model_config.num_classes,
    )
    class_weights = torch.from_numpy(
        class_weights_array
    ).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    history: list[dict] = []
    best_foreground_miou = -1.0
    best_state = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for batch in train_loader:
            images = batch["images"].to(device)
            masks = batch["masks"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            batches += 1

        metrics = evaluate_layout_model(
            model,
            dev_loader,
            device,
            num_classes=model_config.num_classes,
        )
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, batches),
            **metrics,
        }
        history.append(record)

        score = metrics["foreground_mean_iou"]
        if score > best_foreground_miou:
            best_foreground_miou = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "layout-detector.pt"

    checkpoint = {
        "schema_version": "1",
        "model": "TinyLayoutUNet",
        "model_version": MODEL_VERSION,
        "model_config": model_config.to_dict(),
        "training_config": config.to_dict(),
        "class_ids": LAYOUT_CLASS_IDS,
        "class_weights": class_weights_array.tolist(),
        "state_dict": model.state_dict(),
        "history": history,
        "best_dev_foreground_mean_iou": best_foreground_miou,
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = _sha256_file(checkpoint_path)

    metadata = {
        "schema_version": "1",
        "model": "TinyLayoutUNet",
        "model_version": MODEL_VERSION,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "class_ids": LAYOUT_CLASS_IDS,
        "class_weights": class_weights_array.tolist(),
        "model_config": model_config.to_dict(),
        "training_config": config.to_dict(),
        "train_samples": len(train_samples),
        "dev_samples": len(dev_samples),
        "best_dev_foreground_mean_iou": best_foreground_miou,
        "device": str(device),
        "history": history,
        "train_subsets": dict(
            sorted(
                Counter(
                    sample.subset
                    for sample in train_samples
                ).items()
            )
        ),
        "dev_subsets": dict(
            sorted(
                Counter(
                    sample.subset
                    for sample in dev_samples
                ).items()
            )
        ),
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "checkpoint": checkpoint_path,
        "metadata": metadata_path,
        "best_dev_foreground_mean_iou": best_foreground_miou,
    }


def export_layout_detector(
    checkpoint_path: str | Path,
    output_path: str | Path,
) -> Path:
    torch, _, _, _ = _require_torch()
    from lao_document_ocr.layout_model import (
        LayoutSegmentationConfig,
        TinyLayoutUNet,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model_config = LayoutSegmentationConfig(
        **checkpoint["model_config"]
    )
    model = TinyLayoutUNet(model_config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    example = torch.zeros(
        (
            1,
            3,
            model_config.image_height,
            model_config.image_width,
        ),
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
        "model": "TinyLayoutUNet",
        "model_version": MODEL_VERSION,
        "artifact": destination.name,
        "artifact_sha256": artifact_sha256,
        "model_config": checkpoint["model_config"],
        "class_ids": checkpoint["class_ids"],
    }
    destination.with_suffix(
        destination.suffix + ".json"
    ).write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination
