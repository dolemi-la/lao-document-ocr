from __future__ import annotations

import hashlib
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from lao_document_ocr.normalization import normalize_lao_text
from lao_document_ocr.training_manifest import TrainingSample, deterministic_split
from lao_document_ocr.vocabulary import CharacterVocabulary

MODEL_VERSION = "crnn-ctc-v2"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ctc_required_timesteps(token_ids: list[int]) -> int:
    if not token_ids:
        return 0
    repeats = sum(1 for left, right in zip(token_ids, token_ids[1:], strict=False) if left == right)
    return len(token_ids) + repeats


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 1e-3
    dev_ratio: float = 0.1
    seed: int = 20260921
    image_height: int = 48
    max_width: int = 512
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.image_height < 16:
            raise ValueError("image_height must be at least 16")
        if self.max_width < 32:
            raise ValueError("max_width must be at least 32")

    def to_dict(self) -> dict:
        return asdict(self)


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError(
            "Recognizer training requires the optional 'train' dependencies. "
            "Install with: pip install -e '.[train]'"
        ) from exc
    return torch, nn, DataLoader, Dataset


def prepare_line_pil_image(
    image: Image.Image,
    *,
    image_height: int,
    max_width: int,
) -> tuple[np.ndarray, int]:
    image = ImageOps.exif_transpose(image).convert("L")

    if image.width < 1 or image.height < 1:
        raise ValueError("Invalid image dimensions")

    scale = image_height / image.height
    target_width = max(1, int(round(image.width * scale)))

    if target_width > max_width:
        scale = max_width / image.width
        target_width = max_width
        target_height = max(1, int(round(image.height * scale)))
    else:
        target_height = image_height

    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (target_width, image_height), 255)
    offset_y = max(0, (image_height - target_height) // 2)
    canvas.paste(image, (0, offset_y))

    array = np.asarray(canvas, dtype=np.float32)
    array = 1.0 - (array / 255.0)
    return array[None, :, :], target_width


def prepare_line_image(
    path: str | Path,
    *,
    image_height: int,
    max_width: int,
) -> tuple[np.ndarray, int]:
    with Image.open(path) as source:
        return prepare_line_pil_image(
            source,
            image_height=image_height,
            max_width=max_width,
        )


def _build_dataset_type():
    torch, _, _, Dataset = _require_torch()

    class LineDataset(Dataset):
        def __init__(
            self,
            samples: list[TrainingSample],
            vocabulary: CharacterVocabulary,
            *,
            image_height: int,
            max_width: int,
        ) -> None:
            self.samples = samples
            self.vocabulary = vocabulary
            self.image_height = image_height
            self.max_width = max_width

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> dict[str, Any]:
            sample = self.samples[index]
            array, width = prepare_line_image(
                sample.image,
                image_height=self.image_height,
                max_width=self.max_width,
            )
            target = self.vocabulary.encode(sample.text)
            available_timesteps = max(1, width // 4)
            required_timesteps = ctc_required_timesteps(target)
            if required_timesteps > available_timesteps:
                raise ValueError(
                    f"Sample {sample.id} requires {required_timesteps} CTC timesteps "
                    f"but only {available_timesteps} are available. Increase --max-width "
                    "or shorten the training line."
                )
            return {
                "id": sample.id,
                "image": torch.from_numpy(array),
                "width": width,
                "target": torch.tensor(target, dtype=torch.long),
                "text": normalize_lao_text(sample.text),
            }

    return LineDataset


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    torch, _, _, _ = _require_torch()
    height = batch[0]["image"].shape[1]
    max_width = max(int(item["width"]) for item in batch)
    images = torch.zeros((len(batch), 1, height, max_width), dtype=torch.float32)

    targets: list[Any] = []
    target_lengths: list[int] = []
    widths: list[int] = []
    texts: list[str] = []
    ids: list[str] = []

    for index, item in enumerate(batch):
        width = int(item["width"])
        images[index, :, :, :width] = item["image"]
        targets.append(item["target"])
        target_lengths.append(int(item["target"].numel()))
        widths.append(width)
        texts.append(item["text"])
        ids.append(item["id"])

    return {
        "ids": ids,
        "images": images,
        "widths": torch.tensor(widths, dtype=torch.long),
        "targets": torch.cat(targets),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "texts": texts,
    }


def _greedy_decode(log_probs, vocabulary: CharacterVocabulary) -> list[str]:
    predictions = log_probs.argmax(dim=-1).permute(1, 0)
    return [vocabulary.decode_ctc(row.tolist()) for row in predictions]


def _edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            cost = 0 if ref_char == hyp_char else 1
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def evaluate(model, loader, vocabulary: CharacterVocabulary, device) -> dict[str, float]:
    torch, _, _, _ = _require_torch()
    model.eval()
    edits = 0
    characters = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            log_probs = model(images)
            predictions = _greedy_decode(log_probs.cpu(), vocabulary)
            for reference, hypothesis in zip(batch["texts"], predictions, strict=True):
                edits += _edit_distance(reference, hypothesis)
                characters += len(reference)
    cer = edits / characters if characters else 0.0
    return {"cer": cer, "character_edits": edits, "characters": characters}


def train_recognizer(
    samples: list[TrainingSample],
    output_dir: str | Path,
    *,
    training_config: TrainingConfig | None = None,
) -> dict:
    training_config = training_config or TrainingConfig()
    torch, nn, DataLoader, _ = _require_torch()

    from lao_document_ocr.recognizer_model import LaoCrnnRecognizer, RecognizerConfig

    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)

    train_samples, dev_samples = deterministic_split(
        samples,
        dev_ratio=training_config.dev_ratio,
    )
    vocabulary = CharacterVocabulary.from_texts([sample.text for sample in samples])

    model_config = RecognizerConfig(
        image_height=training_config.image_height,
        max_width=training_config.max_width,
    )
    LineDataset = _build_dataset_type()
    train_dataset = LineDataset(
        train_samples,
        vocabulary,
        image_height=model_config.image_height,
        max_width=model_config.max_width,
    )
    dev_dataset = LineDataset(
        dev_samples,
        vocabulary,
        image_height=model_config.image_height,
        max_width=model_config.max_width,
    )

    generator = torch.Generator()
    generator.manual_seed(training_config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        collate_fn=_collate,
        generator=generator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        collate_fn=_collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LaoCrnnRecognizer(vocabulary.size, model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    history: list[dict[str, float | int]] = []
    best_cer = math.inf
    best_state = None

    for epoch in range(1, training_config.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for batch in train_loader:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            widths = batch["widths"].to(device)

            optimizer.zero_grad(set_to_none=True)
            log_probs = model(images)
            input_lengths = model.output_lengths(widths)
            input_lengths = torch.clamp(input_lengths, max=log_probs.shape[0])

            loss = loss_fn(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.item())
            batches += 1

        dev_metrics = evaluate(model, dev_loader, vocabulary, device)
        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, batches),
            "dev_cer": dev_metrics["cer"],
        }
        history.append(epoch_record)

        if dev_metrics["cer"] < best_cer:
            best_cer = dev_metrics["cer"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    vocabulary_path = vocabulary.save(output / "vocab.json")
    checkpoint_path = output / "recognizer.pt"

    checkpoint = {
        "schema_version": "1",
        "model": "LaoCrnnRecognizer",
        "model_version": MODEL_VERSION,
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "vocabulary": vocabulary.to_dict(),
        "vocabulary_checksum": vocabulary.checksum(),
        "state_dict": model.state_dict(),
        "history": history,
        "best_dev_cer": best_cer,
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = _sha256_file(checkpoint_path)

    metadata = {
        "schema_version": "1",
        "model": "LaoCrnnRecognizer",
        "model_version": MODEL_VERSION,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "vocabulary": vocabulary_path.name,
        "vocabulary_checksum": vocabulary.checksum(),
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "train_samples": len(train_samples),
        "dev_samples": len(dev_samples),
        "best_dev_cer": best_cer,
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
        "vocabulary": vocabulary_path,
        "best_dev_cer": best_cer,
    }


def export_recognizer(
    checkpoint_path: str | Path,
    output_path: str | Path,
) -> Path:
    torch, _, _, _ = _require_torch()
    from lao_document_ocr.recognizer_model import LaoCrnnRecognizer, RecognizerConfig

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = RecognizerConfig(**checkpoint["model_config"])
    vocabulary = CharacterVocabulary(tuple(checkpoint["vocabulary"]["characters"]))

    model = LaoCrnnRecognizer(vocabulary.size, model_config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    example = torch.zeros(
        (1, 1, model_config.image_height, model_config.max_width),
        dtype=torch.float32,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The tensor attributes .* were assigned during export.*",
            category=UserWarning,
        )
        exported = torch.export.export(model, (example,))

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported, str(destination))
    artifact_sha256 = _sha256_file(destination)

    metadata = {
        "schema_version": "1",
        "format": "torch-export",
        "model": "LaoCrnnRecognizer",
        "model_version": MODEL_VERSION,
        "artifact": destination.name,
        "artifact_sha256": artifact_sha256,
        "vocabulary": checkpoint["vocabulary"],
        "vocabulary_checksum": checkpoint["vocabulary_checksum"],
        "model_config": checkpoint["model_config"],
        "width_downsample_factor": model.width_downsample_factor,
    }
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
