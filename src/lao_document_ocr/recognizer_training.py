from __future__ import annotations

import hashlib
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from lao_document_ocr.normalization import normalize_lao_text
from lao_document_ocr.training_manifest import (
    SPLIT_STRATEGY,
    TrainingSample,
    deterministic_split,
)
from lao_document_ocr.vocabulary import CharacterVocabulary

MODEL_VERSION = "crnn-ctc-v2"
UNIDIRECTIONAL_MODEL_VERSION = "crnn-ctc-v3"
DEV_EVALUATION_VERSION = "normalized-valid-timestep-v2"
TRAINING_PADDING_STRATEGY = "fixed-max-width-v1"
PREDICTION_DIAGNOSTICS_VERSION = "valid-timestep-greedy-v1"


def _model_version_for_config(model_config) -> str:
    return (
        MODEL_VERSION
        if model_config.bidirectional
        else UNIDIRECTIONAL_MODEL_VERSION
    )


def _normalized_resume_model_config(
    payload: Any,
    *,
    model_version: str,
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    normalized = dict(payload)
    if model_version == MODEL_VERSION and "bidirectional" not in normalized:
        normalized["bidirectional"] = True
    return normalized


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_samples_checksum(samples: list[TrainingSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        image_sha256 = sample.sha256 or _sha256_file(sample.image)
        payload = json.dumps(
            {
                "id": sample.id,
                "text": normalize_lao_text(sample.text),
                "image_sha256": image_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _cpu_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
    }


def _to_cpu(value):
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _atomic_torch_save(torch, payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _capture_rng_state(torch, device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif (
        device.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
    ):
        state["mps"] = torch.mps.get_rng_state()
    return _to_cpu(state)


def _restore_rng_state(torch, device, state: dict[str, Any] | None) -> bool:
    if not isinstance(state, dict):
        return False

    required = {"torch", "python", "numpy"}
    if device.type == "cuda":
        required.add("cuda")
    elif device.type == "mps":
        required.add("mps")
    if not required.issubset(state):
        return False

    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    elif (
        device.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(state["mps"])
    return True


@dataclass(frozen=True)
class LineResizePlan:
    width: int
    height: int
    offset_y: int
    width_capped: bool


def plan_line_resize(
    source_width: int,
    source_height: int,
    *,
    image_height: int,
    max_width: int,
) -> LineResizePlan:
    """Describe the existing aspect-preserving resize, not glyph/ink quality."""
    dimensions = (source_width, source_height, image_height, max_width)
    if any(type(value) is not int or value < 1 for value in dimensions):
        raise ValueError("Line image dimensions must be positive integers")
    target_width = max(1, int(round(source_width * (image_height / source_height))))
    width_capped = target_width > max_width
    if width_capped:
        target_width = max_width
        target_height = max(1, int(round(source_height * (max_width / source_width))))
    else:
        target_height = image_height
    return LineResizePlan(
        width=target_width,
        height=target_height,
        offset_y=max(0, (image_height - target_height) // 2),
        width_capped=width_capped,
    )


def _prepared_line_geometry(
    path: str | Path,
    *,
    image_height: int,
    max_width: int,
) -> LineResizePlan:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        return plan_line_resize(
            image.width, image.height, image_height=image_height, max_width=max_width,
        )


def _prepared_line_width(
    path: str | Path,
    *,
    image_height: int,
    max_width: int,
) -> int:
    return _prepared_line_geometry(path, image_height=image_height, max_width=max_width).width


def preflight_ctc_capacity(
    samples: list[TrainingSample],
    vocabulary: CharacterVocabulary,
    *,
    image_height: int,
    max_width: int,
) -> dict[str, int]:
    if not samples:
        raise ValueError("Training manifest contains no samples")

    min_margin: int | None = None
    max_required = 0
    max_available = 0
    first_failure: tuple[str, int, int] | None = None
    failures = 0
    width_capped_samples = 0
    resized_heights: list[int] = []
    resized_widths: list[int] = []

    for sample in samples:
        target = vocabulary.encode(sample.text)
        required = ctc_required_timesteps(target)
        geometry = _prepared_line_geometry(
            sample.image,
            image_height=image_height,
            max_width=max_width,
        )
        width_capped_samples += int(geometry.width_capped)
        resized_heights.append(geometry.height)
        resized_widths.append(geometry.width)
        available = max(1, geometry.width // 4)
        margin = available - required
        min_margin = margin if min_margin is None else min(min_margin, margin)
        max_required = max(max_required, required)
        max_available = max(max_available, available)
        if margin < 0:
            failures += 1
            if first_failure is None:
                first_failure = (sample.id, required, available)

    if first_failure is not None:
        sample_id, required, available = first_failure
        raise ValueError(
            f"CTC capacity preflight failed for {failures} sample(s); "
            f"first failure {sample_id} requires {required} timesteps "
            f"but only {available} are available. Increase --image-height for "
            "uncapped lines or --max-width for width-capped lines, or regenerate "
            "shorter source lines with matching labels."
        )

    return {
        "samples": len(samples),
        "failures": failures,
        "min_timestep_margin": int(min_margin or 0),
        "max_required_timesteps": max_required,
        "max_available_timesteps": max_available,
        "input_image_height": image_height,
        "input_max_width": max_width,
        "width_capped_samples": width_capped_samples,
        "min_resized_height": min(resized_heights),
        "max_resized_height": max(resized_heights),
        "min_resized_width": min(resized_widths),
        "max_resized_width": max(resized_widths),
    }


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
    max_width: int = 768
    num_workers: int = 0
    device: str = "auto"
    bidirectional: bool = True

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

    geometry = plan_line_resize(
        image.width, image.height, image_height=image_height, max_width=max_width,
    )
    image = image.resize((geometry.width, geometry.height), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (geometry.width, image_height), 255)
    canvas.paste(image, (0, geometry.offset_y))

    array = np.asarray(canvas, dtype=np.float32)
    array = 1.0 - (array / 255.0)
    return array[None, :, :], geometry.width


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


def _collate(
    batch: list[dict[str, Any]],
    *,
    padded_width: int | None = None,
) -> dict[str, Any]:
    torch, _, _, _ = _require_torch()
    height = batch[0]["image"].shape[1]
    batch_max_width = max(int(item["width"]) for item in batch)
    output_width = batch_max_width if padded_width is None else int(padded_width)
    if output_width < batch_max_width:
        raise ValueError("padded_width cannot be smaller than a sample width")
    images = torch.zeros((len(batch), 1, height, output_width), dtype=torch.float32)

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


def _greedy_decode(
    log_probs,
    vocabulary: CharacterVocabulary,
    *,
    input_lengths=None,
) -> list[str]:
    predictions = log_probs.argmax(dim=-1).permute(1, 0)
    if input_lengths is None:
        return [vocabulary.decode_ctc(row.tolist()) for row in predictions]

    lengths = [int(value) for value in input_lengths.tolist()]
    if len(lengths) != len(predictions):
        raise ValueError("input_lengths must match the prediction batch size")
    return [
        vocabulary.decode_ctc(row[:length].tolist())
        for row, length in zip(predictions, lengths, strict=True)
    ]


def _edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            cost = 0 if ref_char == hyp_char else 1
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def evaluate(model, loader, vocabulary: CharacterVocabulary, device) -> dict[str, Any]:
    torch, _, _, _ = _require_torch()
    model.eval()
    edits = 0
    characters = 0
    raw_reference_characters = 0
    samples = 0
    empty_predictions = 0
    predicted_characters = 0
    blank_timesteps = 0
    valid_timesteps = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            log_probs = model(images)
            input_lengths = model.output_lengths(batch["widths"])
            input_lengths = input_lengths.clamp(max=log_probs.shape[0])
            cpu_log_probs = log_probs.cpu()
            predictions = _greedy_decode(
                cpu_log_probs,
                vocabulary,
                input_lengths=input_lengths,
            )
            paths = cpu_log_probs.argmax(dim=-1)
            for index, length in enumerate(input_lengths.tolist()):
                valid_timesteps += length
                blank_timesteps += int((paths[:length, index] == 0).sum().item())
            for reference, hypothesis in zip(batch["texts"], predictions, strict=True):
                # Match benchmark_recognizer's NFC/whitespace policy for model selection.
                normalized_reference = normalize_lao_text(reference)
                normalized_hypothesis = normalize_lao_text(hypothesis)
                edits += _edit_distance(normalized_reference, normalized_hypothesis)
                characters += len(normalized_reference)
                raw_reference_characters += len(reference)
                samples += 1
                empty_predictions += int(not hypothesis)
                predicted_characters += len(hypothesis)
    cer = edits / characters if characters else (0.0 if edits == 0 else 1.0)
    # Counts are aggregated across batches; padded tails never contribute.
    # Keep labels, hypotheses, and sample identifiers out of this telemetry.
    diagnostics = {
        "version": PREDICTION_DIAGNOSTICS_VERSION,
        "samples": samples,
        "empty_predictions": empty_predictions,
        "empty_prediction_ratio": empty_predictions / samples if samples else None,
        "predicted_characters": predicted_characters,
        "reference_characters": raw_reference_characters,
        "predicted_to_reference_character_ratio": (
            predicted_characters / raw_reference_characters if raw_reference_characters else None
        ),
        "blank_timesteps": blank_timesteps,
        "valid_timesteps": valid_timesteps,
        "blank_timestep_ratio": blank_timesteps / valid_timesteps if valid_timesteps else None,
    }
    return {
        "cer": cer,
        "character_edits": edits,
        "characters": characters,
        "prediction_diagnostics": diagnostics,
    }


def _initialize_model_weights(model, source, *, model_config, model_version, vocabulary) -> dict:
    """Load a selected inference checkpoint, never its optimizer/history or latest state."""
    torch, _, _, _ = _require_torch()
    path = Path(source)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        raise ValueError("Initialization requires a weights-only inference checkpoint") from None
    if (not isinstance(checkpoint, dict)
            or checkpoint.get("artifact_type") == "recognizer-training-state"
            or "latest_state_dict" in checkpoint):
        raise ValueError("Initialization requires selected weights, not a training state")
    if checkpoint.get("model") != "LaoCrnnRecognizer":
        raise ValueError("Initialization model mismatch")
    if checkpoint.get("model_version") != model_version:
        raise ValueError("Initialization model version mismatch")
    config = _normalized_resume_model_config(
        checkpoint.get("model_config"), model_version=model_version,
    )
    if config != model_config.to_dict():
        raise ValueError("Initialization model configuration mismatch")
    if (checkpoint.get("vocabulary_checksum") != vocabulary.checksum()
            or checkpoint.get("vocabulary") != vocabulary.to_dict()):
        raise ValueError("Initialization vocabulary mismatch")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Initialization checkpoint has no selected weights")
    if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all().item()
           for value in state.values()):
        raise ValueError("Initialization weights must be finite tensors")
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, ValueError):
        raise ValueError("Initialization weight shapes or keys do not match") from None
    return {
        "mode": "weights-only-new-experiment",
        "source": str(path), "source_sha256": _sha256_file(path),
        "source_best_epoch": checkpoint.get("best_epoch"),
        "optimizer_restored": False, "history_restored": False,
        "rng_restored": False, "vocabulary_verified": True,
    }


def train_recognizer(
    samples: list[TrainingSample],
    output_dir: str | Path,
    *,
    training_config: TrainingConfig | None = None,
    resume_from: str | Path | None = None,
    initialize_from: str | Path | None = None,
    recompute_resume_metrics: bool = False,
) -> dict:
    if initialize_from is not None and resume_from is not None:
        raise ValueError("--initialize-from and --resume-from are mutually exclusive")
    if initialize_from is not None and Path(output_dir).exists():
        target = Path(output_dir)
        if not target.is_dir() or any(target.iterdir()):
            raise ValueError("Initialization requires a new or empty output directory")
    if recompute_resume_metrics and resume_from is None:
        raise ValueError("--recompute-resume-metrics requires --resume-from")
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
    training_samples_checksum = _training_samples_checksum(samples)
    ctc_preflight = preflight_ctc_capacity(
        samples,
        vocabulary,
        image_height=training_config.image_height,
        max_width=training_config.max_width,
    )

    model_config = RecognizerConfig(
        image_height=training_config.image_height,
        max_width=training_config.max_width,
        bidirectional=training_config.bidirectional,
    )
    model_version = _model_version_for_config(model_config)
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

    fixed_width_collate = partial(_collate, padded_width=model_config.max_width)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        collate_fn=fixed_width_collate,
        generator=generator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        collate_fn=fixed_width_collate,
    )

    from lao_document_ocr.recognizer_inference import resolve_torch_device

    device = resolve_torch_device(training_config.device)
    model = LaoCrnnRecognizer(vocabulary.size, model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    output = Path(output_dir)
    training_state_path = output / "training-state.pt"

    history: list[dict[str, Any]] = []
    best_cer = math.inf
    best_epoch: int | None = None
    metric_migrations: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    start_epoch = 1
    resume_metadata: dict[str, Any] | None = None
    initialization_metadata: dict[str, Any] | None = None
    if initialize_from is not None:
        initialization_metadata = _initialize_model_weights(
            model, initialize_from, model_config=model_config,
            model_version=model_version, vocabulary=vocabulary,
        )

    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        initialization_metadata = checkpoint.get("initialization")
        if initialization_metadata is not None and (
            not isinstance(initialization_metadata, dict)
            or initialization_metadata.get("mode") != "weights-only-new-experiment"
        ):
            raise ValueError("Resume checkpoint has invalid initialization provenance")

        if checkpoint.get("model") != "LaoCrnnRecognizer":
            raise ValueError("Resume checkpoint model is not LaoCrnnRecognizer")
        checkpoint_model_version = checkpoint.get("model_version")
        if checkpoint_model_version != model_version:
            raise ValueError("Resume checkpoint model version mismatch")
        resume_model_config = _normalized_resume_model_config(
            checkpoint.get("model_config"),
            model_version=checkpoint_model_version,
        )
        if resume_model_config != model_config.to_dict():
            raise ValueError("Resume checkpoint model configuration mismatch")
        if checkpoint.get("vocabulary_checksum") != vocabulary.checksum():
            raise ValueError("Resume checkpoint vocabulary checksum mismatch")
        if checkpoint.get("split_strategy", SPLIT_STRATEGY) != SPLIT_STRATEGY:
            raise ValueError("Resume checkpoint split strategy mismatch")

        resume_artifact_type = checkpoint.get("artifact_type")
        previous_eval_version = checkpoint.get("dev_evaluation_version")
        known_metric_versions = {
            None, "legacy-unbounded-padding-v0", "valid-timestep-v1", DEV_EVALUATION_VERSION,
        }
        if (
            previous_eval_version is not None and not isinstance(previous_eval_version, str)
        ) or previous_eval_version not in known_metric_versions:
            raise ValueError("Unsupported resume checkpoint dev-evaluation version")
        if (
            resume_artifact_type == "recognizer-training-state"
            and previous_eval_version != DEV_EVALUATION_VERSION
            and not (recompute_resume_metrics and previous_eval_version == "valid-timestep-v1")
        ):
            raise ValueError(
                "Resume training state dev-evaluation version mismatch "
                f"({previous_eval_version!r} != {DEV_EVALUATION_VERSION!r}). "
                "Use --recompute-resume-metrics to explicitly migrate valid-timestep-v1 "
                "scores; unknown or unversioned training states cannot be migrated."
            )
        migrations = checkpoint.get("metric_migrations", [])
        if not isinstance(migrations, list) or not all(
            isinstance(item, dict) for item in migrations
        ):
            raise ValueError("Resume checkpoint metric migrations must be a list of objects")
        metric_migrations = [dict(item) for item in migrations]
        previous_padding_strategy = checkpoint.get("training_padding_strategy")
        if (
            resume_artifact_type == "recognizer-training-state"
            and previous_padding_strategy != TRAINING_PADDING_STRATEGY
        ):
            raise ValueError(
                "Resume training state padding strategy mismatch "
                f"({previous_padding_strategy!r} != {TRAINING_PADDING_STRATEGY!r})"
            )

        previous_samples_checksum = checkpoint.get("training_samples_checksum")
        if (
            resume_artifact_type == "recognizer-training-state"
            and not isinstance(previous_samples_checksum, str)
        ):
            raise ValueError("Resume training state has no training-samples checksum")
        if (
            isinstance(previous_samples_checksum, str)
            and previous_samples_checksum != training_samples_checksum
        ):
            raise ValueError("Resume checkpoint training samples mismatch")

        previous_resolved_device = checkpoint.get("resolved_device")
        if (
            resume_artifact_type == "recognizer-training-state"
            and isinstance(previous_resolved_device, str)
            and previous_resolved_device != str(device)
        ):
            raise ValueError(
                "Resume training state device mismatch "
                f"({previous_resolved_device} != {device})"
            )

        previous_config = checkpoint.get("training_config")
        if not isinstance(previous_config, dict):
            raise ValueError("Resume checkpoint has no training configuration")
        compatibility_fields = (
            "batch_size",
            "learning_rate",
            "dev_ratio",
            "seed",
            "image_height",
            "max_width",
            "num_workers",
        )
        mismatched = [
            field
            for field in compatibility_fields
            if previous_config.get(field) != training_config.to_dict().get(field)
        ]
        if mismatched:
            raise ValueError(
                "Resume checkpoint training configuration mismatch: "
                + ", ".join(mismatched)
            )

        latest_state_value = checkpoint.get("latest_state_dict")
        has_latest_state = isinstance(latest_state_value, dict)
        latest_state = latest_state_value or checkpoint.get("state_dict")
        if not isinstance(latest_state, dict):
            raise ValueError("Resume checkpoint has no model state")
        model.load_state_dict(latest_state)

        best_source = checkpoint.get("best_state_dict") or checkpoint.get("state_dict")
        if isinstance(best_source, dict):
            best_state = _cpu_state_dict(best_source)

        history_value = checkpoint.get("history", [])
        if not isinstance(history_value, list):
            raise ValueError("Resume checkpoint history must be a list")
        history_version = previous_eval_version or "legacy-unbounded-padding-v0"
        history = [
            {
                **dict(item),
                "dev_evaluation_version": dict(item).get(
                    "dev_evaluation_version",
                    history_version,
                ),
            }
            for item in history_value
        ]
        completed_epoch = max(
            (int(item.get("epoch", 0)) for item in history),
            default=0,
        )
        best_cer = float(checkpoint.get("best_dev_cer", math.inf))
        saved_best_epoch = checkpoint.get("best_epoch")
        if saved_best_epoch is not None:
            if type(saved_best_epoch) is not int or not 1 <= saved_best_epoch <= completed_epoch:
                raise ValueError("Resume checkpoint best_epoch is outside its history")
            best_epoch = saved_best_epoch
        elif history and math.isfinite(best_cer):
            matching_best_epochs = [
                int(item.get("epoch", 0))
                for item in history
                if item.get("dev_evaluation_version") == history_version
                and math.isclose(
                    float(item.get("dev_cer", math.inf)), best_cer,
                    rel_tol=1e-12, abs_tol=1e-12,
                )
            ]
            if matching_best_epochs:
                # Strict improvement retains the FIRST minimum, not the last tie.
                best_epoch = min(matching_best_epochs)
        if not has_latest_state and history:
            if best_epoch is None:
                raise ValueError("Cannot identify the retained checkpoint epoch from its history")
            completed_epoch = best_epoch
            history = [item for item in history if int(item.get("epoch", 0)) <= completed_epoch]

        if training_config.epochs <= completed_epoch:
            raise ValueError(
                "epochs must be greater than the completed resume epoch "
                f"({completed_epoch})"
            )
        start_epoch = completed_epoch + 1

        optimizer_restored = False
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if isinstance(optimizer_state, dict):
            optimizer.load_state_dict(optimizer_state)
            optimizer_restored = True

        generator_restored = False
        generator_state = checkpoint.get("data_loader_generator_state")
        if generator_state is not None:
            generator.set_state(generator_state)
            generator_restored = True

        rng_restored = _restore_rng_state(
            torch,
            device,
            checkpoint.get("rng_state"),
        )

        if resume_artifact_type == "recognizer-training-state":
            missing_state = [
                name
                for name, restored in (
                    ("optimizer", optimizer_restored),
                    ("data-loader generator", generator_restored),
                    ("RNG", rng_restored),
                )
                if not restored
            ]
            if missing_state:
                raise ValueError(
                    "Resume training state is incomplete: "
                    + ", ".join(missing_state)
                )

        baseline_dev_cer_recomputed = None
        if previous_eval_version != DEV_EVALUATION_VERSION:
            if resume_artifact_type == "recognizer-training-state" and best_state is None:
                raise ValueError("Resume training state is incomplete: retained best weights")
            # Evaluate only retained states: unavailable historical weights cannot be rescored.
            # Iterating a DataLoader may consume RNG state even in evaluation mode.
            saved_rng = _capture_rng_state(torch, device)
            saved_generator = generator.get_state().clone()
            saved_mode = model.training
            latest_weights = _cpu_state_dict(model.state_dict())
            try:
                latest_cer = evaluate(model, dev_loader, vocabulary, device)["cer"]
                retained_best_cer = None
                if has_latest_state and best_state is not None:
                    model.load_state_dict(best_state)
                    retained_best_cer = evaluate(model, dev_loader, vocabulary, device)["cer"]
                if not math.isfinite(latest_cer) or (
                    retained_best_cer is not None and not math.isfinite(retained_best_cer)
                ):
                    raise ValueError("Cannot migrate non-finite resume metrics")
                if retained_best_cer is not None and retained_best_cer <= latest_cer:
                    best_cer = retained_best_cer
                    selected = "retained-best"
                else:
                    best_cer = latest_cer
                    best_state = latest_weights
                    best_epoch = completed_epoch
                    selected = "latest" if has_latest_state else "weights-only"
                baseline_dev_cer_recomputed = best_cer
            finally:
                model.load_state_dict(latest_weights)
                model.train(saved_mode)
                generator.set_state(saved_generator)
                _restore_rng_state(torch, device, saved_rng)
            metric_migrations.append({
                "source_sha256": _sha256_file(resume_path),
                "completed_epoch": completed_epoch,
                "from_version": history_version,
                "to_version": DEV_EVALUATION_VERSION,
                "selection_scope": "retained-states-only",
                "latest_dev_cer": latest_cer,
                "retained_best_dev_cer": retained_best_cer,
                "selected_state": selected,
                "selected_epoch": best_epoch,
                "historical_scores_rewritten": False,
            })

        resume_metadata = {
            "source": str(resume_path),
            "source_sha256": _sha256_file(resume_path),
            "completed_epoch": completed_epoch,
            "previous_dev_evaluation_version": previous_eval_version,
            "dev_evaluation_version": DEV_EVALUATION_VERSION,
            "training_padding_strategy_verified": (
                previous_padding_strategy == TRAINING_PADDING_STRATEGY
                if isinstance(previous_padding_strategy, str)
                else False
            ),
            "baseline_dev_cer_recomputed": baseline_dev_cer_recomputed,
            "metric_migration_performed": baseline_dev_cer_recomputed is not None,
            "training_samples_checksum_verified": (
                previous_samples_checksum == training_samples_checksum
                if isinstance(previous_samples_checksum, str)
                else False
            ),
            "resolved_device_verified": (
                previous_resolved_device == str(device)
                if isinstance(previous_resolved_device, str)
                else False
            ),
            "optimizer_state_restored": optimizer_restored,
            "data_loader_generator_state_restored": generator_restored,
            "rng_state_restored": rng_restored,
        }

    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, training_config.epochs + 1):
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
            "dev_evaluation_version": DEV_EVALUATION_VERSION,
            "dev_prediction_diagnostics": dev_metrics["prediction_diagnostics"],
        }
        history.append(epoch_record)
        health = dev_metrics["prediction_diagnostics"]
        if health["samples"] and health["empty_predictions"] == health["samples"]:
            warnings.warn(
                f"Recognizer epoch {epoch}: all {health['samples']} development "
                "predictions are empty (CTC blank output); falling loss alone "
                "does not establish OCR learning.",
                RuntimeWarning,
                stacklevel=2,
            )

        if dev_metrics["cer"] < best_cer:
            best_cer = dev_metrics["cer"]
            best_epoch = epoch
            best_state = _cpu_state_dict(model.state_dict())

        if best_state is None:
            best_state = _cpu_state_dict(model.state_dict())

        training_state = {
            "schema_version": "1",
            "artifact_type": "recognizer-training-state",
            "model": "LaoCrnnRecognizer",
            "model_version": model_version,
            "model_config": model_config.to_dict(),
            "training_config": training_config.to_dict(),
            "split_strategy": SPLIT_STRATEGY,
            "dev_evaluation_version": DEV_EVALUATION_VERSION,
            "training_padding_strategy": TRAINING_PADDING_STRATEGY,
            "training_samples_checksum": training_samples_checksum,
            "ctc_preflight": ctc_preflight,
            "resolved_device": str(device),
            "vocabulary": vocabulary.to_dict(),
            "vocabulary_checksum": vocabulary.checksum(),
            "latest_state_dict": _cpu_state_dict(model.state_dict()),
            "best_state_dict": best_state,
            "optimizer_state_dict": _to_cpu(optimizer.state_dict()),
            "data_loader_generator_state": generator.get_state(),
            "rng_state": _capture_rng_state(torch, device),
            "history": history,
            "best_dev_cer": best_cer,
            "best_epoch": best_epoch,
            "metric_migrations": metric_migrations,
            "completed_epoch": epoch,
            "initialization": initialization_metadata,
        }
        _atomic_torch_save(torch, training_state, training_state_path)

    if best_state is not None:
        model.load_state_dict(best_state)

    vocabulary_path = vocabulary.save(output / "vocab.json")
    checkpoint_path = output / "recognizer.pt"

    checkpoint = {
        "schema_version": "1",
        "model": "LaoCrnnRecognizer",
        "model_version": model_version,
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "split_strategy": SPLIT_STRATEGY,
        "dev_evaluation_version": DEV_EVALUATION_VERSION,
        "training_padding_strategy": TRAINING_PADDING_STRATEGY,
        "training_samples_checksum": training_samples_checksum,
        "ctc_preflight": ctc_preflight,
        "resolved_device": str(device),
        "vocabulary": vocabulary.to_dict(),
        "vocabulary_checksum": vocabulary.checksum(),
        "state_dict": model.state_dict(),
        "history": history,
        "best_dev_cer": best_cer,
        "best_epoch": best_epoch,
        "metric_migrations": metric_migrations,
        "resume": resume_metadata,
        "initialization": initialization_metadata,
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = _sha256_file(checkpoint_path)

    metadata = {
        "schema_version": "1",
        "model": "LaoCrnnRecognizer",
        "model_version": model_version,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "vocabulary": vocabulary_path.name,
        "vocabulary_checksum": vocabulary.checksum(),
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "split_strategy": SPLIT_STRATEGY,
        "dev_evaluation_version": DEV_EVALUATION_VERSION,
        "training_padding_strategy": TRAINING_PADDING_STRATEGY,
        "training_samples_checksum": training_samples_checksum,
        "ctc_preflight": ctc_preflight,
        "train_samples": len(train_samples),
        "dev_samples": len(dev_samples),
        "best_dev_cer": best_cer,
        "best_epoch": best_epoch,
        "metric_migrations": metric_migrations,
        "device": str(device),
        "history": history,
        "training_state": training_state_path.name,
        "resume": resume_metadata,
        "initialization": initialization_metadata,
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
        "training_state": training_state_path,
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
        "model_version": checkpoint.get(
            "model_version",
            _model_version_for_config(model_config),
        ),
        "artifact": destination.name,
        "artifact_sha256": artifact_sha256,
        "vocabulary": checkpoint["vocabulary"],
        "vocabulary_checksum": checkpoint["vocabulary_checksum"],
        "model_config": checkpoint["model_config"],
        "split_strategy": checkpoint.get("split_strategy", "legacy-sample-id-sha256"),
        "dev_evaluation_version": checkpoint.get("dev_evaluation_version"),
        "best_epoch": checkpoint.get("best_epoch"),
        "metric_migrations": checkpoint.get("metric_migrations", []),
        "width_downsample_factor": model.width_downsample_factor,
    }
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
