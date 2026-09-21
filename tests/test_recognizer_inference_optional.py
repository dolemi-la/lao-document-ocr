from __future__ import annotations

import hashlib
import json
import warnings

import pytest
from PIL import Image

pytest.importorskip("torch")

import torch  # noqa: E402

from lao_document_ocr.recognizer_inference import ExportedLineRecognizer  # noqa: E402
from lao_document_ocr.recognizer_model import LaoCrnnRecognizer, RecognizerConfig  # noqa: E402
from lao_document_ocr.vocabulary import CharacterVocabulary  # noqa: E402


def test_exported_recognizer_runs_fixed_width_artifact(tmp_path) -> None:
    vocab = CharacterVocabulary.from_texts(["ab"])
    config = RecognizerConfig(
        image_height=48,
        max_width=128,
        cnn_channels=64,
        hidden_size=32,
        lstm_layers=1,
    )
    model = LaoCrnnRecognizer(vocab.size, config)
    model.eval()

    example = torch.zeros((1, 1, config.image_height, config.max_width))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The tensor attributes .* were assigned during export.*",
            category=UserWarning,
        )
        exported = torch.export.export(model, (example,))
    artifact = tmp_path / "recognizer.pt2"
    torch.export.save(exported, artifact)

    metadata = {
        "schema_version": "1",
        "format": "torch-export",
        "model": "LaoCrnnRecognizer",
        "artifact": artifact.name,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "vocabulary": vocab.to_dict(),
        "vocabulary_checksum": vocab.checksum(),
        "model_config": config.to_dict(),
        "width_downsample_factor": model.width_downsample_factor,
    }
    artifact.with_suffix(".pt2.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    image_path = tmp_path / "line.png"
    Image.new("L", (80, 32), 255).save(image_path)

    recognizer = ExportedLineRecognizer(artifact)
    result = recognizer.recognize(image_path)

    assert isinstance(result.text, str)
    assert 0 <= result.confidence <= 1
    assert result.valid_timesteps > 0


def test_exported_recognizer_rejects_tampered_artifact(tmp_path) -> None:
    artifact = tmp_path / "recognizer.pt2"
    artifact.write_bytes(b"tampered")
    metadata = {
        "format": "torch-export",
        "artifact_sha256": "0" * 64,
        "model_config": {"image_height": 48, "max_width": 128},
        "width_downsample_factor": 4,
        "vocabulary": {"characters": ["a"]},
    }
    artifact.with_suffix(".pt2.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        ExportedLineRecognizer(artifact)
