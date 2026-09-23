from __future__ import annotations

import hashlib
import json
import warnings

import pytest
from PIL import Image

pytest.importorskip("torch")

import torch  # noqa: E402

from lao_document_ocr.confidence_calibration import (  # noqa: E402
    CalibrationBin,
    ConfidenceCalibration,
)
from lao_document_ocr.language_model import (  # noqa: E402
    save_language_model,
    train_character_ngram_language_model,
)
from lao_document_ocr.recognizer_inference import (  # noqa: E402
    ExportedLineRecognizer,
    resolve_torch_device,
)
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

    assert recognizer.device.type == "cpu"
    assert recognizer.metadata["runtime_device"] == "cpu"
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


def test_exported_recognizer_applies_calibration(tmp_path) -> None:
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
    artifact.with_suffix(".pt2.json").write_text(json.dumps(metadata), encoding="utf-8")

    calibration = ConfidenceCalibration(
        schema_version="1",
        method="test",
        sample_count=2,
        bins=(
            CalibrationBin(0.0, 1.0, 0.5, 0.42, 2),
        ),
        raw_mae=0.1,
        calibrated_mae=0.05,
    )
    calibration_path = calibration.save(tmp_path / "calibration.json")
    image_path = tmp_path / "line.png"
    Image.new("L", (80, 32), 255).save(image_path)

    recognizer = ExportedLineRecognizer(artifact, calibration_path=calibration_path)
    result = recognizer.recognize(image_path)
    assert result.calibrated_confidence == pytest.approx(0.42)


def test_resolve_torch_device_cpu_and_invalid() -> None:
    assert resolve_torch_device("cpu").type == "cpu"
    with pytest.raises(ValueError, match="device must be one of"):
        resolve_torch_device("tpu")


def test_auto_device_resolves_to_available_backend() -> None:
    device = resolve_torch_device("auto")
    if torch.cuda.is_available():
        assert device.type == "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        assert device.type == "mps"
    else:
        assert device.type == "cpu"


def test_unavailable_explicit_accelerator_has_clear_error() -> None:
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA was requested"):
            resolve_torch_device("cuda")

    mps_available = (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    if not mps_available:
        with pytest.raises(RuntimeError, match="MPS was requested"):
            resolve_torch_device("mps")


def _export_test_recognizer(tmp_path):
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
    artifact = tmp_path / "decoder-recognizer.pt2"
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
    return artifact


def test_recognizer_rejects_invalid_decoder_configuration(tmp_path) -> None:
    artifact = _export_test_recognizer(tmp_path)

    with pytest.raises(ValueError, match="decoder must be one of"):
        ExportedLineRecognizer(artifact, decoder="unknown")
    with pytest.raises(ValueError, match="beam_width"):
        ExportedLineRecognizer(artifact, decoder="beam", beam_width=0)


def test_beam_decoder_is_reported_in_runtime_metadata(tmp_path) -> None:
    artifact = _export_test_recognizer(tmp_path)
    recognizer = ExportedLineRecognizer(
        artifact,
        decoder="beam",
        beam_width=7,
    )

    assert recognizer.metadata["decoder"] == "beam"
    assert recognizer.metadata["beam_width"] == 7


def test_beam_decoder_rejects_greedy_calibration(tmp_path) -> None:
    artifact = _export_test_recognizer(tmp_path)
    calibration = ConfidenceCalibration(
        schema_version="1",
        method="greedy-test",
        sample_count=2,
        bins=(CalibrationBin(0.0, 1.0, 0.5, 0.99, 2),),
        raw_mae=0.1,
        calibrated_mae=0.05,
        decoder="greedy",
    )
    calibration_path = calibration.save(tmp_path / "greedy-calibration.json")

    with pytest.raises(ValueError, match="decoder mismatch"):
        ExportedLineRecognizer(
            artifact,
            calibration_path=calibration_path,
            decoder="beam",
            beam_width=5,
        )


def test_beam_decoder_applies_matching_calibration(tmp_path) -> None:
    artifact = _export_test_recognizer(tmp_path)
    calibration = ConfidenceCalibration(
        schema_version="1",
        method="beam-test",
        sample_count=2,
        bins=(CalibrationBin(0.0, 1.0, 0.5, 0.77, 2),),
        raw_mae=0.1,
        calibrated_mae=0.05,
        decoder="beam",
    )
    calibration_path = calibration.save(tmp_path / "beam-calibration.json")
    image_path = tmp_path / "beam-line.png"
    Image.new("L", (80, 32), 255).save(image_path)

    recognizer = ExportedLineRecognizer(
        artifact,
        calibration_path=calibration_path,
        decoder="beam",
        beam_width=5,
    )
    result = recognizer.recognize(image_path)

    assert isinstance(result.text, str)
    assert 0 <= result.confidence <= 1
    assert result.calibrated_confidence == pytest.approx(0.77)


def _language_model_for_vocab(tmp_path, vocab, lines=None):
    model, _ = train_character_ngram_language_model(
        lines or ["ab", "ab", "aa"],
        vocab,
        order=3,
        alpha=0.1,
    )
    return save_language_model(model, tmp_path / "char-lm.json")


def test_beam_recognizer_loads_matching_language_model(tmp_path) -> None:
    artifact = _export_test_recognizer(tmp_path)
    vocab = CharacterVocabulary.from_texts(["ab"])
    language_model = _language_model_for_vocab(tmp_path, vocab)
    image_path = tmp_path / "lm-line.png"
    Image.new("L", (80, 32), 255).save(image_path)

    recognizer = ExportedLineRecognizer(
        artifact,
        decoder="beam",
        beam_width=5,
        language_model_path=language_model,
        language_model_weight=0.3,
        language_model_token_bonus=0.1,
    )
    result = recognizer.recognize(image_path)

    assert recognizer.metadata["language_model"]["type"] == "character-ngram"
    assert recognizer.metadata["language_model"]["sha256"]
    assert recognizer.metadata["language_model_weight"] == pytest.approx(0.3)
    assert recognizer.metadata["language_model_token_bonus"] == pytest.approx(0.1)
    assert result.decoder_ranking_score is not None
    assert result.language_model_log_probability is not None


def test_language_model_requires_beam_decoder(tmp_path) -> None:
    artifact = _export_test_recognizer(tmp_path)
    vocab = CharacterVocabulary.from_texts(["ab"])
    language_model = _language_model_for_vocab(tmp_path, vocab)

    with pytest.raises(ValueError, match="requires decoder='beam'"):
        ExportedLineRecognizer(
            artifact,
            decoder="greedy",
            language_model_path=language_model,
            language_model_weight=0.3,
        )


def test_language_model_rejects_vocabulary_mismatch(tmp_path) -> None:
    artifact = _export_test_recognizer(tmp_path)
    other_vocab = CharacterVocabulary.from_texts(["abc"])
    language_model = _language_model_for_vocab(
        tmp_path,
        other_vocab,
        lines=["abc", "aba"],
    )

    with pytest.raises(ValueError, match="vocabulary checksum mismatch"):
        ExportedLineRecognizer(
            artifact,
            decoder="beam",
            language_model_path=language_model,
            language_model_weight=0.3,
        )


def test_language_model_fusion_parameters_require_model(tmp_path) -> None:
    artifact = _export_test_recognizer(tmp_path)

    with pytest.raises(ValueError, match="require language_model_path"):
        ExportedLineRecognizer(
            artifact,
            decoder="beam",
            language_model_weight=0.3,
        )
    with pytest.raises(ValueError, match="must be positive"):
        vocab = CharacterVocabulary.from_texts(["ab"])
        language_model = _language_model_for_vocab(tmp_path, vocab)
        ExportedLineRecognizer(
            artifact,
            decoder="beam",
            language_model_path=language_model,
            language_model_weight=0.0,
        )


def test_lm_calibration_must_match_language_model_and_fusion_parameters(tmp_path) -> None:
    from lao_document_ocr.language_model import CharacterNgramLanguageModel

    artifact = _export_test_recognizer(tmp_path)
    vocab = CharacterVocabulary.from_texts(["ab"])
    language_model_path = _language_model_for_vocab(tmp_path, vocab)
    loaded_lm = CharacterNgramLanguageModel.load(language_model_path)

    wrong_lm = ConfidenceCalibration(
        schema_version="1",
        method="lm-test",
        sample_count=2,
        bins=(CalibrationBin(0.0, 1.0, 0.5, 0.7, 2),),
        raw_mae=0.1,
        calibrated_mae=0.05,
        decoder="beam",
        language_model_sha256="0" * 64,
        language_model_weight=0.3,
        language_model_token_bonus=0.1,
    )
    wrong_lm_path = wrong_lm.save(tmp_path / "wrong-lm-calibration.json")

    with pytest.raises(ValueError, match="language model mismatch"):
        ExportedLineRecognizer(
            artifact,
            decoder="beam",
            beam_width=5,
            language_model_path=language_model_path,
            language_model_weight=0.3,
            language_model_token_bonus=0.1,
            calibration_path=wrong_lm_path,
        )

    wrong_weight = ConfidenceCalibration(
        schema_version="1",
        method="lm-test",
        sample_count=2,
        bins=(CalibrationBin(0.0, 1.0, 0.5, 0.7, 2),),
        raw_mae=0.1,
        calibrated_mae=0.05,
        decoder="beam",
        language_model_sha256=loaded_lm.artifact_sha256,
        language_model_weight=0.4,
        language_model_token_bonus=0.1,
    )
    wrong_weight_path = wrong_weight.save(tmp_path / "wrong-weight-calibration.json")

    with pytest.raises(ValueError, match="weight mismatch"):
        ExportedLineRecognizer(
            artifact,
            decoder="beam",
            beam_width=5,
            language_model_path=language_model_path,
            language_model_weight=0.3,
            language_model_token_bonus=0.1,
            calibration_path=wrong_weight_path,
        )

    matching = ConfidenceCalibration(
        schema_version="1",
        method="lm-test",
        sample_count=2,
        bins=(CalibrationBin(0.0, 1.0, 0.5, 0.7, 2),),
        raw_mae=0.1,
        calibrated_mae=0.05,
        decoder="beam",
        language_model_sha256=loaded_lm.artifact_sha256,
        language_model_weight=0.3,
        language_model_token_bonus=0.1,
    )
    matching_path = matching.save(tmp_path / "matching-calibration.json")

    recognizer = ExportedLineRecognizer(
        artifact,
        decoder="beam",
        beam_width=5,
        language_model_path=language_model_path,
        language_model_weight=0.3,
        language_model_token_bonus=0.1,
        calibration_path=matching_path,
    )
    assert recognizer.calibration is not None
