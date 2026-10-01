"""Initialization CLI routing can be tested without loading model weights."""

from __future__ import annotations

import sys

import lao_document_ocr.cli as cli
import lao_document_ocr.recognizer_training as training


def test_cli_forwards_new_experiment_initialization(tmp_path, monkeypatch):
    seen = {}

    def train(samples, output, **kwargs):
        seen.update(kwargs)
        return {
            "checkpoint": output / "recognizer.pt",
            "training_state": output / "state.pt",
            "metadata": output / "metadata.json",
            "best_dev_cer": 0.5,
        }

    monkeypatch.setattr(training, "train_recognizer", train)
    monkeypatch.setattr(cli, "load_training_manifest", lambda *a, **k: [])
    source = tmp_path / "parent.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "train-recognizer",
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--output",
            str(tmp_path / "new"),
            "--initialize-from",
            str(source),
            "--learning-rate",
            "0.0003",
        ],
    )
    assert cli.main() == 0
    assert seen["initialize_from"] == source
    assert seen["resume_from"] is None
    assert seen["training_config"].learning_rate == 0.0003


def test_cli_cannot_confuse_initialization_with_resume(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_training_manifest", lambda *a, **k: [])
    destination = tmp_path / "untouched"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "train-recognizer",
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--output",
            str(destination),
            "--initialize-from",
            str(tmp_path / "parent.pt"),
            "--resume-from",
            str(tmp_path / "state.pt"),
        ],
    )
    assert cli.main() == 1
    assert "mutually exclusive" in capsys.readouterr().err
    assert not destination.exists()
