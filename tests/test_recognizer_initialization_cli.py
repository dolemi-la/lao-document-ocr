"""Initialization CLI routing can be tested without loading model weights."""

from __future__ import annotations

import sys

import pytest

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


@pytest.mark.parametrize("resize", [False, True])
def test_cli_forwards_resolution_transfer_flag(tmp_path, monkeypatch, resize):
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
    monkeypatch.setattr(cli, "load_training_manifest", lambda *args, **kwargs: [])
    argv = [
        "lao-ocr",
        "train-recognizer",
        "--manifest",
        str(tmp_path / "manifest.jsonl"),
        "--output",
        str(tmp_path / "new"),
        "--initialize-from",
        str(tmp_path / "parent.pt"),
        "--image-height",
        "64",
        "--max-width",
        "1024",
    ]
    if resize:
        argv.append("--initialize-resize")
    monkeypatch.setattr(sys, "argv", argv)
    assert cli.main() == 0
    assert seen["initialize_resize"] is resize
    assert seen["training_config"].image_height == 64
    assert seen["training_config"].max_width == 1024


def test_cli_rejects_resize_without_initialization(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_training_manifest", lambda *args, **kwargs: [])
    output = tmp_path / "untouched"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "train-recognizer",
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--output",
            str(output),
            "--initialize-resize",
        ],
    )
    assert cli.main() == 1
    assert "requires --initialize-from" in capsys.readouterr().err
    assert not output.exists()
