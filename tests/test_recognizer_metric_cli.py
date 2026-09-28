"""Resume-metric CLI plumbing remains testable without optional PyTorch."""

from __future__ import annotations

import sys

import pytest

import lao_document_ocr.cli as cli
import lao_document_ocr.recognizer_training as training


@pytest.mark.parametrize("migrate", [False, True])
def test_train_cli_forwards_metric_migration(tmp_path, monkeypatch, migrate):
    seen = {}

    def train(samples, output, **kwargs):
        seen.update(kwargs)
        return {
            "checkpoint": output / "recognizer.pt",
            "training_state": output / "training-state.pt",
            "metadata": output / "metadata.json",
            "best_dev_cer": 0.5,
        }

    monkeypatch.setattr(training, "train_recognizer", train)
    monkeypatch.setattr(cli, "load_training_manifest", lambda *a, **kw: [])
    args = [
        "lao-ocr",
        "train-recognizer",
        "--manifest",
        str(tmp_path / "manifest.jsonl"),
        "--output",
        str(tmp_path / "run"),
        "--resume-from",
        str(tmp_path / "state.pt"),
    ]
    if migrate:
        args.append("--recompute-resume-metrics")
    monkeypatch.setattr(sys, "argv", args)
    assert cli.main() == 0
    assert seen["resume_from"] == tmp_path / "state.pt"
    assert seen["recompute_resume_metrics"] is migrate


def test_metric_migration_without_source_fails_before_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_training_manifest", lambda *a, **kw: [])
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
            "--recompute-resume-metrics",
        ],
    )
    assert cli.main() == 1
    assert "requires --resume-from" in capsys.readouterr().err
    assert not output.exists()
