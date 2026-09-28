"""Training-selection CER must match the exported benchmark's text policy."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from lao_document_ocr.recognizer_benchmark import benchmark_recognizer  # noqa: E402
from lao_document_ocr.recognizer_training import evaluate  # noqa: E402
from lao_document_ocr.training_manifest import TrainingSample  # noqa: E402
from lao_document_ocr.vocabulary import CharacterVocabulary  # noqa: E402


class _RawPrediction(torch.nn.Module):
    def __init__(self, vocabulary, prediction):
        super().__init__()
        tokens = [0]
        for char in prediction:
            tokens.extend([vocabulary.char_to_id[char], 0])
        # A nonblank padded tail must not enter either the metric or health counts.
        self.valid = len(tokens)
        tokens.extend([vocabulary.char_to_id["x"], 0])
        self.register_buffer("paths", torch.tensor(tokens))
        self.classes = vocabulary.size

    def forward(self, images):
        scores = torch.full((len(self.paths), 1, self.classes), -20.0)
        scores.scatter_(2, self.paths[:, None, None], 0.0)
        return scores.log_softmax(dim=-1)

    @staticmethod
    def output_lengths(widths):
        return widths // 4


@pytest.mark.parametrize(
    "reference,prediction",
    [
        ("a b", "  a   b  "),
        ("é", "e\u0301"),
        ("  e\u0301  ", "é"),
        ("ພາສາ ລາວ", "  ພາສາ\t\u00a0ລາວ  "),
        ("ab", "a b"),
        ("ກ", "ຂ"),
        ("a", "   "),
        ("   ", "a"),
        ("", ""),
        ("a\nb", "a\r\n\n\n\nb"),
    ],
)
def test_training_cer_matches_benchmark_normalization(reference, prediction):
    vocabulary = CharacterVocabulary(tuple(sorted(set(reference + prediction + "x"))))
    model = _RawPrediction(vocabulary, prediction)
    batch = {
        "images": torch.zeros((1, 1, 16, len(model.paths) * 4)),
        "widths": torch.tensor([model.valid * 4]),
        "texts": [reference],
    }
    result = evaluate(model, [batch], vocabulary, "cpu")
    report = benchmark_recognizer(
        [TrainingSample("sample", Path("unused.png"), reference)],
        SimpleNamespace(recognize=lambda _: SimpleNamespace(text=prediction)),
    )
    for key in ("cer", "character_edits", "characters"):
        assert result[key] == report["overall"][key]
    # Raw diagnostics remain truthful: spaces are not CTC blank tokens.
    health = result["prediction_diagnostics"]
    assert health["version"] == "valid-timestep-greedy-v1"
    assert health["reference_characters"] == len(reference)
    assert health["predicted_characters"] == len(prediction)
    assert health["empty_predictions"] == int(not prediction)
    assert health["valid_timesteps"] == model.valid


def test_normalized_metric_aggregates_edits_not_per_sample_ratios():
    vocabulary = CharacterVocabulary(tuple(sorted(set("a bx"))))
    batch_a = {
        "images": torch.zeros((1, 1, 16, 20)),
        "widths": torch.tensor([12]),
        "texts": [" a "],
    }
    batch_b = {
        "images": torch.zeros((1, 1, 16, 36)),
        "widths": torch.tensor([28]),
        "texts": [" aaa "],
    }

    class TwoBatches(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.index = 0
            self.models = [_RawPrediction(vocabulary, "a"), _RawPrediction(vocabulary, "aab")]

        def forward(self, images):
            value = self.models[self.index](images)
            self.index += 1
            return value

        output_lengths = staticmethod(lambda widths: widths // 4)

    result = evaluate(TwoBatches(), [batch_a, batch_b], vocabulary, "cpu")
    assert result["character_edits"] == 1
    assert result["characters"] == 4
    assert result["cer"] == 0.25


@pytest.mark.parametrize("metric_version", [None, "normalized-valid-timestep-v2"])
def test_export_preserves_metric_version_without_relabeling_legacy(tmp_path, metric_version):
    import json

    from lao_document_ocr.recognizer_model import LaoCrnnRecognizer, RecognizerConfig
    from lao_document_ocr.recognizer_training import export_recognizer

    vocabulary = CharacterVocabulary.from_texts(["ab"])
    config = RecognizerConfig(max_width=32, cnn_channels=64, hidden_size=16, lstm_layers=1)
    model = LaoCrnnRecognizer(vocabulary.size, config)
    checkpoint = {
        "model_config": config.to_dict(),
        "vocabulary": vocabulary.to_dict(),
        "vocabulary_checksum": vocabulary.checksum(),
        "state_dict": model.state_dict(),
    }
    if metric_version is not None:
        checkpoint["dev_evaluation_version"] = metric_version
        checkpoint["best_epoch"] = 4
        checkpoint["metric_migrations"] = [{"selection_scope": "retained-states-only"}]
    source = tmp_path / "source.pt"
    torch.save(checkpoint, source)
    destination = export_recognizer(source, tmp_path / "export.pt2")
    sidecar = json.loads(destination.with_suffix(".pt2.json").read_text())
    assert sidecar["dev_evaluation_version"] == metric_version
    assert sidecar["best_epoch"] == checkpoint.get("best_epoch")
    assert sidecar["metric_migrations"] == checkpoint.get("metric_migrations", [])
