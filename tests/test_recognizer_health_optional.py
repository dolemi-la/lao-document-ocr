"""Prediction-health diagnostics must not include labels or padded timesteps."""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from lao_document_ocr.recognizer_training import evaluate  # noqa: E402
from lao_document_ocr.vocabulary import CharacterVocabulary  # noqa: E402


class FixedPredictions(torch.nn.Module):
    def __init__(self, batches, classes):
        super().__init__()
        self.batches = iter(batches)
        self.classes = classes

    def forward(self, images):
        paths = torch.tensor(next(self.batches), dtype=torch.long)
        scores = torch.full((*paths.shape, self.classes), -20.0)
        scores.scatter_(2, paths.unsqueeze(-1), 0.0)
        return scores.log_softmax(dim=-1)

    @staticmethod
    def output_lengths(widths):
        return widths // 4


def _batch(texts, widths):
    return {
        "images": torch.zeros((len(texts), 1, 16, max(widths))),
        "widths": torch.tensor(widths),
        "texts": texts,
    }


def test_health_counts_only_valid_timesteps_and_contains_no_text() -> None:
    vocab = CharacterVocabulary.from_texts(["ab"])
    a, b = vocab.encode("ab")
    # First sample is entirely blank in its valid prefix. Its padded tail must
    # neither create a prediction nor inflate the valid/blank timestep counts.
    model = FixedPredictions([
        [[0, a], [0, 0], [a, b], [a, 0]],
    ], vocab.size)
    result = evaluate(model, [_batch(["a", "ab"], [8, 16])], vocab, "cpu")

    assert result["cer"] == pytest.approx(1 / 3)
    assert result["prediction_diagnostics"] == {
        "version": "valid-timestep-greedy-v1",
        "samples": 2,
        "empty_predictions": 1,
        "empty_prediction_ratio": 0.5,
        "predicted_characters": 2,
        "reference_characters": 3,
        "predicted_to_reference_character_ratio": 2 / 3,
        "blank_timesteps": 4,
        "valid_timesteps": 6,
        "blank_timestep_ratio": 4 / 6,
    }
    payload = json.dumps(result, allow_nan=False)
    for field in ('"texts"', '"reference"', '"hypothesis"', '"ids"'):
        assert field not in payload
    assert not model.training


def test_health_aggregates_counts_not_unweighted_batch_ratios() -> None:
    vocab = CharacterVocabulary.from_texts(["ab"])
    a, b = vocab.encode("ab")
    model = FixedPredictions([
        [[0]],
        [[a, b], [0, 0], [b, a]],
    ], vocab.size)
    result = evaluate(
        model,
        [_batch(["a"], [4]), _batch(["ab", "ba"], [12, 12])],
        vocab,
        "cpu",
    )
    health = result["prediction_diagnostics"]
    assert health["samples"] == 3
    assert health["empty_predictions"] == 1
    assert health["empty_prediction_ratio"] == pytest.approx(1 / 3)
    assert health["valid_timesteps"] == 7
    assert health["blank_timesteps"] == 3
    assert health["blank_timestep_ratio"] == pytest.approx(3 / 7)
    assert health["predicted_characters"] == 4
    assert health["reference_characters"] == 5
    assert health["predicted_to_reference_character_ratio"] == pytest.approx(4 / 5)


def test_health_reports_all_blank_predictions_explicitly() -> None:
    vocab = CharacterVocabulary.from_texts(["ab"])
    model = FixedPredictions([[[0, 0], [0, 0]]], vocab.size)
    result = evaluate(model, [_batch(["a", "b"], [8, 8])], vocab, "cpu")
    health = result["prediction_diagnostics"]
    assert result["cer"] == 1.0
    assert health["empty_predictions"] == health["samples"] == 2
    assert health["empty_prediction_ratio"] == 1.0
    assert health["predicted_characters"] == 0
    assert health["blank_timestep_ratio"] == 1.0


def test_empty_evaluation_has_no_fabricated_health_ratios() -> None:
    vocab = CharacterVocabulary.from_texts(["ab"])
    result = evaluate(FixedPredictions([], vocab.size), [], vocab, "cpu")
    health = result["prediction_diagnostics"]
    assert health["samples"] == health["valid_timesteps"] == 0
    assert health["empty_prediction_ratio"] is None
    assert health["blank_timestep_ratio"] is None
    assert health["predicted_to_reference_character_ratio"] is None
    json.dumps(result, allow_nan=False)
