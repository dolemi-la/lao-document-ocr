import itertools
import math
from collections import defaultdict

import numpy as np
import pytest

from lao_document_ocr.ctc_decoding import ctc_prefix_beam_search


def _collapse(path: tuple[int, ...], blank_id: int = 0) -> tuple[int, ...]:
    output: list[int] = []
    previous = blank_id
    for token in path:
        if token == blank_id:
            previous = blank_id
            continue
        if token != previous:
            output.append(token)
        previous = token
    return tuple(output)


def _exact_best(probs: np.ndarray, blank_id: int = 0) -> tuple[int, ...]:
    totals: defaultdict[tuple[int, ...], float] = defaultdict(float)
    timesteps, classes = probs.shape
    for path in itertools.product(range(classes), repeat=timesteps):
        probability = 1.0
        for timestep, token in enumerate(path):
            probability *= float(probs[timestep, token])
        totals[_collapse(path, blank_id)] += probability
    return max(totals.items(), key=lambda item: item[1])[0]


def test_blank_only_decodes_empty() -> None:
    log_probs = np.log(
        np.asarray(
            [
                [0.9, 0.1],
                [0.8, 0.2],
            ],
            dtype=np.float64,
        )
    )

    result = ctc_prefix_beam_search(log_probs, beam_width=4)

    assert result.token_ids == ()
    assert math.isfinite(result.log_probability)


def test_repeated_character_requires_blank_to_duplicate() -> None:
    probs = np.asarray(
        [
            [0.05, 0.95],
            [0.95, 0.05],
            [0.05, 0.95],
        ],
        dtype=np.float64,
    )

    result = ctc_prefix_beam_search(np.log(probs), beam_width=8)

    assert result.token_ids == (1, 1)


def test_beam_search_matches_exhaustive_ctc_probability() -> None:
    probs = np.asarray(
        [
            [0.35, 0.40, 0.25],
            [0.35, 0.25, 0.40],
            [0.40, 0.35, 0.25],
        ],
        dtype=np.float64,
    )

    expected = _exact_best(probs)
    result = ctc_prefix_beam_search(
        np.log(probs),
        beam_width=64,
    )

    assert result.token_ids == expected


def test_beam_search_can_differ_from_greedy_path() -> None:
    probs = np.asarray(
        [
            [0.40, 0.60],
            [0.40, 0.60],
        ],
        dtype=np.float64,
    )
    greedy = _collapse(tuple(probs.argmax(axis=1).tolist()))
    exact = _exact_best(probs)

    result = ctc_prefix_beam_search(
        np.log(probs),
        beam_width=8,
    )

    assert greedy == (1,)
    assert result.token_ids == exact


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="beam_width"):
        ctc_prefix_beam_search(np.zeros((2, 2)), beam_width=0)

    with pytest.raises(ValueError, match="shape"):
        ctc_prefix_beam_search(np.zeros((2, 2, 2)))

    with pytest.raises(ValueError, match="blank_id"):
        ctc_prefix_beam_search(np.zeros((2, 2)), blank_id=2)

    with pytest.raises(ValueError, match="finite"):
        ctc_prefix_beam_search(
            np.full((2, 2), float("-inf")),
        )
