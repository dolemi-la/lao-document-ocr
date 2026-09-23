from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_NEG_INF = float("-inf")


@dataclass(frozen=True)
class BeamSearchResult:
    token_ids: tuple[int, ...]
    log_probability: float


def _logaddexp(left: float, right: float) -> float:
    if left == _NEG_INF:
        return right
    if right == _NEG_INF:
        return left
    high = max(left, right)
    low = min(left, right)
    return high + math.log1p(math.exp(low - high))


def _total_probability(blank_log: float, nonblank_log: float) -> float:
    return _logaddexp(blank_log, nonblank_log)


def ctc_prefix_beam_search(
    log_probs: np.ndarray,
    *,
    beam_width: int = 10,
    blank_id: int = 0,
) -> BeamSearchResult:
    """Decode CTC log probabilities with prefix beam search.

    The returned token sequence is already CTC-collapsed; callers should map
    token IDs directly to characters rather than applying CTC collapse again.
    """
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")

    values = np.asarray(log_probs, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("log_probs must have shape [timesteps, classes]")
    timesteps, classes = values.shape
    if classes < 1:
        raise ValueError("log_probs must contain at least one class")
    if not 0 <= blank_id < classes:
        raise ValueError("blank_id is outside the class range")
    if timesteps == 0:
        return BeamSearchResult(token_ids=(), log_probability=0.0)
    if not np.isfinite(values).any():
        raise ValueError("log_probs contain no finite values")

    # prefix -> (probability ending in blank, probability ending in nonblank)
    beams: dict[tuple[int, ...], tuple[float, float]] = {
        (): (0.0, _NEG_INF)
    }

    for timestep in range(timesteps):
        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}

        for prefix, (blank_log, nonblank_log) in beams.items():
            total_log = _total_probability(blank_log, nonblank_log)

            blank_score = total_log + float(values[timestep, blank_id])
            existing_blank, existing_nonblank = next_beams.get(
                prefix,
                (_NEG_INF, _NEG_INF),
            )
            next_beams[prefix] = (
                _logaddexp(existing_blank, blank_score),
                existing_nonblank,
            )

            for token_id in range(classes):
                if token_id == blank_id:
                    continue
                token_log = float(values[timestep, token_id])

                if prefix and token_id == prefix[-1]:
                    same_score = nonblank_log + token_log
                    existing_blank, existing_nonblank = next_beams.get(
                        prefix,
                        (_NEG_INF, _NEG_INF),
                    )
                    next_beams[prefix] = (
                        existing_blank,
                        _logaddexp(existing_nonblank, same_score),
                    )

                    extended = (*prefix, token_id)
                    repeat_score = blank_log + token_log
                    existing_blank, existing_nonblank = next_beams.get(
                        extended,
                        (_NEG_INF, _NEG_INF),
                    )
                    next_beams[extended] = (
                        existing_blank,
                        _logaddexp(existing_nonblank, repeat_score),
                    )
                    continue

                extended = (*prefix, token_id)
                extended_score = total_log + token_log
                existing_blank, existing_nonblank = next_beams.get(
                    extended,
                    (_NEG_INF, _NEG_INF),
                )
                next_beams[extended] = (
                    existing_blank,
                    _logaddexp(existing_nonblank, extended_score),
                )

        ranked = sorted(
            next_beams.items(),
            key=lambda item: _total_probability(*item[1]),
            reverse=True,
        )
        beams = dict(ranked[:beam_width])

    best_prefix, (blank_log, nonblank_log) = max(
        beams.items(),
        key=lambda item: _total_probability(*item[1]),
    )
    return BeamSearchResult(
        token_ids=best_prefix,
        log_probability=_total_probability(blank_log, nonblank_log),
    )
