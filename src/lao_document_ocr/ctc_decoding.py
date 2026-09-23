from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

_NEG_INF = float("-inf")


@dataclass(frozen=True)
class BeamSearchResult:
    token_ids: tuple[int, ...]
    log_probability: float
    ranking_score: float
    language_model_log_probability: float = 0.0


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
    extension_scorer: Callable[[tuple[int, ...], int], float] | None = None,
    scorer_weight: float = 0.0,
    token_bonus: float = 0.0,
) -> BeamSearchResult:
    """Decode CTC log probabilities with prefix beam search.

    The returned token sequence is already CTC-collapsed; callers should map
    token IDs directly to characters rather than applying CTC collapse again.
    """
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")
    if not math.isfinite(scorer_weight) or scorer_weight < 0:
        raise ValueError("scorer_weight must be finite and non-negative")
    if not math.isfinite(token_bonus):
        raise ValueError("token_bonus must be finite")
    if extension_scorer is None and scorer_weight != 0:
        raise ValueError("scorer_weight requires extension_scorer")

    values = np.asarray(log_probs, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("log_probs must have shape [timesteps, classes]")
    timesteps, classes = values.shape
    if classes < 1:
        raise ValueError("log_probs must contain at least one class")
    if not 0 <= blank_id < classes:
        raise ValueError("blank_id is outside the class range")
    if timesteps == 0:
        return BeamSearchResult(
            token_ids=(),
            log_probability=0.0,
            ranking_score=0.0,
            language_model_log_probability=0.0,
        )
    if not np.isfinite(values).any():
        raise ValueError("log_probs contain no finite values")

    # prefix -> (probability ending in blank, probability ending in nonblank)
    beams: dict[tuple[int, ...], tuple[float, float]] = {
        (): (0.0, _NEG_INF)
    }
    language_scores: dict[tuple[int, ...], float] = {(): 0.0}

    for timestep in range(timesteps):
        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}
        next_language_scores: dict[tuple[int, ...], float] = {}

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
            next_language_scores[prefix] = language_scores[prefix]

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
                    language_score = language_scores[prefix]
                    if extension_scorer is not None:
                        language_score += float(
                            extension_scorer(prefix, token_id)
                        )
                    next_language_scores[extended] = language_score
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
                language_score = language_scores[prefix]
                if extension_scorer is not None:
                    language_score += float(
                        extension_scorer(prefix, token_id)
                    )
                next_language_scores[extended] = language_score

        def ranking_score(
            item,
            scores=next_language_scores,
        ):
            prefix, beam = item
            acoustic = _total_probability(*beam)
            return (
                acoustic
                + scorer_weight * scores.get(prefix, 0.0)
                + token_bonus * len(prefix)
            )

        ranked = sorted(
            next_beams.items(),
            key=ranking_score,
            reverse=True,
        )
        ranked = ranked[:beam_width]
        beams = dict(ranked)
        language_scores = {
            prefix: next_language_scores.get(prefix, 0.0)
            for prefix, _ in ranked
        }

    def final_ranking(item):
        prefix, beam = item
        acoustic = _total_probability(*beam)
        return (
            acoustic
            + scorer_weight * language_scores.get(prefix, 0.0)
            + token_bonus * len(prefix)
        )

    best_prefix, (blank_log, nonblank_log) = max(
        beams.items(),
        key=final_ranking,
    )
    acoustic = _total_probability(blank_log, nonblank_log)
    language_score = language_scores.get(best_prefix, 0.0)
    return BeamSearchResult(
        token_ids=best_prefix,
        log_probability=acoustic,
        ranking_score=(
            acoustic
            + scorer_weight * language_score
            + token_bonus * len(best_prefix)
        ),
        language_model_log_probability=language_score,
    )
