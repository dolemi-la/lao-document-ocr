import pytest

from services.api.app.rate_limit import SlidingWindowRateLimiter


def test_disabled_limiter_always_allows() -> None:
    limiter = SlidingWindowRateLimiter(requests=0, window_seconds=60)

    decision = limiter.check("client-a", now=1.0)

    assert decision.allowed is True
    assert decision.retry_after_seconds == 0
    assert limiter.snapshot()["tracked_clients"] == 0


def test_sliding_window_blocks_after_limit_and_recovers() -> None:
    limiter = SlidingWindowRateLimiter(
        requests=2,
        window_seconds=10,
    )

    first = limiter.check("client-a", now=100.0)
    second = limiter.check("client-a", now=101.0)
    blocked = limiter.check("client-a", now=102.0)
    recovered = limiter.check("client-a", now=111.1)

    assert first.allowed is True
    assert first.remaining == 1
    assert second.allowed is True
    assert second.remaining == 0
    assert blocked.allowed is False
    assert blocked.retry_after_seconds == 8
    assert recovered.allowed is True


def test_clients_are_isolated() -> None:
    limiter = SlidingWindowRateLimiter(
        requests=1,
        window_seconds=60,
    )

    assert limiter.check("client-a", now=1.0).allowed is True
    assert limiter.check("client-a", now=2.0).allowed is False
    assert limiter.check("client-b", now=2.0).allowed is True


def test_client_key_memory_is_bounded() -> None:
    limiter = SlidingWindowRateLimiter(
        requests=1,
        window_seconds=60,
        max_clients=2,
    )

    assert limiter.check("a", now=1.0).allowed is True
    assert limiter.check("b", now=1.0).allowed is True

    blocked = limiter.check("c", now=2.0)

    assert blocked.allowed is False
    assert limiter.snapshot()["tracked_clients"] == 2


def test_stale_clients_are_pruned_at_capacity() -> None:
    limiter = SlidingWindowRateLimiter(
        requests=1,
        window_seconds=10,
        max_clients=2,
    )
    limiter.check("a", now=1.0)
    limiter.check("b", now=1.0)

    decision = limiter.check("c", now=12.0)

    assert decision.allowed is True
    assert limiter.snapshot()["tracked_clients"] == 1


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SlidingWindowRateLimiter(requests=-1, window_seconds=60)
    with pytest.raises(ValueError, match="window_seconds"):
        SlidingWindowRateLimiter(requests=1, window_seconds=0)
    with pytest.raises(ValueError, match="max_clients"):
        SlidingWindowRateLimiter(requests=1, window_seconds=60, max_clients=0)


def test_rate_limit_cost_consumes_multiple_slots_atomically() -> None:
    limiter = SlidingWindowRateLimiter(requests=5, window_seconds=60)

    first = limiter.check("client-a", now=10.0, cost=3)
    blocked = limiter.check("client-a", now=11.0, cost=3)
    second = limiter.check("client-a", now=11.0, cost=2)

    assert first.allowed is True
    assert first.remaining == 2
    assert blocked.allowed is False
    assert second.allowed is True
    assert second.remaining == 0


def test_rate_limit_rejects_cost_larger_than_window_capacity() -> None:
    limiter = SlidingWindowRateLimiter(requests=2, window_seconds=60)

    decision = limiter.check("client-a", now=1.0, cost=3)

    assert decision.allowed is False
    assert limiter.snapshot()["tracked_clients"] == 1


def test_rate_limit_rejects_invalid_cost() -> None:
    limiter = SlidingWindowRateLimiter(requests=2, window_seconds=60)
    with pytest.raises(ValueError, match="cost"):
        limiter.check("client-a", cost=0)
