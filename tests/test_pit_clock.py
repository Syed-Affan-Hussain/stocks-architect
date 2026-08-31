from datetime import datetime, timedelta, timezone

import pytest

from market_agent.pit.clock import FutureInformationError, PointInTimeClock


def test_assert_not_future_passes_for_past_timestamp():
    clock = PointInTimeClock(now=datetime(2024, 6, 1, tzinfo=timezone.utc))
    clock.assert_not_future(datetime(2024, 5, 1, tzinfo=timezone.utc))  # should not raise


def test_assert_not_future_raises_for_future_timestamp():
    clock = PointInTimeClock(now=datetime(2024, 6, 1, tzinfo=timezone.utc))
    with pytest.raises(FutureInformationError):
        clock.assert_not_future(datetime(2024, 7, 1, tzinfo=timezone.utc))


def test_assert_not_future_handles_naive_datetimes_as_utc():
    clock = PointInTimeClock(now=datetime(2024, 6, 1, tzinfo=timezone.utc))
    clock.assert_not_future(datetime(2024, 5, 1))  # naive - should be treated as UTC, not raise


def test_advance_to_moves_forward():
    clock = PointInTimeClock(now=datetime(2024, 6, 1, tzinfo=timezone.utc))
    later = clock.advance_to(datetime(2024, 6, 2, tzinfo=timezone.utc))
    assert later.now == datetime(2024, 6, 2, tzinfo=timezone.utc)
    assert clock.now == datetime(2024, 6, 1, tzinfo=timezone.utc)  # original clock unchanged - immutable


def test_advance_to_rejects_moving_backward():
    clock = PointInTimeClock(now=datetime(2024, 6, 2, tzinfo=timezone.utc))
    with pytest.raises(ValueError):
        clock.advance_to(datetime(2024, 6, 1, tzinfo=timezone.utc))
