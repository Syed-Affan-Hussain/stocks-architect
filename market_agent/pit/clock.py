"""Point-in-time simulation clock - the single gate every data access in
this system must pass through.

Blueprint reference: "Adaptive Market-Intelligence Blueprint" section O
("Chronological backtesting and leakage defenses"). A static backtest
replays one frozen model against history once; this system replays an
ENTIRE adaptation pipeline (recalibration, hypothesis testing, promotion)
sequentially at every historical checkpoint - which means every single
read (episodic memory query, validated-relationship lookup, hypothesis
test, retraining trigger) must be restricted to information published no
later than the simulated "now". Getting this gate wrong is the most
common, most silent way an adaptive-learning backtest lies about its own
results - it doesn't crash, it just quietly looks too good.

This module is deliberately tiny and has no business logic in it at all -
its only job is to be the one place `as_of` comes from, so every other
module can take `clock: PointInTimeClock` as a parameter instead of
reading a real wall clock, and so a single adversarial test here (see
tests/test_pit_clock.py) can assert the gate actually holds.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


class FutureInformationError(Exception):
    """Raised when code asks for or supplies information timestamped after
    the clock's current simulated time. This must never be caught and
    silently ignored anywhere in this system - it means a leakage bug."""


@dataclass
class PointInTimeClock:
    """Wraps a single `now` timestamp. In live operation this is the real
    wall-clock time (advances continuously). In a historical simulation,
    this is set explicitly by the simulation driver and advanced one
    checkpoint at a time - it is NEVER read from the real system clock
    during a simulation, which is what makes replay reproducible."""
    now: datetime

    @classmethod
    def live(cls) -> "PointInTimeClock":
        return cls(now=datetime.now(timezone.utc))

    def assert_not_future(self, timestamp: datetime, label: str = "timestamp") -> None:
        """The one call every reader/writer in this system should make
        before trusting an externally-supplied timestamp. Deliberately
        strict (raises rather than clamps/warns) - a leakage bug should be
        loud, not silently corrected."""
        ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        if ts > self.now:
            raise FutureInformationError(
                f"{label} ({ts.isoformat()}) is after the simulation clock's current time "
                f"({self.now.isoformat()}) - this would leak future information.")

    def advance_to(self, new_now: datetime) -> "PointInTimeClock":
        """Returns a NEW clock at a later time - clocks are immutable so a
        reference held elsewhere can't silently jump forward underneath it."""
        new_now = new_now if new_now.tzinfo else new_now.replace(tzinfo=timezone.utc)
        if new_now < self.now:
            raise ValueError(f"Cannot move the simulation clock backward: {new_now} < {self.now}")
        return PointInTimeClock(now=new_now)
