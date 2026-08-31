"""Common interface for a prediction agent - the thing that turns an
EventRecord into a PredictionRecord. STATIC and ADAPTIVE (agents/
static_agent.py, agents/adaptive_agent.py) are the only two
implementations, and the entire point of the Static-vs-Adaptive
experiment (Blueprint section P) is that they share this interface so
they can be run side by side against the identical event stream.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from market_agent.events.schema import EventRecord, PredictionRecord


class PredictionAgent(ABC):
    model_version: str

    @abstractmethod
    def predict(self, event: EventRecord, horizon_days: int, predicted_at: datetime) -> PredictionRecord:
        raise NotImplementedError
