"""Agent-STATIC: a frozen, unconditional baseline. Blueprint section P:
"STATIC agent: knowledge/calibration remains frozen."

The baseline is copied into this object at construction time and never
read from the live store again - this is what makes "frozen" a real
guarantee rather than a claim: even if validated_relationships changes
underneath it (because Agent-ADAPTIVE promoted something), this agent's
predictions for a previously-unseen event of the same type are
byte-for-byte identical to what they would have been on day one.
"""
from __future__ import annotations

from datetime import datetime

from market_agent.agents.base import PredictionAgent
from market_agent.events.schema import EventRecord, PredictionRecord


class StaticAgent(PredictionAgent):
    def __init__(self, unconditional_baseline: dict[int, float], model_version: str = "STATIC_v1"):
        self.unconditional_baseline = dict(unconditional_baseline)  # copied, not referenced
        self.model_version = model_version

    def predict(self, event: EventRecord, horizon_days: int, predicted_at: datetime) -> PredictionRecord:
        baseline = self.unconditional_baseline.get(horizon_days)
        if baseline is None:
            return PredictionRecord(horizon_days=horizon_days, predicted_impact=None,
                                     predicted_confidence="INSUFFICIENT_PRECEDENT",
                                     basis={"basis": "no_baseline_for_horizon"},
                                     model_version=self.model_version, predicted_at=predicted_at)
        signed_baseline = baseline if event.direction == "positive" else (
            -baseline if event.direction == "negative" else 0.0)
        return PredictionRecord(horizon_days=horizon_days, predicted_impact=signed_baseline,
                                 predicted_confidence="MEDIUM", basis={"basis": "unconditional_baseline"},
                                 model_version=self.model_version, predicted_at=predicted_at)
