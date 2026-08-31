"""Agent-ADAPTIVE: queries category 3 (validated_relationships) live at
prediction time, falling back to the SAME unconditional baseline
Agent-STATIC uses when no validated relationship matches. That shared
fallback matters: when nothing has been learned yet, or nothing relevant
applies to a given event, ADAPTIVE and STATIC must produce byte-for-byte
identical predictions - any observed difference between them is then
attributable ONLY to a governed, promoted update, never to a difference
in their starting assumptions.

This agent never writes to validated_relationships itself - see
learn/governance.py for the only code path allowed to do that. Reading
here is always gated by `published_before` for point-in-time correctness
in a historical replay.

STAGE 6: CONCEPT_FILTER - the four-way experiment (STATIC / CURRENT
ADAPTIVE / TECHNICAL ADAPTIVE / METHODOLOGY-INFORMED ADAPTIVE) needs three
DIFFERENT eligibility rules over the SAME validated_relationships table,
not three duplicated agent implementations. `concept_filter` (default
None = unrestricted, i.e. exactly this class's pre-stage-6 behavior -
every existing caller/test that doesn't pass it is completely unaffected)
selects which relationships this agent instance is even allowed to match
against:

  EVENT_CONTEXT_ONLY          - concept IS NULL. What "CURRENT ADAPTIVE"
      means for a fair four-way comparison: relationships conditioned only
      on regime/prior_return_bucket/vol_bucket, i.e. exactly what this
      agent could see before stage 6 added technical concepts at all.
  ANY_TECHNICAL_CONCEPT       - concept IS NOT NULL. "TECHNICAL ADAPTIVE":
      any relationship that touches a canonical trading concept, whether
      or not any methodology happened to map onto it.
  METHODOLOGY_BACKED_CONCEPT  - concept IS NOT NULL AND methodology_ids_json
      IS NOT NULL. "METHODOLOGY-INFORMED ADAPTIVE": restricted further to
      concepts at least one ingested methodology independently claimed.

See agents/variants.py for the three named factory functions the
experiment harness actually uses - this class itself stays a single,
un-duplicated implementation.

STAGE 7: QUALIFIED_RELATIONSHIP_IDS - the fifth agent, ENSEMBLE_ADAPTIVE
(agents/variants.py::make_ensemble_adaptive_agent), needs a FOURTH
eligibility rule that concept_filter's fixed SQL clauses can't express:
"only a specific, explicitly-verified SET of relationship_ids" - the ones
that have passed the FULL stage-7 diagnostic stack (statistical
significance, learn/incremental_value.py, and
learn/overfitting_diagnostics.py's permutation/temporal-stability checks),
computed periodically by a real governance pass (see
scripts/run_stage7_final_report.py), never per-prediction. When
`qualified_relationship_ids` is provided (default None = no additional
restriction, so every existing caller/test is unaffected), it's ANDed
onto whatever concept_filter already restricts - this is a SELECTION, not
a blend: a matching qualified relationship's own effect_estimate replaces
the baseline outright (weight 1.0), it is never averaged with anything.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from market_agent.agents.base import PredictionAgent
from market_agent.events.schema import EventRecord, PredictionRecord

CONFIDENCE_N_THRESHOLDS = [(50, "HIGH"), (15, "MEDIUM")]  # n_supporting >= threshold -> confidence label

CONCEPT_FILTER_CLAUSES: dict[str | None, str] = {
    None: "",
    "EVENT_CONTEXT_ONLY": " AND concept IS NULL",
    "ANY_TECHNICAL_CONCEPT": " AND concept IS NOT NULL",
    "METHODOLOGY_BACKED_CONCEPT": " AND concept IS NOT NULL AND methodology_ids_json IS NOT NULL",
}


def _confidence_for_n(n: int) -> str:
    for threshold, label in CONFIDENCE_N_THRESHOLDS:
        if n >= threshold:
            return label
    return "LOW"


class AdaptiveAgent(PredictionAgent):
    def __init__(self, conn: sqlite3.Connection, unconditional_baseline: dict[int, float],
                 model_version: str = "ADAPTIVE_v1", concept_filter: str | None = None,
                 qualified_relationship_ids: set[str] | None = None):
        if concept_filter not in CONCEPT_FILTER_CLAUSES:
            raise ValueError(f"concept_filter={concept_filter!r} is not valid - must be one of "
                              f"{list(CONCEPT_FILTER_CLAUSES)}")
        self.conn = conn
        self.unconditional_baseline = dict(unconditional_baseline)
        self.model_version = model_version
        self.concept_filter = concept_filter
        self.qualified_relationship_ids = qualified_relationship_ids

    def predict(self, event: EventRecord, horizon_days: int, predicted_at: datetime) -> PredictionRecord:
        match = self._best_matching_relationship(event, horizon_days)
        if match is not None:
            return PredictionRecord(
                horizon_days=horizon_days, predicted_impact=match["effect_estimate"],
                predicted_confidence=_confidence_for_n(match["n_supporting"]),
                basis={"basis": "validated_relationship", "relationship_id": match["relationship_id"],
                       "n_supporting": match["n_supporting"]},
                model_version=self.model_version, predicted_at=predicted_at,
            )

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

    def _best_matching_relationship(self, event: EventRecord, horizon_days: int) -> sqlite3.Row | None:
        extra_clause = CONCEPT_FILTER_CLAUSES[self.concept_filter]
        params: list = [horizon_days, event.event_type, event.direction]
        if self.qualified_relationship_ids is not None:
            if not self.qualified_relationship_ids:
                return None  # an empty qualified set means nothing is eligible - never silently unrestricted
            placeholders = ",".join("?" for _ in self.qualified_relationship_ids)
            extra_clause += f" AND relationship_id IN ({placeholders})"
            params.extend(self.qualified_relationship_ids)
        rows = self.conn.execute(
            f"""SELECT * FROM validated_relationships
               WHERE status = 'ACTIVE' AND horizon_days = ?
                 AND json_extract(condition_json, '$.event_type') = ?
                 AND json_extract(condition_json, '$.direction') = ?{extra_clause}""",
            params,
        ).fetchall()
        if not rows:
            return None
        candidates = [r for r in rows if _condition_matches_context(json.loads(r["condition_json"]), event.context)]
        if not candidates:
            return None
        # Most specific (most conditioning keys beyond event_type/direction) wins; ties broken by largest N.
        candidates.sort(key=lambda r: (-_specificity(r), -r["n_supporting"]))
        return candidates[0]


def _specificity(row: sqlite3.Row) -> int:
    return len(json.loads(row["condition_json"]))


def _condition_matches_context(condition: dict, context: dict) -> bool:
    for key, value in condition.items():
        if key in ("event_type", "direction"):
            continue  # already filtered in SQL
        if context.get(key) != value:
            return False
    return True
