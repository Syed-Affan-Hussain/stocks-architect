"""Structured retrieval over episodic memory (Blueprint section E).

Stage 1 implements the structured-filter half of the hybrid design only -
same event_type, same regime, comparable prior-return bucket. The
embedding-similarity second pass (for nuance a hard filter can't express)
is deliberately deferred: it needs an embedding model and enough volume
to be worth the added complexity, and the blueprint itself (MVP section)
says to defer vector search past the first working slice. Retrieval here
still does real, useful work on its own - it's what the hypothesis-testing
step (learn/hypothesis_testing.py) actually queries.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from market_agent.store.db import deduplicate_by_real_event

PRIOR_RETURN_BUCKETS = [(-1.0, -0.08, "LARGE_DECLINE"), (-0.08, -0.02, "MILD_DECLINE"),
                         (-0.02, 0.02, "FLAT"), (0.02, 0.08, "MILD_GAIN"), (0.08, 1.0, "LARGE_GAIN")]

# Stage 5: a second real bucketing dimension for hypothesis conditioning (learn/hypothesis.py),
# distinct from regime/prior_return_bucket rather than redundant with them - realized vol captures
# "how turbulent has this specific stock been", which a broad-market regime signal or the stock's
# own directional drift does not.
VOL_BUCKETS = [(0.0, 0.015, "LOW_VOL"), (0.015, 0.035, "NORMAL_VOL"), (0.035, 10.0, "HIGH_VOL")]


def prior_return_bucket(prior_5d_return: float | None) -> str:
    if prior_5d_return is None:
        return "UNKNOWN"
    for lo, hi, label in PRIOR_RETURN_BUCKETS:
        if lo <= prior_5d_return < hi:
            return label
    return "UNKNOWN"


def vol_bucket(realized_vol_20d: float | None) -> str:
    if realized_vol_20d is None:
        return "UNKNOWN"
    for lo, hi, label in VOL_BUCKETS:
        if lo <= realized_vol_20d < hi:
            return label
    return "UNKNOWN"


@dataclass
class SimilarCase:
    event_id: str
    published_at: str
    entity: str
    predicted_impact: float | None
    realized_abnormal_return: float | None
    error_type: str | None


def find_similar_cases(conn: sqlite3.Connection, event_type: str, regime: str,
                        prior_5d_return: float | None, horizon_days: int,
                        published_before: str | None = None,
                        outcome_known_only: bool = True) -> list[SimilarCase]:
    """Structured-filter retrieval: same event_type + same regime + same
    prior-return bucket + same horizon. `published_before` enforces
    point-in-time correctness on the read - a caller running inside a
    historical simulation MUST pass the simulation clock's current time
    here, never leave it None outside of live/current-time use."""
    bucket = prior_return_bucket(prior_5d_return)
    clauses = ["event_type = ?", "horizon_days = ?", "json_extract(context_json, '$.regime') = ?"]
    params: list = [event_type, horizon_days, regime]
    if outcome_known_only:
        # see store/db.py::query_events - outcome_locked=1 alone can still mean a NULL numeric
        # outcome (DATA_ERROR, e.g. a delisted ticker).
        clauses.append("outcome_locked = 1 AND realized_abnormal_return IS NOT NULL")
    if published_before is not None:
        clauses.append("published_at < ?")
        params.append(published_before)
    where = " AND ".join(clauses)
    rows = conn.execute(f"SELECT * FROM episodic_events WHERE {where} ORDER BY published_at", params).fetchall()

    matched = [r for r in rows if prior_return_bucket(_extract_prior_return(r)) == bucket]
    # CRITICAL: same fix as learn/hypothesis_testing.py::_matching_prior_rows - without this, the
    # same real event logged by multiple agents inflates the "similar cases" count, understating
    # novelty_score (pipeline.py: 1/(1+len(similar))) and double-counting retrieved-case evidence.
    matched = deduplicate_by_real_event(matched)
    return [SimilarCase(event_id=r["event_id"], published_at=r["published_at"], entity=r["entity"],
                         predicted_impact=r["predicted_impact"], realized_abnormal_return=r["realized_abnormal_return"],
                         error_type=r["error_type"]) for r in matched]


def _extract_prior_return(row: sqlite3.Row) -> float | None:
    import json
    ctx = json.loads(row["context_json"])
    return ctx.get("prior_5d_return")
