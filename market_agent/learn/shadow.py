"""Shadow deployment - Blueprint section 21's gold-standard safeguard,
not yet implemented in stages 1-3: a confirmed relationship must prove
itself on genuinely NEW data - accumulated strictly AFTER it entered
probation - before it is trusted to influence a single live prediction.

This is the one safeguard immune to leakage-through-implementation-bugs
in a way that even careful historical hypothesis testing can't fully
guarantee: the shadow-period evidence is, by construction, data the
relationship's own original confirmation never touched, and is disjoint
from the data used to confirm it in the first place (published_at >=
shadow_started_at, versus the original test's published_at < the
triggering event). A relationship that only looked good because of a
subtle point-in-time bug or a lucky historical sample is caught here,
prospectively, not just argued away.

AdaptiveAgent (agents/adaptive_agent.py) only ever queries status='ACTIVE'
relationships - a SHADOW relationship has zero effect on any live
prediction until this module promotes it.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from market_agent.learn import governance
from market_agent.learn.hypothesis_testing import _run_significance_test
from market_agent.store import db

MIN_SHADOW_OBSERVATIONS = 10  # new, disjoint matching resolved observations required before a
#                                 shadow relationship is even eligible for a promotion decision -
#                                 deliberately smaller than MIN_N (the original-confirmation
#                                 threshold) since this is a confirmatory check on top of an already-
#                                 passed test, not a first-time discovery.


def evaluate_shadow_relationships(conn: sqlite3.Connection, unconditional_baseline: dict[int, float],
                                   promoted_by: str, clock_now: datetime) -> list[dict]:
    """Checks every SHADOW relationship. Promotes to ACTIVE if enough new,
    disjoint evidence has accumulated AND that new evidence alone still
    passes the same significance/economic-effect test used everywhere
    else in this system; retires directly from SHADOW (never having
    influenced a live prediction) if the new evidence contradicts it;
    leaves it in SHADOW, unresolved, if not enough new evidence exists
    yet - "not enough new evidence" is never treated as either a pass or
    a fail."""
    clock_now_iso = clock_now.isoformat() if hasattr(clock_now, "isoformat") else clock_now
    summary = []
    for row in db.shadow_relationships(conn):
        condition = json.loads(row["condition_json"])
        horizon_days = row["horizon_days"]
        new_rows = _new_shadow_evidence(conn, condition, horizon_days, row["shadow_started_at"], clock_now_iso)

        if len(new_rows) < MIN_SHADOW_OBSERVATIONS:
            summary.append({"relationship_id": row["relationship_id"], "outcome": "STILL_IN_SHADOW",
                             "n_new_observations": len(new_rows)})
            continue

        result = _run_significance_test(
            row["relationship_id"], new_rows, condition, unconditional_baseline.get(horizon_days, 0.0),
            f"(NEW evidence only, published >= shadow start {row['shadow_started_at']}, disjoint from the "
            "evidence that originally confirmed this relationship)")
        new_status = governance.promote_from_shadow(conn, row["relationship_id"], result, promoted_by, clock_now)
        summary.append({"relationship_id": row["relationship_id"], "outcome": new_status,
                         "n_new_observations": len(new_rows), "test_status": result.status})
    return summary


def _new_shadow_evidence(conn: sqlite3.Connection, condition: dict, horizon_days: int, shadow_started_at: str,
                          as_of_iso: str) -> list[sqlite3.Row]:
    rows = conn.execute(
        """SELECT * FROM episodic_events
           WHERE outcome_locked = 1 AND realized_abnormal_return IS NOT NULL AND horizon_days = ?
             AND event_type = ? AND direction = ? AND published_at >= ? AND published_at < ?""",
        (horizon_days, condition["event_type"], condition["direction"], shadow_started_at, as_of_iso),
    ).fetchall()
    extra_keys = {k: v for k, v in condition.items() if k not in ("event_type", "direction")}
    if extra_keys:
        rows = [r for r in rows if all(json.loads(r["context_json"]).get(k) == v for k, v in extra_keys.items())]
    # CRITICAL: same fix as learn/hypothesis_testing.py::_matching_prior_rows - without this,
    # MIN_SHADOW_OBSERVATIONS (10) could be satisfied by as few as 3 real disjoint events logged by
    # 4 agents each, not 10 genuinely new observations.
    return db.deduplicate_by_real_event(rows)
