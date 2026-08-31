"""The revalidation scheduler - Blueprint section M: every ACTIVE
validated_relationship is periodically re-tested against ALL matching
history up to the current point, with the same rigor as a brand-new
hypothesis. A relationship that stops replicating is RETIRED, never
deleted; episodic_events (the permanent historical record) is never
touched by this module at all - only validated_relationships' `status`
and `n_supporting`/`effect_estimate` fields, and model_registry gain an
entry either way.

Since test_relationship() re-tests against ALL matching resolved history
(not just data accumulated since the last revalidation pass), N can only
grow between revalidation runs - a relationship that had enough evidence
to be promoted in the first place should never fall back below MIN_N on
a later check. REJECTED_INSUFFICIENT_N is handled the same as any other
non-CONFIRMED outcome (retirement) here for that reason, not because
insufficient-evidence and disconfirmed-evidence are treated as
equivalent findings in general (learn/hypothesis_testing.py's
HypothesisTestResult.status still distinguishes them, and that
distinction is preserved in the registry entry's statistical_tests_json).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime

from market_agent.learn import governance
from market_agent.learn.hypothesis_testing import test_relationship


def run_revalidation_pass(conn: sqlite3.Connection, unconditional_baseline: dict[int, float],
                           promoted_by: str, clock_now: datetime) -> list[dict]:
    """Re-tests every ACTIVE relationship once. Returns one summary dict
    per relationship: {relationship_id, previous_status, new_status,
    n_before, n_after}. Intended to be called on the blueprint's quarterly
    cadence (see the experiment harness / a future scheduled-task wiring)
    - this function itself has no notion of scheduling, it just does one
    pass whenever called, which keeps it trivially testable."""
    active_rows = conn.execute("SELECT * FROM validated_relationships WHERE status = 'ACTIVE'").fetchall()
    summary = []
    for row in active_rows:
        result = test_relationship(conn, row, unconditional_baseline, clock_now)
        new_status = governance.revalidate(conn, row["relationship_id"], result, promoted_by, clock_now)
        summary.append({
            "relationship_id": row["relationship_id"], "previous_status": "ACTIVE", "new_status": new_status,
            "n_before": row["n_supporting"], "n_after": result.n, "test_status": result.status,
        })
    return summary
