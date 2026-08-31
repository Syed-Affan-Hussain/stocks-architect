"""The only code path allowed to write to validated_relationships
(category 3) or model_registry. Every promotion, rejection outcome, and
retirement is recorded here with a reason, evidence, and (for promotions/
retirements) a full registry entry - Blueprint section N.

Nothing upstream of this module (interpretation, retrieval, prediction,
hypothesis generation) can reach category 3 directly - that's enforced
simply by nobody else importing store.db's upsert_relationship /
register_change functions, which is why this module exists as a single,
narrow choke point rather than scattering promotion logic elsewhere.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime

from market_agent.learn.hypothesis_testing import HypothesisTestResult
from market_agent.store import db


def _find_existing_active_relationship(conn: sqlite3.Connection, condition: dict, horizon_days: int) -> str | None:
    """Two hypotheses proposing the SAME condition (e.g. two different
    WRONG_DIRECTION events both happening under regime=NORMAL) must
    strengthen ONE relationship, not fragment into near-duplicate rows
    with independently-computed effect estimates that then compete at
    prediction time. Found necessary running against real data: the
    rule-based hypothesis generator's one fixed shape (condition on
    current regime) means many different triggering events legitimately
    produce the identical condition, and confirmed hypotheses of the
    identical condition were previously each creating a brand-new
    relationship_id via uuid4() - correct individually, wasteful and
    confusing in aggregate (hundreds of rows saying the same thing with
    different N as of when each happened to be tested). Matches
    regardless of current status (SHADOW/ACTIVE/RETIRED) - a RETIRED
    relationship rediscovered by a fresh, independently-tested
    confirmation re-enters SHADOW probation on the SAME relationship_id
    (apply_test_results always sets status="SHADOW" on upsert), rather
    than silently reviving straight to ACTIVE or fragmenting into a
    second row for a condition already known once."""
    rows = conn.execute(
        "SELECT relationship_id, condition_json FROM validated_relationships WHERE horizon_days = ?",
        (horizon_days,)).fetchall()
    for row in rows:
        if json.loads(row["condition_json"]) == condition:
            return row["relationship_id"]
    return None


def apply_test_results(conn: sqlite3.Connection, results: list[HypothesisTestResult], promoted_by: str,
                        clock_now: datetime) -> list[str]:
    """Applies a batch of already-corrected test results: CONFIRMED
    hypotheses enter SHADOW status - not ACTIVE - reusing an existing
    relationship_id for the SAME condition if one already exists (see
    _find_existing_active_relationship, which now also matches SHADOW
    rows so a relationship already on probation gets strengthened rather
    than duplicated), otherwise creating a new one, plus a SHADOW registry
    entry either way. A SHADOW relationship has passed its historical test
    but has not yet influenced a single live prediction - see
    learn/shadow.py for the probation period and the promotion to ACTIVE.
    Everything else gets marked REJECTED on the hypothesis row itself,
    with the rejection reason preserved there permanently (a rejected
    hypothesis is never deleted or retried silently - see
    candidate_hypotheses' own permanence). Returns the list of
    SHADOW-entered relationship_ids (may contain repeats if multiple
    hypotheses in this batch confirmed the same condition and all
    strengthened the same relationship)."""
    promoted_ids = []
    for r in results:
        hyp_row = conn.execute("SELECT * FROM candidate_hypotheses WHERE hypothesis_id = ?",
                                (r.hypothesis_id,)).fetchone()
        db.set_hypothesis_result(conn, r.hypothesis_id, "CONFIRMED" if r.status == "CONFIRMED" else "REJECTED",
                                  tested_at=clock_now,
                                  test_result=_result_to_dict(r))
        if r.status != "CONFIRMED":
            continue

        condition = json.loads(hyp_row["condition_json"])
        horizon_days = hyp_row["horizon_days"]
        existing_id = _find_existing_active_relationship(conn, condition, horizon_days)
        relationship_id = existing_id or str(uuid.uuid4())
        action = "STRENGTHEN_RELATIONSHIP" if existing_id else "CREATE_RELATIONSHIP"
        # Stage 6/7 provenance carried through from the hypothesis row - without this, a confirmed
        # technical-concept or methodology-backed hypothesis would silently lose its `concept`/
        # `methodology_ids` tag on promotion, making it invisible to agents/variants.py's
        # ANY_TECHNICAL_CONCEPT/METHODOLOGY_BACKED_CONCEPT filters (both query concept IS NOT NULL).
        methodology_ids = json.loads(hyp_row["methodology_ids_json"]) if hyp_row["methodology_ids_json"] else None
        db.upsert_relationship(
            conn, relationship_id=relationship_id, condition=condition, horizon_days=horizon_days,
            effect_estimate=r.mean_effect, ci_low=r.ci_low, ci_high=r.ci_high, n_supporting=r.n, status="SHADOW",
            source_hypothesis_id=r.hypothesis_id, created_at=clock_now,
            last_revalidated_at=clock_now if existing_id else None,
            shadow_started_at=clock_now if not existing_id else None,
            concept=hyp_row["concept"], methodology_ids=methodology_ids,
        )
        db.register_change(
            conn, version_id=str(uuid.uuid4()), reason=f"Hypothesis {r.hypothesis_id} confirmed - entering shadow",
            change={"action": action, "relationship_id": relationship_id, "condition": condition},
            performance_before=None, performance_after=None, statistical_tests=_result_to_dict(r),
            promoted_by=promoted_by, promotion_status="SHADOW", created_at=clock_now,
        )
        promoted_ids.append(relationship_id)
    return promoted_ids


def revalidate(conn: sqlite3.Connection, relationship_id: str, new_result: HypothesisTestResult,
               promoted_by: str, clock_now: datetime) -> str:
    """Quarterly re-validation (Blueprint section M) of an already-ACTIVE
    relationship, using the SAME test used for a brand-new hypothesis - no
    special leniency for something already promoted. If it no longer
    passes, it is marked RETIRED (never deleted) with a full registry
    entry explaining why and when."""
    row = conn.execute("SELECT * FROM validated_relationships WHERE relationship_id = ?",
                        (relationship_id,)).fetchone()
    if row is None:
        raise KeyError(f"No validated_relationships row with relationship_id={relationship_id!r}")

    if new_result.status == "CONFIRMED":
        db.upsert_relationship(
            conn, relationship_id=relationship_id, condition=json.loads(row["condition_json"]),
            horizon_days=row["horizon_days"], effect_estimate=new_result.mean_effect,
            ci_low=new_result.ci_low, ci_high=new_result.ci_high,
            n_supporting=new_result.n, status="ACTIVE", source_hypothesis_id=row["source_hypothesis_id"],
            created_at=row["created_at"], last_revalidated_at=clock_now,
        )
        new_status = "ACTIVE"
    else:
        db.upsert_relationship(
            conn, relationship_id=relationship_id, condition=json.loads(row["condition_json"]),
            horizon_days=row["horizon_days"], effect_estimate=row["effect_estimate"], ci_low=row["ci_low"],
            ci_high=row["ci_high"], n_supporting=row["n_supporting"], status="RETIRED",
            source_hypothesis_id=row["source_hypothesis_id"], created_at=row["created_at"],
            last_revalidated_at=clock_now,
        )
        new_status = "RETIRED"

    db.register_change(
        conn, version_id=str(uuid.uuid4()), reason=f"Quarterly revalidation of {relationship_id}: {new_status}",
        change={"action": "REVALIDATE_RELATIONSHIP", "relationship_id": relationship_id, "new_status": new_status},
        performance_before={"n_supporting": row["n_supporting"], "effect_estimate": row["effect_estimate"]},
        performance_after=_result_to_dict(new_result), statistical_tests=_result_to_dict(new_result),
        promoted_by=promoted_by, promotion_status="PROMOTED" if new_status == "ACTIVE" else "ROLLED_BACK",
        created_at=clock_now,
    )
    return new_status


def promote_from_shadow(conn: sqlite3.Connection, relationship_id: str, shadow_result: HypothesisTestResult,
                         promoted_by: str, clock_now: datetime) -> str:
    """The only path from status='SHADOW' to status='ACTIVE' (learn/
    shadow.py calls this once a shadow relationship has accumulated
    enough new, disjoint evidence). If the new evidence doesn't support
    it, retires directly from SHADOW - this relationship never influenced
    a single live prediction, and that fact is preserved in the registry
    entry's reason text, distinct from an ACTIVE relationship later being
    retired by ordinary revalidation."""
    row = conn.execute("SELECT * FROM validated_relationships WHERE relationship_id = ?",
                        (relationship_id,)).fetchone()
    if row is None:
        raise KeyError(f"No validated_relationships row with relationship_id={relationship_id!r}")
    if row["status"] != "SHADOW":
        raise ValueError(f"promote_from_shadow called on relationship_id={relationship_id!r} with status "
                          f"{row['status']!r}, not SHADOW - use revalidate() for an already-ACTIVE relationship.")

    if shadow_result.status == "CONFIRMED":
        db.upsert_relationship(
            conn, relationship_id=relationship_id, condition=json.loads(row["condition_json"]),
            horizon_days=row["horizon_days"], effect_estimate=shadow_result.mean_effect,
            ci_low=shadow_result.ci_low, ci_high=shadow_result.ci_high, n_supporting=shadow_result.n,
            status="ACTIVE", source_hypothesis_id=row["source_hypothesis_id"], created_at=row["created_at"],
            last_revalidated_at=clock_now, shadow_promoted_at=clock_now,
        )
        new_status, reason = "ACTIVE", f"Shadow probation passed on new evidence - promoting {relationship_id}"
    else:
        db.upsert_relationship(
            conn, relationship_id=relationship_id, condition=json.loads(row["condition_json"]),
            horizon_days=row["horizon_days"], effect_estimate=row["effect_estimate"], ci_low=row["ci_low"],
            ci_high=row["ci_high"], n_supporting=row["n_supporting"], status="RETIRED",
            source_hypothesis_id=row["source_hypothesis_id"], created_at=row["created_at"],
            last_revalidated_at=clock_now,
        )
        new_status = "RETIRED"
        reason = f"Shadow probation failed on new evidence - retiring {relationship_id} (never went live)"

    db.register_change(
        conn, version_id=str(uuid.uuid4()), reason=reason,
        change={"action": "SHADOW_PROMOTION_DECISION", "relationship_id": relationship_id, "new_status": new_status},
        performance_before={"n_supporting": row["n_supporting"], "effect_estimate": row["effect_estimate"]},
        performance_after=_result_to_dict(shadow_result), statistical_tests=_result_to_dict(shadow_result),
        promoted_by=promoted_by, promotion_status="PROMOTED" if new_status == "ACTIVE" else "ROLLED_BACK",
        created_at=clock_now,
    )
    return new_status


def _result_to_dict(r: HypothesisTestResult) -> dict:
    return {"status": r.status, "n": r.n, "mean_effect": r.mean_effect, "baseline_effect": r.baseline_effect,
            "p_value": r.p_value, "p_value_corrected": r.p_value_corrected, "ci_low": r.ci_low,
            "ci_high": r.ci_high, "evidence": r.evidence}
