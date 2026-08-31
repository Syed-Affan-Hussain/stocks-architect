"""The live knowledge-state report - a read-only view over the six-
category store (category 1 is never written here, only queried). Answers
"what does the system currently believe, and how well-supported is it" -
item 8 of stage 4.

DECAY STATE, disclosed thresholds: a relationship is FRESH if revalidated
within REVALIDATION_DUE_DAYS, DUE_FOR_REVALIDATION between that and
REVALIDATION_OVERDUE_DAYS, OVERDUE beyond that, or NEVER_REVALIDATED if
`last_revalidated_at` is still null (freshly promoted, hasn't hit a
quarterly check yet). These are the same rough cadence as the walk-
forward harness's own quarterly revalidation trigger - not independently
tuned, not claimed to be optimal, just consistent with the rest of the
system. This is TIME-based staleness (`RelationshipSummary.decay_state`,
unchanged from stage 4) - see DECAYING below for a separate,
PERFORMANCE-based warning signal stage 6 adds.

STAGE 6 - "DECAYING" AS A REPORTING CLASSIFICATION, NOT A GOVERNANCE
STATUS: nothing in learn/governance.py or learn/revalidation.py ever
writes a "DECAYING" status to validated_relationships.status - that
column stays exactly SHADOW | ACTIVE | RETIRED, unchanged (see
store/schema.py). A relationship whose real, out-of-sample track record
since promotion (n_predictions_contradicted / total >
DECAY_WARNING_CONTRADICTION_RATE, with at least DECAY_WARNING_MIN_N
predictions to make the ratio meaningful - both fixed, disclosed
thresholds) is flagged `is_decaying=True` here, in the REPORT ONLY. This
NEVER changes the relationship's real status or retires it automatically
- that stays governed exclusively by learn/revalidation.py's real
statistical re-test (see that module's own docstring for why an
insufficient-evidence-on-retest and a disconfirmed-on-retest are both
retired the same way, but neither happens silently or early). DECAYING
is an early-warning surfaced to a human/dashboard reader, never a
self-triggering action.

CONCEPT-LEVEL AND METHODOLOGY-LEVEL SECTIONS (stage 6): see
_build_concept_summaries/_build_methodology_summaries below for the
UNTESTED/REJECTED/SHADOW/ACTIVE/DECAYING/RETIRED breakdown per canonical
trading concept, and per methodology, with sample size, effect estimate,
confidence interval, out-of-sample performance, and provenance.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime

from market_agent.concepts.ontology import TradingConcept

REVALIDATION_DUE_DAYS = 100
REVALIDATION_OVERDUE_DAYS = 200

DECAY_WARNING_MIN_N = 5             # fewer than this many live predictions and the contradiction
#                                      rate isn't meaningful yet - fixed, disclosed, not tuned.
DECAY_WARNING_CONTRADICTION_RATE = 0.5  # more than half of live predictions using this relationship
#                                          have been a learnable error -> flagged, never auto-retired.


@dataclass
class RelationshipSummary:
    relationship_id: str
    condition: dict
    status: str
    horizon_days: int
    effect_estimate: float
    ci_low: float | None
    ci_high: float | None
    n_supporting: int
    created_at: str
    last_revalidated_at: str | None
    shadow_started_at: str | None
    shadow_promoted_at: str | None
    decay_state: str
    n_predictions_supported: int
    n_predictions_contradicted: int
    is_decaying: bool          # see module docstring - performance-based warning, never a stored status
    concept: str | None
    methodology_ids: list[str] = field(default_factory=list)


@dataclass
class SourceReliabilitySummary:
    source: str
    n_resolved_predictions: int
    n_learnable_errors: int
    hit_rate: float | None  # fraction of resolved predictions NOT classified as a learnable error


@dataclass
class HorizonCalibration:
    """Direction accuracy and mean absolute error for one horizon, split by
    agent (STATIC vs ADAPTIVE) - the per-horizon slice item 8 explicitly
    asks for, computed with the SAME machinery (experiment/metrics.py) the
    governed walk-forward evaluation uses, so a number here means the same
    thing it means everywhere else in this system."""
    horizon_days: int
    static_n: int
    static_direction_accuracy: float | None
    static_mae: float | None
    adaptive_n: int
    adaptive_direction_accuracy: float | None
    adaptive_mae: float | None


@dataclass
class OperationalCounts:
    """Item 8's ledger-level operational snapshot - how much has actually
    happened, not what the system currently believes (that's the
    relationship summaries below). All counts are plain SELECTs over
    episodic_events/candidate_hypotheses/validated_relationships - nothing
    here is itself a statistical claim."""
    n_events_ingested: int                    # total episodic_events rows (= total predictions logged)
    n_events_resolved: int                    # outcome_locked=1 AND realized_abnormal_return IS NOT NULL
    n_events_insufficient_precedent: int      # predicted_confidence = 'INSUFFICIENT_PRECEDENT'
    n_events_data_error: int                  # resolved with error_type = 'DATA_ERROR'
    error_type_distribution: dict[str, int]   # error_type -> count, resolved rows only
    n_hypotheses_generated: int
    n_hypotheses_confirmed: int
    n_hypotheses_rejected: int
    n_hypotheses_untested: int
    n_relationships_active: int
    n_relationships_shadow: int
    n_relationships_retired: int
    n_governance_changes: int


@dataclass
class ConceptHypothesisEntry:
    hypothesis_id: str
    condition: dict
    status: str          # UNTESTED | TESTING | REJECTED | CONFIRMED
    n: int | None
    methodology_ids: list[str] = field(default_factory=list)


@dataclass
class ConceptKnowledgeSummary:
    """One canonical TradingConcept's full status breakdown - UNTESTED/
    REJECTED (from candidate_hypotheses) and SHADOW/ACTIVE/DECAYING/RETIRED
    (from validated_relationships, DECAYING derived per-relationship - see
    RelationshipSummary.is_decaying), each carrying sample size, effect
    estimate, confidence interval, out-of-sample performance
    (n_predictions_supported/contradicted), and provenance
    (methodology_ids). Always present for all 20 ontology concepts, even
    ones with zero hypotheses/relationships - that absence is itself the
    honest answer to "which concepts are UNTESTED" for a concept nothing
    has ever proposed a hypothesis about."""
    concept: str
    computable: bool
    shadow_relationships: list[RelationshipSummary] = field(default_factory=list)
    active_relationships: list[RelationshipSummary] = field(default_factory=list)
    decaying_relationships: list[RelationshipSummary] = field(default_factory=list)
    retired_relationships: list[RelationshipSummary] = field(default_factory=list)
    untested_hypotheses: list[ConceptHypothesisEntry] = field(default_factory=list)
    rejected_hypotheses: list[ConceptHypothesisEntry] = field(default_factory=list)
    contributing_methodologies: list[dict] = field(default_factory=list)  # {methodology_id, name, practitioner}


@dataclass
class MethodologyKnowledgeSummary:
    """A methodology's own provenance rollup - NEVER a statistical
    verdict on the methodology itself (see methodology/schema.py's module
    docstring: only real, out-of-sample market data ever validates a
    concept). `concepts_with_active_evidence`/`concepts_with_no_active_evidence`
    describe what happened to the CONCEPTS this methodology claimed, not
    a test performed on the methodology as a unit - a methodology can
    (and often does) claim multiple concepts with different fates."""
    methodology_id: str
    name: str
    practitioner: str
    source_type: str
    extractor_name: str
    ingested_at: str
    concepts_claimed: list[str] = field(default_factory=list)
    concepts_with_active_evidence: list[str] = field(default_factory=list)
    concepts_with_no_active_evidence: list[str] = field(default_factory=list)


@dataclass
class KnowledgeStateReport:
    generated_at: str
    knowledge_version: int
    llm_status: str
    active_relationships: list[RelationshipSummary] = field(default_factory=list)
    shadow_relationships: list[RelationshipSummary] = field(default_factory=list)
    retired_relationships: list[RelationshipSummary] = field(default_factory=list)
    rejected_hypotheses: list[dict] = field(default_factory=list)
    source_reliability: list[SourceReliabilitySummary] = field(default_factory=list)
    operational_counts: OperationalCounts | None = None
    calibration_by_horizon: list[HorizonCalibration] = field(default_factory=list)
    concepts: list[ConceptKnowledgeSummary] = field(default_factory=list)
    methodologies: list[MethodologyKnowledgeSummary] = field(default_factory=list)


def _decay_state(last_revalidated_at: str | None, now: datetime) -> str:
    if last_revalidated_at is None:
        return "NEVER_REVALIDATED"
    last_dt = datetime.fromisoformat(last_revalidated_at)
    # Defensive: stored timestamps are always tz-aware (store/db.py's _iso), but `now` is whatever
    # the caller passed - normalize both to naive-UTC-equivalent here rather than let a caller's
    # naive datetime.now() (as opposed to datetime.now(timezone.utc)) crash this comparison, found
    # running this report from a script that had made exactly that mistake.
    if last_dt.tzinfo is not None:
        last_dt = last_dt.replace(tzinfo=None)
    if now.tzinfo is not None:
        now = now.replace(tzinfo=None)
    age_days = (now - last_dt).total_seconds() / 86400.0
    if age_days <= REVALIDATION_DUE_DAYS:
        return "FRESH"
    if age_days <= REVALIDATION_OVERDUE_DAYS:
        return "DUE_FOR_REVALIDATION"
    return "OVERDUE"


def _prediction_support_counts(conn: sqlite3.Connection, relationship_id: str) -> tuple[int, int]:
    """(n_predictions_supported, n_predictions_contradicted) - every
    resolved episodic_events row whose prediction_basis_json points at
    this relationship_id, and of those, how many turned out to be a
    learnable error (WRONG_DIRECTION/WRONG_MAGNITUDE) - i.e. cases where
    trusting this relationship led the prediction astray."""
    rows = conn.execute(
        """SELECT * FROM episodic_events
           WHERE outcome_locked = 1 AND realized_abnormal_return IS NOT NULL
             AND json_extract(prediction_basis_json, '$.relationship_id') = ?""",
        (relationship_id,)).fetchall()
    # CRITICAL: same fix as learn/hypothesis_testing.py::_matching_prior_rows - a technical-concept
    # relationship that's ALSO methodology-backed can be independently selected by both
    # TECHNICAL_ADAPTIVE and METHODOLOGY_ADAPTIVE for the same real event, double-counting support.
    from market_agent.store.db import deduplicate_by_real_event
    rows = deduplicate_by_real_event(rows)
    supported = len(rows)
    contradicted = sum(1 for r in rows if r["error_type"] in ("WRONG_DIRECTION", "WRONG_MAGNITUDE"))
    return supported, contradicted


def _is_decaying(status: str, supported: int, contradicted: int) -> bool:
    if status != "ACTIVE" or supported < DECAY_WARNING_MIN_N:
        return False
    return (contradicted / supported) > DECAY_WARNING_CONTRADICTION_RATE


def _summarize_relationship(conn: sqlite3.Connection, row: sqlite3.Row, now: datetime) -> RelationshipSummary:
    supported, contradicted = _prediction_support_counts(conn, row["relationship_id"])
    methodology_ids = json.loads(row["methodology_ids_json"]) if row["methodology_ids_json"] else []
    return RelationshipSummary(
        relationship_id=row["relationship_id"], condition=json.loads(row["condition_json"]),
        status=row["status"], horizon_days=row["horizon_days"], effect_estimate=row["effect_estimate"],
        ci_low=row["ci_low"], ci_high=row["ci_high"], n_supporting=row["n_supporting"],
        created_at=row["created_at"], last_revalidated_at=row["last_revalidated_at"],
        shadow_started_at=row["shadow_started_at"], shadow_promoted_at=row["shadow_promoted_at"],
        decay_state=_decay_state(row["last_revalidated_at"], now),
        n_predictions_supported=supported, n_predictions_contradicted=contradicted,
        is_decaying=_is_decaying(row["status"], supported, contradicted),
        concept=row["concept"], methodology_ids=methodology_ids,
    )


def _llm_status() -> str:
    """Item 8's explicit 'LLM-connection status' field. Reads the same
    env vars llm/select.py's provider switch reads (default 'rule_based')
    rather than assuming - a deployment could set INTERPRETER_PROVIDER
    without HYPOTHESIS_PROVIDER, or vice versa. Also checks whether an
    LLM SDK is even importable in this environment, so the report
    distinguishes "configured for rule-based" from "configured for LLM
    but no SDK present" if that combination is ever reached."""
    import os
    interpreter_provider = os.environ.get("INTERPRETER_PROVIDER", "rule_based").strip().lower()
    hypothesis_provider = os.environ.get("HYPOTHESIS_PROVIDER", "rule_based").strip().lower()
    sdk_available = False
    try:
        import anthropic  # noqa: F401
        sdk_available = True
    except ImportError:
        pass
    parts = [f"INTERPRETER_PROVIDER={interpreter_provider}", f"HYPOTHESIS_PROVIDER={hypothesis_provider}",
             f"anthropic SDK installed: {sdk_available}"]
    if interpreter_provider == "rule_based" and hypothesis_provider == "rule_based":
        parts.append("NO LLM reasoning is occurring in this deployment - both providers are rule-based stand-ins.")
    return " | ".join(parts)


def _operational_counts(conn: sqlite3.Connection) -> OperationalCounts:
    def _count(where: str = "") -> int:
        return conn.execute(f"SELECT COUNT(*) c FROM episodic_events {where}").fetchone()["c"]

    error_rows = conn.execute(
        "SELECT error_type, COUNT(*) c FROM episodic_events WHERE outcome_locked = 1 AND error_type IS NOT NULL "
        "GROUP BY error_type").fetchall()

    def _hyp_count(status: str | None = None) -> int:
        where = f"WHERE status = '{status}'" if status else ""
        return conn.execute(f"SELECT COUNT(*) c FROM candidate_hypotheses {where}").fetchone()["c"]

    def _rel_count(status: str) -> int:
        return conn.execute("SELECT COUNT(*) c FROM validated_relationships WHERE status = ?",
                             (status,)).fetchone()["c"]

    from market_agent.store.db import count_governance_changes
    return OperationalCounts(
        n_events_ingested=_count(),
        n_events_resolved=_count("WHERE outcome_locked = 1 AND realized_abnormal_return IS NOT NULL"),
        n_events_insufficient_precedent=_count("WHERE predicted_confidence = 'INSUFFICIENT_PRECEDENT'"),
        n_events_data_error=_count("WHERE outcome_locked = 1 AND error_type = 'DATA_ERROR'"),
        error_type_distribution={r["error_type"]: r["c"] for r in error_rows},
        n_hypotheses_generated=_hyp_count(), n_hypotheses_confirmed=_hyp_count("CONFIRMED"),
        n_hypotheses_rejected=_hyp_count("REJECTED"), n_hypotheses_untested=_hyp_count("UNTESTED"),
        n_relationships_active=_rel_count("ACTIVE"), n_relationships_shadow=_rel_count("SHADOW"),
        n_relationships_retired=_rel_count("RETIRED"), n_governance_changes=count_governance_changes(conn),
    )


def _calibration_by_horizon(conn: sqlite3.Connection) -> list[HorizonCalibration]:
    """Per-horizon STATIC vs ADAPTIVE direction accuracy/MAE, computed with
    the SAME experiment/metrics.py machinery the governed walk-forward
    evaluation uses - so a number here means the same thing it means
    everywhere else, not a second, competing definition of accuracy."""
    from market_agent.experiment.metrics import PredictionOutcome, compute_metrics

    horizons = [r["horizon_days"] for r in
                conn.execute("SELECT DISTINCT horizon_days FROM episodic_events ORDER BY horizon_days").fetchall()]

    def _metrics_for(horizon_days: int, prefix: str):
        rows = conn.execute(
            """SELECT predicted_impact, predicted_confidence, realized_abnormal_return FROM episodic_events
               WHERE horizon_days = ? AND model_version LIKE ? AND outcome_locked = 1
                 AND realized_abnormal_return IS NOT NULL""",
            (horizon_days, f"{prefix}%")).fetchall()
        return compute_metrics([PredictionOutcome(r["predicted_impact"], r["predicted_confidence"],
                                                    r["realized_abnormal_return"]) for r in rows])

    results = []
    for h in horizons:
        static_metrics = _metrics_for(h, "STATIC")
        adaptive_metrics = _metrics_for(h, "ADAPTIVE")
        results.append(HorizonCalibration(
            horizon_days=h, static_n=static_metrics.n, static_direction_accuracy=static_metrics.direction_accuracy,
            static_mae=static_metrics.mae, adaptive_n=adaptive_metrics.n,
            adaptive_direction_accuracy=adaptive_metrics.direction_accuracy, adaptive_mae=adaptive_metrics.mae,
        ))
    return results


def _source_reliability(conn: sqlite3.Connection) -> list[SourceReliabilitySummary]:
    sources = conn.execute("SELECT DISTINCT source FROM episodic_events").fetchall()
    summaries = []
    for s in sources:
        source = s["source"]
        rows = conn.execute(
            """SELECT error_type FROM episodic_events
               WHERE source = ? AND outcome_locked = 1 AND realized_abnormal_return IS NOT NULL""",
            (source,)).fetchall()
        n = len(rows)
        n_errors = sum(1 for r in rows if r["error_type"] in ("WRONG_DIRECTION", "WRONG_MAGNITUDE"))
        hit_rate = (1 - n_errors / n) if n > 0 else None
        summaries.append(SourceReliabilitySummary(source=source, n_resolved_predictions=n,
                                                    n_learnable_errors=n_errors, hit_rate=hit_rate))
    return summaries


def _build_concept_summaries(conn: sqlite3.Connection, now: datetime) -> list[ConceptKnowledgeSummary]:
    from market_agent.concepts.ontology import CONCEPT_REGISTRY
    from market_agent.store.db import methodologies_for_concept

    summaries = []
    for concept in TradingConcept:
        summary = ConceptKnowledgeSummary(concept=concept.value, computable=CONCEPT_REGISTRY[concept].computable)

        for row in conn.execute("SELECT * FROM validated_relationships WHERE concept = ?",
                                 (concept.value,)).fetchall():
            rel_summary = _summarize_relationship(conn, row, now)
            if rel_summary.status == "SHADOW":
                summary.shadow_relationships.append(rel_summary)
            elif rel_summary.status == "ACTIVE":
                (summary.decaying_relationships if rel_summary.is_decaying else summary.active_relationships) \
                    .append(rel_summary)
            elif rel_summary.status == "RETIRED":
                summary.retired_relationships.append(rel_summary)

        for row in conn.execute("SELECT * FROM candidate_hypotheses WHERE concept = ?", (concept.value,)).fetchall():
            test_result = json.loads(row["test_result_json"]) if row["test_result_json"] else {}
            methodology_ids = json.loads(row["methodology_ids_json"]) if row["methodology_ids_json"] else []
            entry = ConceptHypothesisEntry(hypothesis_id=row["hypothesis_id"], condition=json.loads(row["condition_json"]),
                                            status=row["status"], n=test_result.get("n"),
                                            methodology_ids=methodology_ids)
            if row["status"] == "UNTESTED":
                summary.untested_hypotheses.append(entry)
            elif row["status"] == "REJECTED":
                summary.rejected_hypotheses.append(entry)

        summary.contributing_methodologies = [
            {"methodology_id": r["methodology_id"], "name": r["name"], "practitioner": r["practitioner"]}
            for r in methodologies_for_concept(conn, concept.value)]
        summaries.append(summary)
    return summaries


def _build_methodology_summaries(conn: sqlite3.Connection) -> list[MethodologyKnowledgeSummary]:
    from market_agent.store.db import all_methodologies, concept_links_for_methodology

    active_concepts = {r["concept"] for r in
                        conn.execute("SELECT DISTINCT concept FROM validated_relationships "
                                     "WHERE status = 'ACTIVE' AND concept IS NOT NULL").fetchall()}

    summaries = []
    for row in all_methodologies(conn):
        links = concept_links_for_methodology(conn, row["methodology_id"])
        claimed = sorted({link["concept"] for link in links})
        with_active = [c for c in claimed if c in active_concepts]
        without_active = [c for c in claimed if c not in active_concepts]
        summaries.append(MethodologyKnowledgeSummary(
            methodology_id=row["methodology_id"], name=row["name"], practitioner=row["practitioner"],
            source_type=row["source_type"], extractor_name=row["extractor_name"], ingested_at=row["ingested_at"],
            concepts_claimed=claimed, concepts_with_active_evidence=with_active,
            concepts_with_no_active_evidence=without_active,
        ))
    return summaries


def build_knowledge_state_report(conn: sqlite3.Connection, now: datetime | None = None) -> KnowledgeStateReport:
    now = now or datetime.now()
    from market_agent.store.db import count_governance_changes
    report = KnowledgeStateReport(generated_at=now.isoformat(), knowledge_version=count_governance_changes(conn),
                                   llm_status=_llm_status())

    for row in conn.execute("SELECT * FROM validated_relationships WHERE status = 'ACTIVE'").fetchall():
        report.active_relationships.append(_summarize_relationship(conn, row, now))
    for row in conn.execute("SELECT * FROM validated_relationships WHERE status = 'SHADOW'").fetchall():
        report.shadow_relationships.append(_summarize_relationship(conn, row, now))
    for row in conn.execute("SELECT * FROM validated_relationships WHERE status = 'RETIRED'").fetchall():
        report.retired_relationships.append(_summarize_relationship(conn, row, now))

    for row in conn.execute("SELECT * FROM candidate_hypotheses WHERE status = 'REJECTED'").fetchall():
        test_result = json.loads(row["test_result_json"]) if row["test_result_json"] else {}
        report.rejected_hypotheses.append({
            "hypothesis_id": row["hypothesis_id"], "condition": json.loads(row["condition_json"]),
            "explanation_text": row["explanation_text"], "tested_at": row["tested_at"],
            "reason": test_result.get("status"), "n": test_result.get("n"),
        })

    report.source_reliability = _source_reliability(conn)
    report.operational_counts = _operational_counts(conn)
    report.calibration_by_horizon = _calibration_by_horizon(conn)
    report.concepts = _build_concept_summaries(conn, now)
    report.methodologies = _build_methodology_summaries(conn)
    return report
