"""Typed read/write access to the six-category store. This is the ONLY
module in the system allowed to execute SQL - every other module goes
through the functions here, which is what lets the append-only guarantee
on episodic_events be enforced in code rather than trusted by convention.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from market_agent.events.schema import EventRecord, PredictionRecord
from market_agent.store.schema import init_db


class AppendOnlyViolation(Exception):
    """Raised on any attempt to modify an episodic_events row whose outcome
    has already been recorded (outcome_locked = 1). Category 1 is
    append-only by design (Blueprint section B) - this is that rule
    enforced at the data-access layer, not left to convention."""


def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def _iso(dt: datetime | str) -> str:
    """Accepts either a datetime or an already-ISO string - the latter
    happens whenever a caller re-upserts a value it just read back out of
    the store (e.g. governance.revalidate() carrying forward an existing
    row's created_at), which sqlite always returns as text, not a
    datetime object."""
    if isinstance(dt, str):
        return dt
    dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


# --- episodic_events (category 1) ---

def log_prediction(conn: sqlite3.Connection, event: EventRecord, prediction: PredictionRecord) -> str:
    """The ONE way a new episodic_events row is ever created. Returns the
    new event_id. There is no corresponding `update_prediction` - a
    prediction, once logged, is immutable; only the outcome fields (via
    record_outcome, exactly once) may ever be added afterward."""
    event_id = str(uuid.uuid4())
    predicted_direction = prediction.predicted_direction
    if predicted_direction is None and prediction.predicted_impact is not None:
        predicted_direction = ("positive" if prediction.predicted_impact > 0 else
                                "negative" if prediction.predicted_impact < 0 else "unclear")
    conn.execute(
        """INSERT INTO episodic_events
           (event_id, entity, event_type, direction, source, source_reliability_snapshot,
            raw_text, published_at, ingested_at, context_json, horizon_days,
            predicted_impact, predicted_confidence, prediction_basis_json, model_version,
            predicted_at, predicted_direction, uncertainty, retrieved_cases_json, knowledge_version,
            novelty_score, outcome_locked)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)""",
        (event_id, event.entity, event.event_type, event.direction, event.source,
         event.source_reliability_snapshot, event.raw_text, _iso(event.published_at),
         _iso(event.ingested_at), json.dumps(event.context), prediction.horizon_days,
         prediction.predicted_impact, prediction.predicted_confidence,
         json.dumps(prediction.basis), prediction.model_version, _iso(prediction.predicted_at),
         predicted_direction, prediction.uncertainty, json.dumps(prediction.retrieved_cases),
         prediction.knowledge_version, prediction.novelty_score),
    )
    conn.commit()
    return event_id


def record_outcome(conn: sqlite3.Connection, event_id: str, realized_abnormal_return: float,
                    observed_at: datetime, error_value: float, error_type: str) -> None:
    """Sets the outcome fields on an episodic_events row EXACTLY ONCE.
    A second call for the same event_id raises AppendOnlyViolation rather
    than silently overwriting - a prediction's realized outcome is a fact
    about history and must never be allowed to change after the fact."""
    row = conn.execute("SELECT outcome_locked FROM episodic_events WHERE event_id = ?",
                        (event_id,)).fetchone()
    if row is None:
        raise KeyError(f"No episodic_events row with event_id={event_id!r}")
    if row["outcome_locked"]:
        raise AppendOnlyViolation(
            f"event_id={event_id!r} already has a recorded outcome - outcomes are immutable once set.")
    conn.execute(
        """UPDATE episodic_events
           SET realized_abnormal_return = ?, outcome_observed_at = ?, error_value = ?,
               error_type = ?, outcome_locked = 1
           WHERE event_id = ?""",
        (realized_abnormal_return, _iso(observed_at), error_value, error_type, event_id),
    )
    conn.commit()


def get_event(conn: sqlite3.Connection, event_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM episodic_events WHERE event_id = ?", (event_id,)).fetchone()


def deduplicate_by_real_event(rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
    """CRITICAL STATISTICAL-VALIDITY FIX (found inspecting a real stage-7
    four-agent walk-forward run directly): every agent scored for the same
    underlying market event/horizon (STATIC, CURRENT_ADAPTIVE,
    TECHNICAL_ADAPTIVE, METHODOLOGY_ADAPTIVE, ...) logs its OWN
    episodic_events row via log_prediction - same entity, same
    published_at, same context_json, same realized_abnormal_return once
    resolved, but a DIFFERENT event_id and model_version. A query that
    matches on event_type/direction/horizon_days/context alone (as every
    hypothesis-testing, shadow-evaluation, and similar-case-retrieval
    query in this system does) picks up ALL of them - confirmed by direct
    inspection of a real run: 560 "matching rows" for one condition were
    exactly 140 distinct real events x 4 agents.

    This does not bias a MEAN (duplicates share the identical realized
    return), but it does inflate reported N and - critically - understate
    the true standard error a one-sample t-test computes, since
    scipy.stats.ttest_1samp assumes every row is an independent
    observation. Four non-independent copies of the same observation are
    not four observations. This was already present with 2 agents
    (stages 1-5, 2x inflation) and became 4x with stage 6's four-agent
    harness - the earlier scale simply looked less alarming.

    The fix: keep exactly ONE row per distinct (entity, published_at)
    pair - deterministically (the smallest event_id, not "whichever
    agent's row looks best") - REGARDLESS of how many agents logged a
    prediction for it, or what they're named. This makes the fix correct
    for real multi-agent walk-forward runs AND a no-op for single-
    prediction-per-event test fixtures (nothing to deduplicate there)."""
    seen: dict[tuple, sqlite3.Row] = {}
    for row in sorted(rows, key=lambda r: r["event_id"]):
        key = (row["entity"], row["published_at"], row["horizon_days"], row["event_type"], row["direction"])
        if key not in seen:
            seen[key] = row
    return list(seen.values())


def query_events(conn: sqlite3.Connection, event_type: str | None = None, entity: str | None = None,
                  published_before: datetime | None = None, outcome_known_only: bool = False) -> list[sqlite3.Row]:
    """The read path every retrieval/testing/recalibration job uses.
    `published_before` is how point-in-time discipline actually gets
    enforced on reads of category 1 - callers pass the simulation clock's
    current time, never omit it in a historical-replay context."""
    clauses, params = [], []
    if event_type is not None:
        clauses.append("event_type = ?"); params.append(event_type)
    if entity is not None:
        clauses.append("entity = ?"); params.append(entity)
    if published_before is not None:
        clauses.append("published_at < ?"); params.append(_iso(published_before))
    if outcome_known_only:
        # outcome_locked=1 alone can mean "resolved as DATA_ERROR" (e.g. no price history at the
        # horizon date) - realized_abnormal_return IS NOT NULL is required for a genuinely KNOWN
        # numeric outcome. See learn/hypothesis_testing.py's _matching_prior_rows for the same fix,
        # found by running against real (sometimes delisted) tickers.
        clauses.append("outcome_locked = 1 AND realized_abnormal_return IS NOT NULL")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return conn.execute(f"SELECT * FROM episodic_events {where} ORDER BY published_at",
                         params).fetchall()


# --- validated_relationships (category 3) ---

def upsert_relationship(conn: sqlite3.Connection, relationship_id: str, condition: dict, horizon_days: int,
                         effect_estimate: float, ci_low: float | None, ci_high: float | None,
                         n_supporting: int, status: str, created_at: datetime,
                         source_hypothesis_id: str | None = None,
                         last_revalidated_at: datetime | None = None,
                         shadow_started_at: datetime | None = None,
                         shadow_promoted_at: datetime | None = None,
                         concept: str | None = None,
                         methodology_ids: list[str] | None = None) -> None:
    conn.execute(
        """INSERT INTO validated_relationships
           (relationship_id, condition_json, horizon_days, effect_estimate, ci_low, ci_high,
            n_supporting, status, source_hypothesis_id, created_at, last_revalidated_at,
            shadow_started_at, shadow_promoted_at, concept, methodology_ids_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(relationship_id) DO UPDATE SET
             effect_estimate=excluded.effect_estimate, ci_low=excluded.ci_low, ci_high=excluded.ci_high,
             n_supporting=excluded.n_supporting, status=excluded.status,
             last_revalidated_at=excluded.last_revalidated_at,
             shadow_promoted_at=COALESCE(excluded.shadow_promoted_at, validated_relationships.shadow_promoted_at),
             concept=COALESCE(excluded.concept, validated_relationships.concept),
             methodology_ids_json=COALESCE(excluded.methodology_ids_json, validated_relationships.methodology_ids_json)""",
        (relationship_id, json.dumps(condition), horizon_days, effect_estimate, ci_low, ci_high,
         n_supporting, status, source_hypothesis_id, _iso(created_at),
         _iso(last_revalidated_at) if last_revalidated_at else None,
         _iso(shadow_started_at) if shadow_started_at else None,
         _iso(shadow_promoted_at) if shadow_promoted_at else None,
         concept, json.dumps(methodology_ids) if methodology_ids else None),
    )
    conn.commit()


def active_relationships(conn: sqlite3.Connection, event_type: str, horizon_days: int) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT * FROM validated_relationships
           WHERE status = 'ACTIVE' AND horizon_days = ?
             AND json_extract(condition_json, '$.event_type') = ?""",
        (horizon_days, event_type),
    ).fetchall()


def shadow_relationships(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """All relationships currently on probation - confirmed by a
    statistical test but not yet trusted for live prediction. See
    learn/shadow.py."""
    return conn.execute("SELECT * FROM validated_relationships WHERE status = 'SHADOW'").fetchall()


def count_governance_changes(conn: sqlite3.Connection) -> int:
    """A monotonic counter of every registry entry ever written - used as
    `knowledge_version` on a logged prediction, so two predictions of the
    same event can be compared against exactly how much the system had
    learned by each point in time."""
    return conn.execute("SELECT COUNT(*) c FROM model_registry").fetchone()["c"]


# --- candidate_hypotheses (category 4) ---

def add_hypothesis(conn: sqlite3.Connection, source_event_id: str, condition: dict, horizon_days: int,
                    explanation_text: str, proposed_at: datetime, concept: str | None = None,
                    methodology_ids: list[str] | None = None) -> str:
    hypothesis_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO candidate_hypotheses
           (hypothesis_id, source_event_id, condition_json, horizon_days, explanation_text,
            proposed_at, status, concept, methodology_ids_json)
           VALUES (?,?,?,?,?,?,'UNTESTED',?,?)""",
        (hypothesis_id, source_event_id, json.dumps(condition), horizon_days, explanation_text,
         _iso(proposed_at), concept, json.dumps(methodology_ids) if methodology_ids else None),
    )
    conn.commit()
    return hypothesis_id


def set_hypothesis_result(conn: sqlite3.Connection, hypothesis_id: str, status: str,
                           tested_at: datetime, test_result: dict) -> None:
    if status not in ("REJECTED", "CONFIRMED", "TESTING"):
        raise ValueError(f"Invalid hypothesis status: {status!r}")
    conn.execute(
        """UPDATE candidate_hypotheses SET status = ?, tested_at = ?, test_result_json = ?
           WHERE hypothesis_id = ?""",
        (status, _iso(tested_at), json.dumps(test_result), hypothesis_id),
    )
    conn.commit()


def untested_hypotheses(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM candidate_hypotheses WHERE status = 'UNTESTED'").fetchall()


# --- model_registry (category 5 + governance) ---

def register_change(conn: sqlite3.Connection, version_id: str, reason: str, change: dict,
                     performance_before: dict | None, performance_after: dict | None,
                     statistical_tests: dict, promoted_by: str, promotion_status: str,
                     created_at: datetime, rollback_of: str | None = None,
                     training_data_range: tuple[datetime, datetime] | None = None) -> None:
    conn.execute(
        """INSERT INTO model_registry
           (version_id, created_at, reason, change_json, training_data_range_start,
            training_data_range_end, performance_before_json, performance_after_json,
            statistical_tests_json, promoted_by, rollback_of, promotion_status)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (version_id, _iso(created_at), reason, json.dumps(change),
         _iso(training_data_range[0]) if training_data_range else None,
         _iso(training_data_range[1]) if training_data_range else None,
         json.dumps(performance_before) if performance_before else None,
         json.dumps(performance_after) if performance_after else None,
         json.dumps(statistical_tests), promoted_by, rollback_of, promotion_status),
    )
    conn.commit()


# --- trading_methodologies / methodology_concept_links (stage 6 provenance - see store/schema.py's
# module-level comment on these two tables: never read by prediction or hypothesis-testing code,
# only by the ingestion layer and reporting) ---

def add_methodology(conn: sqlite3.Connection, methodology_id: str, name: str, practitioner: str,
                     source_type: str, source_description: str, extractor_name: str,
                     ingested_at: datetime) -> None:
    conn.execute(
        """INSERT INTO trading_methodologies
           (methodology_id, name, practitioner, source_type, source_description, extractor_name, ingested_at)
           VALUES (?,?,?,?,?,?,?)
           ON CONFLICT(methodology_id) DO NOTHING""",
        (methodology_id, name, practitioner, source_type, source_description, extractor_name, _iso(ingested_at)),
    )
    conn.commit()


def add_methodology_concept_link(conn: sqlite3.Connection, link_id: str, methodology_id: str, concept: str,
                                  rationale: str, created_at: datetime) -> None:
    conn.execute(
        """INSERT INTO methodology_concept_links (link_id, methodology_id, concept, rationale, created_at)
           VALUES (?,?,?,?,?)""",
        (link_id, methodology_id, concept, rationale, _iso(created_at)),
    )
    conn.commit()


def get_methodology(conn: sqlite3.Connection, methodology_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM trading_methodologies WHERE methodology_id = ?",
                         (methodology_id,)).fetchone()


def all_methodologies(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM trading_methodologies ORDER BY ingested_at").fetchall()


def concept_links_for_methodology(conn: sqlite3.Connection, methodology_id: str) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM methodology_concept_links WHERE methodology_id = ?",
                         (methodology_id,)).fetchall()


def methodologies_for_concept(conn: sqlite3.Connection, concept: str) -> list[sqlite3.Row]:
    """Every methodology that mapped to this concept, joined with its link
    rationale - the "multiple independent methodologies may contribute
    evidence for the same concept" provenance view. NEVER used to decide
    whether a concept is valid (see module docstring) - only real
    hypothesis-testing results (validated_relationships) do that."""
    return conn.execute(
        """SELECT m.*, l.rationale, l.link_id FROM trading_methodologies m
           JOIN methodology_concept_links l ON l.methodology_id = m.methodology_id
           WHERE l.concept = ? ORDER BY m.ingested_at""",
        (concept,)).fetchall()


# --- setup_observations (stage 8 - schema v4, category-1-equivalent for the continuous scan) ---

def log_setup_observation(conn: sqlite3.Connection, entity: str, as_of: datetime, regime: str,
                           technical: dict, horizon_days: int) -> str:
    """The ONE way a new setup_observations row is ever created - see
    log_prediction's identical discipline for episodic_events.
    `technical` is an already-serialized TechnicalMarketContext.to_dict()
    snapshot, kept as a plain dict here (not the dataclass type) so this
    module never needs to import concepts/technical_context.py."""
    observation_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO setup_observations
           (observation_id, entity, as_of, regime, technical_json, horizon_days, outcome_locked)
           VALUES (?,?,?,?,?,?,0)""",
        (observation_id, entity, _iso(as_of), regime, json.dumps(technical), horizon_days),
    )
    conn.commit()
    return observation_id


def record_setup_outcome(conn: sqlite3.Connection, observation_id: str, realized_abnormal_return: float | None,
                          observed_at: datetime) -> None:
    """Sets the outcome exactly once - same AppendOnlyViolation discipline
    as episodic_events' record_outcome."""
    row = conn.execute("SELECT outcome_locked FROM setup_observations WHERE observation_id = ?",
                        (observation_id,)).fetchone()
    if row is None:
        raise KeyError(f"No setup_observations row with observation_id={observation_id!r}")
    if row["outcome_locked"]:
        raise AppendOnlyViolation(
            f"observation_id={observation_id!r} already has a recorded outcome - outcomes are immutable once set.")
    conn.execute(
        """UPDATE setup_observations SET realized_abnormal_return = ?, outcome_observed_at = ?, outcome_locked = 1
           WHERE observation_id = ?""",
        (realized_abnormal_return, _iso(observed_at), observation_id),
    )
    conn.commit()


def get_setup_observation(conn: sqlite3.Connection, observation_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM setup_observations WHERE observation_id = ?", (observation_id,)).fetchone()


def query_setup_observations(conn: sqlite3.Connection, horizon_days: int | None = None,
                              as_of_before: datetime | None = None, as_of_after: datetime | None = None,
                              outcome_known_only: bool = False) -> list[sqlite3.Row]:
    clauses, params = [], []
    if horizon_days is not None:
        clauses.append("horizon_days = ?"); params.append(horizon_days)
    if as_of_before is not None:
        clauses.append("as_of < ?"); params.append(_iso(as_of_before))
    if as_of_after is not None:
        clauses.append("as_of >= ?"); params.append(_iso(as_of_after))
    if outcome_known_only:
        clauses.append("outcome_locked = 1 AND realized_abnormal_return IS NOT NULL")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return conn.execute(f"SELECT * FROM setup_observations {where} ORDER BY as_of", params).fetchall()


# --- discovered_setups (stage 8 - schema v4, category-3-equivalent for composite technical setups) ---

def upsert_discovered_setup(conn: sqlite3.Connection, setup_id: str, regime: str | None,
                             technical_conditions: dict, horizon_days: int, invalidation_pct: float | None,
                             train_result: dict | None, validate_result: dict | None,
                             shadow_result: dict | None, test_result: dict | None, status: str,
                             created_at: datetime) -> None:
    conn.execute(
        """INSERT INTO discovered_setups
           (setup_id, regime, technical_conditions_json, horizon_days, invalidation_pct,
            train_result_json, validate_result_json, shadow_result_json, test_result_json, status, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(setup_id) DO UPDATE SET
             train_result_json=excluded.train_result_json, validate_result_json=excluded.validate_result_json,
             shadow_result_json=excluded.shadow_result_json, test_result_json=excluded.test_result_json,
             status=excluded.status""",
        (setup_id, regime, json.dumps(technical_conditions), horizon_days, invalidation_pct,
         json.dumps(train_result) if train_result is not None else None,
         json.dumps(validate_result) if validate_result is not None else None,
         json.dumps(shadow_result) if shadow_result is not None else None,
         json.dumps(test_result) if test_result is not None else None,
         status, _iso(created_at)),
    )
    conn.commit()


def get_discovered_setup(conn: sqlite3.Connection, setup_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM discovered_setups WHERE setup_id = ?", (setup_id,)).fetchone()


def discovered_setups_by_status(conn: sqlite3.Connection, status: str) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM discovered_setups WHERE status = ?", (status,)).fetchall()


# --- research_reports / research_watchlist (schema v5 - the AI Market Research & Analysis product) ---

def save_research_report(conn: sqlite3.Connection, entity: str, generated_at: datetime, assessment: str,
                          report_json: dict) -> str:
    report_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO research_reports (report_id, entity, generated_at, assessment, report_json) VALUES (?,?,?,?,?)",
        (report_id, entity, _iso(generated_at), assessment, json.dumps(report_json)),
    )
    conn.commit()
    return report_id


def latest_research_report(conn: sqlite3.Connection, entity: str) -> sqlite3.Row | None:
    """Most recent report for `entity`, or None if this is the first ever
    research pass - see research/change_detection.py, which is the ONLY
    consumer of this and treats None as "no prior report to diff against",
    never as an error."""
    return conn.execute(
        "SELECT * FROM research_reports WHERE entity = ? ORDER BY generated_at DESC LIMIT 1",
        (entity,)).fetchone()


def research_report_history(conn: sqlite3.Connection, entity: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM research_reports WHERE entity = ? ORDER BY generated_at", (entity,)).fetchall()


def add_to_watchlist(conn: sqlite3.Connection, entity: str, added_at: datetime) -> None:
    conn.execute(
        "INSERT INTO research_watchlist (entity, added_at) VALUES (?,?) "
        "ON CONFLICT(entity) DO NOTHING", (entity, _iso(added_at)))
    conn.commit()


def remove_from_watchlist(conn: sqlite3.Connection, entity: str) -> None:
    conn.execute("DELETE FROM research_watchlist WHERE entity = ?", (entity,))
    conn.commit()


def get_watchlist(conn: sqlite3.Connection) -> list[str]:
    return [r["entity"] for r in conn.execute("SELECT entity FROM research_watchlist ORDER BY added_at").fetchall()]


# --- news_event_vectors / news_company_states (schema v6 - the News State Engine) ---

def save_news_event_vector(conn: sqlite3.Connection, event_vector_id: str, entity: str, as_of: str,
                            computed_at: datetime, event_vector_json: dict) -> None:
    conn.execute(
        "INSERT INTO news_event_vectors (event_vector_id, entity, as_of, computed_at, event_vector_json) "
        "VALUES (?,?,?,?,?) ON CONFLICT(event_vector_id) DO NOTHING",
        (event_vector_id, entity, as_of, _iso(computed_at), json.dumps(event_vector_json)),
    )
    conn.commit()


def prior_news_event_vectors(conn: sqlite3.Connection, entity: str, before: str) -> list[sqlite3.Row]:
    """Every EventVector ever persisted for `entity` strictly before
    `before` (ISO) - the ONLY legitimate source for novelty comparison
    (see news_state/aggregation.py). Empty if this is the first time this
    entity has ever been processed - callers must treat that as
    "no baseline yet", never fabricate one."""
    return conn.execute(
        "SELECT * FROM news_event_vectors WHERE entity = ? AND as_of < ? ORDER BY as_of",
        (entity, before)).fetchall()


def save_news_company_state(conn: sqlite3.Connection, entity: str, as_of: str, computed_at: datetime,
                             state_json: dict) -> str:
    state_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO news_company_states (state_id, entity, as_of, computed_at, state_json) VALUES (?,?,?,?,?)",
        (state_id, entity, as_of, _iso(computed_at), json.dumps(state_json)),
    )
    conn.commit()
    return state_id


def latest_news_company_state(conn: sqlite3.Connection, entity: str, before: str | None = None) -> sqlite3.Row | None:
    if before is not None:
        return conn.execute(
            "SELECT * FROM news_company_states WHERE entity = ? AND as_of < ? ORDER BY as_of DESC LIMIT 1",
            (entity, before)).fetchone()
    return conn.execute(
        "SELECT * FROM news_company_states WHERE entity = ? ORDER BY as_of DESC LIMIT 1", (entity,)).fetchone()


# --- prediction_log (schema v7 - market_agent/research/evaluation/) ---

def save_prediction(conn: sqlite3.Connection, entity: str, mode: str, triggered_at: datetime, model_version: str,
                     decision_label: str | None, predicted_impact: float | None, predicted_confidence: float | None,
                     inputs_snapshot: dict) -> str:
    prediction_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO prediction_log (prediction_id, entity, mode, triggered_at, model_version, decision_label, "
        "predicted_impact, predicted_confidence, inputs_snapshot_json) VALUES (?,?,?,?,?,?,?,?,?)",
        (prediction_id, entity, mode, _iso(triggered_at), model_version, decision_label, predicted_impact,
         predicted_confidence, json.dumps(inputs_snapshot)),
    )
    conn.commit()
    return prediction_id


def unresolved_predictions(conn: sqlite3.Connection, horizon_column: str) -> list[sqlite3.Row]:
    """Every logged prediction whose `realized_return_{horizon}` is still
    NULL - outcome_resolution.py's own candidate list. `horizon_column`
    must be one of the four literal column names (never user/caller
    input assembled into SQL from anything but this module's own fixed
    HORIZON_COLUMNS)."""
    from market_agent.research.evaluation.outcome_resolution import HORIZON_COLUMNS
    if horizon_column not in HORIZON_COLUMNS:
        raise ValueError(f"Unknown horizon column: {horizon_column!r}")
    return conn.execute(
        f"SELECT * FROM prediction_log WHERE {horizon_column} IS NULL ORDER BY triggered_at").fetchall()


def record_prediction_outcome(conn: sqlite3.Connection, prediction_id: str, return_column: str,
                               resolved_column: str, realized_return: float, resolved_at: datetime) -> None:
    """NOT `record_outcome` (that name already exists above, for
    episodic_events' category-1 outcome locking - a different table, a
    different system). Named distinctly on purpose after the two were
    found to collide (a real bug caught by the full test suite, not
    hypothetical)."""
    from market_agent.research.evaluation.outcome_resolution import HORIZON_COLUMNS
    if return_column not in HORIZON_COLUMNS or resolved_column not in HORIZON_COLUMNS.values():
        raise ValueError(f"Unknown outcome column pair: {return_column!r}/{resolved_column!r}")
    conn.execute(f"UPDATE prediction_log SET {return_column} = ?, {resolved_column} = ? WHERE prediction_id = ?",
                 (realized_return, _iso(resolved_at), prediction_id))
    conn.commit()


def prediction_log_for_entity(conn: sqlite3.Connection, entity: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM prediction_log WHERE entity = ? ORDER BY triggered_at", (entity,)).fetchall()


def all_predictions(conn: sqlite3.Connection, mode: str | None = None) -> list[sqlite3.Row]:
    if mode is not None:
        return conn.execute(
            "SELECT * FROM prediction_log WHERE mode = ? ORDER BY triggered_at", (mode,)).fetchall()
    return conn.execute("SELECT * FROM prediction_log ORDER BY triggered_at").fetchall()
