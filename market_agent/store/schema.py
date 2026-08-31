"""SQLite schema for the six-category data model (Blueprint section B).

Only THREE tables actually exist on disk. The other three categories are
deliberately NOT separate tables:

  Category 1 (permanent historical experience) -> `episodic_events`, append-only.
  Category 2 (current statistical influence)    -> NOT a table. A derived,
      recomputed-on-query weighting over category 1 (see learn/influence.py) -
      persisting it would let it silently drift out of sync with its own
      definition. It is recomputed from category 1 every time it's needed.
  Category 3 (validated relationships)          -> `validated_relationships`.
  Category 4 (candidate hypotheses)              -> `candidate_hypotheses`.
  Category 5 (model parameters) + governance     -> `model_registry`.
  Category 6 (temporary contextual information)  -> NOT a table. Captured as
      an immutable JSON snapshot INSIDE each episodic_events row at
      prediction time (what the regime/rumor-status/portfolio looked like
      right then) - it never gets its own row, and is never updated after
      the fact. That immutability is what keeps it "temporary" (irrelevant
      once its moment has passed) rather than a second, competing notion of
      "current state".

STORAGE CHOICE: SQLite, not Postgres+pgvector as the target-scale blueprint
recommends. This is a deliberate, disclosed MVP simplification - see
scripts/README or the implementation-plan chat message this module was
introduced in. SQLite has no concurrent-writer story and no native vector
type; both are fine for a single-process, single-event-type MVP and both
become real constraints the moment this needs to run continuously against
a broad universe. Upgrading later means changing db.py's connection layer,
not this schema's logical shape.
"""
from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 7

# Columns added in SCHEMA_VERSION 2 (continuous-operation ledger fields). Listed here, not just in
# the DDL, so _migrate() below can add them to a database created under SCHEMA_VERSION 1 without
# losing its existing episodic_events history - a continuously operating system must be able to
# evolve its schema without discarding permanent record (Blueprint category 1's own append-only
# guarantee extends to "the table itself is never dropped and recreated to add a column").
NEW_EPISODIC_COLUMNS: dict[str, str] = {
    "predicted_direction": "TEXT",       # 'positive' | 'negative' | 'unclear' - the AGENT's predicted
    #                                       direction, stored explicitly rather than re-derived from
    #                                       predicted_impact's sign, so a future model that could predict
    #                                       a direction different from the raw event's own classified
    #                                       direction has somewhere to record that distinction.
    "uncertainty": "REAL",               # half-width of the relationship's CI (or a baseline-derived
    #                                       default) - a numeric companion to predicted_confidence's
    #                                       coarse HIGH/MEDIUM/LOW label.
    "retrieved_cases_json": "TEXT",      # event_ids of the similar historical cases actually shown/used
    #                                       for this prediction (retrieval/similarity.py) - audit trail.
    "knowledge_version": "INTEGER",      # monotonic counter of how many governance changes (promotions/
    #                                       retirements) had been applied by the time this prediction was
    #                                       made - lets two predictions of the same event be compared
    #                                       against "what the system knew" at each point.
    "novelty_score": "REAL",             # 0-1, from retrieval coverage - see experiment/context.py.
}

# Columns added in SCHEMA_VERSION 3 (stage 6 - methodology/concept provenance). NULLABLE on both
# tables: a hypothesis/relationship conditioned purely on event-context dimensions (regime,
# prior_return_bucket, vol_bucket - stage 1-5) has no concept or methodology behind it at all, and
# that must stay representable, not forced into an empty-string placeholder.
NEW_HYPOTHESIS_COLUMNS: dict[str, str] = {
    "concept": "TEXT",              # a market_agent.concepts.ontology.TradingConcept value, or NULL
    "methodology_ids_json": "TEXT",  # JSON list of trading_methodologies.methodology_id that motivated
    #                                   this hypothesis via methodology_concept_links - provenance only,
    #                                   never itself evidence (see store schema module docstring below).
}
NEW_RELATIONSHIP_COLUMNS_V3: dict[str, str] = {
    "concept": "TEXT",
    "methodology_ids_json": "TEXT",
}

DDL = """
-- Category 1: permanent historical experience. Append-only by convention
-- AND by enforcement (see db.py's outcome_locked guard) - once outcome_*
-- fields are set, no further UPDATE to this row is permitted.
CREATE TABLE IF NOT EXISTS episodic_events (
    event_id                TEXT PRIMARY KEY,
    entity                  TEXT NOT NULL,
    event_type              TEXT NOT NULL,
    direction                TEXT,                 -- 'positive' | 'negative' | 'unclear'
    source                   TEXT NOT NULL,
    source_reliability_snapshot REAL,               -- reliability estimate AS KNOWN at prediction time
    raw_text                  TEXT,
    published_at               TEXT NOT NULL,        -- when the information became public
    ingested_at                 TEXT NOT NULL,        -- when this system saw it
    context_json                 TEXT NOT NULL,        -- category-6 snapshot, frozen at prediction time
    horizon_days                  INTEGER NOT NULL,

    -- prediction (set once, at prediction time)
    predicted_impact                REAL,
    predicted_confidence              TEXT,           -- 'HIGH' | 'MEDIUM' | 'LOW' | 'INSUFFICIENT_PRECEDENT'
    prediction_basis_json              TEXT,           -- which validated_relationship_id or 'unconditional_baseline'
    model_version                       TEXT NOT NULL,  -- which agent + registry version produced this prediction
    predicted_at                         TEXT NOT NULL,
    predicted_direction                   TEXT,          -- see NEW_EPISODIC_COLUMNS above (schema v2)
    uncertainty                            REAL,
    retrieved_cases_json                    TEXT,
    knowledge_version                        INTEGER,
    novelty_score                             REAL,

    -- outcome (set exactly once, later, when the horizon closes)
    realized_abnormal_return               REAL,
    outcome_observed_at                     TEXT,
    error_value                              REAL,
    error_type                                TEXT,     -- see learn/error_taxonomy.py
    outcome_locked                             INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_episodic_entity ON episodic_events(entity);
CREATE INDEX IF NOT EXISTS idx_episodic_event_type ON episodic_events(event_type);
CREATE INDEX IF NOT EXISTS idx_episodic_published_at ON episodic_events(published_at);
CREATE INDEX IF NOT EXISTS idx_episodic_outcome_locked ON episodic_events(outcome_locked);

-- Category 3: validated relationships. Live predictions read ONLY this
-- table (plus the unconditional baseline seed row) - never candidate_hypotheses.
CREATE TABLE IF NOT EXISTS validated_relationships (
    relationship_id       TEXT PRIMARY KEY,
    condition_json          TEXT NOT NULL,     -- structured filter, e.g. {"event_type": "...", "regime": "..."}
    horizon_days              INTEGER NOT NULL,
    effect_estimate             REAL NOT NULL,
    ci_low                       REAL,
    ci_high                       REAL,
    n_supporting                   INTEGER NOT NULL,
    status                          TEXT NOT NULL,  -- 'SHADOW' | 'ACTIVE' | 'RETIRED'
    source_hypothesis_id             TEXT,           -- NULL for the seeded unconditional baseline
    created_at                        TEXT NOT NULL,
    last_revalidated_at                TEXT,
    shadow_started_at                  TEXT,          -- set when first confirmed (status=SHADOW); see learn/shadow.py
    shadow_promoted_at                  TEXT,          -- set when SHADOW -> ACTIVE (schema v2)
    concept                              TEXT,          -- a TradingConcept value, or NULL (schema v3)
    methodology_ids_json                  TEXT           -- JSON list of contributing methodology_ids (schema v3)
);
CREATE INDEX IF NOT EXISTS idx_relationships_status ON validated_relationships(status);

-- Category 4: candidate hypotheses. NEVER read by a live prediction -
-- only by the governed testing job (learn/hypothesis_testing.py).
CREATE TABLE IF NOT EXISTS candidate_hypotheses (
    hypothesis_id          TEXT PRIMARY KEY,
    source_event_id           TEXT NOT NULL,
    condition_json              TEXT NOT NULL,
    horizon_days                  INTEGER NOT NULL,
    explanation_text                TEXT NOT NULL,   -- LLM/rule-proposed prose - audit trail only, never trusted as fact
    proposed_at                       TEXT NOT NULL,
    status                             TEXT NOT NULL,  -- 'UNTESTED' | 'TESTING' | 'REJECTED' | 'CONFIRMED'
    tested_at                           TEXT,
    test_result_json                     TEXT,
    concept                               TEXT,          -- a TradingConcept value, or NULL (schema v3)
    methodology_ids_json                   TEXT,          -- JSON list of contributing methodology_ids (schema v3)
    FOREIGN KEY (source_event_id) REFERENCES episodic_events(event_id)
);
CREATE INDEX IF NOT EXISTS idx_hypotheses_status ON candidate_hypotheses(status);

-- Category 5 + governance: every promotion/retirement/rollback.
CREATE TABLE IF NOT EXISTS model_registry (
    version_id                TEXT PRIMARY KEY,
    created_at                   TEXT NOT NULL,
    reason                         TEXT NOT NULL,
    change_json                     TEXT NOT NULL,   -- what changed: relationship created/updated/retired
    training_data_range_start        TEXT,
    training_data_range_end           TEXT,
    performance_before_json            TEXT,
    performance_after_json              TEXT,
    statistical_tests_json               TEXT,
    promoted_by                           TEXT NOT NULL,
    rollback_of                            TEXT,      -- version_id this reverts, if any
    promotion_status                        TEXT NOT NULL  -- 'SHADOW' | 'PROMOTED' | 'ROLLED_BACK'
);

-- Stage 6: methodology ingestion + provenance. NEITHER table is read by any prediction or
-- hypothesis-TESTING path - they exist purely so a candidate_hypotheses/validated_relationships row
-- can point back at which methodology/methodologies motivated it (see NEW_HYPOTHESIS_COLUMNS /
-- NEW_RELATIONSHIP_COLUMNS_V3 above). A methodology or a methodology-concept link is NEVER itself
-- statistical evidence - only learn/hypothesis_testing.py's real, out-of-sample significance test
-- against episodic_events ever changes a concept's validated_relationships status. Multiple
-- methodologies can (and often do) link to the SAME concept - that agreement is recorded here for
-- provenance/audit only, and must never be read as corroboration by any governance code.
CREATE TABLE IF NOT EXISTS trading_methodologies (
    methodology_id        TEXT PRIMARY KEY,
    name                    TEXT NOT NULL,
    practitioner              TEXT NOT NULL,
    source_type                 TEXT NOT NULL,   -- e.g. 'book' | 'published_research' | 'publicly_documented_system'
    source_description            TEXT NOT NULL,   -- short, paraphrased summary - audit trail only, never
    --                                                verbatim reproduction of copyrighted source text
    extractor_name                  TEXT NOT NULL,  -- which MethodologyExtractor produced this ingestion
    ingested_at                       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS methodology_concept_links (
    link_id             TEXT PRIMARY KEY,
    methodology_id        TEXT NOT NULL,
    concept                 TEXT NOT NULL,   -- a TradingConcept value
    rationale                 TEXT NOT NULL,   -- short, paraphrased justification - audit trail only
    created_at                  TEXT NOT NULL,
    FOREIGN KEY (methodology_id) REFERENCES trading_methodologies(methodology_id)
);
CREATE INDEX IF NOT EXISTS idx_methodology_links_concept ON methodology_concept_links(concept);
CREATE INDEX IF NOT EXISTS idx_methodology_links_methodology ON methodology_concept_links(methodology_id);

-- Stage 8 (schema v4): continuous market-state setup discovery. `setup_observations` is a SEPARATE,
-- PARALLEL ledger to episodic_events (category 1) - one row per (entity, as_of) point sampled from a
-- continuous calendar scan (market_agent/setups/market_scan.py), NEVER triggered by a discrete news
-- event. Kept as its own table rather than merged into episodic_events, because the two have
-- fundamentally different sampling schemes (dense calendar scan vs. sparse event arrival) and mixing
-- them would silently change what "one row = one independent observation" means for every existing
-- statistical test built around episodic_events. Same append-only outcome discipline
-- (outcome_locked, set exactly once - see db.py's record_setup_outcome).
CREATE TABLE IF NOT EXISTS setup_observations (
    observation_id          TEXT PRIMARY KEY,
    entity                    TEXT NOT NULL,
    as_of                       TEXT NOT NULL,     -- the scanned REAL trading day (ISO)
    regime                        TEXT NOT NULL,
    technical_json                   TEXT NOT NULL,     -- full TechnicalMarketContext.to_dict() snapshot
    horizon_days                       INTEGER NOT NULL,

    realized_abnormal_return              REAL,
    outcome_observed_at                     TEXT,
    outcome_locked                            INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_setup_obs_entity ON setup_observations(entity);
CREATE INDEX IF NOT EXISTS idx_setup_obs_as_of ON setup_observations(as_of);
CREATE INDEX IF NOT EXISTS idx_setup_obs_outcome_locked ON setup_observations(outcome_locked);

-- Category-3-equivalent for stage 8: a DISCOVERED, hierarchically-tested COMPOSITE setup (a
-- conjunction over TechnicalMarketContext fields, optionally + regime) - see
-- setups/setup_discovery.py. Distinct from validated_relationships (which is always conditioned on
-- a discrete event_type/direction): a Setup has no event at all, and carries its own
-- TRAIN/VALIDATE/SHADOW/TEST per-segment expectancy rather than one effect_estimate, since stage 8's
-- discovery loop reports how the SAME candidate's expectancy evolves across chronological segments,
-- not just a single confirm/reject decision.
CREATE TABLE IF NOT EXISTS discovered_setups (
    setup_id                  TEXT PRIMARY KEY,
    regime                      TEXT,              -- NULL = regime-agnostic
    technical_conditions_json     TEXT NOT NULL,     -- {"trend_direction": "UP", "breakout_state": "BREAKOUT_UP", ...}
    horizon_days                    INTEGER NOT NULL,
    invalidation_pct                  REAL,           -- fixed reference stop, same convention as strategy/decision_process.py
    train_result_json                   TEXT,
    validate_result_json                  TEXT,
    shadow_result_json                      TEXT,
    test_result_json                          TEXT,
    status                                      TEXT NOT NULL,  -- see setups/setup_discovery.py's SETUP_STATUSES
    created_at                                    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_discovered_setups_status ON discovered_setups(status);

-- Schema v5: the AI Market Research & Analysis product (market_agent/research/). A DELIBERATE
-- PRODUCT PIVOT - see research/schema.py's module docstring. `research_reports` stores one
-- COMPLETE, self-contained JSON snapshot per research pass (the whole ResearchReport.to_dict()) -
-- deliberately NOT normalized into one table per sub-entity (source documents, narratives, risks,
-- ...): every research pass recomputes its full evidence set fresh from live providers each time,
-- so what needs to persist is simply "what did the LAST report say" (for change_detection.py's
-- "what changed since last time" - item 11/18), not a queryable relational history of every
-- narrative/risk ever seen. `research_watchlist` is the simple persistent watch-list (item 23).
CREATE TABLE IF NOT EXISTS research_reports (
    report_id            TEXT PRIMARY KEY,
    entity                  TEXT NOT NULL,
    generated_at               TEXT NOT NULL,
    assessment                    TEXT NOT NULL,
    report_json                     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_research_reports_entity ON research_reports(entity);
CREATE INDEX IF NOT EXISTS idx_research_reports_generated_at ON research_reports(generated_at);

CREATE TABLE IF NOT EXISTS research_watchlist (
    entity            TEXT PRIMARY KEY,
    added_at             TEXT NOT NULL
);

-- Schema v6: the News State Engine (market_agent/research/news_state/). `news_event_vectors`
-- persists every EventVector ever computed for an entity, append-only - this is BOTH the audit trail
-- (every number traces back to a real extraction pass) AND the ONLY legitimate source for novelty
-- (an event's implication-shape compared against this entity's OWN prior history, never fabricated -
-- see news_state/aggregation.py). `news_company_states` persists one CompanyNewsState snapshot per
-- aggregation pass, enabling real ΔN(t)/velocity/acceleration against the PRIOR persisted state -
-- never a synthetic/interpolated one.
CREATE TABLE IF NOT EXISTS news_event_vectors (
    event_vector_id       TEXT PRIMARY KEY,
    entity                   TEXT NOT NULL,
    as_of                       TEXT NOT NULL,
    computed_at                    TEXT NOT NULL,
    event_vector_json                 TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_news_event_vectors_entity ON news_event_vectors(entity);
CREATE INDEX IF NOT EXISTS idx_news_event_vectors_as_of ON news_event_vectors(as_of);

CREATE TABLE IF NOT EXISTS news_company_states (
    state_id            TEXT PRIMARY KEY,
    entity                 TEXT NOT NULL,
    as_of                     TEXT NOT NULL,
    computed_at                  TEXT NOT NULL,
    state_json                      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_news_company_states_entity ON news_company_states(entity);
CREATE INDEX IF NOT EXISTS idx_news_company_states_as_of ON news_company_states(as_of);

-- Schema v7: the prospective evaluation harness (market_agent/research/evaluation/). ONE row per
-- (entity, mode, triggered_at) - never overwritten, never deleted, so the log is a real audit trail,
-- not a mutable "current belief". `mode` is one of A_NO_NEWS / B_BLENDED / C_NEWS_ONLY (see
-- evaluation/modes.py). `predicted_impact`/`predicted_confidence` are NULL when the underlying
-- decision carried no tradeable signal (e.g. mode A/B's assessment was INSUFFICIENT_EVIDENCE) -
-- never coerced to 0, which would be a false "no change expected" claim rather than "no signal at
-- all". The four realized_return_* / resolved_*_at column pairs stay NULL until that many TRADING
-- days have genuinely elapsed since triggered_at AND outcome_resolution.py has actually computed
-- them from real price data - see that module for why this can only ever be filled in
-- prospectively, never backfilled from a value computed before the horizon passed.
CREATE TABLE IF NOT EXISTS prediction_log (
    prediction_id           TEXT PRIMARY KEY,
    entity                     TEXT NOT NULL,
    mode                          TEXT NOT NULL,
    triggered_at                     TEXT NOT NULL,
    model_version                       TEXT NOT NULL,
    decision_label                         TEXT,
    predicted_impact                          REAL,
    predicted_confidence                         REAL,
    inputs_snapshot_json                            TEXT NOT NULL,
    realized_return_1d REAL, resolved_1d_at TEXT,
    realized_return_5d REAL, resolved_5d_at TEXT,
    realized_return_20d REAL, resolved_20d_at TEXT,
    realized_return_60d REAL, resolved_60d_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_prediction_log_entity ON prediction_log(entity);
CREATE INDEX IF NOT EXISTS idx_prediction_log_mode ON prediction_log(mode);
CREATE INDEX IF NOT EXISTS idx_prediction_log_triggered_at ON prediction_log(triggered_at);
"""


NEW_RELATIONSHIP_COLUMNS: dict[str, str] = {
    "shadow_started_at": "TEXT",
    "shadow_promoted_at": "TEXT",
}


def _migrate(conn: sqlite3.Connection) -> None:
    """Adds SCHEMA_VERSION 2/3 columns to a database created under an
    older schema, without touching any existing row. Idempotent - checks
    PRAGMA table_info before every ALTER, safe to call on every connect().
    The two brand-new v3 TABLES (trading_methodologies,
    methodology_concept_links) don't need an entry here - their
    `CREATE TABLE IF NOT EXISTS` in the DDL string above is already
    idempotent on its own."""
    for table, columns in (("episodic_events", NEW_EPISODIC_COLUMNS),
                            ("validated_relationships", NEW_RELATIONSHIP_COLUMNS),
                            ("candidate_hypotheses", NEW_HYPOTHESIS_COLUMNS),
                            ("validated_relationships", NEW_RELATIONSHIP_COLUMNS_V3)):
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for col, coltype in columns.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
    conn.commit()


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)
    conn.commit()
    _migrate(conn)
