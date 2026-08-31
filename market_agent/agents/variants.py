"""The three ADAPTIVE agent variants stage 6's four-way experiment
compares (alongside STATIC, agents/static_agent.py) - each is the SAME
AdaptiveAgent implementation (agents/adaptive_agent.py), differing only in
which relationships it's eligible to match against (see that module's
CONCEPT_FILTER_CLAUSES docstring for exactly what each filter means and
why). No agent logic is duplicated three times; only the eligibility rule
differs.
"""
from __future__ import annotations

import sqlite3

from market_agent.agents.adaptive_agent import AdaptiveAgent

CURRENT_ADAPTIVE_MODEL_VERSION = "CURRENT_ADAPTIVE_v1"
TECHNICAL_ADAPTIVE_MODEL_VERSION = "TECHNICAL_ADAPTIVE_v1"
METHODOLOGY_ADAPTIVE_MODEL_VERSION = "METHODOLOGY_ADAPTIVE_v1"
ENSEMBLE_ADAPTIVE_MODEL_VERSION = "ENSEMBLE_ADAPTIVE_v1"


def make_current_adaptive_agent(conn: sqlite3.Connection, unconditional_baseline: dict[int, float],
                                 model_version: str = CURRENT_ADAPTIVE_MODEL_VERSION) -> AdaptiveAgent:
    """"CURRENT ADAPTIVE": relationships conditioned only on the stage 1-5
    event-context dimensions (regime/prior_return_bucket/vol_bucket) -
    exactly what this agent could see before stage 6 added technical
    concepts, so the four-way comparison isolates what technical concepts
    actually change rather than comparing against a moving target."""
    return AdaptiveAgent(conn, unconditional_baseline, model_version=model_version,
                          concept_filter="EVENT_CONTEXT_ONLY")


def make_technical_adaptive_agent(conn: sqlite3.Connection, unconditional_baseline: dict[int, float],
                                   model_version: str = TECHNICAL_ADAPTIVE_MODEL_VERSION) -> AdaptiveAgent:
    """"TECHNICAL ADAPTIVE": any relationship that touches a canonical
    trading concept (concepts/ontology.py), regardless of whether any
    methodology happened to map onto it."""
    return AdaptiveAgent(conn, unconditional_baseline, model_version=model_version,
                          concept_filter="ANY_TECHNICAL_CONCEPT")


def make_methodology_informed_adaptive_agent(conn: sqlite3.Connection, unconditional_baseline: dict[int, float],
                                              model_version: str = METHODOLOGY_ADAPTIVE_MODEL_VERSION
                                              ) -> AdaptiveAgent:
    """"METHODOLOGY-INFORMED ADAPTIVE": restricted further to concepts at
    least one ingested methodology (methodology/) independently claimed.
    Methodology provenance is NEVER itself evidence (see
    methodology/schema.py) - this only changes which ALREADY-VALIDATED
    (real, out-of-sample-tested) relationships the agent is eligible to
    use, never how a relationship gets validated in the first place."""
    return AdaptiveAgent(conn, unconditional_baseline, model_version=model_version,
                          concept_filter="METHODOLOGY_BACKED_CONCEPT")


def make_ensemble_adaptive_agent(conn: sqlite3.Connection, unconditional_baseline: dict[int, float],
                                  qualified_relationship_ids: set[str],
                                  model_version: str = ENSEMBLE_ADAPTIVE_MODEL_VERSION) -> AdaptiveAgent:
    """"ENSEMBLE ADAPTIVE" (stage 7 item 6) - NOT a weighted average of
    the other agents' predictions, which would be an arbitrary model with
    no principled basis for its weights. Instead a GOVERNED SELECTION:
    eligible ONLY for relationships whose relationship_id is in
    `qualified_relationship_ids` - the set that has passed the FULL
    stage-7 diagnostic stack (the existing significance test,
    learn/incremental_value.py, and learn/overfitting_diagnostics.py's
    permutation/temporal-stability checks), computed by a real governance
    pass using only information available as of that point in the walk
    (see scripts/run_stage7_final_report.py for where that set is
    actually built - periodically, from real data, never per-prediction).
    An unqualified relationship contributes NOTHING here even if it is
    ACTIVE and would ordinarily be used by TECHNICAL_ADAPTIVE or
    METHODOLOGY_ADAPTIVE - falls back to the SAME unconditional baseline
    as every other agent when nothing qualified matches."""
    return AdaptiveAgent(conn, unconditional_baseline, model_version=model_version,
                          qualified_relationship_ids=qualified_relationship_ids)
