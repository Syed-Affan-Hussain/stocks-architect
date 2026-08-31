"""Turning a failed prediction into candidate hypotheses (category 4).

Blueprint section G: an LLM proposes the explanation; a SEPARATE
formalization step expresses it as a structured, testable condition over
existing schema fields. There is no LLM wired into this execution
environment (see events/interpret.py's module docstring for the same
constraint, and market_agent/llm/select.py for the explicit
HYPOTHESIS_PROVIDER switch) - `RuleBasedHypothesisGenerator` is the
honest stand-in, active until a real provider is configured.
LLMHypothesisGenerator (market_agent/llm/hypothesis_generator.py) is a
second implementation of `HypothesisGenerator`; nothing in
learn/hypothesis_testing.py or learn/governance.py needs to change when
it's used, because both of those only ever consume the already-formalized
condition_json, never the explanation prose.

STAGE 5: BOUNDED COMBINATORIAL CONDITIONING. Earlier stages proposed at
most 2 fixed shapes (regime alone; regime + prior-return-bucket).
RuleBasedHypothesisGenerator now proposes every non-empty SUBSET, up to
MAX_CONDITIONING_VARS in size, of the available real conditioning
dimensions - still a small, FIXED, DISCLOSED search space, not
unrestricted combinatorial explosion over every schema field. Every
candidate is logged to candidate_hypotheses before testing
(formalize_and_store writes each one), and all of them are tested
TOGETHER in the same Holm-Bonferroni batch (test_hypotheses_batch) - more
candidates costs more correction, exactly as it should; this module does
not get to propose more ideas for free.

STAGE 6: TECHNICAL TRADING CONCEPTS JOIN THE SAME BOUNDED POOL, DOUBLY
BOUNDED SO THE POOL NEVER EXPLODES. The 3 stage-1-5 event-context
dimensions (regime, prior_return_bucket, vol_bucket) are joined by up to
MAX_TECHNICAL_DIMENSIONS_PER_EVENT (3) technical-concept dimensions drawn
from concepts/technical_context.py's 18 computable states (stage 7 item 7
added 3 more, 15 -> 18) - NOT all 18 at once (that alone, even capped at
MAX_CONDITIONING_VARS=3 per combination, would mean choosing from a
21-dimension pool: C(21,1)+C(21,2)+C(21,3) = 1561 candidates per learnable
error, which is unrestricted feature mining in every meaningful sense
regardless of the arity cap). Instead: which
technical dimensions are even ELIGIBLE for a given event is itself bounded
to at most 3, selected by TECHNICAL_DIMENSION_PRIORITY - a FIXED, disclosed
order set before any hypothesis testing happens, never reordered based on
which dimension looks more promising after seeing results. With the
resulting <=6-dimension pool (3 event-context + <=3 technical,
MAX_CONDITIONING_VARS=3 still applies to arity), the worst case is
C(6,1)+C(6,2)+C(6,3) = 41 candidates - larger than stage 5's 7, but a
disclosed, fixed multiple of it, not an open-ended search. This is also
how "bounded combinations of trading concepts AND interactions with event
context" gets satisfied: a technical dimension can combine with another
technical dimension, with an event-context dimension, or stand alone -
all through the SAME powerset mechanism, all subject to the SAME arity cap.

Every technical-concept-touching hypothesis also records WHICH TradingConcept(s)
it operationalizes and which methodologies (methodology/) independently mapped
to that concept (see _resolve_provenance below) - PROVENANCE ONLY. Multiple
methodologies contributing to a concept is recorded for audit; it is never
treated as evidence, and never changes MIN_N, ALPHA, or any other gate in
learn/hypothesis_testing.py.
"""
from __future__ import annotations

import itertools
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from market_agent.concepts.technical_context import (
    DIMENSION_TO_CONCEPT, TECHNICAL_DEFAULT_VALUES, TECHNICAL_STATE_FIELD_NAMES,
)
from market_agent.learn.error_taxonomy import MAY_LEARN_FROM
from market_agent.retrieval.similarity import prior_return_bucket, vol_bucket

MAX_CONDITIONING_VARS = 3  # hard cap - see module docstring. Change this only as a disclosed,
#                            versioned decision (it changes VARIANT_LABEL's implicit meaning in
#                            experiment/walkforward.py), never silently to chase a better result.

MAX_TECHNICAL_DIMENSIONS_PER_EVENT = 3  # second, independent cap - see module docstring's stage 6 note.

CONDITIONING_DIMENSIONS: dict[str, callable] = {
    "regime": lambda context: context.get("regime") if context.get("regime") not in (None, "UNKNOWN") else None,
    "prior_return_bucket": lambda context: (
        lambda b: b if b != "UNKNOWN" else None)(prior_return_bucket(context.get("prior_5d_return"))),
    "vol_bucket": lambda context: (lambda b: b if b != "UNKNOWN" else None)(vol_bucket(context.get("realized_vol_20d"))),
}

def _make_technical_dimension_fn(field_name: str, default_values: tuple[str, ...]):
    def _fn(context: dict):
        value = context.get(field_name)
        return value if value is not None and value not in default_values else None
    return _fn


# Order fixed by TECHNICAL_STATE_FIELD_NAMES (concepts/technical_context.py) - disclosed, and never
# reordered after seeing which dimension "worked" (see module docstring). "Uninteresting"/default
# values per field come from the SAME TECHNICAL_DEFAULT_VALUES stage 7's hierarchical Level 1
# screening (learn/hierarchical_research.py) uses - one shared definition, not two that could drift.
TECHNICAL_CONDITIONING_DIMENSIONS: dict[str, callable] = {
    field_name: _make_technical_dimension_fn(field_name, TECHNICAL_DEFAULT_VALUES[field_name])
    for field_name in TECHNICAL_STATE_FIELD_NAMES
}
TECHNICAL_DIMENSION_PRIORITY: list[str] = list(TECHNICAL_CONDITIONING_DIMENSIONS.keys())


@dataclass
class ProposedHypothesis:
    condition: dict          # structured, testable filter - e.g. {"event_type":..,"direction":..,"regime":..}
    explanation_text: str    # audit-trail prose only - never consumed as fact downstream


class HypothesisGenerator(ABC):
    """NAME is a required class attribute for the same reason
    events.interpret.Interpreter.NAME is - so it's always obvious, from
    any run's output, which generator actually produced a given
    hypothesis. See market_agent/llm for the LLM-backed implementation."""
    NAME: str

    @abstractmethod
    def generate(self, event_row, error_type: str) -> list[ProposedHypothesis]:
        raise NotImplementedError


class RuleBasedHypothesisGenerator(HypothesisGenerator):
    """Proposes every non-empty subset (up to MAX_CONDITIONING_VARS) of
    whichever real conditioning dimensions are available for this event -
    the 3 event-context dimensions (stage 1-5) plus, when
    `include_technical_dimensions=True`, up to
    MAX_TECHNICAL_DIMENSIONS_PER_EVENT technical-concept dimensions
    (stage 6) - see module docstring for why the technical pool is
    ADDITIONALLY bounded before the powerset is even taken. Returns [] for
    error types §F already excludes from learning (MAY_LEARN_FROM gate,
    checked here as a second, defense-in-depth guard - the caller should
    already have checked this too).

    STAGE 7: `include_technical_dimensions` defaults to True for backward
    compatibility with stage 6, but the stage-6 real run showed this flat,
    reactive, per-event approach dilutes Holm-Bonferroni correction so
    severely across tens of thousands of simultaneously-tested technical
    combinations that literally zero ever got confirmed (see
    scripts/run_stage6_experiment.py's commit message). Stage 7's
    hierarchical research procedure (learn/hierarchical_research.py) is
    the REPLACEMENT mechanism for technical/methodology concept discovery
    - a periodic, budget-bounded, level-gated batch pass, not a reactive
    per-prediction-error trigger. A caller wiring up the stage-7 agents
    should construct this generator with `include_technical_dimensions=False`
    so the reactive path stays exactly what it was in stage 1-5
    (event-context-only, driving EVENT_ADAPTIVE) while technical/
    methodology discovery happens entirely through the hierarchical pass
    instead - two mechanisms, not one diluting the other's correction
    batch."""

    NAME = "RULE_BASED"

    def __init__(self, include_technical_dimensions: bool = True):
        self.include_technical_dimensions = include_technical_dimensions

    def generate(self, event_row, error_type: str) -> list[ProposedHypothesis]:
        if not MAY_LEARN_FROM.get(error_type, False):
            return []
        context = json.loads(event_row["context_json"])

        available = {name: fn(context) for name, fn in CONDITIONING_DIMENSIONS.items()}
        available = {name: value for name, value in available.items() if value is not None}

        if self.include_technical_dimensions:
            technical_available = {name: fn(context) for name, fn in TECHNICAL_CONDITIONING_DIMENSIONS.items()}
            technical_available = {name: value for name, value in technical_available.items() if value is not None}
            # Bounded pool: at most MAX_TECHNICAL_DIMENSIONS_PER_EVENT technical dimensions are even
            # eligible, chosen by the FIXED TECHNICAL_DIMENSION_PRIORITY order - never by which one
            # would make the hypothesis look most promising.
            technical_added = 0
            for name in TECHNICAL_DIMENSION_PRIORITY:
                if technical_added >= MAX_TECHNICAL_DIMENSIONS_PER_EVENT:
                    break
                if name in technical_available:
                    available[name] = technical_available[name]
                    technical_added += 1

        if not available:
            return []

        base_fields = {"event_type": event_row["event_type"], "direction": event_row["direction"]}
        proposals = []
        dims = list(available.keys())
        for size in range(1, min(MAX_CONDITIONING_VARS, len(dims)) + 1):
            for combo in itertools.combinations(dims, size):
                extra = {name: available[name] for name in combo}
                description = " AND ".join(f"{name}={value!r}" for name, value in extra.items())
                proposals.append(ProposedHypothesis(
                    condition={**base_fields, **extra},
                    explanation_text=self._explain(event_row, error_type, description),
                ))
        return proposals

    @staticmethod
    def _explain(event_row, error_type: str, condition_description: str) -> str:
        return (
            f"Rule-based stand-in (no LLM wired in this environment - see module docstring): prediction for "
            f"{event_row['entity']} ({event_row['event_type']}/{event_row['direction']}) missed "
            f"({error_type}) while {condition_description}. Candidate hypothesis: this event type's effect "
            f"differs from the unconditional baseline specifically under this condition. This is a prose "
            "explanation for audit purposes only - it is not treated as true until "
            "learn/hypothesis_testing.py tests it against prior, held-out history."
        )


def _resolve_provenance(conn, condition: dict) -> tuple[str | None, list[str] | None]:
    """Inspects a FINAL proposed condition dict for keys that correspond to
    a canonical TradingConcept (via DIMENSION_TO_CONCEPT) and looks up
    which methodologies (methodology/) independently mapped to each one -
    generic over any HypothesisGenerator implementation, since it only
    reads condition KEYS, never the generator's internals. PROVENANCE
    ONLY - see module docstring: this never influences the condition
    itself or any statistical gate."""
    from market_agent.store.db import methodologies_for_concept  # local import avoids a store<->learn cycle

    concepts = sorted({DIMENSION_TO_CONCEPT[k].value for k in condition if k in DIMENSION_TO_CONCEPT})
    if not concepts:
        return None, None
    methodology_ids: list[str] = []
    seen: set[str] = set()
    for concept_value in concepts:
        for row in methodologies_for_concept(conn, concept_value):
            if row["methodology_id"] not in seen:
                seen.add(row["methodology_id"])
                methodology_ids.append(row["methodology_id"])
    return ",".join(concepts), (methodology_ids or None)


def formalize_and_store(conn, generator: HypothesisGenerator, event_row, error_type: str,
                         horizon_days: int, proposed_at: datetime) -> list[str]:
    """Runs the generator and writes every proposed hypothesis to
    candidate_hypotheses (category 4) - EVERY candidate, before any
    testing happens, so the full search space this run actually explored
    is always auditable, win or lose. Returns the list of new
    hypothesis_ids (empty if this error type/event doesn't yield any)."""
    from market_agent.store.db import add_hypothesis  # local import avoids a store<->learn import cycle

    hypothesis_ids = []
    for proposed in generator.generate(event_row, error_type):
        concept, methodology_ids = _resolve_provenance(conn, proposed.condition)
        hid = add_hypothesis(conn, source_event_id=event_row["event_id"], condition=proposed.condition,
                              horizon_days=horizon_days, explanation_text=proposed.explanation_text,
                              proposed_at=proposed_at, concept=concept, methodology_ids=methodology_ids)
        hypothesis_ids.append(hid)
    return hypothesis_ids
