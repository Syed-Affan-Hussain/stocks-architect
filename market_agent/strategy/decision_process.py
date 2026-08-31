"""Canonical methodology decision-process representation - stage 7 item 1:
REGIME -> SETUP -> ENTRY -> CONFIRMATION -> INVALIDATION -> EXIT -> RISK,
structured, never free-form text.

TWO CONSTRUCTORS, TWO EVIDENCE LEVELS - the core discipline this module
exists to enforce ("do not claim a methodology is profitable merely
because a practitioner describes it"):

  build_hypothesis_only_decision_process() - built directly from a
    methodology's OWN CLAIMED concept (methodology_concept_links,
    methodology/ package), before any statistical test has run.
    evidence_status="HYPOTHESIS_ONLY". This is what a methodology's
    description IS - a hypothesis about a mechanism - never evidence.
    StrategyAgent (strategy/strategy_agent.py) MUST NEVER act on a
    HYPOTHESIS_ONLY decision process - see that module's own docstring.

  build_validated_decision_process() - built from a REAL
    validated_relationships row that has already passed
    learn/hypothesis_testing.py's significance test (SHADOW or ACTIVE
    status). evidence_status="STATISTICALLY_VALIDATED". This is the ONLY
    kind of decision process StrategyAgent is allowed to consume.

WHAT'S GENUINELY EXTRACTED VS. WHAT'S A DISCLOSED SYSTEM DEFAULT: REGIME
and SETUP come directly from the validated relationship's own
condition_json (real, tested, structural data - e.g. {"breakout_state":
"BREAKOUT_UP", "regime": "RISK_ON"}). ENTRY, CONFIRMATION, INVALIDATION,
EXIT, and RISK do NOT come from genuinely parsed per-methodology rules -
this project's methodology corpus (methodology/seed_corpus.py) is
paraphrased prose run through a keyword extractor (RuleBasedMethodologyExtractor),
which maps text to canonical CONCEPTS only, never to structured
entry/exit/risk parameters (that would need an LLM actually reading and
formalizing free-form trading rules, which - as everywhere else in this
project - is not available in this environment). Inventing specific
entry/stop/target numbers and presenting them as "the trader's own rules"
would be fabrication. Instead, ENTRY/CONFIRMATION/INVALIDATION/EXIT/RISK
here are a single, FIXED, DISCLOSED "reference execution policy" applied
uniformly to every validated relationship - never tuned per-relationship,
never presented as a specific practitioner's actual rule:

  ENTRY: at the next available price after the triggering event (the
    predicted_at timestamp already used everywhere else in this system -
    no intraday fill-price modeling exists).
  CONFIRMATION: none required beyond the setup/regime condition itself
    already matching (this system has no intraday confirmation-bar data).
  INVALIDATION: a fixed multiple (INVALIDATION_BASELINE_MULTIPLE) of the
    unconditional baseline magnitude AT THIS HORIZON - scales sensibly
    with horizon/typical volatility without being fit to any one
    relationship's own effect size.
  EXIT: fixed-horizon, at EXACTLY the relationship's own tested
    horizon_days - never a different, untested holding period.
  RISK: a fixed, standard position-sizing convention
    (DEFAULT_MAX_POSITION_RISK_PCT), not derived from or tuned to any
    result.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

INVALIDATION_BASELINE_MULTIPLE = 2.0    # fixed, disclosed - see module docstring
DEFAULT_MAX_POSITION_RISK_PCT = 0.01    # 1% notional risk per position - standard convention, not tuned


@dataclass(frozen=True)
class RegimeCondition:
    dimension: str
    value: str


@dataclass(frozen=True)
class SetupCondition:
    concept: str
    dimension: str
    value: str


@dataclass(frozen=True)
class EntryTrigger:
    trigger_type: str  # "NEXT_AVAILABLE_PRICE" - the only kind this system can honestly support
    description: str


@dataclass(frozen=True)
class ConfirmationRequirement:
    required: bool
    description: str


@dataclass(frozen=True)
class InvalidationCondition:
    max_adverse_excursion_pct: float | None  # None only if no baseline magnitude was available
    description: str


@dataclass(frozen=True)
class ExitCondition:
    exit_type: str  # "FIXED_HORIZON" - the only kind directly supported
    horizon_days: int
    description: str


@dataclass(frozen=True)
class RiskConstraint:
    max_position_risk_pct: float
    description: str


@dataclass
class MethodologyDecisionProcess:
    concept: str
    horizon_days: int
    event_type: str
    direction: str
    regime: RegimeCondition | None
    setup: SetupCondition
    entry: EntryTrigger
    confirmation: ConfirmationRequirement
    invalidation: InvalidationCondition
    exit: ExitCondition
    risk: RiskConstraint
    technical_concepts_used: list[str]
    provenance_methodology_ids: list[str] = field(default_factory=list)
    evidence_status: str = "HYPOTHESIS_ONLY"  # "HYPOTHESIS_ONLY" | "STATISTICALLY_VALIDATED"
    source_relationship_id: str | None = None
    effect_estimate: float | None = None
    n_supporting: int | None = None
    ci_low: float | None = None
    ci_high: float | None = None


def _condition_to_setup_and_regime(condition: dict) -> tuple[SetupCondition | None, RegimeCondition | None]:
    """Splits a relationship's condition_json into the one technical-
    concept-bearing key (the SETUP) and, if present, `regime` (the
    REGIME) - a relationship's condition may have neither, either, or
    both, plus event_type/direction which aren't part of either."""
    from market_agent.concepts.technical_context import DIMENSION_TO_CONCEPT

    setup, regime = None, None
    for key, value in condition.items():
        if key in ("event_type", "direction"):
            continue
        if key == "regime":
            regime = RegimeCondition(dimension="regime", value=value)
        elif key in DIMENSION_TO_CONCEPT:
            setup = SetupCondition(concept=DIMENSION_TO_CONCEPT[key].value, dimension=key, value=value)
    return setup, regime


def build_validated_decision_process(conn: sqlite3.Connection, relationship_row: sqlite3.Row,
                                      unconditional_baseline: dict[int, float]) -> MethodologyDecisionProcess | None:
    """The ONLY constructor StrategyAgent is allowed to consume - see
    module docstring. Returns None if the relationship's condition has no
    technical-concept-bearing key at all (a pure event-context
    relationship has no SETUP in the trading-concept sense this decision
    process represents)."""
    condition = json.loads(relationship_row["condition_json"])
    setup, regime = _condition_to_setup_and_regime(condition)
    if setup is None:
        return None

    horizon_days = relationship_row["horizon_days"]
    baseline_mag = unconditional_baseline.get(horizon_days)
    invalidation_pct = (INVALIDATION_BASELINE_MULTIPLE * baseline_mag) if baseline_mag is not None else None

    methodology_ids = (json.loads(relationship_row["methodology_ids_json"])
                        if relationship_row["methodology_ids_json"] else [])

    return MethodologyDecisionProcess(
        concept=setup.concept, horizon_days=horizon_days, event_type=condition["event_type"],
        direction=condition["direction"], regime=regime, setup=setup,
        entry=EntryTrigger("NEXT_AVAILABLE_PRICE",
                            "Entry at the next available price after the triggering event - this system has "
                            "no intraday fill-price model."),
        confirmation=ConfirmationRequirement(False, "No additional confirmation beyond the setup/regime "
                                              "condition itself matching - no intraday confirmation-bar data."),
        invalidation=InvalidationCondition(invalidation_pct,
                                            f"Fixed reference stop: {INVALIDATION_BASELINE_MULTIPLE}x the "
                                            f"unconditional baseline magnitude at this horizon "
                                            f"({baseline_mag!r}) - a system-wide default, not specific to this "
                                            "relationship or tuned to its own effect size."),
        exit=ExitCondition("FIXED_HORIZON", horizon_days,
                            f"Exit at exactly {horizon_days} days - the SAME horizon this relationship was "
                            "statistically tested at, never a different, untested holding period."),
        risk=RiskConstraint(DEFAULT_MAX_POSITION_RISK_PCT,
                             f"Fixed {DEFAULT_MAX_POSITION_RISK_PCT:.1%} notional risk per position - a "
                             "standard position-sizing convention, not derived from or tuned to any result."),
        technical_concepts_used=[setup.concept], provenance_methodology_ids=methodology_ids,
        evidence_status="STATISTICALLY_VALIDATED", source_relationship_id=relationship_row["relationship_id"],
        effect_estimate=relationship_row["effect_estimate"], n_supporting=relationship_row["n_supporting"],
        ci_low=relationship_row["ci_low"], ci_high=relationship_row["ci_high"],
    )


def build_hypothesis_only_decision_process(conn: sqlite3.Connection, methodology_id: str, concept: str,
                                            event_type: str, direction: str,
                                            horizon_days: int) -> MethodologyDecisionProcess:
    """Built directly from a methodology's own CLAIMED concept
    (methodology_concept_links) - NO statistical test has run. Every
    numeric field that would require real evidence (effect_estimate, N,
    CI) stays None; entry/confirmation/invalidation/exit/risk use the
    SAME disclosed reference-policy defaults as the validated constructor
    (there is no more genuine a per-methodology rule available here
    either), explicitly labeled hypothesis-only throughout. Never
    consumed by StrategyAgent - see module docstring."""
    setup = SetupCondition(concept=concept, dimension="(unspecified - methodology claim only)", value="(any)")
    return MethodologyDecisionProcess(
        concept=concept, horizon_days=horizon_days, event_type=event_type, direction=direction, regime=None,
        setup=setup,
        entry=EntryTrigger("NEXT_AVAILABLE_PRICE", "Hypothesis only - no statistical test has run."),
        confirmation=ConfirmationRequirement(False, "Hypothesis only - no statistical test has run."),
        invalidation=InvalidationCondition(None, "Hypothesis only - no statistical test has run."),
        exit=ExitCondition("FIXED_HORIZON", horizon_days, "Hypothesis only - no statistical test has run."),
        risk=RiskConstraint(DEFAULT_MAX_POSITION_RISK_PCT, "Hypothesis only - no statistical test has run."),
        technical_concepts_used=[concept], provenance_methodology_ids=[methodology_id],
        evidence_status="HYPOTHESIS_ONLY", source_relationship_id=None, effect_estimate=None, n_supporting=None,
        ci_low=None, ci_high=None,
    )
