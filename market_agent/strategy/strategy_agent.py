"""StrategyAgent - stage 7 item 2: consumes a structured
MethodologyDecisionProcess (strategy/decision_process.py) and produces an
explicit trading ACTION, never a bare return prediction.

SEPARATES "prediction of return" FROM "decision to trade" - the core
requirement this module exists to enforce. A decision process can have a
statistically CONFIRMED, positive expected return and StrategyAgent can
still ABSTAIN, because:

  1. It NEVER acts on a HYPOTHESIS_ONLY decision process - only
     evidence_status="STATISTICALLY_VALIDATED" (see decision_process.py's
     own docstring for why methodology claims alone are never evidence).
  2. The setup/regime conditions must actually be met in the CURRENT
     context - a validated relationship says nothing about a context it
     doesn't match.
  3. The relationship's own confidence interval must not span zero - a
     statistically "CONFIRMED" effect (p < ALPHA after correction) can
     still carry a wide CI; if the interval includes zero, the DIRECTION
     of the edge isn't actually pinned down closely enough to trade.
  4. The economic gate: the effect must clear transaction costs by a
     fixed, disclosed margin (COST_MARGIN_MULTIPLE x
     TRANSACTION_COST_PER_TRADE, reusing experiment/portfolio_metrics.py's
     own disclosed cost assumption, not a new tunable). A statistically
     real but economically tiny edge is not tradeable after costs.

Only if ALL FOUR checks pass does StrategyAgent commit to LONG or SHORT.
Every other outcome is ABSTAIN, with the specific reason recorded - never
a forced trade.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from market_agent.agents.adaptive_agent import CONFIDENCE_N_THRESHOLDS
from market_agent.experiment.portfolio_metrics import TRANSACTION_COST_PER_TRADE
from market_agent.strategy.decision_process import MethodologyDecisionProcess

COST_MARGIN_MULTIPLE = 2.0  # fixed, disclosed - the effect must exceed this multiple of the round-trip
#                             transaction cost assumption to justify trading net of costs, not just
#                             clearing zero.


def _confidence_for_n(n: int | None) -> str:
    if n is None:
        return "LOW"
    for threshold, label in CONFIDENCE_N_THRESHOLDS:
        if n >= threshold:
            return label
    return "LOW"


@dataclass
class StrategyDecision:
    action: str  # "LONG" | "SHORT" | "FLAT" | "ABSTAIN"
    entity: str
    concept: str | None
    entry_condition: str | None
    invalidation_level: float | None       # signed return-space threshold, not an absolute price
    exit_condition: str | None
    expected_holding_days: int | None
    confidence: str | None                  # "HIGH" | "MEDIUM" | "LOW" | None if ABSTAIN before evidence exists
    predicted_return: float | None          # the RAW statistical prediction - kept separate from `action`
    position_risk_pct: float | None
    reasoning: list[str] = field(default_factory=list)
    source_relationship_id: str | None = None
    abstain_reason: str | None = None


class StrategyAgent:
    def __init__(self, transaction_cost: float = TRANSACTION_COST_PER_TRADE,
                 cost_margin_multiple: float = COST_MARGIN_MULTIPLE):
        self.transaction_cost = transaction_cost
        self.cost_margin_multiple = cost_margin_multiple

    def _abstain(self, entity: str, concept: str | None, reason: str,
                 predicted_return: float | None = None, source_relationship_id: str | None = None) -> StrategyDecision:
        return StrategyDecision(action="ABSTAIN", entity=entity, concept=concept, entry_condition=None,
                                 invalidation_level=None, exit_condition=None, expected_holding_days=None,
                                 confidence=None, predicted_return=predicted_return, position_risk_pct=None,
                                 reasoning=[reason], source_relationship_id=source_relationship_id,
                                 abstain_reason=reason)

    def decide(self, decision_process: MethodologyDecisionProcess, entity: str,
                current_context: dict) -> StrategyDecision:
        if decision_process.evidence_status != "STATISTICALLY_VALIDATED":
            return self._abstain(entity, decision_process.concept,
                                  "Decision process is HYPOTHESIS_ONLY - a methodology's own claim is never "
                                  "traded on directly, only a statistically validated relationship.")

        if current_context.get(decision_process.setup.dimension) != decision_process.setup.value:
            return self._abstain(entity, decision_process.concept,
                                  f"Setup condition not currently met: {decision_process.setup.dimension}="
                                  f"{decision_process.setup.value!r} required, context has "
                                  f"{current_context.get(decision_process.setup.dimension)!r}.",
                                  source_relationship_id=decision_process.source_relationship_id)

        if decision_process.regime is not None and current_context.get("regime") != decision_process.regime.value:
            return self._abstain(entity, decision_process.concept,
                                  f"Regime condition not currently met: {decision_process.regime.value!r} "
                                  f"required, context has {current_context.get('regime')!r}.",
                                  source_relationship_id=decision_process.source_relationship_id)

        if decision_process.ci_low is not None and decision_process.ci_high is not None:
            if decision_process.ci_low <= 0 <= decision_process.ci_high:
                return self._abstain(entity, decision_process.concept,
                                      f"95% CI [{decision_process.ci_low:+.2%}, {decision_process.ci_high:+.2%}] "
                                      "spans zero - the direction of the edge isn't pinned down closely enough "
                                      "to trade, even though the batch-corrected significance test passed.",
                                      predicted_return=decision_process.effect_estimate,
                                      source_relationship_id=decision_process.source_relationship_id)

        effect = decision_process.effect_estimate
        cost_threshold = self.cost_margin_multiple * self.transaction_cost
        if effect is None or abs(effect) < cost_threshold:
            return self._abstain(entity, decision_process.concept,
                                  f"Expected edge {effect!r} does not clear the required "
                                  f"{self.cost_margin_multiple}x transaction-cost margin "
                                  f"({cost_threshold:.2%}) - statistically real does not mean economically "
                                  "tradeable after costs.",
                                  predicted_return=effect, source_relationship_id=decision_process.source_relationship_id)

        action = "LONG" if effect > 0 else "SHORT"
        invalidation_pct = decision_process.invalidation.max_adverse_excursion_pct
        invalidation_level = (-invalidation_pct if action == "LONG" else invalidation_pct) \
            if invalidation_pct is not None else None

        reasoning = [
            f"Validated relationship {decision_process.source_relationship_id}: {decision_process.concept} "
            f"({decision_process.setup.dimension}={decision_process.setup.value!r}"
            + (f", regime={decision_process.regime.value!r}" if decision_process.regime else "") + ")",
            f"Effect estimate {effect:+.2%} (95% CI [{decision_process.ci_low:+.2%}, "
            f"{decision_process.ci_high:+.2%}]), N={decision_process.n_supporting}.",
            f"Clears the {self.cost_margin_multiple}x transaction-cost margin ({cost_threshold:.2%}).",
        ]
        return StrategyDecision(
            action=action, entity=entity, concept=decision_process.concept,
            entry_condition=decision_process.entry.description, invalidation_level=invalidation_level,
            exit_condition=decision_process.exit.description, expected_holding_days=decision_process.horizon_days,
            confidence=_confidence_for_n(decision_process.n_supporting), predicted_return=effect,
            position_risk_pct=decision_process.risk.max_position_risk_pct, reasoning=reasoning,
            source_relationship_id=decision_process.source_relationship_id, abstain_reason=None,
        )
