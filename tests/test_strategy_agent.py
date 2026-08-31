from market_agent.strategy.decision_process import (
    ConfirmationRequirement, EntryTrigger, ExitCondition, InvalidationCondition, MethodologyDecisionProcess,
    RegimeCondition, RiskConstraint, SetupCondition,
)
from market_agent.strategy.strategy_agent import COST_MARGIN_MULTIPLE, StrategyAgent
from market_agent.experiment.portfolio_metrics import TRANSACTION_COST_PER_TRADE


def _validated_dp(effect=0.05, ci_low=0.02, ci_high=0.08, n=40, regime_value="RISK_ON",
                   setup_dim="breakout_state", setup_value="BREAKOUT_UP", horizon=20):
    return MethodologyDecisionProcess(
        concept="BREAKOUT", horizon_days=horizon, event_type="GUIDANCE_CHANGE", direction="positive",
        regime=RegimeCondition("regime", regime_value) if regime_value else None,
        setup=SetupCondition("BREAKOUT", setup_dim, setup_value),
        entry=EntryTrigger("NEXT_AVAILABLE_PRICE", "entry desc"),
        confirmation=ConfirmationRequirement(False, "no confirmation"),
        invalidation=InvalidationCondition(0.10, "fixed stop"),
        exit=ExitCondition("FIXED_HORIZON", horizon, "exit desc"),
        risk=RiskConstraint(0.01, "risk desc"),
        technical_concepts_used=["BREAKOUT"], provenance_methodology_ids=["meth-1"],
        evidence_status="STATISTICALLY_VALIDATED", source_relationship_id="rel-1",
        effect_estimate=effect, n_supporting=n, ci_low=ci_low, ci_high=ci_high,
    )


def _hypothesis_dp():
    return MethodologyDecisionProcess(
        concept="BREAKOUT", horizon_days=20, event_type="GUIDANCE_CHANGE", direction="positive", regime=None,
        setup=SetupCondition("BREAKOUT", "(unspecified)", "(any)"),
        entry=EntryTrigger("NEXT_AVAILABLE_PRICE", "hypothesis only"),
        confirmation=ConfirmationRequirement(False, "hypothesis only"),
        invalidation=InvalidationCondition(None, "hypothesis only"),
        exit=ExitCondition("FIXED_HORIZON", 20, "hypothesis only"),
        risk=RiskConstraint(0.01, "hypothesis only"),
        technical_concepts_used=["BREAKOUT"], provenance_methodology_ids=["meth-1"],
        evidence_status="HYPOTHESIS_ONLY",
    )


def test_never_trades_a_hypothesis_only_decision_process():
    agent = StrategyAgent()
    decision = agent.decide(_hypothesis_dp(), "ACME", {"breakout_state": "BREAKOUT_UP", "regime": "RISK_ON"})
    assert decision.action == "ABSTAIN"
    assert "HYPOTHESIS_ONLY" in decision.abstain_reason


def test_abstains_when_setup_condition_not_currently_met():
    agent = StrategyAgent()
    dp = _validated_dp()
    decision = agent.decide(dp, "ACME", {"breakout_state": "NONE", "regime": "RISK_ON"})
    assert decision.action == "ABSTAIN"
    assert "Setup condition not currently met" in decision.abstain_reason


def test_abstains_when_regime_condition_not_currently_met():
    agent = StrategyAgent()
    dp = _validated_dp()
    decision = agent.decide(dp, "ACME", {"breakout_state": "BREAKOUT_UP", "regime": "RISK_OFF"})
    assert decision.action == "ABSTAIN"
    assert "Regime condition not currently met" in decision.abstain_reason


def test_abstains_when_confidence_interval_spans_zero():
    agent = StrategyAgent()
    dp = _validated_dp(effect=0.03, ci_low=-0.01, ci_high=0.07)
    decision = agent.decide(dp, "ACME", {"breakout_state": "BREAKOUT_UP", "regime": "RISK_ON"})
    assert decision.action == "ABSTAIN"
    assert "spans zero" in decision.abstain_reason
    assert decision.predicted_return == 0.03  # the raw prediction is still surfaced, separate from the decision


def test_abstains_when_edge_does_not_clear_transaction_costs():
    agent = StrategyAgent()
    tiny_effect = COST_MARGIN_MULTIPLE * TRANSACTION_COST_PER_TRADE * 0.5  # below the required margin
    dp = _validated_dp(effect=tiny_effect, ci_low=tiny_effect * 0.5, ci_high=tiny_effect * 1.5)
    decision = agent.decide(dp, "ACME", {"breakout_state": "BREAKOUT_UP", "regime": "RISK_ON"})
    assert decision.action == "ABSTAIN"
    assert "transaction-cost margin" in decision.abstain_reason


def test_commits_to_long_when_everything_clears():
    agent = StrategyAgent()
    dp = _validated_dp(effect=0.05, ci_low=0.02, ci_high=0.08, n=60)
    decision = agent.decide(dp, "ACME", {"breakout_state": "BREAKOUT_UP", "regime": "RISK_ON"})
    assert decision.action == "LONG"
    assert decision.predicted_return == 0.05
    assert decision.confidence == "HIGH"  # n=60 clears the HIGH threshold (>=50)
    assert decision.expected_holding_days == 20
    assert decision.invalidation_level == -0.10  # negative for a LONG position
    assert decision.source_relationship_id == "rel-1"
    assert len(decision.reasoning) >= 2


def test_commits_to_short_for_a_negative_effect():
    agent = StrategyAgent()
    dp = _validated_dp(effect=-0.05, ci_low=-0.08, ci_high=-0.02, n=60)
    decision = agent.decide(dp, "ACME", {"breakout_state": "BREAKOUT_UP", "regime": "RISK_ON"})
    assert decision.action == "SHORT"
    assert decision.invalidation_level == 0.10  # positive (adverse-to-a-short direction) for a SHORT position


def test_decision_process_with_no_regime_specified_ignores_context_regime():
    agent = StrategyAgent()
    dp = _validated_dp(regime_value=None)
    decision = agent.decide(dp, "ACME", {"breakout_state": "BREAKOUT_UP", "regime": "RISK_OFF"})
    assert decision.action == "LONG"  # no regime constraint to fail
