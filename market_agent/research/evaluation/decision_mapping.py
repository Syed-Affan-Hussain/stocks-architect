"""Maps the research product's qualitative ASSESSMENT to a signed
predicted_impact in [-1,1], purely for THIS evaluation harness - the
product's own report (assessment.py) is completely unchanged by this
module and stays exactly what it has always been: a disclosed, transparent
scoring rule producing a non-trading-verb label, never touched here.

WHY THIS EXISTS, AND WHY IT'S KEPT SEPARATE: assessment.py's own docstring
is explicit - "NOT a buy/sell signal... none of which are trading verbs."
That is a deliberate product decision and this module does not reverse it.
But Sharpe/Sortino/drawdown/turnover (portfolio_metrics.py, reused
unchanged) are mathematically meaningless without SOME signed quantity to
score outcomes against. This module is that bridge, confined to the
evaluation harness, fully disclosed, and versioned - never imported by
assessment.py, report_format.py, or anything the end-user-facing report
renders.

LITERATURE GROUNDING (verified by direct extraction of the actual paper
text, not assumed from title/abstract):

Pontes et al., "Backtesting Sentiment Signals for Trading: Evaluating the
Viability of Alpha Generation from Sentiment Analysis" (arXiv:2507.03350),
Section 3.1.3-3.1.4: converts categorical sentiment labels (negative/
neutral/positive) to signed values (-1/0/+1) exactly this way ("As the
FinBERT and DualGCN predictions are only labels, we replace the negative,
neutral, and positive labels by the values -1, 0, and +1"), then applies a
symmetric two-threshold rule (their tuned BUY_SIGNAL/SELL_SIGNAL bands) to
decide Buy/Sell/Neutral, backtested with Sharpe, Sortino, and max
drawdown - the exact metric set this project's own portfolio_metrics.py
(from the pre-existing trading-research system) already computes. This
module borrows that label-to-signed-value convention and applies it to
this project's own 7-way ASSESSMENTS scale instead of a 3-way sentiment
label; it does not borrow their specific threshold VALUES (40/60 on a
0-100 scale, tuned to their own data) - those numbers are that paper's
domain-specific calibration, not a universal constant, and this project
has no comparable tuning dataset to fit its own thresholds against either.
This is disclosed as OUR OWN choice below, not claimed as literature-
specified.

INSUFFICIENT_EVIDENCE MAPS TO NO SIGNAL AT ALL (predicted_impact=None),
never to 0.0 - a 0.0 would claim "the model expects no change", which is a
real, substantive prediction; "the model had nothing to go on" is a
different claim entirely, and conflating them would be exactly the kind
of fabricated-precision this project's SOURCE_UNAVAILABLE discipline
exists to prevent everywhere else.
"""
from __future__ import annotations

from dataclasses import dataclass

from market_agent.research.schema import ASSESSMENTS

MODEL_VERSION = "decision_mapping_v1"

# OUR OWN calibration - see module docstring. Ordered strongest-favorable to strongest-unfavorable;
# symmetric around NEUTRAL/UNCERTAIN=0.0 by construction, not fit to any outcome data.
ASSESSMENT_TO_SIGNED_IMPACT: dict[str, float | None] = {
    "FAVORABLE": 1.0,
    "CAUTIOUSLY_FAVORABLE": 0.5,
    "NEUTRAL": 0.0,
    "UNCERTAIN": 0.0,
    "CAUTIOUS": -0.5,
    "NEGATIVE": -1.0,
    "INSUFFICIENT_EVIDENCE": None,   # no signal - see module docstring, never coerced to 0.0
}
assert set(ASSESSMENT_TO_SIGNED_IMPACT) == set(ASSESSMENTS), \
    "ASSESSMENT_TO_SIGNED_IMPACT must cover every schema.ASSESSMENTS value - fail loudly, not silently"

# metrics.py's Brier-score machinery (reused, unmodified) expects a HIGH/MEDIUM/LOW confidence label,
# not our [0,1] float - a disclosed, fixed-threshold adapter, not a re-calibration of that module.
CONFIDENCE_HIGH_THRESHOLD = 0.7
CONFIDENCE_MEDIUM_THRESHOLD = 0.4


@dataclass(frozen=True)
class MappedDecision:
    decision_label: str                  # the source ASSESSMENT string, unchanged - full audit trail
    predicted_impact: float | None       # signed [-1,1], None if no tradeable signal
    predicted_confidence: float | None   # the source assessment_confidence, passed through unchanged


def map_assessment_to_decision(assessment: str, assessment_confidence: float | None) -> MappedDecision:
    if assessment not in ASSESSMENT_TO_SIGNED_IMPACT:
        raise ValueError(f"Unknown assessment label: {assessment!r}")
    impact = ASSESSMENT_TO_SIGNED_IMPACT[assessment]
    confidence = assessment_confidence if impact is not None else None
    return MappedDecision(decision_label=assessment, predicted_impact=impact, predicted_confidence=confidence)


def confidence_float_to_bucket(confidence: float | None) -> str:
    """Adapter for market_agent/experiment/metrics.py's CONFIDENCE_TO_PROB
    dict (HIGH/MEDIUM/LOW), which that module already uses for its own
    Brier-score calculation - reused unmodified, not re-implemented."""
    if confidence is None:
        return "LOW"
    if confidence >= CONFIDENCE_HIGH_THRESHOLD:
        return "HIGH"
    if confidence >= CONFIDENCE_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def news_state_to_decision(news_state: dict | None) -> MappedDecision:
    """Mode C (news-only): predicted_impact derived PURELY from
    CompanyNewsState's own quantified axes - no narratives/risks/
    consistency checks at all. Each IMPLICATION_AXES value is already
    bounded to [-1,1] by the (frozen) Event Quantifier's own anchor
    tables, so the mean of the non-null axes is itself already in
    [-1,1] with no further rescaling needed. None (no signal) if the
    news state is unavailable or carried no signal on any axis at all -
    same no-fabrication discipline as INSUFFICIENT_EVIDENCE above."""
    if news_state is None:
        return MappedDecision(decision_label="NEWS_UNAVAILABLE", predicted_impact=None, predicted_confidence=None)
    dims = [v for v in news_state.get("dimensions", {}).values() if v is not None]
    if not dims:
        return MappedDecision(decision_label="NEWS_NO_SIGNAL", predicted_impact=None, predicted_confidence=None)
    impact = round(sum(dims) / len(dims), 4)
    label = "NEWS_ONLY_UP" if impact > 0 else ("NEWS_ONLY_DOWN" if impact < 0 else "NEWS_ONLY_FLAT")
    return MappedDecision(decision_label=label, predicted_impact=impact,
                           predicted_confidence=news_state.get("confidence"))
