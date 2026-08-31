"""Error classification - Blueprint section F. The taxonomy is a GATE, not
a label: `MAY_LEARN_FROM` on each error type controls whether it's even
eligible to become a hypothesis (learn/hypothesis.py checks this before
doing anything else). Stage 1 implements the subset reachable without a
confounding-event detector or a full attribution engine (both later
stages) - CONFOUNDING_EVENT and REGIME_MISMATCH are defined here but not
yet automatically detected; they exist so the schema doesn't need to
change when detection for them is added.
"""
from __future__ import annotations

from dataclasses import dataclass

# error_type -> may this error type ever become a learning hypothesis?
# Four of these are hard NO regardless of how large the residual was -
# see the blueprint's own error-taxonomy table for why: conflating these
# with genuine MODEL_ERROR is exactly how an adaptive system starts
# "learning" from noise and confounds instead of real relationships.
MAY_LEARN_FROM: dict[str, bool] = {
    "OK": False,                     # no error - nothing to learn from
    "WRONG_DIRECTION": True,
    "WRONG_MAGNITUDE": True,
    "NOVEL_EVENT": False,            # correct low-confidence output, not a model failure
    "CONFOUNDING_EVENT": False,      # another event likely drove the outcome - not this model's error
    "DATA_ERROR": False,             # operational; including it would poison training data
    "INSUFFICIENT_DATA": False,      # no prediction was made with confidence - nothing to diagnose
}

DIRECTION_MATTERS_THRESHOLD = 0.005  # 0.5% - below this, "direction" is not economically meaningful to score


@dataclass
class ErrorClassification:
    error_type: str
    error_value: float | None
    may_learn_from: bool
    evidence: str


def classify_error(predicted_impact: float | None, predicted_confidence: str,
                    realized_abnormal_return: float | None, outcome_status: str) -> ErrorClassification:
    """Stage-1 classifier. Confounding-event and regime-mismatch detection
    are NOT implemented here yet (both need an attribution/isolation-quality
    engine this stage doesn't build) - a genuinely confounded case will
    currently be classified as WRONG_DIRECTION/WRONG_MAGNITUDE, which
    means it CAN incorrectly become a hypothesis candidate today. This is
    a disclosed Stage-1 gap, not a silent one: the hypothesis-testing step
    (learn/hypothesis_testing.py) is the actual backstop, since a spurious
    hypothesis born from an unflagged confound still has to survive a
    held-out statistical test before it can be promoted."""
    if outcome_status != "OK" or realized_abnormal_return is None:
        return ErrorClassification("DATA_ERROR", None, MAY_LEARN_FROM["DATA_ERROR"],
                                    "Outcome could not be observed - not attributable to the model.")
    if predicted_confidence == "INSUFFICIENT_PRECEDENT" or predicted_impact is None:
        return ErrorClassification("INSUFFICIENT_DATA", None, MAY_LEARN_FROM["INSUFFICIENT_DATA"],
                                    "No confident prediction was made - correctly abstained, not an error.")

    error_value = realized_abnormal_return - predicted_impact
    predicted_sign = 1 if predicted_impact > 0 else (-1 if predicted_impact < 0 else 0)
    realized_sign = 1 if realized_abnormal_return > DIRECTION_MATTERS_THRESHOLD else (
        -1 if realized_abnormal_return < -DIRECTION_MATTERS_THRESHOLD else 0)

    if predicted_sign != 0 and realized_sign != 0 and predicted_sign != realized_sign:
        return ErrorClassification("WRONG_DIRECTION", error_value, MAY_LEARN_FROM["WRONG_DIRECTION"],
                                    f"Predicted {predicted_impact:+.2%}, realized {realized_abnormal_return:+.2%} - "
                                    "opposite sign.")

    magnitude_error_pct = abs(error_value) / max(abs(realized_abnormal_return), DIRECTION_MATTERS_THRESHOLD)
    if magnitude_error_pct > 0.5:  # off by more than 50% of the realized move, same direction - project-specific,
        #                            disclosed threshold, not fit to any dataset; revisit once Stage 2 has real data.
        return ErrorClassification("WRONG_MAGNITUDE", error_value, MAY_LEARN_FROM["WRONG_MAGNITUDE"],
                                    f"Predicted {predicted_impact:+.2%}, realized {realized_abnormal_return:+.2%} - "
                                    "right direction, magnitude materially off.")

    return ErrorClassification("OK", error_value, MAY_LEARN_FROM["OK"],
                                f"Predicted {predicted_impact:+.2%}, realized {realized_abnormal_return:+.2%} - "
                                "within tolerance.")
