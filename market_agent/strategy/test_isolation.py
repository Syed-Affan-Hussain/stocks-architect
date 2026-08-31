"""Automatic detection/rejection of TEST-informed parameter selection -
stage 7 item 4's explicit requirement: "Any parameter chosen using TEST
performance is a methodological violation and must be detected/rejected
automatically."

WHAT "AUTOMATIC" HONESTLY MEANS HERE: there is no way for a Python
process to omnisciently detect that a human read a printed number and
then, in an unrelated later action, "was influenced by" it - that is not
a mechanically checkable property. What IS mechanically enforceable, and
what this module provides, is a single, explicit, STATEFUL BOUNDARY: once
`mark_test_observed()` is called (by whichever code path first computes
or reports TEST-segment performance), every SUBSEQUENT call to
`assert_parameter_selection_allowed()` raises TestIsolationViolation
immediately - so a pipeline/script that tries to reconstruct a
StrategyAgent with different constants, re-filter which decision
processes/relationships to trade, or re-run governance AFTER computing
TEST metrics is stopped in code, not just by convention. Every parameter-
selecting call site in the stage-7 pipeline (StrategyAgent construction,
decision-process filtering, ResearchBudget selection) is expected to call
the guard first - see scripts/run_stage7_final_report.py for where.

This is a SEPARATE, complementary mechanism to
experiment/four_way_walkforward.py's `freeze_governance_during_test` -
that flag stops the GOVERNED RELATIONSHIP pipeline (hypothesis testing,
promotion, shadow evaluation) from writing anything once chronological
TEST time is reached, regardless of whether anyone has "looked at" TEST
results yet. This guard instead stops STRATEGY-LEVEL parameter selection
specifically once TEST results HAVE been observed - the two together
cover both the "chronological" and "informational" leakage vectors.
"""
from __future__ import annotations


class TestIsolationViolation(Exception):
    """Raised when strategy-level parameter selection is attempted after
    TEST-segment performance has already been observed."""


class TestIsolationGuard:
    def __init__(self):
        self._test_observed = False
        self._test_observed_context: str | None = None

    @property
    def test_observed(self) -> bool:
        return self._test_observed

    @property
    def test_observed_context(self) -> str | None:
        return self._test_observed_context

    def mark_test_observed(self, context: str) -> None:
        """Called ONCE, by whichever code path first computes or prints a
        TEST-segment strategy metric. Idempotent - a second call just
        keeps the ORIGINAL context (the first observation is what
        matters for the audit trail)."""
        if not self._test_observed:
            self._test_observed = True
            self._test_observed_context = context

    def assert_parameter_selection_allowed(self, action_description: str) -> None:
        """Call this BEFORE any action that selects/changes a parameter -
        constructing a StrategyAgent with specific constants, choosing
        which decision processes or relationships to trade, picking a
        ResearchBudget, retuning a threshold. Raises immediately if TEST
        results were already observed."""
        if self._test_observed:
            raise TestIsolationViolation(
                f"Refused: {action_description!r} was attempted after TEST-segment performance was already "
                f"observed (first observed during: {self._test_observed_context!r}). Any parameter chosen "
                "using TEST performance is a methodological violation - see this module's docstring.")

    def reset(self) -> None:
        """Only for constructing a FRESH pipeline run (a new guard should
        normally be created per run instead) - never call this to
        'un-observe' TEST results within the same run."""
        self._test_observed = False
        self._test_observed_context = None
