"""log_predictions_for_entity's own new logic is thin (call research_company,
compute the 3 modes, persist each) - research_company itself already has no
direct test coverage anywhere in this project (pipeline.py is a pure
orchestrator over already-tested pieces), so consistent with that
convention, this stubs research_company rather than hitting real
SEC/Yahoo/Google News network calls."""
from market_agent.research.evaluation.modes import MODE_A, MODE_B, MODE_C
from market_agent.research.evaluation.run import log_predictions_for_entity
from market_agent.research.schema import ChangeSummary, CompanyProfile, ResearchReport
from market_agent.store import db


def _fake_report():
    return ResearchReport(
        entity="ACME", generated_at="2024-06-01T00:00:00+00:00", llm_status="UNAVAILABLE",
        research_period_days=30, assessment="FAVORABLE", assessment_confidence=0.75,
        assessment_reasoning="test", executive_summary="test", profile=CompanyProfile(entity="ACME", cik=None,
        name="Acme Corp"), timeline=[], narratives=[], consistency_checks=[], contradictions=[], risks=[],
        positive_factors=[], market_context=None, historical_reactions=[],
        change=ChangeSummary(has_prior_report=False), strongest_evidence="none", weakest_evidence="none",
        major_uncertainty="none", what_would_strengthen="none", what_would_weaken="none", what_to_watch=[],
        sources=[], news_state={"dimensions": {"growth": 0.5}, "confidence": 0.4, "contradiction_axes": []},
    )


def test_logs_exactly_three_modes_from_one_report(monkeypatch):
    monkeypatch.setattr("market_agent.research.evaluation.run.research_company", lambda *a, **k: _fake_report())
    conn = db.connect(":memory:")
    ids = log_predictions_for_entity("ACME", conn)
    assert len(ids) == 3
    rows = db.prediction_log_for_entity(conn, "ACME")
    assert {r["mode"] for r in rows} == {MODE_A, MODE_B, MODE_C}


def test_each_logged_row_carries_a_full_inputs_snapshot(monkeypatch):
    monkeypatch.setattr("market_agent.research.evaluation.run.research_company", lambda *a, **k: _fake_report())
    conn = db.connect(":memory:")
    log_predictions_for_entity("ACME", conn)
    rows = db.prediction_log_for_entity(conn, "ACME")
    import json
    for r in rows:
        snapshot = json.loads(r["inputs_snapshot_json"])
        assert snapshot["entity"] == "ACME"
        assert snapshot["assessment"] == "FAVORABLE"
        assert "mode_reasoning" in snapshot
        assert snapshot["news_state"]["dimensions"]["growth"] == 0.5


def test_mode_a_and_mode_c_disagree_when_assessment_and_news_diverge(monkeypatch):
    monkeypatch.setattr("market_agent.research.evaluation.run.research_company", lambda *a, **k: _fake_report())
    conn = db.connect(":memory:")
    log_predictions_for_entity("ACME", conn)
    rows = {r["mode"]: r for r in db.prediction_log_for_entity(conn, "ACME")}
    assert rows[MODE_A]["predicted_impact"] == 1.0   # FAVORABLE
    assert rows[MODE_C]["predicted_impact"] == 0.5   # news_state's own growth=0.5


def test_one_failing_entity_does_not_abort_the_watchlist_batch(monkeypatch):
    from market_agent.research.evaluation.run import log_predictions_for_watchlist

    def flaky(entity, conn=None, **kwargs):
        if entity == "BAD":
            raise RuntimeError("simulated provider failure")
        return _fake_report()

    monkeypatch.setattr("market_agent.research.evaluation.run.research_company", flaky)
    conn = db.connect(":memory:")
    results = log_predictions_for_watchlist(conn, ["ACME", "BAD"], prices=object())
    assert len(results["ACME"]) == 3
    assert "FAILED" in results["BAD"][0]
