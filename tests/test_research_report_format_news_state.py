"""Validates the QUANTIFIED NEWS STATE section render_text() gained when
the News State Engine was wired into the main research report
(market_agent/research/pipeline.py). Tested against hand-built dicts
shaped exactly like news_state.schema.CompanyNewsState.to_dict()'s real
output, rather than constructing a full ResearchReport - report_format.py
has never had per-section unit tests (this product's other sections are
verified via real end-to-end runs instead), so this stays narrowly scoped
to the one new function rather than retrofitting the whole file."""
from market_agent.research.report_format import _render_news_state_lines


def _state(**overrides):
    base = {
        "dimensions": {"growth": None, "demand": None, "risk": None}, "dispersion": {},
        "confidence": 0.359, "news_volume": 40, "independent_event_count": 3,
        "contradiction_axes": [], "half_life_days": 7.0, "state_change": None,
    }
    base.update(overrides)
    return base


def test_none_news_state_reports_source_unavailable():
    assert _render_news_state_lines(None) == ["  SOURCE_UNAVAILABLE - no news retrieved this pass."]


def test_populated_axes_render_signed_and_sorted():
    ns = _state(dimensions={"growth": 0.5, "demand": -0.18, "risk": None})
    lines = _render_news_state_lines(ns)
    assert "    demand: -0.18" in lines
    assert "    growth: +0.50" in lines
    assert not any("risk" in l for l in lines)  # None axis never rendered as a fake 0
    assert lines.index("    demand: -0.18") < lines.index("    growth: +0.50")  # alphabetical


def test_no_signal_is_disclosed_not_silently_empty():
    ns = _state(dimensions={"growth": None, "demand": None, "risk": None})
    lines = _render_news_state_lines(ns)
    assert any("No axis carried a signal" in l for l in lines)


def test_contradiction_axis_is_flagged():
    ns = _state(dimensions={"demand": 0.0}, contradiction_axes=["demand"])
    lines = _render_news_state_lines(ns)
    assert any(l.startswith("  Contradiction: demand") for l in lines)


def test_no_contradiction_axes_produces_no_contradiction_line():
    ns = _state(dimensions={"growth": 0.5}, contradiction_axes=[])
    lines = _render_news_state_lines(ns)
    assert not any(l.startswith("  Contradiction:") for l in lines)


def test_confidence_volume_and_half_life_always_shown():
    ns = _state(confidence=0.5, news_volume=12, half_life_days=7.0)
    lines = _render_news_state_lines(ns)
    assert any("Confidence: 50%" in l and "News volume: 12 document(s)" in l and "Half-life: 7d" in l
               for l in lines)


def test_state_change_rendered_when_present():
    ns = _state(dimensions={"growth": 0.8}, state_change={"growth": 0.6})
    lines = _render_news_state_lines(ns)
    assert any("Change vs. prior pass: growth +0.60" in l for l in lines)


def test_state_change_absent_when_no_prior_report():
    ns = _state(dimensions={"growth": 0.8}, state_change=None)
    lines = _render_news_state_lines(ns)
    assert not any(l.startswith("  Change vs. prior pass:") for l in lines)


def test_excluded_by_role_is_surfaced_when_present():
    ns = _state(dimensions={"growth": 1.0}, excluded_by_role={"COMPETITOR": 2, "INDUSTRY": 1})
    lines = _render_news_state_lines(ns)
    assert any(l.startswith("  Excluded (not attributed to this company") and "competitor: 2" in l
               and "industry: 1" in l for l in lines)


def test_no_excluded_by_role_line_when_nothing_was_excluded():
    ns = _state(dimensions={"growth": 1.0}, excluded_by_role={})
    lines = _render_news_state_lines(ns)
    assert not any(l.startswith("  Excluded (not attributed") for l in lines)
