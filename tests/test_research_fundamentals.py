from market_agent.research.fundamentals import build_fundamental_facts, explain_fundamentals


def _xbrl(tag, entries):
    return {"facts": {"us-gaap": {tag: {"units": {"USD": entries}}}}}


def _entry(start, end, val, form="10-Q", fp="Q1", fy=2024, filed=None):
    return {"start": start, "end": end, "val": val, "form": form, "fp": fp, "fy": fy, "filed": filed or end}


def test_none_facts_json_returns_empty_facts_and_disclosed_evidence():
    assert build_fundamental_facts(None, "src") == []
    lines = explain_fundamentals(None)
    assert any("SOURCE_UNAVAILABLE" in line for line in lines)


def test_picks_most_recent_candidate_tag_not_first_with_any_data():
    facts = {"facts": {"us-gaap": {
        "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [
            _entry("2019-01-01", "2019-04-01", 100, filed="2019-05-01"),
        ]}},
        "PaymentsToAcquireProductiveAssets": {"units": {"USD": [
            _entry("2024-01-01", "2024-04-01", 500, filed="2024-05-01"),
        ]}},
    }}}
    ff = build_fundamental_facts(facts, "src")
    capex = next(f for f in ff if f.tag == "CapitalExpenditure")
    assert capex.value == 500
    assert capex.period_end == "2024-04-01"


def test_six_month_cumulative_entries_are_excluded_from_latest():
    entries = [
        _entry("2024-01-01", "2024-04-01", 1000),      # ~90 days - quarterly, kept
        _entry("2024-01-01", "2024-07-01", 2100),       # ~180 days - cumulative, excluded
        _entry("2024-04-02", "2024-07-01", 1100),         # ~90 days - quarterly, kept, latest
    ]
    facts = _xbrl("Revenues", entries)
    ff = build_fundamental_facts(facts, "src")
    revenue = next(f for f in ff if f.tag == "Revenues")
    assert revenue.value == 1100
    assert revenue.period_end == "2024-07-01"


def test_yoy_growth_compares_same_length_period_one_year_prior():
    entries = [
        _entry("2023-04-02", "2023-07-01", 1000, filed="2023-08-01"),  # prior-year same quarter
        _entry("2024-01-01", "2024-04-01", 1200, filed="2024-05-01"),  # a DIFFERENT quarter, not prior-year
        _entry("2024-04-02", "2024-07-01", 1500, filed="2024-08-01"),  # latest
    ]
    facts = _xbrl("Revenues", entries)
    lines = explain_fundamentals(facts)
    assert any("+50.0%" in line for line in lines)  # (1500-1000)/1000


def test_free_cash_flow_is_derived_only_when_both_inputs_share_the_same_period():
    facts = {"facts": {"us-gaap": {
        "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
            _entry("2024-01-01", "2024-04-01", 1000, filed="2024-05-01")]}},
        "PaymentsToAcquireProductiveAssets": {"units": {"USD": [
            _entry("2024-01-01", "2024-04-01", 300, filed="2024-05-01")]}},
    }}}
    ff = build_fundamental_facts(facts, "src")
    fcf = next(f for f in ff if f.tag == "FreeCashFlow")
    assert fcf.value == 700


def test_missing_concept_produces_a_fact_with_none_value_not_a_fabricated_number():
    facts = _xbrl("Revenues", [_entry("2024-01-01", "2024-04-01", 1000)])
    ff = build_fundamental_facts(facts, "src")
    ni = next(f for f in ff if f.tag == "NetIncomeLoss")
    assert ni.value is None
    assert ni.source_ids == []
