"""Items 4/9: real, disclosed fundamental data - pulled directly from
SEC's XBRL "company facts" API (providers.py's SECProvider), never
fabricated or estimated. A metric this system cannot find a matching XBRL
tag for stays SOURCE_UNAVAILABLE (schema.py's FundamentalFact with
value=None), not a guessed number.

TAG CANDIDATES, NOT ONE FIXED TAG PER METRIC: different filers use
different us-gaap tags for economically equivalent concepts (e.g. some
report "Revenues", newer filers report
"RevenueFromContractWithCustomerExcludingAssessedTax" since ASC 606
adoption) - each metric below tries a short, disclosed, fixed list of
candidate tags in order and uses the first one with real data. This is
NOT free-form tag guessing; the candidate lists are fixed before any
company is looked at.

EXPLAIN, DON'T JUST LIST (item 9's explicit requirement): explain_fundamentals
produces a handful of rule-based, DATA-DRIVEN sentences ("free-cash-flow
growth has slowed because capital expenditure increased") only when the
underlying numbers actually support that specific relationship - never a
templated sentence with no real number behind it.
"""
from __future__ import annotations

from datetime import datetime, timezone

from market_agent.research.schema import FundamentalFact, SOURCE_UNAVAILABLE

# Fixed, disclosed candidate tag lists - tried in order, first with real data wins.
TAG_CANDIDATES: dict[str, tuple[str, ...]] = {
    "Revenues": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                 "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"),
    "GrossProfit": ("GrossProfit",),
    "OperatingIncomeLoss": ("OperatingIncomeLoss",),
    "NetIncomeLoss": ("NetIncomeLoss", "ProfitLoss"),
    "EarningsPerShareDiluted": ("EarningsPerShareDiluted",),
    "CashAndCashEquivalents": ("CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
    "LongTermDebt": ("LongTermDebtNoncurrent", "LongTermDebt"),
    "OperatingCashFlow": ("NetCashProvidedByUsedInOperatingActivities",),
    "CapitalExpenditure": ("PaymentsToAcquireProductiveAssets", "PaymentsToAcquirePropertyPlantAndEquipment",
                           "PaymentsForCapitalImprovements"),
    "DividendsPaid": ("PaymentsOfDividendsCommonStock", "PaymentsOfDividends"),
    "StockRepurchases": ("PaymentsForRepurchaseOfCommonStock",),
    "TotalAssets": ("Assets",),
    "TotalLiabilities": ("Liabilities",),
}

LABELS: dict[str, str] = {
    "Revenues": "Revenue", "GrossProfit": "Gross profit", "OperatingIncomeLoss": "Operating income",
    "NetIncomeLoss": "Net income", "EarningsPerShareDiluted": "Diluted EPS",
    "CashAndCashEquivalents": "Cash and equivalents", "LongTermDebt": "Long-term debt",
    "OperatingCashFlow": "Operating cash flow", "CapitalExpenditure": "Capital expenditure",
    "DividendsPaid": "Dividends paid", "StockRepurchases": "Stock repurchases",
    "TotalAssets": "Total assets", "TotalLiabilities": "Total liabilities",
    "FreeCashFlow": "Free cash flow (derived: operating cash flow - capex)",
}

INSTANT_CONCEPTS = {"CashAndCashEquivalents", "LongTermDebt", "TotalAssets", "TotalLiabilities"}
PREFERRED_FORMS = ("10-K", "10-Q")


def _plausible_duration(r: dict) -> bool:
    """Keeps only QUARTERLY (~80-100 days) or ANNUAL (~350-380 days)
    duration entries for a concept that has a start/end period (revenue,
    cash flow, ...) - XBRL duration facts also legitimately include
    6-month/9-month CUMULATIVE entries (e.g. a Q2 10-Q reporting H1
    figures alongside the Q2-only figures under the SAME tag), and mixing
    those into a "latest"/YoY comparison would silently compare a
    half-year figure against a quarterly one. Instant concepts (no
    'start' at all - cash, debt, assets) always pass, since this
    filter doesn't apply to them."""
    if "start" not in r or not r["start"]:
        return True
    days = (datetime.strptime(r["end"], "%Y-%m-%d") - datetime.strptime(r["start"], "%Y-%m-%d")).days
    return 80 <= days <= 100 or 350 <= days <= 380


def _extract_series(facts: dict, concept: str, unit: str = "USD") -> list[dict]:
    """Tries every candidate tag for `concept` and returns the series from
    whichever candidate has the MOST RECENT coverage - a company can (and
    does) switch which specific us-gaap tag it reports a concept under
    over time; picking "the first candidate with ANY data" would silently
    freeze onto a tag the company stopped using years ago (found live: for
    a real company, PaymentsToAcquirePropertyPlantAndEquipment's most
    recent entry was from 2020, years stale, while a plausible newer tag
    had current data - always compare candidates by recency, never take
    the first match blindly)."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    best_series: list[dict] = []
    for tag in TAG_CANDIDATES.get(concept, (concept,)):
        entry = us_gaap.get(tag)
        if not entry:
            continue
        raw = entry.get("units", {}).get(unit) or entry.get("units", {}).get("USD/shares")
        if not raw:
            continue
        filtered = [r for r in raw if r.get("form") in PREFERRED_FORMS and _plausible_duration(r)]
        if not filtered:
            continue
        # dedupe by (start, end) - keep the MOST RECENTLY FILED value for that period (a later
        # filing restating an earlier period's figure supersedes the original)
        by_period: dict[tuple, dict] = {}
        for r in filtered:
            key = (r.get("start"), r["end"])
            existing = by_period.get(key)
            if existing is None or r["filed"] > existing["filed"]:
                by_period[key] = r
        series = sorted(by_period.values(), key=lambda r: r["end"])
        if series and (not best_series or series[-1]["end"] > best_series[-1]["end"]):
            best_series = series
    return best_series


def _latest(series: list[dict]) -> dict | None:
    return series[-1] if series else None


def _same_period_prior_year(series: list[dict], latest: dict) -> dict | None:
    """The entry covering the SAME fiscal period one year earlier - a
    real year-over-year comparison, not just 'whatever is second to
    last' (which could be a different-length period). Requires BOTH the
    end date to fall ~350-380 days before `latest`'s end date AND the
    candidate's OWN duration to be similar to `latest`'s duration (both
    ~quarterly or both ~annual) - comparing a quarter's end date to an
    annual period that happens to end on a similar calendar date would
    silently produce a nonsensical "YoY growth" figure otherwise."""
    latest_end = datetime.strptime(latest["end"], "%Y-%m-%d")
    latest_start = datetime.strptime(latest["start"], "%Y-%m-%d") if latest.get("start") else None
    latest_duration = (latest_end - latest_start).days if latest_start else None
    for r in reversed(series[:-1]):
        r_end = datetime.strptime(r["end"], "%Y-%m-%d")
        if not (350 <= (latest_end - r_end).days <= 380):
            continue
        if latest_duration is None:
            return r  # instant concept (no start date) - end-date proximity alone is enough
        r_start = datetime.strptime(r["start"], "%Y-%m-%d") if r.get("start") else None
        if r_start is None:
            continue
        r_duration = (r_end - r_start).days
        if abs(r_duration - latest_duration) <= 15:  # both quarterly (~90d) or both annual (~365d)
            return r
    return None


def build_fundamental_facts(facts_json: dict | None, source_id: str | None) -> list[FundamentalFact]:
    """Item 4/9: one FundamentalFact per concept with real XBRL data
    found, plus a derived FreeCashFlow fact when both its inputs are
    available. `facts_json=None` (SOURCE_UNAVAILABLE from the provider)
    returns an empty list - never fabricated zeros."""
    if facts_json is None:
        return []
    results: list[FundamentalFact] = []
    series_by_concept: dict[str, list[dict]] = {}
    for concept in TAG_CANDIDATES:
        series = _extract_series(facts_json, concept)
        series_by_concept[concept] = series
        latest = _latest(series)
        results.append(FundamentalFact(
            tag=concept, label=LABELS[concept], period_end=latest["end"] if latest else None,
            value=latest["val"] if latest else None, unit="USD",
            fiscal_period_type="Q" if latest and latest.get("fp") != "FY" else "FY",
            source_ids=[source_id] if latest and source_id else [],
        ))

    ocf, capex = _latest(series_by_concept.get("OperatingCashFlow", [])), _latest(series_by_concept.get("CapitalExpenditure", []))
    if ocf and capex and ocf["end"] == capex["end"]:
        fcf = ocf["val"] - capex["val"]
        results.append(FundamentalFact(tag="FreeCashFlow", label=LABELS["FreeCashFlow"], period_end=ocf["end"],
                                        value=fcf, unit="USD", fiscal_period_type="Q" if ocf.get("fp") != "FY" else "FY",
                                        source_ids=[source_id] if source_id else []))
    return results


def _yoy_growth(series: list[dict]) -> tuple[float | None, dict | None, dict | None]:
    latest = _latest(series)
    if latest is None:
        return None, None, None
    prior = _same_period_prior_year(series, latest)
    if prior is None or prior["val"] == 0:
        return None, latest, prior
    return (latest["val"] - prior["val"]) / abs(prior["val"]), latest, prior


def explain_fundamentals(facts_json: dict | None) -> list[str]:
    """Item 9: a handful of rule-based, DATA-DRIVEN explanatory
    sentences - each one only fires when the underlying numbers actually
    support it (real growth-rate comparisons, not a templated sentence
    with nothing behind it)."""
    if facts_json is None:
        return [f"Fundamental data: {SOURCE_UNAVAILABLE} - could not retrieve SEC XBRL company facts."]

    revenue_series = _extract_series(facts_json, "Revenues")
    ocf_series = _extract_series(facts_json, "OperatingCashFlow")
    capex_series = _extract_series(facts_json, "CapitalExpenditure")
    ni_series = _extract_series(facts_json, "NetIncomeLoss")

    rev_growth, rev_latest, _ = _yoy_growth(revenue_series)
    ocf_growth, ocf_latest, _ = _yoy_growth(ocf_series)
    capex_growth, capex_latest, _ = _yoy_growth(capex_series)
    ni_growth, _, _ = _yoy_growth(ni_series)

    lines: list[str] = []
    if rev_growth is not None:
        direction = "grew" if rev_growth > 0 else "declined"
        lines.append(f"Revenue {direction} {rev_growth:+.1%} year-over-year as of {rev_latest['end']} "
                     f"(${rev_latest['val']:,.0f}).")
    else:
        lines.append("Revenue year-over-year comparison unavailable - insufficient matching prior-year period "
                      "in disclosed XBRL data.")

    if ocf_growth is not None and capex_growth is not None:
        if ocf_growth < (rev_growth or 0) and capex_growth > (rev_growth or 0):
            lines.append(f"Operating cash flow growth ({ocf_growth:+.1%} YoY) lagged revenue growth while capital "
                         f"expenditure grew {capex_growth:+.1%} YoY - free-cash-flow growth has been diluted by "
                         "rising capex, not by weaker operating performance.")
        elif ocf_growth is not None:
            lines.append(f"Operating cash flow grew {ocf_growth:+.1%} year-over-year.")
    if ni_growth is not None and rev_growth is not None and abs(ni_growth - rev_growth) > 0.05:
        wider = "faster" if ni_growth > rev_growth else "slower"
        lines.append(f"Net income grew {ni_growth:+.1%} YoY, {wider} than revenue growth ({rev_growth:+.1%}) - "
                     "consistent with a change in margins rather than in top-line demand alone.")

    if not lines:
        lines.append("Insufficient disclosed data to generate a year-over-year fundamentals narrative.")
    return lines
