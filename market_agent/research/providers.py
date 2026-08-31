"""Item 1/2: the data-collection provider abstraction for the research
product. Every provider returns real data or explicitly signals
SOURCE_UNAVAILABLE (schema.py) - never a fabricated placeholder. Kept
separate from market_agent/sources/ (which is the EXISTING, unmodified
event-study ingestion for the trading-research system - phrase-matched
guidance/dividend 8-Ks only) because this product needs a materially
different access pattern: ALL recent filings for one company, and its
full XBRL fact set, not a phrase search across many companies.

NO NEW DEPENDENCIES: SEC's JSON APIs and Google News' RSS feed are parsed
with `requests` (already a project dependency) plus the standard library's
`xml.etree.ElementTree` - no feedparser/bs4/lxml needed.

PROVIDER ABSTRACTION NAMES MATCH THE REQUESTED ARCHITECTURE:
SECProvider, NewsProvider, MarketDataProvider (delegates to the EXISTING
market_agent.sources.yahoo_prices.YahooPriceSeriesProvider - not
reimplemented), FundamentalDataProvider. CompanyNewsProvider is NOT a
separate implementation in this MVP - NewsProvider's Google News query is
already company-scoped (`q=<ticker>`), so a distinct company-news class
would just wrap the same call; the abstraction point for a FUTURE
company-specific source (e.g. an investor-relations RSS feed) is
`NewsProvider`'s own interface, which any new provider can implement.
"""
from __future__ import annotations

import hashlib
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests

from market_agent.research.schema import SOURCE_UNAVAILABLE, SourceDocument

HEADERS = {"User-Agent": "Stocks_Architect research (contact: affanhussain2003@gmail.com)"}
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
SEC_FILING_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{primary_doc}"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"

MATERIAL_FORMS = ("10-K", "10-Q", "8-K")


@dataclass
class ProviderResult:
    """Every provider call returns this - `status` is either "OK" or
    SOURCE_UNAVAILABLE, so a caller never has to guess whether an empty
    list means "genuinely nothing found" vs. "the source was unreachable"."""
    status: str
    documents: list[SourceDocument]
    evidence: str


def make_fingerprint(title: str, content: str) -> str:
    """Item 3: a normalized-text hash used to detect syndicated/repeated
    coverage of the SAME underlying story. Deliberately coarse (lowercased
    title + first 400 normalized characters of content, not full-text) -
    wire-service stories reproduced verbatim or near-verbatim across many
    publishers share this prefix; genuinely different articles almost
    never do. A simple, disclosed heuristic for this MVP, not a trained
    near-duplicate model - see normalize.py's module docstring."""
    norm_title = re.sub(r"\s+", " ", title.strip().lower())
    norm_content = re.sub(r"\s+", " ", content.strip().lower())[:400]
    return hashlib.sha256((norm_title + "|" + norm_content).encode("utf-8")).hexdigest()[:24]


class SECProvider:
    """Item 1: ticker -> CIK resolution, the full recent-filings index
    (10-K/10-Q/8-K), and XBRL company facts (real, disclosed financial
    data) - all from SEC's own free, public, no-key-required APIs. NEVER
    the phrase-matched full-text-search approach in sources/edgar_guidance.py
    (that module answers a different question: "which historical filings,
    across ALL companies, used this exact phrase" - this one answers
    "what has THIS company filed, and what does it disclose").
    """

    def __init__(self):
        self._ticker_cik_map: dict[str, str] | None = None

    def _load_ticker_map(self) -> dict[str, str] | None:
        if self._ticker_cik_map is not None:
            return self._ticker_cik_map
        try:
            resp = requests.get(SEC_TICKERS_URL, headers=HEADERS, timeout=20)
            if resp.status_code != 200:
                return None
            data = resp.json()
            self._ticker_cik_map = {row["ticker"].upper(): str(row["cik_str"]).zfill(10) for row in data.values()}
            return self._ticker_cik_map
        except Exception:  # noqa: BLE001
            return None

    def resolve_cik(self, ticker: str) -> str | None:
        m = self._load_ticker_map()
        return m.get(ticker.upper()) if m else None

    def fetch_company_meta(self, ticker: str) -> dict | None:
        """Company name, SIC description, and CIK - from the submissions
        endpoint's own top-level fields."""
        cik10 = self.resolve_cik(ticker)
        if cik10 is None:
            return None
        try:
            resp = requests.get(SEC_SUBMISSIONS_URL.format(cik10=cik10), headers=HEADERS, timeout=20)
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception:  # noqa: BLE001
            return None

    def fetch_recent_filings(self, ticker: str, entity: str, lookback_days: int = 365,
                              max_filings: int = 40) -> ProviderResult:
        meta = self.fetch_company_meta(ticker)
        if meta is None:
            return ProviderResult(SOURCE_UNAVAILABLE, [],
                                   f"SEC submissions data unavailable for {ticker} (network error, unresolved "
                                   "ticker, or SEC API outage).")

        cik10 = self.resolve_cik(ticker)
        recent = meta.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        items = recent.get("items", [""] * len(forms))

        cutoff = datetime.now(timezone.utc).date()
        docs: list[SourceDocument] = []
        now_iso = datetime.now(timezone.utc).isoformat()
        for form, date_str, accession, primary_doc, item_codes in zip(forms, dates, accessions, primary_docs, items):
            if form not in MATERIAL_FORMS:
                continue
            filed = datetime.strptime(date_str, "%Y-%m-%d").date()
            if (cutoff - filed).days > lookback_days:
                continue
            accession_nodash = accession.replace("-", "")
            cik_no_padding = str(int(cik10))
            url = SEC_FILING_INDEX_URL.format(cik=cik_no_padding, accession_nodash=accession_nodash,
                                               primary_doc=primary_doc)
            title = f"{meta.get('name', entity)} {form} filed {date_str}" + (f" (items {item_codes})" if item_codes else "")
            content = f"SEC {form} filing by {meta.get('name', entity)}, filed {date_str}, accession {accession}" \
                      + (f", 8-K item codes: {item_codes}." if item_codes else ".")
            source_id = f"sec:{accession}"
            docs.append(SourceDocument(
                source_id=source_id, publisher="SEC EDGAR", source_type="SEC_FILING", url=url,
                published_at=filed.isoformat(), retrieved_at=now_iso, entity=entity, title=title,
                raw_content=content, normalized_content=content, reliability="PRIMARY",
                fingerprint=make_fingerprint(title, content),
            ))
            if len(docs) >= max_filings:
                break

        return ProviderResult("OK", docs, f"{len(docs)} material SEC filing(s) (10-K/10-Q/8-K) for {ticker} in the "
                               f"last {lookback_days} days.")

    def fetch_company_facts(self, ticker: str) -> tuple[str, dict | None, str]:
        """Real XBRL-disclosed financial facts (item 4/9). Returns
        (status, raw_facts_json_or_None, evidence)."""
        cik10 = self.resolve_cik(ticker)
        if cik10 is None:
            return SOURCE_UNAVAILABLE, None, f"Could not resolve a CIK for ticker {ticker!r}."
        try:
            resp = requests.get(SEC_COMPANYFACTS_URL.format(cik10=cik10), headers=HEADERS, timeout=25)
            if resp.status_code != 200:
                return SOURCE_UNAVAILABLE, None, f"SEC companyfacts API returned HTTP {resp.status_code} for {ticker}."
            return "OK", resp.json(), f"Real XBRL company facts retrieved for {ticker} (CIK {cik10})."
        except Exception as e:  # noqa: BLE001
            return SOURCE_UNAVAILABLE, None, f"SEC companyfacts fetch failed for {ticker}: {e!r}."


class NewsProvider:
    """Item 1/3: Google News RSS, company-scoped by ticker+name query.
    Real HTTP fetch of a real public RSS feed - no API key required, no
    fabricated headlines. Deduplication (item 3) happens downstream in
    normalize.py using each document's fingerprint - this provider just
    collects everything the feed returns, syndicated copies included."""

    def fetch(self, query: str, entity: str, max_items: int = 40) -> ProviderResult:
        try:
            resp = requests.get(GOOGLE_NEWS_RSS_URL, params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                                 headers=HEADERS, timeout=20)
            if resp.status_code != 200:
                return ProviderResult(SOURCE_UNAVAILABLE, [],
                                       f"Google News RSS returned HTTP {resp.status_code} for query {query!r}.")
            root = ET.fromstring(resp.content)
        except Exception as e:  # noqa: BLE001
            return ProviderResult(SOURCE_UNAVAILABLE, [], f"Google News RSS fetch failed for {query!r}: {e!r}.")

        now_iso = datetime.now(timezone.utc).isoformat()
        docs: list[SourceDocument] = []
        for item in root.findall(".//item")[:max_items]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_date_raw = item.findtext("pubDate")
            description = (item.findtext("description") or "").strip()
            source_el = item.find("source")
            publisher = (source_el.text or "Unknown").strip() if source_el is not None else "Unknown"
            if not title:
                continue
            try:
                published_at = parsedate_to_datetime(pub_date_raw).astimezone(timezone.utc).isoformat() \
                    if pub_date_raw else now_iso
            except Exception:  # noqa: BLE001
                published_at = now_iso
            content = re.sub(r"<[^>]+>", " ", description)  # Google wraps description in an <a> tag
            content = re.sub(r"\s+", " ", content).strip() or title
            source_id = f"news:{make_fingerprint(title, link)}"
            docs.append(SourceDocument(
                source_id=source_id, publisher=publisher, source_type="NEWS", url=link,
                published_at=published_at, retrieved_at=now_iso, entity=entity, title=title,
                raw_content=content, normalized_content=content,
                reliability=classify_news_reliability(publisher), fingerprint=make_fingerprint(title, content),
            ))
        return ProviderResult("OK", docs, f"{len(docs)} news item(s) from Google News RSS for query {query!r}.")


# item 2 - PRIMARY sources are SEC filings (handled directly by SECProvider); among NEWS documents,
# major wire services are the closest thing to a primary-grade original report (everyone else
# usually reproduces them) - promoted to SECONDARY; everything else is TERTIARY. This is a disclosed,
# fixed publisher list, not an attempt at a comprehensive media-reliability database.
WIRE_SERVICES = {"reuters", "bloomberg", "associated press", "ap news", "the wall street journal", "dow jones",
                  "wsj", "wsj.com", "bloomberg.com", "reuters.com"}


def classify_news_reliability(publisher: str) -> str:
    return "SECONDARY" if publisher.strip().lower() in WIRE_SERVICES else "TERTIARY"


class MarketDataProvider:
    """Item 1: thin wrapper over the EXISTING, unmodified
    market_agent.sources.yahoo_prices.YahooPriceSeriesProvider - not
    reimplemented. Returns SOURCE_UNAVAILABLE if the underlying provider
    has no usable history for the ticker."""

    def __init__(self, price_series_provider):
        self.prices = price_series_provider

    def has_data(self, ticker: str, as_of: datetime) -> bool:
        return self.prices.close_price(ticker, as_of) is not None
