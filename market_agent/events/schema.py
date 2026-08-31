"""Event and prediction data structures - the in-memory shape of a
category-1 row before it's written to the store, and the typed input the
Interpreter/agents pass between each other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RawItem:
    """A single unprocessed piece of information, before interpretation -
    the input to events.interpret.Interpreter. Deliberately minimal:
    this is what any information source (news, filing, analyst note,
    social post) reduces to before the interpreter runs."""
    text: str
    source: str
    entity: str
    published_at: datetime


@dataclass
class ContextSnapshot:
    """Category-6 (temporary contextual information) captured as an
    IMMUTABLE snapshot at prediction time - see store/schema.py's module
    docstring for why this never gets its own mutable table. Context is
    stored as JSON specifically so it can grow without a schema migration
    - this dataclass grew in stage 4 to cover more of the blueprint's
    candidate explanatory variables (regime alone was stage 1-3's set).

    NOT ALL BLUEPRINT-REQUESTED VARIABLES ARE HERE, DISCLOSED RATHER THAN
    FAKED: sector_return/momentum has no real data source wired in (no
    sector classification or sector-benchmark provider exists yet - stays
    "UNKNOWN", never a guessed value). event_surprise (deviation from a
    consensus estimate) and earnings_proximity (days to/from the nearest
    earnings date) have no consensus-estimate or earnings-calendar source
    wired in either - both stay None. Source identity/reliability are
    NOT duplicated here since EventRecord already carries `source` and
    `source_reliability_snapshot` directly - this dataclass only holds
    what's genuinely "surrounding context", not fields that already exist
    elsewhere on the record.

    THESE ARE CANDIDATE EXPLANATORY VARIABLES, NOT ASSERTED SIGNAL - per
    this stage's explicit instruction. Computing and storing a field here
    does not mean any hypothesis generator is required to condition on
    it; learn/hypothesis.py's rule-based generator currently only
    conditions on `regime` and `prior_return_bucket` (retrieval/
    similarity.py's existing bucketing, reused rather than reinvented -
    see that module for why). Everything else is available for a future,
    more capable (e.g. LLM-backed) generator to evaluate, and for direct
    inspection via the knowledge-state report."""
    # Original three (stage 1-3) fields, kept in their original order/position so existing
    # positional constructor calls (ContextSnapshot("NORMAL", -0.05, "NEGATIVE")) keep working
    # unchanged. Every stage-4 addition below is appended AFTER these, all defaulted to None, for
    # the same reason - inserting a field in the middle would silently reinterpret old positional
    # arguments as the wrong field.
    regime: str                       # "RISK_ON" | "RISK_OFF" | "NORMAL" | "UNKNOWN" - retrieval/regime.py
    prior_5d_return: float | None     # entity's own trailing return, for "already priced in" style conditioning
    sector_momentum: str | None       # "UNKNOWN" - no sector data source wired in, see class docstring

    # --- stage 4 additions, all optional/defaulted ---
    prior_1d_return: float | None = None
    prior_20d_return: float | None = None
    prior_60d_return: float | None = None
    realized_vol_20d: float | None = None    # stdev of the entity's own trailing 20 daily returns
    market_return_20d: float | None = None   # benchmark's trailing 20d return (distinct from regime's 60d lookback)
    published_weekday: int | None = None     # 0=Monday .. 6=Sunday
    published_hour_utc: int | None = None
    days_since_last_same_entity_event: float | None = None  # from episodic_events - a real "recent related events" signal
    competing_events_same_day: int | None = None             # other entities' events on the same UTC date - a light
    #                                                           attribution/isolation-quality proxy (see error_taxonomy.py's
    #                                                           disclosed CONFOUNDING_EVENT gap - this does not close that
    #                                                           gap, it's a cheap partial signal toward it)
    extra: dict = field(default_factory=dict)  # room to grow further without another migration

    def to_dict(self) -> dict:
        return {"regime": self.regime, "prior_1d_return": self.prior_1d_return,
                "prior_5d_return": self.prior_5d_return, "prior_20d_return": self.prior_20d_return,
                "prior_60d_return": self.prior_60d_return, "realized_vol_20d": self.realized_vol_20d,
                "market_return_20d": self.market_return_20d, "sector_momentum": self.sector_momentum,
                "published_weekday": self.published_weekday, "published_hour_utc": self.published_hour_utc,
                "days_since_last_same_entity_event": self.days_since_last_same_entity_event,
                "competing_events_same_day": self.competing_events_same_day, **self.extra}


@dataclass
class EventRecord:
    """The interpreted, structured form of a RawItem - what actually gets
    logged to episodic_events. `direction` and `event_type` come from the
    Interpreter (rule-based in Stage 1, LLM-backed later - see
    events/interpret.py); nothing here is asserted by this dataclass
    itself, it's just the typed container."""
    entity: str
    event_type: str
    direction: str            # "positive" | "negative" | "unclear"
    source: str
    source_reliability_snapshot: float | None
    raw_text: str
    published_at: datetime
    ingested_at: datetime
    context: dict             # ContextSnapshot.to_dict() - stored as JSON


@dataclass
class PredictionRecord:
    """What an agent (static or adaptive) produces for one event at one
    horizon - logged alongside the EventRecord in the same episodic_events
    row via store.db.log_prediction(). Fields below `predicted_at` were
    added in stage 4 (the prediction-ledger expansion) and are all
    optional so existing 6-field construction (used throughout stage 1-3
    code/tests) keeps working unchanged."""
    horizon_days: int
    predicted_impact: float | None       # None means "no prediction possible" (e.g. INSUFFICIENT_PRECEDENT)
    predicted_confidence: str            # "HIGH" | "MEDIUM" | "LOW" | "INSUFFICIENT_PRECEDENT"
    basis: dict                          # {"relationship_id": ...} or {"basis": "unconditional_baseline"}
    model_version: str
    predicted_at: datetime

    # --- stage 4 ledger additions, all optional/defaulted ---
    predicted_direction: str | None = None   # "positive" | "negative" | "unclear" - see store/schema.py
    uncertainty: float | None = None         # half-width of the relationship's CI, if used
    retrieved_cases: list[str] = field(default_factory=list)  # event_ids of similar cases shown/used
    knowledge_version: int | None = None     # governance-change counter as of prediction time
    novelty_score: float | None = None       # 0-1, from retrieval coverage
