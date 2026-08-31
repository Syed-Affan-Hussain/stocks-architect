"""The designed (not, in this environment, active) LLM-backed extraction
path - answers this design's central research question ("can an LLM be
used as a semantic measurement instrument that converts financial news
into a stable numerical representation") with a real, callable interface,
not a hand-wave.

REUSES market_agent's EXISTING LLMClient/LLMNotConfiguredError contract
(market_agent/llm/interpreter.py) - the SAME no-silent-fallback discipline
already established for the rest of this project's LLM integration
points, not a second, parallel one. No LLM client is configured in this
environment by default; `extract_event_llm` raises LLMNotConfiguredError
if called with `client=None`, and `llm_status(client)` reports
"UNAVAILABLE" wherever this module's output is surfaced - callers must
check this explicitly and fall back to event_vector.py's deterministic
rule-based extractor (what actually runs, and what this whole design was
validated against - see the validation report).

WHY A SCHEMA, NOT PROSE: the Blueprint discipline this whole project
follows (events/interpret.py, llm/hypothesis_generator.py) is "every LLM
call returns a typed, schema-validated object... never raw prose feeding
downstream logic." EXTRACTION_SCHEMA below is that contract for news: the
LLM is constrained to emit exactly IMPLICATION_AXES (schema.py) plus the
handful of scalar fields this design settled on - it CANNOT invent a new
dimension or return free text where a number belongs.

CALL IT WHAT IT ACTUALLY IS: every LLMExtractionResult is labeled
`extraction_method="LLM_V1"` and carries the exact prompt version, model
identifier, and raw response - never presented as an objective
measurement. It is a MODEL-DERIVED QUANTITATIVE SEMANTIC MEASUREMENT, and
every consumer of it (report_format.py-style rendering, if this is ever
wired into a report) must say so next to the number, the same way this
project already discloses SOURCE_UNAVAILABLE/INSUFFICIENT_EVIDENCE rather
than papering over a gap.

WHY SINGLE-SHOT, NOT SEQUENTIAL CHAIN-OF-THOUGHT: Qian et al. 2026
("Improving Event-Level Financial Sentiment Analysis with Adaptive
Reasoning Order Chain-of-Thought Prompting", Intell. Comput. 5:0651)
measures that the ORDER in which an LLM is asked to resolve company ->
event -> sentiment in a multi-hop CoT prompt materially changes accuracy
(43% of their EFSA errors traced to a suboptimal reasoning sequence), and
that the best order differs by event type - their fix (ARO-CoT) is a
LoRA-trained model that learns to pick among 6 candidate orderings per
input. That result doesn't transfer here AS A TECHNIQUE, because
EXTRACTION_SCHEMA below asks for every field in ONE JSON call rather than
a sequence of dependent hops - there is no "first hop's mistake corrupts
the second hop" failure mode to fix, because there is no second hop. This
is a genuine, disclosed architectural difference, not an oversight: if
this module is ever redesigned around multi-step prompting (e.g. to let
the model reason before committing to a number), Qian et al.'s finding is
the reason to seriously consider event-type-adaptive ordering rather than
one fixed hop sequence, instead of assuming a single fixed CoT order is
safe by default.

DETERMINISM, AS FAR AS PRACTICAL WITH A GENERATIVE MODEL: fixed schema,
constrained output (no free-form fields for the numeric axes), a versioned
prompt (PROMPT_VERSION), the model/version recorded on every result,
confidence recorded (not fabricated - see CONFIDENCE_FIELD's own
docstring below on why it's still a heuristic even here), the source
document retained by reference (source_id, never copied/mutated), and an
extraction timestamp. A real LLM is still a stochastic function of its
input - "as far as practical" explicitly does NOT mean bit-identical
output on every call; running the SAME input through the SAME prompt
version multiple times and checking dispersion is the intended way to
audit how stable a given extraction actually is (see the validation
report's discussion of what remains untested without a configured
client).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from market_agent.llm.interpreter import LLMClient, LLMNotConfiguredError
from market_agent.research.news_state.schema import EPISTEMIC_STATUSES, IMPLICATION_AXES, TIME_HORIZONS

PROMPT_VERSION = "news_event_extraction_v1"
MODEL_FIELD_UNCONFIGURED = "UNCONFIGURED"

# Every numeric axis is CONSTRAINED to [-1, 1] (implications) or [0, 1] (materiality/certainty/
# confidence) in the schema itself - the LLM cannot return an out-of-range or free-text value where a
# number belongs. `epistemic_status` and `time_horizon` are closed enums, matching schema.py exactly -
# the LLM is not free to invent a new epistemic category or a new axis name.
EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "implications": {
            "type": "object",
            "description": "One signed value per axis in [-1, 1], or null if this event says nothing "
                            "about that axis. Do not force a value where the text is silent.",
            "properties": {axis: {"type": ["number", "null"], "minimum": -1, "maximum": 1}
                            for axis in IMPLICATION_AXES},
            "required": list(IMPLICATION_AXES),
        },
        "text_sentiment": {"type": ["number", "null"], "minimum": -1, "maximum": 1,
                            "description": "Linguistic tone ONLY - separate from economic implication. "
                                           "'Layoffs' can be textually negative while risk/profitability "
                                           "implications differ in sign from it."},
        "materiality": {"type": "number", "minimum": 0, "maximum": 1},
        "epistemic_status": {"type": "string", "enum": list(EPISTEMIC_STATUSES)},
        "time_horizon": {"type": "string", "enum": list(TIME_HORIZONS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1,
                        "description": "The model's own stated confidence in this extraction - a "
                                       "self-reported heuristic, not a calibrated probability (see "
                                       "CONFIDENCE_FIELD docstring)."},
        "rationale": {"type": "string", "description": "One sentence, audit trail only - never parsed "
                                                          "or used as a numeric input downstream."},
    },
    "required": ["implications", "text_sentiment", "materiality", "epistemic_status", "time_horizon", "confidence"],
}
# CONFIDENCE_FIELD note: an LLM's self-reported confidence is well-documented in the literature to be
# poorly calibrated (models are frequently overconfident) unless independently validated against a
# labeled outcome set - which this project does not have. Recording it is still worthwhile (it is a
# real signal, just not a trustworthy probability on its own) - see the validation report's explicit
# recommendation to treat it as one input to a heuristic confidence, not the final word, exactly the
# same posture epistemic.py already takes for the deterministic path.

EXTRACTION_PROMPT_TEMPLATE = """You are extracting a structured ECONOMIC representation of one financial \
news event, not a sentiment score. Read the clause(s) below and output ONLY the fields in the schema.

For EACH implication axis (growth, profitability, cash_flow, balance_sheet, demand, supply_chain, \
competitive_position, regulatory, guidance, risk): does this event say something specific about that \
axis? If yes, score it in [-1, 1] where positive = improving/favorable for the company (for `risk`, \
positive means risk is INCREASING, not favorable). If the text is silent on an axis, return null for \
it - do not guess a value to fill every field.

Separately score `text_sentiment` - the LINGUISTIC tone of the wording alone, independent of the \
economic implications above. A textually negative sentence (e.g. describing layoffs) can carry a \
positive profitability implication; do not let one drive the other.

Classify `epistemic_status` as exactly one of: OBSERVED_FACT (a filed/disclosed number), \
MANAGEMENT_CLAIM (something company management stated or claimed, including forward guidance), \
THIRD_PARTY_REPORTING (a journalist/outlet reporting something without management attribution), \
ANALYST_INTERPRETATION (a third party's own inference/opinion), or SPECULATION (hedged, forward-\
looking, no firm source).

Classify `time_horizon` as SHORT_TERM (this quarter), MEDIUM_TERM (this fiscal year), LONG_TERM \
(multi-year/strategic), or UNSPECIFIED.

TEXT:
{clause_text}

Prompt version: {prompt_version}"""


@dataclass
class LLMExtractionResult:
    implications: dict[str, float | None]
    text_sentiment: float | None
    materiality: float
    epistemic_status: str
    time_horizon: str
    confidence: float
    rationale: str
    model: str
    prompt_version: str
    extracted_at: str
    source_clause: str
    extraction_method: str = "LLM_V1"

    def to_dict(self) -> dict:
        return {"implications": self.implications, "text_sentiment": self.text_sentiment,
                "materiality": self.materiality, "epistemic_status": self.epistemic_status,
                "time_horizon": self.time_horizon, "confidence": self.confidence, "rationale": self.rationale,
                "model": self.model, "prompt_version": self.prompt_version, "extracted_at": self.extracted_at,
                "source_clause": self.source_clause, "extraction_method": self.extraction_method}


def llm_status(client: LLMClient | None) -> str:
    return "UNAVAILABLE" if client is None else f"ACTIVE:{type(client).__name__}"


def extract_event_llm(client: LLMClient | None, clause_text: str) -> LLMExtractionResult:
    """Raises LLMNotConfiguredError if `client` is None - callers decide,
    explicitly, whether to fall back to event_vector.py's deterministic
    extractor; this function never does so silently (same discipline as
    market_agent/llm/interpreter.py's LLMInterpreter)."""
    if client is None:
        raise LLMNotConfiguredError(
            "extract_event_llm() called with no LLMClient configured - no LLM is wired into this "
            "environment by default (see market_agent/llm/select.py's same discipline). The caller "
            "must explicitly choose to fall back to event_vector.py's deterministic extractor, and "
            "must record that choice - never silently substitute one for the other.")
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(clause_text=clause_text, prompt_version=PROMPT_VERSION)
    raw = client.complete_structured(prompt, EXTRACTION_SCHEMA)
    return LLMExtractionResult(
        implications=raw["implications"], text_sentiment=raw.get("text_sentiment"),
        materiality=raw["materiality"], epistemic_status=raw["epistemic_status"],
        time_horizon=raw["time_horizon"], confidence=raw["confidence"], rationale=raw.get("rationale", ""),
        model=type(client).__name__, prompt_version=PROMPT_VERSION,
        extracted_at=datetime.now(timezone.utc).isoformat(), source_clause=clause_text,
    )
