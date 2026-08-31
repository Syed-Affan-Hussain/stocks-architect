import pytest

from market_agent.llm.interpreter import LLMClient, LLMNotConfiguredError
from market_agent.research.news_state.llm_schema import (
    EXTRACTION_SCHEMA, PROMPT_VERSION, extract_event_llm, llm_status,
)
from market_agent.research.news_state.schema import IMPLICATION_AXES


class FakeLLMClient(LLMClient):
    def __init__(self, response):
        self.response = response
        self.last_prompt = None
        self.last_schema = None

    def complete_structured(self, prompt, schema):
        self.last_prompt = prompt
        self.last_schema = schema
        return self.response


def _valid_response(**overrides):
    resp = {
        "implications": {axis: None for axis in IMPLICATION_AXES},
        "text_sentiment": -0.4, "materiality": 0.8, "epistemic_status": "MANAGEMENT_CLAIM",
        "time_horizon": "MEDIUM_TERM", "confidence": 0.7, "rationale": "test",
    }
    resp["implications"].update(overrides)
    return resp


def test_no_client_raises_not_configured():
    with pytest.raises(LLMNotConfiguredError):
        extract_event_llm(None, "Management raised guidance.")


def test_llm_status_reflects_client_presence():
    assert llm_status(None) == "UNAVAILABLE"
    assert llm_status(FakeLLMClient({})) == "ACTIVE:FakeLLMClient"


def test_extraction_calls_client_with_schema_and_versioned_prompt():
    client = FakeLLMClient(_valid_response(demand=-0.6, risk=0.5))
    result = extract_event_llm(client, "Company announces layoffs due to declining demand.")
    assert client.last_schema is EXTRACTION_SCHEMA
    assert PROMPT_VERSION in client.last_prompt
    assert "layoffs" in client.last_prompt.lower()
    assert result.implications["demand"] == -0.6
    assert result.implications["risk"] == 0.5
    assert result.model == "FakeLLMClient"
    assert result.extraction_method == "LLM_V1"
    assert result.prompt_version == PROMPT_VERSION


def test_extraction_result_carries_provenance_fields():
    client = FakeLLMClient(_valid_response())
    result = extract_event_llm(client, "Some clause.")
    assert result.source_clause == "Some clause."
    assert result.extracted_at  # non-empty ISO timestamp
    assert result.confidence == 0.7


def test_schema_constrains_every_implication_axis_to_minus1_1():
    for axis in IMPLICATION_AXES:
        prop = EXTRACTION_SCHEMA["properties"]["implications"]["properties"][axis]
        assert prop["minimum"] == -1
        assert prop["maximum"] == 1


def test_schema_epistemic_status_is_a_closed_enum():
    from market_agent.research.news_state.schema import EPISTEMIC_STATUSES
    assert set(EXTRACTION_SCHEMA["properties"]["epistemic_status"]["enum"]) == set(EPISTEMIC_STATUSES)
