"""Explicit provider selection for methodology extraction - mirrors
llm/select.py's METHODOLOGY_PROVIDER-equivalent pattern exactly (same
no-silent-fallback contract): `METHODOLOGY_EXTRACTOR_PROVIDER` defaults to
'rule_based'; an explicit 'llm' value with no client configured raises
LLMNotConfiguredError the first time `.extract()` is actually called, not
silently substituted.
"""
from __future__ import annotations

import os

from market_agent.methodology.extractor import (
    LLMMethodologyExtractor, MethodologyExtractor, RuleBasedMethodologyExtractor,
)

VALID_PROVIDERS = ("rule_based", "llm")


def _read_provider(env_var: str) -> str:
    value = os.environ.get(env_var, "rule_based").strip().lower()
    if value not in VALID_PROVIDERS:
        raise ValueError(f"{env_var}={value!r} is not a valid provider - must be one of {VALID_PROVIDERS}.")
    return value


def select_methodology_extractor(use_llm: bool, client=None) -> MethodologyExtractor:
    if use_llm:
        return LLMMethodologyExtractor(client=client)
    return RuleBasedMethodologyExtractor()


def select_methodology_extractor_from_env(client=None) -> MethodologyExtractor:
    return select_methodology_extractor(_read_provider("METHODOLOGY_EXTRACTOR_PROVIDER") == "llm", client=client)


def describe_active_extractor(extractor: MethodologyExtractor) -> str:
    narrow = "  <- narrow, keyword-pattern coverage only, see extractor.py" if extractor.coverage_is_narrow else ""
    return f"Methodology extractor: {extractor.NAME}{narrow}"
