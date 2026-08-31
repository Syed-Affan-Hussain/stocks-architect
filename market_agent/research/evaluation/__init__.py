"""Prospective evaluation harness for the AI Market Research & Analysis
product - see this package's module docstrings for the full design.
Deliberately SEPARATE from market_agent/research/'s core report pipeline:
nothing here changes assessment.py, consistency.py, risk.py, or the News
State Engine (frozen as of Event Quantifier v1.1) - this package only
CONSUMES their already-published output to log and later score decisions.
"""
