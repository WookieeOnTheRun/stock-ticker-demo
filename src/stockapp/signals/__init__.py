"""Signal registry: every entry is a causal Strategy and must pass the look-ahead guard test."""

from stockapp.backtest.strategies import BUY_AND_HOLD, Strategy
from stockapp.signals.library import SIGNAL_STRATEGIES

SIGNALS: dict[str, Strategy] = {s.name: s for s in [BUY_AND_HOLD, *SIGNAL_STRATEGIES]}
"""Registered signals by display name. earnings_drift is not registered: Tier 4 surprise data is not available."""
