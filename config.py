"""
Central configuration for the AI Hedge Fund simulation.

All magic numbers and tunable constants live here so callers
never hard-code values that may need changing across runs.
"""

# ── Pipeline ──────────────────────────────────────────────────────────────────

# Number of PM → Monitor → What-If debate iterations
TOTAL_ITERATIONS: int = 10
TOTAL_ITERATIONS_DEBUG: int = 3

# Tickers used in --debug mode and as the "default" preset
DEFAULT_TICKERS: list[str] = ["AAPL", "MSFT", "NVDA", "GOOGL", "META"]

# ── Backtesting benchmarks ────────────────────────────────────────────────────

# US 3-month T-bill rate used as the risk-free benchmark.
# Update periodically; current Fed funds / T-bill environment: ~4.5%.
RISK_FREE_ANNUAL: float = 0.045
