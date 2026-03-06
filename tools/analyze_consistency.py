from langchain.tools import tool
from classes.financial_summary import FinancialSummary
import math


@tool(description="Checks for a track record of consistent and growing earnings using multi-period historical data.")
def analyze_consistency(summary: FinancialSummary) -> dict:
    """
    Scores the consistency of earnings growth over multiple periods.

    Uses historical_net_income (most-recent first) when available; falls back to
    the single-period earnings_growth metric.  Max score: 4.

    Scoring:
    - +2 if ≥80% of historical periods show year-over-year net income growth
    - +2 if CAGR (oldest→latest) > 10%, +1 if CAGR > 5%
    - Fallback: +3 if single-period earnings_growth > 5%

    Args:
        summary: A FinancialSummary with at least one earnings-related field.

    Returns:
        dict with keys ``score`` (int) and ``details`` (str).
    """
    score = 0
    reasoning = []

    hist = summary.historical_net_income
    if hist and len(hist) >= 2:
        # Periods are most-recent first; check each older→newer growth
        growing = 0
        comparisons = 0
        for i in range(len(hist) - 1):
            newer = hist[i]
            older = hist[i + 1]
            if older and older != 0:
                comparisons += 1
                if newer > older:
                    growing += 1

        if comparisons > 0 and (growing / comparisons) >= 0.8:
            score += 2
            reasoning.append(f"Earnings grew in {growing}/{comparisons} historical periods.")

        # CAGR from oldest (last element) to latest (first element)
        latest = hist[0]
        oldest = hist[-1]
        n_years = len(hist) - 1
        if oldest and oldest > 0 and latest and n_years > 0:
            try:
                cagr = (latest / oldest) ** (1 / n_years) - 1
                if cagr > 0.10:
                    score += 2
                    reasoning.append(f"Strong earnings CAGR of {cagr:.1%} over {n_years} years.")
                elif cagr > 0.05:
                    score += 1
                    reasoning.append(f"Moderate earnings CAGR of {cagr:.1%} over {n_years} years.")
            except (ValueError, ZeroDivisionError):
                pass
    elif summary.earnings_growth and summary.earnings_growth > 0.05:
        score += 3
        reasoning.append("Consistent earnings growth (single-period metric).")

    return {"score": score, "details": "; ".join(reasoning) if reasoning else "Insufficient data."}
