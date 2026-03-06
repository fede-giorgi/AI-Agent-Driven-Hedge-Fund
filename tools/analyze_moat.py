from langchain.tools import tool
from classes.financial_summary import FinancialSummary


@tool(description="Evaluates the company's durable competitive advantage using historical ROE consistency, margin stability, and capital efficiency.")
def analyze_moat(summary: FinancialSummary) -> dict:
    """
    Scores the strength of a company's economic moat.

    Uses historical_return_on_equity and historical_operating_margin when
    available for a richer multi-period view.  Max score: 4.

    Scoring:
    - ROE consistency: +2 if ≥70% of historical ROE periods exceed 15%,
      or +1 if single-period ROIC > 15%
    - Operating margin stability: +1 if recent avg margin ≥ older avg margin
      (i.e. margins are not deteriorating)
    - Asset efficiency (ROIC): +1 if return_on_invested_capital > 15%

    Args:
        summary: A FinancialSummary with capital efficiency and margin fields.

    Returns:
        dict with keys ``score`` (int) and ``details`` (str).
    """
    moat_score = 0
    reasoning = []

    # ROE consistency across historical periods
    hist_roe = summary.historical_return_on_equity
    if hist_roe and len(hist_roe) >= 2:
        above_threshold = sum(1 for r in hist_roe if r and r > 0.15)
        pct = above_threshold / len(hist_roe)
        if pct >= 0.70:
            moat_score += 2
            reasoning.append(f"ROE exceeded 15% in {above_threshold}/{len(hist_roe)} periods.")
        elif pct >= 0.40:
            moat_score += 1
            reasoning.append(f"ROE exceeded 15% in {above_threshold}/{len(hist_roe)} periods (partial).")
    elif summary.return_on_invested_capital and summary.return_on_invested_capital > 0.15:
        moat_score += 1
        reasoning.append("High ROIC suggests a strong moat (single period).")

    # Operating margin stability: split history in half; recent avg vs older avg
    hist_om = summary.historical_operating_margin
    if hist_om and len(hist_om) >= 4:
        mid = len(hist_om) // 2
        recent_avg = sum(m for m in hist_om[:mid] if m is not None) / mid
        older_avg = sum(m for m in hist_om[mid:] if m is not None) / (len(hist_om) - mid)
        if recent_avg >= older_avg:
            moat_score += 1
            reasoning.append(f"Operating margins stable/improving (recent avg {recent_avg:.1%} vs older {older_avg:.1%}).")
    elif summary.gross_margin and summary.gross_margin > 0.4:
        moat_score += 1
        reasoning.append("High gross margins indicate pricing power.")

    # Asset efficiency via ROIC
    if summary.return_on_invested_capital and summary.return_on_invested_capital > 0.15:
        moat_score += 1
        reasoning.append(f"ROIC of {summary.return_on_invested_capital:.1%} signals efficient capital deployment.")

    return {"score": moat_score, "details": "; ".join(reasoning) if reasoning else "Insufficient data."}
