from langchain.tools import tool
from classes.financial_summary import FinancialSummary


@tool(description="Assesses pricing power by analysing gross margin trends and absolute levels across multiple periods.")
def analyze_pricing_power(summary: FinancialSummary) -> dict:
    """
    Scores a company's pricing power using gross margin history.

    Derives gross margins from historical_gross_profit / historical_revenue
    when available.  Falls back to the single-period gross_margin field.
    Max score: 5.

    Scoring (trend, up to +3):
    - +3 if recent avg margin ≥ older avg margin by >2pp (improving)
    - +2 if recent avg margin ≥ older avg margin (stable)
    - +1 if margins are only slightly declining (<2pp drop)

    Scoring (absolute level, up to +2):
    - +2 if latest gross margin > 50%
    - +1 if latest gross margin > 30%

    Args:
        summary: A FinancialSummary with gross profit and revenue fields.

    Returns:
        dict with keys ``score`` (int) and ``details`` (str).
    """
    score = 0
    reasoning = []

    hist_gp = summary.historical_gross_profit
    hist_rev = summary.historical_revenue

    margins = []
    if hist_gp and hist_rev and len(hist_gp) >= 2 and len(hist_rev) == len(hist_gp):
        for gp, rev in zip(hist_gp, hist_rev):
            if rev and rev > 0 and gp is not None:
                margins.append(gp / rev)
            else:
                margins.append(None)
        margins = [m for m in margins if m is not None]

    if len(margins) >= 4:
        mid = len(margins) // 2
        recent_avg = sum(margins[:mid]) / mid
        older_avg = sum(margins[mid:]) / (len(margins) - mid)
        diff = recent_avg - older_avg

        if diff > 0.02:
            score += 3
            reasoning.append(f"Gross margins improving: recent {recent_avg:.1%} vs older {older_avg:.1%}.")
        elif diff >= 0:
            score += 2
            reasoning.append(f"Gross margins stable: recent {recent_avg:.1%} vs older {older_avg:.1%}.")
        elif diff > -0.02:
            score += 1
            reasoning.append(f"Gross margins slightly declining: recent {recent_avg:.1%} vs older {older_avg:.1%}.")

        latest_margin = margins[0]
    elif summary.gross_margin:
        latest_margin = summary.gross_margin
    else:
        latest_margin = None

    # Absolute level bonus
    if latest_margin is not None:
        if latest_margin > 0.50:
            score += 2
            reasoning.append(f"Excellent gross margin of {latest_margin:.1%}.")
        elif latest_margin > 0.30:
            score += 1
            reasoning.append(f"Decent gross margin of {latest_margin:.1%}.")

    return {"score": score, "details": "; ".join(reasoning) if reasoning else "Insufficient data."}
