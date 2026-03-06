from langchain.tools import tool
from classes.financial_summary import FinancialSummary


@tool(description="Analyzes the growth and consistency of book value per share using historical equity and share-count data.")
def analyze_book_value_growth(summary: FinancialSummary) -> dict:
    """
    Scores book value per share growth over multiple periods.

    Derives per-share book value from historical_shareholders_equity /
    historical_outstanding_shares when available.  Falls back to
    book_value_growth (single-period metric).  Max score: 5.

    Scoring:
    - CAGR > 15%: +3, CAGR > 10%: +2, CAGR > 5%: +1
    - Consistency (≥70% of periods growing): +2

    Args:
        summary: A FinancialSummary with equity and share-count fields.

    Returns:
        dict with keys ``score`` (int) and ``details`` (str).
    """
    score = 0
    reasoning = []

    hist_equity = summary.historical_shareholders_equity
    hist_shares = summary.historical_outstanding_shares

    if hist_equity and hist_shares and len(hist_equity) >= 2 and len(hist_shares) == len(hist_equity):
        bvps = []
        for eq, sh in zip(hist_equity, hist_shares):
            if sh and sh > 0 and eq is not None:
                bvps.append(eq / sh)
            else:
                bvps.append(None)

        valid_bvps = [v for v in bvps if v is not None]
        if len(valid_bvps) >= 2:
            latest = valid_bvps[0]
            oldest = valid_bvps[-1]
            n_years = len(valid_bvps) - 1

            if oldest > 0:
                try:
                    cagr = (latest / oldest) ** (1 / n_years) - 1
                    if cagr > 0.15:
                        score += 3
                        reasoning.append(f"Book value CAGR of {cagr:.1%} over {n_years} years (strong).")
                    elif cagr > 0.10:
                        score += 2
                        reasoning.append(f"Book value CAGR of {cagr:.1%} over {n_years} years.")
                    elif cagr > 0.05:
                        score += 1
                        reasoning.append(f"Book value CAGR of {cagr:.1%} over {n_years} years (moderate).")
                except (ValueError, ZeroDivisionError):
                    pass

            # Consistency: count periods where BVPS grew
            growing = sum(1 for i in range(len(valid_bvps) - 1) if valid_bvps[i] > valid_bvps[i + 1])
            comparisons = len(valid_bvps) - 1
            if comparisons > 0 and (growing / comparisons) >= 0.70:
                score += 2
                reasoning.append(f"Book value grew in {growing}/{comparisons} periods.")
    elif summary.book_value_growth and summary.book_value_growth > 0.1:
        score += 2
        reasoning.append("Strong book value growth (single-period metric).")

    return {"score": score, "details": "; ".join(reasoning) if reasoning else "Insufficient data."}
