from langchain.tools import tool
from classes.financial_summary import FinancialSummary


@tool(description="Assesses management's shareholder-friendliness via multi-period buyback track record and dividend history.")
def analyze_management_quality(summary: FinancialSummary) -> dict:
    """
    Scores management quality based on capital-return actions.

    Uses historical_issuance_or_purchase_of_equity_shares for a multi-period
    buyback track record; falls back to the single-period field.  Max score: 2.

    Scoring:
    - +1 if consistent buybacks: ≥70% of historical equity-issuance periods
      are negative (net buybacks), or single-period issuance < 0
    - +1 if company pays dividends (payout_ratio > 0 or distributions < 0)

    Args:
        summary: A FinancialSummary with shareholder-return fields.

    Returns:
        dict with keys ``score`` (int) and ``details`` (str).
    """
    mgmt_score = 0
    reasoning = []

    # Buyback track record
    hist_equity = summary.historical_issuance_or_purchase_of_equity_shares
    if hist_equity and len(hist_equity) >= 2:
        buyback_periods = sum(1 for v in hist_equity if v and v < 0)
        pct = buyback_periods / len(hist_equity)
        if pct >= 0.70:
            mgmt_score += 1
            reasoning.append(f"Consistent buybacks: {buyback_periods}/{len(hist_equity)} periods had net share repurchases.")
    elif summary.issuance_or_purchase_of_equity_shares and summary.issuance_or_purchase_of_equity_shares < 0:
        mgmt_score += 1
        reasoning.append("Company has been repurchasing shares.")

    # Dividend track record
    pays_dividend = (
        (summary.payout_ratio and summary.payout_ratio > 0)
        or (summary.dividends_and_other_cash_distributions and summary.dividends_and_other_cash_distributions < 0)
    )
    if pays_dividend:
        mgmt_score += 1
        reasoning.append("Company pays dividends.")

    return {"score": mgmt_score, "details": "; ".join(reasoning) if reasoning else "No shareholder-return signals found."}
