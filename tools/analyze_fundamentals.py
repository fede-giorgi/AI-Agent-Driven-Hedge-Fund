from langchain.tools import tool
from classes.financial_summary import FinancialSummary


@tool(description="Analyzes key financial health metrics like ROE, ROIC, debt, margins, and liquidity with null-safe checks.")
def analyze_fundamentals(summary: FinancialSummary) -> dict:
    """
    Scores a company's fundamental financial health across four dimensions.

    All field accesses are null-safe; missing data is silently skipped.
    Max score: 9.

    Scoring:
    - ROE > 15%: +2
    - ROIC > 15%: +2 (capital efficiency bonus)
    - Debt/equity < 0.5: +2
    - Operating margin > 15%: +2
    - Current ratio > 1.5: +1

    Args:
        summary: A FinancialSummary with profitability, leverage, and liquidity fields.

    Returns:
        dict with keys ``score`` (int) and ``details`` (str).
    """
    score = 0
    reasoning = []

    if summary.return_on_equity is not None and summary.return_on_equity > 0.15:
        score += 2
        reasoning.append(f"Strong ROE of {summary.return_on_equity:.1%}.")

    if summary.return_on_invested_capital is not None and summary.return_on_invested_capital > 0.15:
        score += 2
        reasoning.append(f"High ROIC of {summary.return_on_invested_capital:.1%}.")

    if summary.debt_to_equity is not None and summary.debt_to_equity < 0.5:
        score += 2
        reasoning.append("Conservative debt levels.")

    if summary.operating_margin is not None and summary.operating_margin > 0.15:
        score += 2
        reasoning.append("Strong operating margins.")

    if summary.current_ratio is not None and summary.current_ratio > 1.5:
        score += 1
        reasoning.append("Good liquidity position.")

    return {"score": score, "details": "; ".join(reasoning) if reasoning else "Insufficient data."}
