from langchain.tools import tool
from classes.financial_summary import FinancialSummary, WarrenBuffettSignal
import math
import json

@tool(description="Evaluates the company's durable competitive advantage by looking at return on capital and margin stability.")
def analyze_moat(summary: FinancialSummary) -> dict:
    moat_score = 0
    reasoning = []

    if summary.return_on_invested_capital and summary.return_on_invested_capital > 0.15:
        moat_score += 2
        reasoning.append("High ROIC suggests a strong moat.")

    if summary.gross_margin and summary.gross_margin > 0.4:
        moat_score += 1
        reasoning.append("High gross margins indicate pricing power.")

    return {"score": moat_score, "details": "; ".join(reasoning)}