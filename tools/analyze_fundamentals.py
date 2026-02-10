from langchain.tools import tool
from classes.financial_summary import FinancialSummary, WarrenBuffettSignal
import math
import json


@tool(description="Analyzes key financial health metrics like ROE, debt, margins, and liquidity.")
def analyze_fundamentals(summary: FinancialSummary) -> dict:
    score = 0
    reasoning = []

    if summary.return_on_equity > 0.15:
        score += 2
        reasoning.append(f"Strong ROE of {summary.return_on_equity:.1%}")
    
    if summary.debt_to_equity < 0.5:
        score += 2
        reasoning.append("Conservative debt levels.")

    if summary.operating_margin and summary.operating_margin > 0.15:
        score += 2
        reasoning.append("Strong operating margins.")

    if summary.current_ratio and summary.current_ratio > 1.5:
        score += 1
        reasoning.append("Good liquidity position.")

    return {"score": score, "details": "; ".join(reasoning)}