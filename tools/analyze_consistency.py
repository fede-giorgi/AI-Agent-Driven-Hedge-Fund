from langchain.tools import tool
from classes.financial_summary import FinancialSummary, WarrenBuffettSignal
import math
import json

@tool(description="Checks for a track record of consistent and growing earnings.")
def analyze_consistency(summary: FinancialSummary) -> dict:
    score = 0
    reasoning = []

    if summary.earnings_growth and summary.earnings_growth > 0.05:
        score += 3
        reasoning.append("Consistent earnings growth.")
    
    return {"score": score, "details": "; ".join(reasoning)}