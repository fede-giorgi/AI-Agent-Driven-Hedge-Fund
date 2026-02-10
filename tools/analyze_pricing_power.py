from langchain.tools import tool
from classes.financial_summary import FinancialSummary, WarrenBuffettSignal
import math
import json

@tool(description="Assesses the company's ability to raise prices by analyzing its gross margin trends.")
def analyze_pricing_power(summary: FinancialSummary) -> dict:
    score = 0
    reasoning = []

    if summary.gross_margin and summary.gross_margin > 0.4:
        score += 2
        reasoning.append("High gross margins suggest strong pricing power.")
    
    return {"score": score, "details": "; ".join(reasoning)}