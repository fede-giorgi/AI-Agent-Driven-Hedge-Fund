from langchain.tools import tool
from classes.financial_summary import FinancialSummary, WarrenBuffettSignal
import math
import json

@tool(description="Analyzes the growth of book value per share over time.")
def analyze_book_value_growth(summary: FinancialSummary) -> dict:
    score = 0
    reasoning = []
    
    if summary.book_value_growth and summary.book_value_growth > 0.1:
        score += 2
        reasoning.append("Strong book value growth.")

    return {"score": score, "details": "; ".join(reasoning)}