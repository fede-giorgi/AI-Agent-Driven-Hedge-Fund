from langchain.tools import tool
from classes.financial_summary import FinancialSummary, WarrenBuffettSignal
import math
import json

@tool(description="Assesses management's shareholder-friendliness by checking for share buybacks and dividends.")
def analyze_management_quality(summary: FinancialSummary) -> dict:
    mgmt_score = 0
    reasoning = []

    if summary.issuance_or_purchase_of_equity_shares and summary.issuance_or_purchase_of_equity_shares < 0:
        mgmt_score += 1
        reasoning.append("Company has been repurchasing shares.")

    if summary.payout_ratio and summary.payout_ratio > 0:
        mgmt_score += 1
        reasoning.append("Company pays dividends.")

    return {"score": mgmt_score, "details": "; ".join(reasoning)}