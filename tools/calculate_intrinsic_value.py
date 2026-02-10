from langchain.tools import tool
from classes.financial_summary import FinancialSummary, WarrenBuffettSignal
import math
import json

# Helper functions
def estimate_maintenance_capex(summary: FinancialSummary) -> float:
    """Estimates the capital expenditures required to maintain current operations."""
    # Simplified estimation using TTM data as we rely on the summary
    if summary.depreciation_and_amortization:
        return summary.depreciation_and_amortization
    return 0.0


def calculate_owner_earnings(summary: FinancialSummary) -> dict:
    """
    Calculates "owner earnings," Buffett's preferred measure of profitability.
    """
    if not all([summary.net_income, summary.depreciation_and_amortization, summary.capital_expenditure]):
        return {"owner_earnings": None, "details": "Missing data for owner earnings calculation."}

    maintenance_capex = estimate_maintenance_capex(summary)
    # Owner Earnings = Net Income + Depreciation - Maintenance Capex
    # Note: capital_expenditure is usually negative in data, but maintenance_capex is a cost (positive magnitude).
    owner_earnings = summary.net_income + summary.depreciation_and_amortization - maintenance_capex
    
    return {"owner_earnings": owner_earnings, "details": f"Owner earnings estimated at ${owner_earnings:,.0f}."}

@tool(description="Estimates the company's intrinsic value using a DCF model.")
def calculate_intrinsic_value(summary: FinancialSummary) -> dict:
    owner_earnings_data = calculate_owner_earnings(summary)
    if not owner_earnings_data["owner_earnings"]:
        return {"intrinsic_value": None, "details": "Could not calculate owner earnings."}
    
    owner_earnings = owner_earnings_data["owner_earnings"]
    
    # Simplified DCF model
    growth_rate = summary.earnings_growth if summary.earnings_growth and summary.earnings_growth > 0 else 0.03
    discount_rate = 0.09
    terminal_growth_rate = 0.02

    # 10-year, 2-stage DCF
    dcf_value = 0
    for i in range(1, 11):
        dcf_value += owner_earnings * ((1 + growth_rate) ** i) / ((1 + discount_rate) ** i)

    terminal_value = (dcf_value * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
    intrinsic_value = (dcf_value + terminal_value)

    if summary.outstanding_shares:
        intrinsic_value_per_share = intrinsic_value / summary.outstanding_shares
    else:
        intrinsic_value_per_share = 0

    return {"intrinsic_value": intrinsic_value, "intrinsic_value_per_share": intrinsic_value_per_share, "details": f"Intrinsic value estimated at ${intrinsic_value:,.0f}."}
