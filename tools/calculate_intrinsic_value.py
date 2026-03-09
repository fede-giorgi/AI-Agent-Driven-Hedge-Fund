from langchain.tools import tool
from classes.financial_summary import FinancialSummary


def estimate_maintenance_capex(summary: FinancialSummary) -> float:
    """
    Estimates maintenance capex as the higher of 85% of total capex or 100% of D&A.

    This is a conservative estimate: maintenance capex is what the company must
    spend just to keep its existing asset base intact.

    Args:
        summary: A FinancialSummary with capital_expenditure and
                 depreciation_and_amortization fields.

    Returns:
        Estimated maintenance capex as a positive float (magnitude only).
    """
    capex = summary.capital_expenditure or 0.0
    da = summary.depreciation_and_amortization or 0.0
    # capex is typically stored as a negative value in financial data
    capex_magnitude = abs(capex)
    return max(0.85 * capex_magnitude, da)


def estimate_wacc(summary: FinancialSummary) -> float:
    """
    Estimates the Weighted Average Cost of Capital (WACC) from available summary fields.

    Uses the CAPM for cost of equity (assuming beta = 1.0, risk-free rate = 4.5%,
    equity risk premium = 5.5%) and derives cost of debt from interest_coverage.
    Falls back to 10% if key fields are unavailable.

    Args:
        summary: A FinancialSummary with market_cap, debt_to_equity, and
                 interest_coverage fields.

    Returns:
        WACC as a float in the range [0.06, 0.18].
    """
    risk_free = 0.045
    erp = 0.055  # equity risk premium
    beta = 1.0   # market-neutral assumption
    cost_of_equity = risk_free + beta * erp  # ≈ 10%

    # Cost of debt: infer from interest coverage (lower coverage → higher spread)
    ic = summary.interest_coverage
    if ic and ic > 0:
        # Rough mapping: IC 1x → +8%, IC 5x → +3%, IC 15x → +1.5%
        spread = max(0.015, min(0.08, 10.0 / (ic + 1)))
        cost_of_debt = risk_free + spread
    else:
        cost_of_debt = risk_free + 0.05  # default 9.5%

    # Capital structure weights from debt_to_equity ratio
    de = summary.debt_to_equity
    if de and de >= 0 and summary.market_cap and summary.market_cap > 0:
        total_debt = summary.market_cap * de  # approximation
        total_value = summary.market_cap + total_debt
        w_equity = summary.market_cap / total_value
        w_debt = total_debt / total_value
        tax_rate = 0.25
        wacc = w_equity * cost_of_equity + w_debt * cost_of_debt * (1 - tax_rate)
    else:
        wacc = cost_of_equity  # no debt data — use cost of equity only

    return float(min(max(wacc, 0.06), 0.18))


def calculate_owner_earnings(summary: FinancialSummary) -> dict:
    """
    Calculates Buffett's "owner earnings" — a truer measure of free cash generation.

    Owner Earnings = Net Income + D&A − Maintenance Capex

    Args:
        summary: A FinancialSummary with net_income, depreciation_and_amortization,
                 and capital_expenditure fields.

    Returns:
        dict with ``owner_earnings`` (float or None) and ``details`` (str).
    """
    if not all([summary.net_income, summary.depreciation_and_amortization, summary.capital_expenditure]):
        return {"owner_earnings": None, "details": "Missing data for owner earnings calculation."}

    maintenance_capex = estimate_maintenance_capex(summary)
    owner_earnings = summary.net_income + summary.depreciation_and_amortization - maintenance_capex

    return {"owner_earnings": owner_earnings, "details": f"Owner earnings estimated at ${owner_earnings:,.0f}."}


@tool(description="Estimates intrinsic value using a 3-stage DCF model based on owner earnings with a 15% conservatism haircut.")
def calculate_intrinsic_value(summary: FinancialSummary) -> dict:
    """
    Computes intrinsic value via a 3-stage Discounted Cash Flow model.

    Stages:
    - Stage 1 (years 1-5): growth capped at 8%, derived from historical_net_income CAGR
    - Stage 2 (years 6-10): half the Stage 1 growth rate
    - Terminal: 2.5% perpetual growth rate

    A 15% conservatism haircut is applied to the raw DCF total.
    Discount rate: WACC estimated from debt_to_equity + interest_coverage (falls back to ~10%).

    Args:
        summary: A FinancialSummary with earnings, cash-flow, and share-count fields.

    Returns:
        dict with ``intrinsic_value``, ``intrinsic_value_per_share``,
        ``margin_of_safety``, and ``details`` keys.
    """
    owner_earnings_data = calculate_owner_earnings(summary)
    if not owner_earnings_data["owner_earnings"]:
        return {"intrinsic_value": None, "intrinsic_value_per_share": None, "margin_of_safety": None,
                "details": "Could not calculate owner earnings."}

    owner_earnings = owner_earnings_data["owner_earnings"]

    # Derive growth rate from historical net income CAGR (capped at 8%)
    hist = summary.historical_net_income
    growth_rate = 0.03  # default conservative
    if hist and len(hist) >= 2:
        latest = hist[0]
        oldest = hist[-1]
        n_years = len(hist) - 1
        if oldest and oldest > 0 and latest and n_years > 0:
            try:
                cagr = (latest / oldest) ** (1 / n_years) - 1
                growth_rate = min(max(cagr, 0.0), 0.08)
            except (ValueError, ZeroDivisionError):
                pass
    elif summary.earnings_growth and summary.earnings_growth > 0:
        growth_rate = min(summary.earnings_growth, 0.08)

    # Use WACC as the discount rate (more rigorous than a flat 10%).
    # For most large-caps this will be 8–12%; the floor/cap is [6%, 18%].
    discount_rate = estimate_wacc(summary)
    terminal_growth_rate = 0.025
    stage2_growth = growth_rate / 2

    # 3-stage DCF
    dcf_value = 0.0
    current_earnings = owner_earnings

    # Stage 1: years 1-5
    for i in range(1, 6):
        current_earnings *= (1 + growth_rate)
        dcf_value += current_earnings / ((1 + discount_rate) ** i)

    # Stage 2: years 6-10
    for i in range(6, 11):
        current_earnings *= (1 + stage2_growth)
        dcf_value += current_earnings / ((1 + discount_rate) ** i)

    # Terminal value
    terminal_value = (current_earnings * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
    terminal_pv = terminal_value / ((1 + discount_rate) ** 10)
    intrinsic_value = (dcf_value + terminal_pv) * 0.85  # 15% conservatism haircut

    intrinsic_value_per_share = None
    margin_of_safety = None

    if summary.outstanding_shares and summary.outstanding_shares > 0:
        intrinsic_value_per_share = intrinsic_value / summary.outstanding_shares

    if intrinsic_value_per_share and summary.price and summary.price > 0:
        margin_of_safety = (intrinsic_value_per_share - summary.price) / summary.price

    return {
        "intrinsic_value": intrinsic_value,
        "intrinsic_value_per_share": intrinsic_value_per_share,
        "margin_of_safety": margin_of_safety,
        "details": (
            f"Intrinsic value ~${intrinsic_value:,.0f} "
            f"(stage1 growth {growth_rate:.1%}, stage2 {stage2_growth:.1%}, "
            f"terminal 2.5%, WACC {discount_rate:.1%}, 15% haircut)."
        ),
    }
