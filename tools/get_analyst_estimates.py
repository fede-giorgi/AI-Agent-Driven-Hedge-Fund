import os
import requests
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

FINDAT_API_KEY = os.getenv("FINDAT_API_KEY")


@tool(description="Get Wall Street analyst consensus revenue and EPS estimates for a given ticker from FinancialDatasets.ai.")
def get_analyst_estimates(
    ticker: str,
    period: str = "annual",
    limit: int = 4,
    end_date: str = None,
) -> dict:
    """
    Fetches analyst consensus forward estimates for revenue and earnings per share.

    Analyst estimates provide a market-consensus baseline for valuation.
    Comparing actual results vs. estimates reveals whether a company
    consistently beats or misses expectations (earnings quality signal).

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        period: "annual" or "quarterly" (default "annual").
        limit: Number of estimate periods to return (default 4).
        end_date: Optional cutoff date (YYYY-MM-DD) to simulate historical view.

    Returns:
        dict with an ``analyst_estimates`` list.  Each item contains:
        - ``fiscal_period`` (str): Target fiscal period (e.g. "FY2025").
        - ``period`` (str): "annual" or "quarterly".
        - ``revenue`` (float): Consensus revenue estimate.
        - ``earnings_per_share`` (float): Consensus EPS estimate.
    """
    if not FINDAT_API_KEY:
        raise ValueError("FINDAT_API_KEY not set.")

    headers = {"X-API-KEY": FINDAT_API_KEY}
    url = (
        f"https://api.financialdatasets.ai/analyst-estimates"
        f"?ticker={ticker}&period={period}&limit={limit}"
    )

    response = requests.get(url, headers=headers, timeout=15)
    if response.status_code != 200:
        return {"error": f"API error {response.status_code} - {response.text}"}

    data = response.json()

    # If backtesting: filter estimates to those with fiscal_period before end_date
    if end_date:
        estimates = data.get("analyst_estimates", [])
        filtered = [e for e in estimates if e.get("fiscal_period", "") <= end_date[:4] + "Z"]
        if filtered:
            return {"analyst_estimates": filtered}
        return {"analyst_estimates": estimates}

    return data
