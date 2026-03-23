"""Fetch historical financial ratios and valuation metrics from FinancialDatasets.ai."""

import os

import requests
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

FINDAT_API_KEY = os.getenv("FINDAT_API_KEY")


@tool(description="Get historical financial metrics for a given ticker symbol")
def get_metrics(
    ticker: str,
    period: str = "annual",
    limit: int = 10,
    end_date: str | None = None,
) -> dict:
    """
    Fetch historical financial ratios and metrics for a ticker.

    Covers 35+ metrics across valuation (P/E, P/B, EV/EBITDA), profitability
    (ROE, ROIC, gross/operating/net margin), liquidity (current ratio, quick ratio),
    leverage (debt/equity, interest coverage), and growth (revenue, earnings, FCF).
    Use limit=8 to pull 8 annual periods for multi-year trend analysis.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        period: Reporting period — "annual", "quarterly", or "ttm". Defaults to "annual".
        limit: Number of periods to return. Defaults to 10.
        end_date: Optional cutoff date (YYYY-MM-DD); filters to periods ≤ this date.

    Returns:
        dict with a ``financial_metrics`` list, each entry containing all
        available ratio fields for one reporting period.
        Returns ``{"error": ...}`` on API failure.
    """
    if not FINDAT_API_KEY:
        raise ValueError("FINDAT_API_KEY not set. Add it to your .env file.")

    headers = {"X-API-KEY": FINDAT_API_KEY}
    url = (
        f"https://api.financialdatasets.ai/financial-metrics"
        f"?ticker={ticker}&period={period}&limit={limit}"
    )

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return {"error": f"API error {response.status_code} - {response.text}"}

    data = response.json()
    metrics = data.get("financial_metrics")

    if end_date and metrics:
        filtered = [
            m for m in metrics
            if m.get("report_period") and m.get("report_period") <= end_date
        ]
        return {"financial_metrics": filtered} if filtered else {"error": f"No data found before {end_date}"}

    return data
