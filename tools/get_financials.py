"""Fetch combined financial statements (income, balance sheet, cash flow) from FinancialDatasets.ai."""

import os

import requests
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

FINDAT_API_KEY = os.getenv("FINDAT_API_KEY")


@tool(description="Get financial data for a given ticker symbol")
def get_financials(
    ticker: str,
    period: str = "annual",
    limit: int = 10,
    end_date: str | None = None,
) -> dict:
    """
    Fetch income statement, balance sheet, and cash flow statements for a ticker.

    Returns up to `limit` reporting periods of combined financial statements.
    When `end_date` is provided, only periods on or before that date are returned,
    enabling point-in-time backtesting.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        period: Reporting period — "annual", "quarterly", or "ttm". Defaults to "annual".
        limit: Maximum number of periods to return. Defaults to 10.
        end_date: Optional cutoff date (YYYY-MM-DD); filters to periods ≤ this date.

    Returns:
        dict with keys ``income_statements``, ``balance_sheets``, and
        ``cash_flow_statements``, each containing a list of period dicts.
        Returns ``{"error": ...}`` on API failure.
    """
    if not FINDAT_API_KEY:
        raise ValueError(
            "FINDAT_API_KEY not set. Add it to your .env file."
        )

    headers = {"X-API-KEY": FINDAT_API_KEY}
    url = (
        f"https://api.financialdatasets.ai/financials/"
        f"?ticker={ticker}&period={period}&limit={limit}"
    )

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return {"error": f"API error {response.status_code} - {response.text}"}

    data = response.json()
    financials = data.get("financials")

    if end_date and financials:
        # API may return either a list of period dicts or a single aggregator dict
        aggregator = financials[0] if isinstance(financials, list) else financials
        statement_types = ["income_statements", "balance_sheets", "cash_flow_statements"]
        selected: dict = {}

        for key in statement_types:
            filtered = [
                f for f in aggregator.get(key, [])
                if f.get("report_period") and f.get("report_period") <= end_date
            ]
            if filtered:
                selected[key] = filtered
            else:
                print(f"Warning: No data found for {key} before {end_date}")

        return selected if selected else {"error": f"No financial data found before {end_date}"}

    return data
