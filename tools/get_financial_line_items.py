"""Fetch granular financial line items for one or more tickers from FinancialDatasets.ai."""

import os

import requests
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

FINDAT_API_KEY = os.getenv("FINDAT_API_KEY")


@tool(description="Get financial line items for a given ticker symbol")
def get_financial_line_items(
    tickers: list[str],
    line_items: list[str],
    period: str = "annual",
    limit: int = 30,
    end_date: str | None = None,
) -> dict:
    """
    Retrieve specific financial line items for a list of tickers.

    Supports any line item available on FinancialDatasets.ai, including
    capital_expenditure, net_income, revenue, free_cash_flow, shareholders_equity,
    outstanding_shares, and more. Use limit=8 to pull 8 years of history.

    Args:
        tickers: List of stock ticker symbols (e.g. ["AAPL", "NVDA"]).
        line_items: Line item names to retrieve (e.g. ["net_income", "revenue"]).
        period: Reporting period — "annual", "quarterly", or "ttm". Defaults to "annual".
        limit: Number of periods to retrieve per ticker. Defaults to 30.
        end_date: Optional cutoff date (YYYY-MM-DD); filters to periods ≤ this date.

    Returns:
        dict with a ``search_results`` list, each entry containing the
        requested line items for one ticker/period combination.
        Returns ``{"error": ...}`` on API failure.
    """
    if not FINDAT_API_KEY:
        raise ValueError("FINDAT_API_KEY not set. Add it to your .env file.")

    headers = {
        "X-API-KEY": FINDAT_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "tickers": tickers,
        "line_items": line_items,
        "period": period,
        "limit": limit,
    }

    response = requests.post(
        "https://api.financialdatasets.ai/financials/search/line-items",
        headers=headers,
        json=payload,
    )
    if response.status_code != 200:
        return {"error": f"API error {response.status_code} - {response.text}"}

    data = response.json()
    search_results = data.get("search_results")

    if end_date and search_results:
        filtered = [
            f for f in search_results
            if f.get("report_period") and f.get("report_period") <= end_date
        ]
        return {"search_results": filtered} if filtered else {"error": f"No data found before {end_date}"}

    return data
