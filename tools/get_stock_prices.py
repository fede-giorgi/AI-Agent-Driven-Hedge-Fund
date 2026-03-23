"""Fetch historical OHLCV price data for a ticker from FinancialDatasets.ai."""

import os
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

FINDAT_API_KEY = os.getenv("FINDAT_API_KEY")


@tool(description="Get historical stock prices for a given ticker symbol")
def get_stock_prices(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    interval: str = "day",
    interval_multiplier: int = 1,
) -> dict:
    """
    Retrieve historical OHLCV (open, high, low, close, volume) prices for a ticker.

    When no dates are provided, defaults to the 7-day window ending today —
    sufficient to capture the most recent closing price even over weekends.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        interval: Time interval — "minute", "day", "week", "month", "year". Defaults to "day".
        interval_multiplier: Multiplier applied to the interval. Defaults to 1.

    Returns:
        dict with a ``prices`` list, each entry containing open, high, low,
        close, volume, and date fields.
        Returns ``{"error": ...}`` on API failure.
    """
    if not FINDAT_API_KEY:
        raise ValueError("FINDAT_API_KEY not set. Add it to your .env file.")

    dt_end = datetime.now() if not end_date else datetime.strptime(end_date, "%Y-%m-%d")
    # Default to 7-day lookback to ensure at least one trading day is captured
    dt_start = dt_end - timedelta(days=7) if not start_date else datetime.strptime(start_date, "%Y-%m-%d")

    url = (
        f"https://api.financialdatasets.ai/prices/"
        f"?ticker={ticker}"
        f"&interval={interval}"
        f"&interval_multiplier={interval_multiplier}"
        f"&start_date={dt_start.strftime('%Y-%m-%d')}"
        f"&end_date={dt_end.strftime('%Y-%m-%d')}"
    )

    headers = {"X-API-KEY": FINDAT_API_KEY}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return {"error": f"API error {response.status_code} - {response.text}"}

    return response.json()
