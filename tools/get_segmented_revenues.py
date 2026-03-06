import os
import requests
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

FINDAT_API_KEY = os.getenv("FINDAT_API_KEY")


@tool(description="Get business-segment revenue breakdown (product lines, geographies) for a given ticker from FinancialDatasets.ai.")
def get_segmented_revenues(
    ticker: str,
    period: str = "annual",
    limit: int = 4,
    end_date: str = None,
) -> dict:
    """
    Fetches segmented (business-unit / geographic) revenue data for a ticker.

    Segmented revenues reveal which product lines or geographies drive growth,
    helping assess concentration risk and competitive positioning.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        period: "annual" or "quarterly" (default "annual").
        limit: Number of reporting periods to return (default 4).
        end_date: Optional cutoff date (YYYY-MM-DD); filters to periods ≤ this date.

    Returns:
        dict with a ``segmented_revenues`` list.  Each item contains:
        - ``report_period`` (str): Period end date.
        - ``items`` (list): Each item has ``line_item``, ``amount``, and ``segments``
          (a dict mapping segment name → amount).
    """
    if not FINDAT_API_KEY:
        raise ValueError("FINDAT_API_KEY not set.")

    headers = {"X-API-KEY": FINDAT_API_KEY}
    url = (
        f"https://api.financialdatasets.ai/financials/segmented-revenues"
        f"?ticker={ticker}&period={period}&limit={limit}"
    )
    if end_date:
        url += f"&report_period_lte={end_date}"

    response = requests.get(url, headers=headers, timeout=15)
    if response.status_code != 200:
        return {"error": f"API error {response.status_code} - {response.text}"}

    return response.json()
