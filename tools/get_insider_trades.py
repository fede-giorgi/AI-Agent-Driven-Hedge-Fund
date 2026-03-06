import os
import requests
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

FINDAT_API_KEY = os.getenv("FINDAT_API_KEY")


@tool(description="Get recent insider trades (Form 4 filings by executives and directors) for a given ticker from FinancialDatasets.ai.")
def get_insider_trades(
    ticker: str,
    limit: int = 20,
    end_date: str = None,
) -> dict:
    """
    Fetches recent insider buy/sell transactions for a ticker.

    Insider buying is a strong positive signal (insiders buy with conviction);
    heavy insider selling warrants scrutiny.  Use this to gauge insider sentiment.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        limit: Number of transactions to return (max 100, default 20).
        end_date: Optional cutoff date (YYYY-MM-DD); filters to filings ≤ this date.

    Returns:
        dict with an ``insider_trades`` list.  Each trade contains:
        - ``name`` (str): Insider's name.
        - ``title`` (str): Role (CEO, CFO, Director, etc.).
        - ``transaction_date`` (str): Date of transaction.
        - ``transaction_shares`` (float): Shares bought (+) or sold (−).
        - ``transaction_price_per_share`` (float): Price per share.
        - ``transaction_value`` (float): Total dollar value (+ buy, − sell).
        - ``shares_owned_after_transaction`` (float): Ownership after trade.
        - ``is_board_director`` (bool): True if the insider is a board member.
    """
    if not FINDAT_API_KEY:
        raise ValueError("FINDAT_API_KEY not set.")

    headers = {"X-API-KEY": FINDAT_API_KEY}
    url = (
        f"https://api.financialdatasets.ai/insider-trades"
        f"?ticker={ticker}&limit={min(limit, 100)}"
    )
    if end_date:
        url += f"&filing_date_lte={end_date}"

    response = requests.get(url, headers=headers, timeout=15)
    if response.status_code != 200:
        return {"error": f"API error {response.status_code} - {response.text}"}

    return response.json()
