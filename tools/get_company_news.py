import os
import requests
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

FINDAT_API_KEY = os.getenv("FINDAT_API_KEY")


@tool(description="Get recent company news articles for a given ticker from FinancialDatasets.ai (max 10 articles).")
def get_company_news(ticker: str, limit: int = 5) -> dict:
    """
    Fetches recent news articles for a stock ticker from FinancialDatasets.ai.

    Use this to surface qualitative context such as earnings announcements,
    product launches, regulatory actions, or macro events that may affect the
    company's outlook.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        limit: Number of articles to return (1–10, default 5).

    Returns:
        dict with a ``news`` list.  Each item contains:
        - ``title`` (str): Article headline.
        - ``source`` (str): Publication name.
        - ``date`` (str): Publication date (YYYY-MM-DD).
        - ``url`` (str): Link to the full article.
    """
    if not FINDAT_API_KEY:
        raise ValueError("FINDAT_API_KEY not set.")

    headers = {"X-API-KEY": FINDAT_API_KEY}
    url = f"https://api.financialdatasets.ai/news?ticker={ticker}&limit={min(limit, 10)}"

    response = requests.get(url, headers=headers, timeout=15)
    if response.status_code != 200:
        return {"error": f"API error {response.status_code} - {response.text}"}

    return response.json()
