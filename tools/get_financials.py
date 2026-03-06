import os
import requests
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

FINDAT_API_KEY = os.getenv("FINDAT_API_KEY")

@tool(description="Get financial data for a given ticker symbol")
def get_financials(ticker: str,
                   period: str = 'annual',
                   limit: int = 10,
                   end_date: str = None
                   ) -> dict:
    """
    Fetches income statement, balance sheet, and cash flow statements for a ticker.

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
    # check if API key is set
    if not FINDAT_API_KEY:
        raise ValueError(
            "API key for Financial Datasets not found. Please set the FINDAT_API_KEY environment variable."
        )
    
    # add your API key to the headers
    headers = {
        "X-API-KEY": FINDAT_API_KEY
    }

    # create the URL
    url = (
        f'https://api.financialdatasets.ai/financials/'
        f'?ticker={ticker}'
        f'&period={period}'
        f'&limit={limit}'
    )

    # make API request
    response = requests.get(url, headers=headers)

    # if status code is 400, 401, 402 or 404 return error message
    if response.status_code != 200:
        return {"error": f"API error {response.status_code} - {response.text}"}

    # parse data from the response
    data = response.json()
    financials = data.get('financials')

    selected_financials = {}

    if end_date and financials:
        # Handle list vs. dict format from the API response
        aggregator = financials[0] if isinstance(financials, list) else financials
        
        statement_types = ['income_statements', 'balance_sheets', 'cash_flow_statements']
        
        for key in statement_types:
            # Get the list of reports for the specific type (e.g., income statements)
            statements_list = aggregator.get(key, [])
            
            # Keep only past reports (relative to the end_date)
            filtered = [
                f for f in statements_list
                if f.get('report_period') and f.get('report_period') <= end_date
            ]
            
            if filtered:
                selected_financials[key] = filtered 
            else:
                print(f"Warning: No data found for {key} before {end_date}")

        if selected_financials:
            return selected_financials
        else:
            return {"error": f"No financial data found before {end_date}"}

    # return all financials if no end_date is specified
    return data