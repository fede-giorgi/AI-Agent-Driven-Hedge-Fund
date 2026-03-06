# tools/mcp.py
#
# Brave Search MCP has been replaced by the native FinancialDatasets.ai news
# endpoint (tools/get_company_news.py).  This module is kept as a placeholder
# so any stale imports don't hard-crash.  fetch_ticker_news is a no-op.


async def fetch_ticker_news(ticker: str) -> str:  # noqa: D401
    """No-op shim.  News is now fetched via the get_company_news LangChain tool."""
    return ""
