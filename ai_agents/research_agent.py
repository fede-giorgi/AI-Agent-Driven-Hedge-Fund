import json
from typing import List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from rich.console import Console
from llm import get_llm

_console = Console()
from classes.financial_summary import FinancialSummary, ToolStatus, Error, Result, ResearchAgentOutput
from tools.get_financials import get_financials
from tools.get_metrics import get_metrics
from tools.get_financial_line_items import get_financial_line_items
from tools.get_stock_prices import get_stock_prices
from tools.get_company_news import get_company_news
from tools.get_segmented_revenues import get_segmented_revenues
from tools.get_insider_trades import get_insider_trades
from tools.get_analyst_estimates import get_analyst_estimates


# All tools available to the research agent
_ALL_TOOLS = [
    get_financials,
    get_metrics,
    get_financial_line_items,
    get_stock_prices,
    get_company_news,
    get_segmented_revenues,
    get_insider_trades,
    get_analyst_estimates,
]

# Tools that should only be called once per ticker (prevent redundant API calls)
_SINGLE_CALL_TOOLS = {t.name for t in _ALL_TOOLS}


_REQUIRED_LINE_ITEMS = [
    "capital_expenditure", "depreciation_and_amortization", "net_income",
    "outstanding_shares", "total_assets", "total_liabilities", "shareholders_equity",
    "dividends_and_other_cash_distributions", "issuance_or_purchase_of_equity_shares",
    "gross_profit", "revenue", "free_cash_flow", "current_assets", "current_liabilities",
]

_REQUIRED_METRICS = [
    "return_on_invested_capital", "gross_margin", "operating_margin", "debt_to_equity",
    "return_on_equity", "current_ratio", "interest_coverage", "revenue_growth",
    "earnings_growth", "book_value_growth", "payout_ratio", "free_cash_flow_per_share",
    "earnings_per_share",
]


async def run_research_agent(
        tickers: List[str],
        backtesting_date: str = None,
) -> ResearchAgentOutput:
    """
    Runs the research agent to gather and structure financial data for a list of tickers.

    For each ticker the agent runs a tool-calling loop (financial data + news +
    segmented revenues + insider trades + analyst estimates), then uses a
    structured-output LLM call to compile all data into a Result model.

    All data comes directly from FinancialDatasets.ai — no third-party MCP
    server or Brave Search required.

    Args:
        tickers: List of ticker symbols to research.
        backtesting_date: Optional date string (YYYY-MM-DD). When provided, all
                          tool calls use this as the end_date.

    Returns:
        A ResearchAgentOutput with a Result per ticker.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(Result)
    agent_output = ResearchAgentOutput(requested_tickers=tickers)

    await _process_tickers(tickers, backtesting_date, llm, structured_llm, agent_output)
    return agent_output


async def _process_tickers(tickers, backtesting_date, llm, structured_llm, agent_output):
    """
    Processes each ticker sequentially: runs the data-gathering tool loop, then
    compiles a structured Result via a final structured-output LLM call.

    Args:
        tickers: Ticker symbols to process.
        backtesting_date: Optional end-date string for all tool calls.
        llm: Base LLM instance for the agentic loop.
        structured_llm: LLM bound to the Result output schema.
        agent_output: Mutable ResearchAgentOutput to append results/errors into.
    """
    for ticker in tickers:
        _console.print(f"\n[bold cyan]Researching {ticker}...[/bold cyan]")

        tools = _ALL_TOOLS
        llm_with_tools = llm.bind_tools(tools)
        tool_map = {t.name: t for t in tools}

        date_instruction = f"Pass `end_date='{backtesting_date}'` to all tools." if backtesting_date else ""

        messages = [
            SystemMessage(content=f"""You are a Data Research Agent feeding a Warren Buffett-style investment pipeline. \
Your sole job is raw data collection for '{ticker}' — do NOT interpret or score the data. \
Downstream agents will perform all analysis.

CRITICAL DATA REQUIREMENTS (the Warren Buffett Agent needs these to score all 8 dimensions):
- 8 YEARS of historical line items (net_income, revenue, gross_profit, shareholders_equity,
  outstanding_shares, issuance_or_purchase_of_equity_shares) — use limit=8 in get_financial_line_items.
- 8 YEARS of financial metrics (return_on_equity, operating_margin) — use limit=8 in get_metrics.
- Current price from get_stock_prices (most recent close).
- News headlines from get_company_news for qualitative context.
- Segment breakdown from get_segmented_revenues (most recent period).
- Insider trades from get_insider_trades (limit=20, recent transactions only).
- Analyst estimates from get_analyst_estimates (most forward annual period).

TOOLS — call ALL of them exactly once:
- `get_stock_prices`          — current price (default 7-day window is fine)
- `get_financial_line_items`  — MUST include: {json.dumps(_REQUIRED_LINE_ITEMS)} with limit=8
- `get_metrics`               — MUST include: {json.dumps(_REQUIRED_METRICS)} with limit=8
- `get_financials`            — income statement, balance sheet, cash flow
- `get_company_news`          — recent headlines (limit=5)
- `get_segmented_revenues`    — business-segment / geographic revenue (limit=4)
- `get_insider_trades`        — Form 4 buy/sell filings (limit=20)
- `get_analyst_estimates`     — consensus revenue & EPS (limit=4)

{date_instruction}
Call each tool at most once. Stop once all 8 tools have been called.
"""),
            HumanMessage(content=f"Start research for {ticker}."),
        ]

        tools_called = set()

        while True:
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                args = tool_call["args"]

                if tool_name in _SINGLE_CALL_TOOLS and tool_name in tools_called:
                    _console.print(f"  [dim]--> Skipping redundant call: {tool_name}[/dim]")
                    tool_result = {"error": f"'{tool_name}' already called. Do not call again."}
                else:
                    _console.print(f"  [dim]-->[/dim] [cyan]{tool_name}[/cyan]  [dim]args={json.dumps(args)}[/dim]")
                    try:
                        tool_result = await tool_map[tool_name].ainvoke(args) if tool_name in tool_map \
                            else {"error": f"Unknown tool: {tool_name}"}
                        if isinstance(tool_result, dict) and "error" not in tool_result:
                            tools_called.add(tool_name)
                    except Exception as e:
                        tool_result = {"error": str(e)}

                messages.append(ToolMessage(content=json.dumps(tool_result), tool_call_id=tool_call["id"]))

        # --- Structured compilation ---
        try:
            system_message = SystemMessage(content="""You are a Data Structuring Expert. Parse the conversation history
and populate the `Result` model precisely.

STANDARD FIELDS:
- Extract all financial ratios, line items, and price from tool outputs.
- Use null for any missing field; never use NaN or Infinity.
- Set `price` to the latest closing price from get_stock_prices.

HISTORICAL ARRAYS (most-recent first, up to 8 entries):
- historical_net_income            ← net_income from each get_financial_line_items period
- historical_revenue               ← revenue from each period
- historical_gross_profit          ← gross_profit from each period
- historical_return_on_equity      ← return_on_equity from financial_metrics periods
- historical_operating_margin      ← operating_margin from financial_metrics periods
- historical_shareholders_equity   ← shareholders_equity from each period
- historical_outstanding_shares    ← outstanding_shares from each period
- historical_issuance_or_purchase_of_equity_shares ← from each period

NEWS:
- recent_news: format as a readable multi-line string from get_company_news results,
  e.g. "[2025-01-15] Headline (Source)\\n[2025-01-10] Headline (Source)"
  Set tool_status.get_company_news = "ok" if news was successfully fetched.

SEGMENTED REVENUES:
- segmented_revenue: extract the most-recent period's segment breakdown as a flat dict
  {segment_name: amount}, e.g. {"iPhone": 200.58e9, "Services": 85.2e9}.
  Use the `segments` sub-dict from get_segmented_revenues items.
  Set tool_status.get_segmented_revenues = "ok" if fetched.

INSIDER TRADES:
- net_insider_buying: sum of all transaction_value fields (positive = net buying).
- insider_buy_count: count of transactions where transaction_value > 0.
- insider_sell_count: count of transactions where transaction_value < 0.
  Set tool_status.get_insider_trades = "ok" if fetched.

ANALYST ESTIMATES:
- analyst_revenue_estimate: revenue from the first (most-forward) annual estimate.
- analyst_eps_estimate: earnings_per_share from the first annual estimate.
- analyst_estimate_period: fiscal_period of that estimate (e.g. "FY2026").
  Set tool_status.get_analyst_estimates = "ok" if fetched.

TOOL STATUS:
- Set get_financials / get_metrics / get_financial_line_items / get_stock_prices
  to "ok" or "error" based on whether the tool succeeded.

Output valid JSON matching the Result model.
""")
            result = structured_llm.invoke([system_message] + messages)
            agent_output.results.append(result)
            _console.print(f"  [green]✓ {ticker} structured successfully.[/green]")

        except Exception as e:
            agent_output.errors.append(Error(tool="processing_chain", message=str(e), ticker=ticker))
            _console.print(f"  [red]✗ {ticker} structuring failed: {e}[/red]")
