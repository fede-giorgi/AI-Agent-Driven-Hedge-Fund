import json
from typing import List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from llm import get_llm
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


async def run_research_agent(
        tickers: List[str],
        research_config: Dict[str, Any],
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
        research_config: Dict produced by get_research_brief() with focus_areas,
                         required_metrics, required_line_items, and search_queries.
        backtesting_date: Optional date string (YYYY-MM-DD). When provided, all
                          tool calls use this as the end_date.

    Returns:
        A ResearchAgentOutput with a Result per ticker.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(Result)
    agent_output = ResearchAgentOutput(requested_tickers=tickers)

    await _process_tickers(tickers, backtesting_date, llm, structured_llm, agent_output, research_config)
    return agent_output


async def _process_tickers(tickers, backtesting_date, llm, structured_llm, agent_output, research_config):
    """
    Processes each ticker sequentially: runs the data-gathering tool loop, then
    compiles a structured Result via a final structured-output LLM call.

    Args:
        tickers: Ticker symbols to process.
        backtesting_date: Optional end-date string for all tool calls.
        llm: Base LLM instance for the agentic loop.
        structured_llm: LLM bound to the Result output schema.
        agent_output: Mutable ResearchAgentOutput to append results/errors into.
        research_config: Research brief dict.
    """
    for ticker in tickers:
        print(f"\nResearching {ticker}...")

        tools = _ALL_TOOLS
        llm_with_tools = llm.bind_tools(tools)
        tool_map = {t.name: t for t in tools}

        date_instruction = f"Pass `end_date='{backtesting_date}'` to all tools." if backtesting_date else ""

        messages = [
            SystemMessage(content=f"""You are a Research Agent. Your goal is to gather comprehensive financial data
for '{ticker}' to populate a FinancialSummary.

RESEARCH BRIEF — focus areas:
{json.dumps(research_config.get('focus_areas', []), indent=2)}

TOOLS AVAILABLE:
- `get_financials`            — income statement, balance sheet, cash flow
- `get_metrics`               — 50+ ratios; prioritise: {json.dumps(research_config.get('required_metrics', []))}
- `get_financial_line_items`  — granular line items (call with limit=8 for 8 years of history);
                                 MUST fetch: {json.dumps(research_config.get('required_line_items', []))}
- `get_stock_prices`          — OHLCV history (fetch current price)
- `get_company_news`          — recent news headlines from FinancialDatasets.ai
- `get_segmented_revenues`    — business-segment / geographic revenue breakdown
- `get_insider_trades`        — recent Form 4 insider buy/sell filings
- `get_analyst_estimates`     — Wall Street consensus revenue & EPS estimates

INSTRUCTIONS:
1. Call `get_stock_prices` to fetch current price.
2. Call `get_financial_line_items` with limit=8 to get 8 years of multi-period history.
3. Call `get_metrics`, `get_financials` for ratios and statements.
4. Call `get_company_news` for qualitative context.
5. Call `get_segmented_revenues` to understand revenue mix.
6. Call `get_insider_trades` (limit=20) to assess insider sentiment.
7. Call `get_analyst_estimates` for forward consensus.
8. {date_instruction}
9. Each tool should be called at most once. Stop when all tools have been called.
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
                    print(f"  --> Skipping redundant call: {tool_name}")
                    tool_result = {"error": f"'{tool_name}' already called. Do not call again."}
                else:
                    print(f"  --> Calling: {tool_name}  args={json.dumps(args)}")
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
            print(f"  ✓ {ticker} structured successfully.")

        except Exception as e:
            agent_output.errors.append(Error(tool="processing_chain", message=str(e), ticker=ticker))
            print(f"  ✗ {ticker} structuring failed: {e}")
