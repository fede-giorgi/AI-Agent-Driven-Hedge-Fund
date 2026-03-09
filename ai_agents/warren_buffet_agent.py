"""
This script defines an investment agent that analyzes stocks according to Warren Buffett's value investing principles.
"""
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

import math
import json

from classes.financial_summary import FinancialSummary, WarrenBuffettSignal
from rich.console import Console
from llm import get_llm

_console = Console()

from tools.analyze_book_value_growth import analyze_book_value_growth
from tools.analyze_consistency import analyze_consistency
from tools.analyze_fundamentals import analyze_fundamentals
from tools.analyze_management_quality import analyze_management_quality
from tools.analyze_moat import analyze_moat
from tools.analyze_pricing_power import analyze_pricing_power
from tools.calculate_intrinsic_value import calculate_intrinsic_value

def warren_buffett_agent(summary: FinancialSummary) -> dict:
    """
    Runs the Warren Buffett analysis agent for a single ticker.

    The agent runs a tool-calling loop using seven domain analysis tools
    (fundamentals, consistency, moat, management quality, book value growth,
    intrinsic value DCF, pricing power).  After the loop completes, a
    structured-output LLM call converts the conversation into a
    WarrenBuffettSignal with signal (bullish/bearish/neutral), confidence (0-100),
    and reasoning.

    Args:
        summary: A fully-populated FinancialSummary for the ticker to analyse.

    Returns:
        dict mapping ``{ticker: WarrenBuffettSignal.model_dump()}``.
    """
    _console.print(f"[bold yellow]Analyzing {summary.ticker} with Warren Buffett agent...[/bold yellow]")
    
    llm = get_llm()
    
    # 1. Define tools that are bound to the specific FinancialSummary of this stock.
    # This allows the LLM to "call" the analysis functions without needing to pass the complex object.
    
    @tool
    def check_fundamentals():
        """Analyzes key financial health metrics like ROE, debt, margins, and liquidity."""
        return analyze_fundamentals.func(summary=summary)

    @tool
    def check_consistency():
        """Checks for a track record of consistent and growing earnings."""
        return analyze_consistency.func(summary=summary)

    @tool
    def check_moat():
        """Evaluates the company's durable competitive advantage (moat)."""
        return analyze_moat.func(summary=summary)

    @tool
    def check_management():
        """Assesses management's shareholder-friendliness (buybacks, dividends)."""
        return analyze_management_quality.func(summary=summary)

    @tool
    def check_book_value_growth():
        """Analyzes the growth of book value per share over time."""
        return analyze_book_value_growth.func(summary=summary)

    @tool
    def check_intrinsic_value():
        """Estimates the company's intrinsic value using a DCF model."""
        return calculate_intrinsic_value.func(summary=summary)

    @tool
    def check_pricing_power():
        """Assesses the company's ability to raise prices (gross margins)."""
        return analyze_pricing_power.func(summary=summary)

    @tool
    def check_qualitative_factors():
        """Reviews recent news headlines, insider buying/selling activity, and analyst consensus estimates for qualitative signals."""
        headlines = []
        if summary.recent_news:
            headlines = [line.strip() for line in summary.recent_news.split("\n") if line.strip()][:6]

        net_buying = summary.net_insider_buying or 0
        insider_sentiment = "NET BUYER" if net_buying > 0 else "NET SELLER" if net_buying < 0 else "NEUTRAL"

        return {
            "recent_news_headlines": headlines,
            "insider_activity": {
                "net_buying_usd": net_buying,
                "buy_transactions": summary.insider_buy_count or 0,
                "sell_transactions": summary.insider_sell_count or 0,
                "sentiment": insider_sentiment,
            },
            "analyst_consensus": {
                "period": summary.analyst_estimate_period,
                "revenue_estimate": summary.analyst_revenue_estimate,
                "eps_estimate": summary.analyst_eps_estimate,
            },
            "details": (
                f"Insider: {insider_sentiment} (${abs(net_buying):,.0f} net). "
                f"Analyst EPS est: {summary.analyst_eps_estimate} for {summary.analyst_estimate_period}. "
                f"{len(headlines)} recent headlines to review."
            ),
        }

    tools = [
        check_fundamentals,
        check_consistency,
        check_moat,
        check_management,
        check_book_value_growth,
        check_intrinsic_value,
        check_pricing_power,
        check_qualitative_factors,
    ]

    llm_with_tools = llm.bind_tools(tools)

    # 2. Agent Loop
    messages = [
        SystemMessage(content=f"""You are a Warren Buffett-style investment analyst. Evaluate {summary.ticker} using ALL eight domain tools before forming your signal. Do not skip any tool — each captures a distinct dimension of business quality.

TOOLS — call ALL of them (order does not matter):
1. check_fundamentals        — ROE, ROIC, debt levels, operating margin, liquidity (max score: 9)
2. check_consistency         — multi-year earnings CAGR + monotonic growth (max score: 4)
3. check_moat                — historical ROE consistency, margin stability, ROIC (max score: 4)
4. check_management          — multi-year buyback track record, dividend history (max score: 2)
5. check_book_value_growth   — BVPS CAGR + period-by-period consistency (max score: 5)
6. check_intrinsic_value     — 3-stage DCF owner-earnings; yields margin_of_safety vs current price
7. check_pricing_power       — gross margin trend + absolute level (max score: 5)
8. check_qualitative_factors — recent news headlines, insider buy/sell activity, analyst EPS/revenue consensus

SIGNAL CALIBRATION:
- BULLISH 70-100 confidence: Strong moat + consistent earnings + margin_of_safety > 25%. "Wonderful company at fair price."
- BULLISH 40-69 confidence: Good fundamentals but margin_of_safety modest (<25%) or one weak dimension.
- NEUTRAL: Mixed — e.g. strong moat but stock is fairly/fully valued, OR improving trajectory but short history.
- BEARISH: Deteriorating fundamentals, negative multi-year earnings CAGR, trading well above intrinsic value, OR concerning insider selling with negative news.

PORTFOLIO MANAGER CONTEXT: Your confidence score directly controls position sizing. High confidence BULLISH → larger allocation. Low confidence or NEUTRAL → minimal/no new position. BEARISH → reduce exposure. Calibrate honestly — overconfident signals destroy the portfolio.

Buffett's rule: "It is far better to buy a wonderful company at a fair price than a fair company at a wonderful price."\
"""),
        HumanMessage(content=f"Analyse {summary.ticker} and provide an investment signal. Call all eight tools first."),
    ]

    while True:
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            break
            
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            # Map tool names to functions
            tool_map = {t.name: t for t in tools}
            
            if tool_name in tool_map:
                try:
                    # Execute the tool (no args needed as they are bound to summary)
                    result = tool_map[tool_name].invoke({})
                except Exception as e:
                    result = f"Error executing {tool_name}: {e}"
            else:
                result = f"Unknown tool: {tool_name}"
            
            messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tool_call["id"]))

    # 3. Final Decision
    # We use a structured output LLM to parse the final conversation into the signal format
    structured_llm = llm.with_structured_output(WarrenBuffettSignal)
    
    final_instruction = HumanMessage(content="""Based on the analysis you performed above, determine a bullish, bearish, or neutral signal.
    - Assign a confidence score (0-100).
    - Provide a brief, decisive reasoning.""")
    
    final_signal = structured_llm.invoke(messages + [final_instruction])
    
    return {summary.ticker: final_signal.model_dump()}
