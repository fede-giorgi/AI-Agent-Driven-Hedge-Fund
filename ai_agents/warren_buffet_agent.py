"""Warren Buffett-style investment agent using all domain-specific analysis tools."""

import json
from typing import Any

from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from classes.financial_summary import FinancialSummary, WarrenBuffettSignal
from llm import get_llm
from tools.analyze_book_value_growth import analyze_book_value_growth
from tools.analyze_consistency import analyze_consistency
from tools.analyze_fundamentals import analyze_fundamentals
from tools.analyze_management_quality import analyze_management_quality
from tools.analyze_moat import analyze_moat
from tools.analyze_pricing_power import analyze_pricing_power
from shared_console import console as _console

# ── Why tools are no-arg closures ────────────────────────────────────────────
# Every analysis tool in tools/ takes `summary: FinancialSummary` as its first
# parameter.  If we passed that parameter through LangChain's tool interface,
# it would appear in the JSON schema sent to the LLM, and the model would try
# to supply a string value for it — which breaks.  The solution is to wrap each
# tool in a no-arg closure that closes over the ticker-specific `summary`
# object.  The actual logic stays in the @tool-decorated files under tools/.
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a Warren Buffett-style investment analyst. Evaluate {ticker} and produce \
a bullish, bearish, or neutral signal with a confidence score (0-100).

AVAILABLE TOOLS (call the ones relevant to your thesis):
1. check_fundamentals       — ROE, ROIC, debt, margins, liquidity (max score 9)
2. check_consistency        — multi-year earnings CAGR + monotonic growth (max 4)
3. check_moat               — historical ROE consistency, margin stability (max 4)
4. check_management_quality — buyback track record, dividend history (max 2)
5. check_book_value_growth  — BVPS CAGR + period consistency (max 5)
6. check_pricing_power      — gross margin trend + absolute level (max 5)
7. check_intrinsic_value    — 3-stage DCF owner-earnings; yields margin_of_safety vs current price
8. get_qualitative_signals  — news headlines, insider buy/sell activity, analyst consensus

APPROACH — be genuinely analytical:
- Start with the tools most relevant to your initial hypothesis about this company.
- You do NOT need to call all tools. Skip tools whose output would not change your conclusion.
- Use check_intrinsic_value to anchor the valuation; only a margin_of_safety > 25% warrants \
high-confidence BULLISH.
- Use check_consistency and check_moat together to assess trajectory and durability.
- Use get_qualitative_signals if insider activity or news may be material.

SIGNAL CALIBRATION:
- BULLISH (70-100): Strong moat + consistent earnings + margin_of_safety > 25%.
- BULLISH (40-69): Good fundamentals but margin_of_safety modest or one weak dimension.
- NEUTRAL: Mixed — strong moat but fully valued, or improving but short track record.
- BEARISH: Deteriorating fundamentals, negative earnings CAGR, trading well above \
intrinsic value, or concerning insider selling with negative news.

{debug_note}
Buffett: "It is far better to buy a wonderful company at a fair price than \
a fair company at a wonderful price."
"""

_DEBUG_NOTE = (
    "NOTE: Debug/testing run. Assign BULLISH ≥ 55 confidence if the company has "
    "any positive qualities — the goal is to exercise the full trading pipeline."
)


async def warren_buffett_agent(
    summary: FinancialSummary,
    debug_mode: bool = False,
) -> dict[str, Any]:
    """
    Analyzes a single ticker using all Warren Buffett domain tools.

    Runs an agentic tool-calling loop where the LLM decides which of the nine
    domain tools to invoke.  A structured-output call converts the conversation
    into a WarrenBuffettSignal.

    Args:
        summary: Fully-populated FinancialSummary for the ticker to analyse.
        debug_mode: When True, nudges the LLM toward a BULLISH signal so the
                    full trading pipeline can be exercised in testing.

    Returns:
        dict mapping ``{ticker: WarrenBuffettSignal.model_dump()}``.
    """
    _console.print(f"[bold yellow]Analyzing {summary.ticker} with Warren Buffett agent...[/bold yellow]")

    llm = get_llm()

    # ── No-arg closures — each calls the @tool function from tools/ ───────────

    @tool
    def check_fundamentals() -> dict:
        """Scores ROE, ROIC, debt levels, operating margin, and current ratio. Max score: 9."""
        return analyze_fundamentals.func(summary=summary)

    @tool
    def check_consistency() -> dict:
        """Scores multi-year earnings CAGR and period-by-period monotonic growth. Max score: 4."""
        return analyze_consistency.func(summary=summary)

    @tool
    def check_moat() -> dict:
        """Scores durable competitive advantage via historical ROE consistency and margin stability. Max score: 4."""
        return analyze_moat.func(summary=summary)

    @tool
    def check_management_quality() -> dict:
        """Scores management's shareholder-friendliness: buyback track record and dividends. Max score: 2."""
        return analyze_management_quality.func(summary=summary)

    @tool
    def check_book_value_growth() -> dict:
        """Scores BVPS CAGR and period-by-period consistency. Max score: 5."""
        return analyze_book_value_growth.func(summary=summary)

    @tool
    def check_pricing_power() -> dict:
        """Scores gross margin trend (improving/stable/declining) and absolute level. Max score: 5."""
        return analyze_pricing_power.func(summary=summary)

    @tool
    def check_intrinsic_value() -> dict:
        """Estimates intrinsic value via a 3-stage DCF on owner earnings. Returns intrinsic_value_per_share and margin_of_safety vs current price."""
        from tools.calculate_intrinsic_value import calculate_intrinsic_value
        return calculate_intrinsic_value.func(summary=summary)

    @tool
    def get_qualitative_signals() -> dict:
        """Returns recent news headlines, insider buy/sell activity, and analyst consensus revenue and EPS estimates."""
        headlines = []
        if summary.recent_news:
            headlines = [l.strip() for l in summary.recent_news.split("\n") if l.strip()][:6]
        net = summary.net_insider_buying or 0
        sentiment = "NET BUYER" if net > 0 else "NET SELLER" if net < 0 else "NEUTRAL"
        return {
            "recent_news_headlines": headlines,
            "insider_activity": {
                "net_buying_usd": net,
                "buy_transactions": summary.insider_buy_count or 0,
                "sell_transactions": summary.insider_sell_count or 0,
                "sentiment": sentiment,
            },
            "analyst_consensus": {
                "period": summary.analyst_estimate_period,
                "revenue_estimate": summary.analyst_revenue_estimate,
                "eps_estimate": summary.analyst_eps_estimate,
            },
        }

    tools = [
        check_fundamentals,
        check_consistency,
        check_moat,
        check_management_quality,
        check_book_value_growth,
        check_pricing_power,
        check_intrinsic_value,
        get_qualitative_signals,
    ]
    llm_with_tools = llm.bind_tools(tools)

    # ── Agent loop ────────────────────────────────────────────────────────────
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT.format(
            ticker=summary.ticker,
            debug_note=_DEBUG_NOTE if debug_mode else "",
        )),
        HumanMessage(content=f"Analyse {summary.ticker}. Use tools as needed, then provide your investment signal."),
    ]

    tool_map = {t.name: t for t in tools}
    while True:
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            try:
                result = tool_map[tool_name].invoke({}) if tool_name in tool_map \
                    else f"Unknown tool: {tool_name}"
            except Exception as e:
                result = f"Error executing {tool_name}: {e}"
            messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tool_call["id"]))

    # ── Structured output ─────────────────────────────────────────────────────
    structured_llm = llm.with_structured_output(WarrenBuffettSignal)
    final_instruction = HumanMessage(content=(
        "Based on your analysis above, provide a final investment signal.\n"
        "- signal: bullish, bearish, or neutral\n"
        "- confidence: 0-100 (calibrate honestly)\n"
        "- reasoning: brief, decisive, grounded in the data you examined"
    ))

    final_signal = await structured_llm.ainvoke(messages + [final_instruction])
    _console.print(f"[green]✓ {summary.ticker}: {final_signal.signal.upper()} {final_signal.confidence}%[/green]")
    return {summary.ticker: final_signal.model_dump()}
