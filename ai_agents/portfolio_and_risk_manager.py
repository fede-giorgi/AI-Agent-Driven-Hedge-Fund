"""Portfolio Manager Agent — translates analyst signals into executable trade orders."""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from llm import get_llm
from tools.calculate_position_size import calculate_position_size


class _PMOutput(BaseModel):
    """Structured output produced by the Portfolio Manager Agent."""

    proposed_trades: list[dict] = Field(description="List of trade dicts with action, ticker, and shares.")
    notes: list[str] = Field(default_factory=list, description="Brief reasoning per decision.")
    errors: list[str] = Field(default_factory=list)


def run_portfolio_manager_agent(
    current_portfolio: dict[str, int],
    available_capital: float,
    risk_profile: int,
    warren_signals: dict[str, Any],
    price_map: dict[str, float],
    current_iteration: int,
    total_iterations: int,
    history: list[dict] | None = None,
    force_trades: bool = False,
) -> dict:
    """
    Runs the Portfolio Manager Agent to propose trades based on signals and risk profile.

    The agent calls calculate_position_size for each ticker it wants to trade,
    which offloads all arithmetic to Python.  A structured-output LLM call
    converts the conversation into concrete trade orders.

    Args:
        current_portfolio: Dict of ticker → current share count.
        available_capital: Cash available for buys.
        risk_profile: Portfolio risk level (1-10).
        warren_signals: Dict of ticker → WarrenBuffettSignal dict.
        price_map: Dict of ticker → current price.
        current_iteration: Current loop iteration number (1-based).
        total_iterations: Total number of loop iterations.
        history: Previous iteration dicts; last entry is used as feedback.
        force_trades: When True (debug mode), the agent is instructed to propose
                      at least one trade even if signals are weak, so the full
                      pipeline can be exercised.

    Returns:
        dict with keys ``agent``, ``proposed_trades``, ``notes``, and ``errors``.
    """
    llm = get_llm()
    previous_feedback = history[-1] if history else None

    total_capital = available_capital + sum(
        current_portfolio.get(t, 0) * price_map.get(t, 0) for t in current_portfolio
    )

    tools = [calculate_position_size]
    llm_with_tools = llm.bind_tools(tools)
    structured_llm = llm.with_structured_output(_PMOutput)

    messages = [
        SystemMessage(content=f"""You are PortfolioManagerAgent — the trading desk of a Warren Buffett-style hedge fund.
Context: Iteration {current_iteration} of {total_iterations}.

Your job: translate Warren Buffett analyst signals into concrete, executable trade orders.

SIGNAL INTERPRETATION:
- BULLISH (confidence 70-100): High conviction → call calculate_position_size, use the shares_to_buy result.
- BULLISH (confidence 40-69): Moderate conviction → call calculate_position_size; respect the lower allocation.
- NEUTRAL: Do NOT add. Hold existing shares; sell only if significantly overweight.
- BEARISH: Reduce or exit the position.

TOOL:
- calculate_position_size(confidence, risk_profile, total_capital, price, current_shares)
  → returns target_shares, shares_to_buy, shares_to_sell, and a plain-English breakdown.
  Use this for every ticker where you intend to buy or sell. Do NOT compute share counts yourself.

HARD CONSTRAINTS (Monitor will reject violations):
- No shorting: sell_shares ≤ currently_held.
- Budget: Σ(buy × price) − Σ(sell × price) ≤ {available_capital:.2f} (available cash).
- Only trade tickers with price > 0 in price_map.
- Never put >30% of total capital in one ticker.

RISK PROFILE {risk_profile}/10:
- Low (1-3): Capital preservation; large cash buffer. Profile 1 = NO buys.
- Mid (4-7): Balanced; modest positions.
- High (8-10): Aggressive; concentrate in top bullish ideas.

REFINEMENT:
- If previous Monitor check was INVALID, fix that exact violation this iteration.
- Adopt What-If counter only if it reduces risk or fixes a Monitor violation.

Total capital (cash + holdings): ${total_capital:,.2f}
{"" if not force_trades else chr(10) + "DEBUG MODE: You MUST propose at least 2 buy trades to exercise the full pipeline. Treat any NEUTRAL signal with confidence ≥ 50 as BULLISH for sizing purposes."}
"""),
        HumanMessage(content=f"""Propose trades for this situation:

Current Portfolio: {json.dumps(current_portfolio)}
Available Capital: {available_capital}
Risk Profile: {risk_profile}
Warren Signals: {json.dumps(warren_signals)}
Price Map: {json.dumps(price_map)}
Previous Iteration Feedback: {json.dumps(previous_feedback) if previous_feedback else "None (first iteration)"}

For each ticker you want to trade, call calculate_position_size first."""),
    ]

    tool_map = {t.name: t for t in tools}
    while True:
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            args = tool_call.get("args", {})
            try:
                result = tool_map[tool_name].invoke(args) if tool_name in tool_map \
                    else {"error": f"Unknown tool: {tool_name}"}
            except Exception as e:
                result = {"error": str(e)}
            messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tool_call["id"]))

    final_instruction = HumanMessage(content="""Based on the position-sizing results above, output the final trade list.
- proposed_trades: list of {action, ticker, shares} dicts using the shares from calculate_position_size
- notes: one brief note per trade explaining the rationale
- errors: any issues encountered""")

    output = structured_llm.invoke(messages + [final_instruction])
    return {
        "agent": "portfolio_manager",
        "proposed_trades": output.proposed_trades,
        "notes": output.notes,
        "errors": output.errors,
    }
