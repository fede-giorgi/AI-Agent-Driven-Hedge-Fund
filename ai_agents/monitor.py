"""Monitor Agent — validates proposed trades against hard mathematical constraints."""

import json

from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from llm import get_llm


class _MonitorOutput(BaseModel):
    """Structured output produced by the Monitor Agent."""

    is_valid: bool = Field(description="True only if every trade passes all constraint checks.")
    summary: dict = Field(description="Cash-flow summary: buy_cost, sell_proceeds, required_cash, available_capital.")
    violations: list[dict] = Field(default_factory=list, description="List of violation dicts with type, ticker, detail.")
    approved_trades: list[dict] = Field(default_factory=list, description="Full trade list if valid; empty list if invalid.")
    notes: list[str] = Field(default_factory=list, description="Optional explanatory notes.")


def run_monitor_agent(
    proposed_trades: list[dict[str, str | int | float]],
    current_portfolio: dict[str, int],
    available_capital: float,
    price_map: dict[str, float],
    current_iteration: int,
    total_iterations: int,
    history: list[dict] | None = None,
) -> dict:
    """
    Runs the Monitor Agent to validate proposed trades against hard constraints.

    The agent calls a deterministic validate_trades tool that computes exact
    cash flows and violations in Python, then interprets the results and
    formats the final response.

    Checks performed:
    - Schema: each trade has action ∈ {buy, sell}, a valid ticker, and shares > 0.
    - Known ticker + positive price: ticker must exist in price_map.
    - No shorting: sell quantity may not exceed current_portfolio[ticker].
    - Budget: buy_cost − sell_proceeds must not exceed available_capital.

    Args:
        proposed_trades: List of trade dicts with action, ticker, and shares keys.
        current_portfolio: Dict of ticker → current share count.
        available_capital: Cash available for buys.
        price_map: Dict of ticker → current price.
        current_iteration: Current loop iteration number (1-based).
        total_iterations: Total number of loop iterations.
        history: Previous iteration dicts (unused by monitor, kept for API consistency).

    Returns:
        dict with keys ``agent``, ``is_valid``, ``summary``, ``violations``, and ``notes``.
    """
    llm = get_llm()

    @tool
    def validate_trades(trades: list[dict], portfolio: dict, capital: float, prices: dict) -> dict:
        """
        Deterministically validates trades and computes exact cash flows.

        Runs all constraint checks in Python and returns a structured result
        with violations, cash-flow summary, and whether the set is valid.

        Args:
            trades: Proposed trade list (action, ticker, shares).
            portfolio: Current holdings (ticker → shares).
            capital: Available cash.
            prices: Price map (ticker → price).
        """
        violations = []
        buy_cost = 0.0
        sell_proceeds = 0.0

        for trade in trades:
            action = trade.get("action", "")
            ticker = trade.get("ticker", "")
            shares = trade.get("shares", 0)

            # Schema check
            if action not in ("buy", "sell"):
                violations.append({"type": "BadAction", "ticker": ticker,
                                    "detail": f"action must be 'buy' or 'sell', got '{action}'"})
                continue
            if not ticker or not isinstance(ticker, str):
                violations.append({"type": "BadTicker", "ticker": str(ticker),
                                    "detail": "ticker must be a non-empty string"})
                continue
            if not isinstance(shares, (int, float)) or shares <= 0:
                violations.append({"type": "BadShares", "ticker": ticker,
                                    "detail": f"shares must be > 0, got {shares}"})
                continue

            # Known ticker with positive price
            price = prices.get(ticker)
            if price is None or price <= 0:
                violations.append({"type": "UnknownTicker", "ticker": ticker,
                                    "detail": f"ticker not in price_map or price ≤ 0"})
                continue

            trade_value = shares * price

            if action == "sell":
                held = portfolio.get(ticker, 0)
                if shares > held:
                    violations.append({"type": "NoShort", "ticker": ticker,
                                        "detail": f"trying to sell {shares} but only holds {held}"})
                sell_proceeds += trade_value
            else:
                buy_cost += trade_value

        required_cash = buy_cost - sell_proceeds
        if required_cash > capital:
            violations.append({"type": "InsufficientFunds", "ticker": "ALL",
                                "detail": f"needs ${required_cash:,.0f} but only ${capital:,.0f} available"})

        return {
            "is_valid": len(violations) == 0,
            "summary": {
                "buy_cost": buy_cost,
                "sell_proceeds": sell_proceeds,
                "required_cash": required_cash,
                "available_capital": capital,
            },
            "violations": violations,
        }

    tools = [validate_trades]
    llm_with_tools = llm.bind_tools(tools)
    structured_llm = llm.with_structured_output(_MonitorOutput)

    messages = [
        SystemMessage(content=f"""You are MonitorAgent — the compliance officer of a Warren Buffett-style hedge fund.
Context: Iteration {current_iteration} of {total_iterations}.

Your job:
1. Call validate_trades with the proposed trades, current portfolio, available capital, and price map.
2. The tool returns exact cash flows and all constraint violations computed in Python.
3. Interpret the results and report them faithfully — do NOT change the numbers.
4. Do NOT suggest fixes; the Portfolio Manager handles corrections in the next iteration.
"""),
        HumanMessage(content=f"""Validate these proposed trades:

Proposed Trades: {json.dumps(proposed_trades)}
Current Portfolio: {json.dumps(current_portfolio)}
Available Capital: {available_capital}
Price Map: {json.dumps(price_map)}

Call validate_trades now."""),
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

    final_instruction = HumanMessage(content="""Based on the validate_trades result above, produce the final monitor report.
- is_valid: true only if there are zero violations
- summary: the exact cash-flow numbers from the tool
- violations: the exact violation list from the tool (empty if none)
- approved_trades: the full proposed_trades list if is_valid, else empty list
- notes: any additional observations""")

    output = structured_llm.invoke(messages + [final_instruction])
    return {
        "agent": "monitor",
        "is_valid": output.is_valid,
        "summary": output.summary,
        "violations": output.violations,
        "approved_trades": output.approved_trades,
        "notes": output.notes,
    }
