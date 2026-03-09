import json
from typing import Dict, List, Union, Tuple, Any


def _text(content) -> str:
    """Return plain text from an LLM response content (handles str or list of blocks)."""
    if isinstance(content, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return content if isinstance(content, str) else str(content)
from classes.financial_summary import FinancialSummary
from langchain_core.messages import SystemMessage, HumanMessage
from llm import get_llm

def run_monitor_agent(
    proposed_trades: List[Dict[str, Union[str, int, float]]],
    current_portfolio: Dict[str, int],
    available_capital: float,
    price_map: Dict[str, float],
    current_iteration: int,
    total_iterations: int,
    history: List[Dict[str, Any]] = None # Added history for standardization
    ) -> dict:
    """
    Runs the Monitor Agent to validate proposed trades against hard constraints.

    Checks performed (all must pass):
    - Schema: each trade has action in {buy, sell}, a valid ticker string, and shares > 0.
    - Known ticker + positive price: ticker must exist in price_map.
    - Holdings: sell quantity may not exceed current_portfolio[ticker].
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
        dict with keys ``agent``, ``is_valid`` (bool), ``summary`` (cash flows),
        ``violations`` (list), and ``notes`` (list).
    """
    llm = get_llm()

    system_instruction = SystemMessage(f"""You are MonitorAgent — the compliance officer of a Warren Buffett-style hedge fund.
Your ONLY job is to validate proposed trades against hard mathematical rules. Do NOT express opinions on the strategy.

CONTEXT: Iteration {current_iteration} of {total_iterations}.

VALIDATION CHECKLIST (ALL must pass for is_valid=true):
1. Schema: each trade has action ∈ {{buy, sell}}, a non-empty ticker string, and shares > 0 (integer).
2. Known ticker + positive price: ticker must exist in price_map with price > 0.
3. No shorting: for every sell, shares ≤ current_portfolio.get(ticker, 0).
4. Budget (evaluate sells first, then buys):
     sell_proceeds = Σ(sell_shares × price)
     buy_cost      = Σ(buy_shares  × price)
     required_cash = buy_cost − sell_proceeds
     Must satisfy: required_cash ≤ available_capital
5. No NaN / Infinity in any numeric field.

REPORTING RULES:
- If is_valid=false: list EVERY violation with type, ticker, and a precise detail (e.g. 'needs $12,450 but only $8,000 available').
- Do NOT suggest fixes. The Portfolio Manager will fix violations in the next iteration.
- approved_trades = full list of trades if valid, empty list if invalid.

Output JSON ONLY:
{{
  "agent": "monitor",
  "is_valid": true|false,
  "summary": {{
    "buy_cost": number,
    "sell_proceeds": number,
    "required_cash": number,
    "available_capital": number
  }},
  "violations": [{{"type": "...", "ticker": "...", "detail": "..."}}],
  "approved_trades": [...],
  "notes": [...]
}}
""")

    user_content = HumanMessage(f"""Please validate the following input configuration:

    Inputs:
    - Proposed Trades: {json.dumps(proposed_trades)}
    - Current Portfolio: {json.dumps(current_portfolio)}
    - Available Capital: {available_capital}
    - Price Map: {json.dumps(price_map)}
    """)
    
    response = llm.invoke([system_instruction, user_content])
    try:
        content = _text(response.content).strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "agent": "monitor",
            "is_valid": False,
            "summary": {},
            "violations": [{"type": "ParseError", "ticker": "ALL", "detail": "Failed to parse LLM response"}],
            "notes": [str(response.content)]
        }
