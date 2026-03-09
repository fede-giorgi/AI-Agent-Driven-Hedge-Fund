import json
from typing import Dict, List, Any, Union
from llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage


def _text(content) -> str:
    """Return plain text from an LLM response content (handles str or list of blocks)."""
    if isinstance(content, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return content if isinstance(content, str) else str(content)

def run_what_if_agent(
    current_portfolio: Dict[str, int],
    available_capital: float,
    proposed_trades: List[Dict[str, Union[str, int, float]]],
    price_map: Dict[str, float],
    current_iteration: int,
    total_iterations: int,
    warren_signals: Dict[str, Any] = None,
    history: List[Dict[str, Any]] = None
    ) -> dict:
    """
    Runs the What-If Agent to simulate the portfolio after applying trades.
    """
    llm = get_llm()
    
    system_message = SystemMessage(
        content=f"""You are WhatIfAgent — the risk stress-tester of a Warren Buffett-style hedge fund.
You challenge the Portfolio Manager's proposal by constructing a specific, executable counter-scenario.

CONTEXT: Iteration {current_iteration} of {total_iterations}.
The Portfolio Manager will see your critique and may adopt it next iteration.
The Final Orchestrator reviews all {total_iterations} iterations and picks the best overall strategy.

YOUR MANDATE:
1. Read the proposed trades and the Warren Buffett signals carefully.
2. Identify the SINGLE biggest risk or inefficiency in the proposal.
3. Construct a concrete, executable alternative. Examples:
   - "Buy 40% fewer NVDA shares to retain a $15k cash buffer"
   - "Sell MSFT entirely — the BEARISH signal outweighs the small position size"
   - "Hold all — the Monitor flagged a budget violation; no trades is the safe path"
4. Your alternative MUST respect: no shorting, available_capital constraint.

WHEN TO PUSH BACK vs. ACCEPT:
- Push back: over-concentration, cash too thin, buying a BEARISH-signal stock.
- Accept ("hold everything"): the proposal is sound and Monitor already approved it.
- Do NOT be contrarian for its own sake.

Output JSON ONLY:
{{
  "agent": "what_if",
  "critique": "Specific risk or flaw in the proposed trades",
  "alternative_scenario": {{
    "description": "Concrete description of the alternative",
    "proposed_trades": [{{"action": "buy|sell", "ticker": "XXX", "shares": int}}]
  }},
  "reasoning": "Why this alternative better manages risk or capital efficiency"
}}
"""
    )
    
    human_message = HumanMessage(
        content=f"""
        Inputs:
        - Current Portfolio: {json.dumps(current_portfolio)}
        - Available Capital: {available_capital}
        - Proposed Trades: {json.dumps(proposed_trades)}
        - Price Map: {json.dumps(price_map)}
        - Warren Signals: {json.dumps(warren_signals) if warren_signals else "None"}
        """
    )
    
    response = llm.invoke([system_message, human_message])
    try:
        content = _text(response.content).strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "agent": "what_if",
            "critique": "Error parsing response",
            "alternative_scenario": {},
            "reasoning": str(response.content)
        }
