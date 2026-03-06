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
        content=f"""You are WhatIfAgent. Your goal is to CHALLENGE the proposed trades from the Portfolio Manager. You act as a "Devil's Advocate" or Scenario Planner.
        
        CONTEXT: Iteration {current_iteration} of {total_iterations}.
        
        Your Task:
        1. Analyze the proposed trades.
        2. Generate a "What-If" scenario that challenges the proposal. Examples:
           - "What if we bought 50% less of X to keep more cash?"
           - "What if we sold Y instead of Z?"
           - "What if we did nothing?"
        3. Provide a concrete alternative trade suggestion if you think it's better.

        Output JSON ONLY:
        {{
          "agent": "what_if",
          "critique": "Brief critique of the proposed trades",
          "alternative_scenario": {{
             "description": "Description of the alternative",
             "proposed_trades": [{{"action":"...","ticker":"...","shares":...}}]
          }},
          "reasoning": "Why this alternative might be safer or better"
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
