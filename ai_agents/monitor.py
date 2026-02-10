import json
from typing import Dict, List, Union, Tuple
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
    Runs the Monitor Agent to validate the proposed configuration.
    """
    llm = get_llm()

    system_instruction = SystemMessage(f"""You are MonitorAgent. Your job is to validate that the proposed stock trading configuration is within bounds and executable.
    
    CONTEXT: Iteration {current_iteration} of {total_iterations}.

    Checks (must all pass):
    - Schema: each trade has action in {{buy,sell}}, ticker string, shares integer > 0.
    - Known ticker + price: ticker exists in price_map AND price > 0.
    - Holdings: for sells, shares <= current_portfolio[ticker].
    - Budget: compute expected_cash_change assuming sells first:
        sell_proceeds = Σ(sell_shares * price)
        buy_cost      = Σ(buy_shares  * price)
        required_cash = buy_cost - sell_proceeds
        Must satisfy required_cash <= available_capital.
    - No NaN/Infinity; treat missing data as invalid.

    If invalid: do NOT “fix” trades unless requested; just report violations clearly.

    Output JSON ONLY in this format:
    {{
    "agent":"monitor",
    "is_valid": true|false,
    "summary":{{
        "buy_cost": number,
        "sell_proceeds": number,
        "required_cash": number,
        "available_capital": number
    }},
    "violations":[{{"type":"...","ticker":"...","detail":"..."}}],
    "notes":[...]
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
        content = response.content.strip()
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
