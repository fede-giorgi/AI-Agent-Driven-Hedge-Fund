import json
from typing import Dict, List, Any
from llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage
from rich.console import Console
from rich.table import Table

def run_final_orchestrator_agent(
    initial_portfolio: Dict[str, int],
    initial_capital: float,
    warren_signals: Dict[str, Any],
    price_map: Dict[str, float],
    history: List[Dict[str, Any]]
) -> dict:
    """
    Runs the Final Orchestrator Agent to make the definitive trading decision after the iteration loop.
    """
    llm = get_llm()
    
    system_message = SystemMessage(
        content="""You are the FinalOrchestratorAgent. You have overseen a simulation loop where a Portfolio Manager, a Monitor, and a What-If Challenger have debated trading strategies for 10 iterations.
        
        Your Goal: Make the FINAL, definitive trading decision to be executed on the user's account.

        Instructions:
        1. Review the history. Look for the most robust, risk-adjusted proposal that passed the Monitor's checks.
        2. Consider the "What-If" critiques. Did they raise valid points?
        3. Decide on the final set of trades. You can choose one of the proposals from the history or synthesize a new one based on the insights.
        4. Ensure the final trades are valid (sufficient capital, no shorting).

        Output JSON ONLY:
        {
          "agent": "final_orchestrator",
          "final_decision_reasoning": "Explanation of why this strategy was chosen over others",
          "final_trades": [{"action":"buy|sell","ticker":"XXX","shares":int}],
          "expected_portfolio": {"TICKER": int},
          "expected_capital": float
        }
        """
    )

    human_message = HumanMessage(
        content=f"""
        Inputs:
        - Initial Portfolio: {json.dumps(initial_portfolio)}
        - Initial Capital: {initial_capital}
        - Warren Buffett Signals: {json.dumps(warren_signals)}
        - Price Map: {json.dumps(price_map)}
        - Iteration History (The debate): {json.dumps(history)}
        """
    )
    
    response = llm.invoke([system_message, human_message])
    try:
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "agent": "final_orchestrator",
            "final_decision_reasoning": "Error parsing LLM response",
            "final_trades": []
        }

def generate_ascii_chart(history: List[Dict[str, Any]]) -> Table:
    """
    Generates a Rich Table showing trade proposals (quantities) over iterations for both agents.
    """
    table = Table(title="Trade Proposals Over Iterations (PM vs What-If)")
    table.add_column("Iter", justify="center", style="cyan")
    table.add_column("Ticker", style="magenta")
    table.add_column("PM Proposal", justify="right", style="green")
    table.add_column("What-If Proposal", justify="right", style="blue")
    
    # Iterate through history to populate the table
    for iteration in history:
        iter_num = iteration.get("iteration", "?")
        
        # Collect trades from PM
        pm_trades = {}
        if "pm_proposal" in iteration and "proposed_trades" in iteration["pm_proposal"]:
            for trade in iteration["pm_proposal"]["proposed_trades"]:
                qty = trade["shares"] if trade["action"] == "buy" else -trade["shares"]
                pm_trades[trade["ticker"]] = qty

        # Collect trades from What-If
        wi_trades = {}
        if "what_if_critique" in iteration and "alternative_scenario" in iteration["what_if_critique"]:
            alt = iteration["what_if_critique"]["alternative_scenario"]
            if alt and "proposed_trades" in alt:
                for trade in alt["proposed_trades"]:
                    qty = trade["shares"] if trade["action"] == "buy" else -trade["shares"]
                    wi_trades[trade["ticker"]] = qty

        # Union of tickers for this iteration
        all_tickers = set(pm_trades.keys()) | set(wi_trades.keys())
        
        if not all_tickers:
            table.add_row(str(iter_num), "-", "-", "-")
            continue

        for i, ticker in enumerate(sorted(all_tickers)):
            pm_qty = f"{pm_trades.get(ticker, 0):+d}" if ticker in pm_trades else "-"
            wi_qty = f"{wi_trades.get(ticker, 0):+d}" if ticker in wi_trades else "-"
            
            # Only show iteration number on the first row of the group
            row_iter = str(iter_num) if i == 0 else ""
            table.add_row(row_iter, ticker, pm_qty, wi_qty)
        
        table.add_section()

    return table