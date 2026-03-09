import json
from typing import Dict, List, Any
from llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage
from rich.console import Console
from rich.table import Table
from rich import box


def _text(content) -> str:
    """Return plain text from an LLM response content (handles str or list of blocks)."""
    if isinstance(content, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return content if isinstance(content, str) else str(content)

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
        content="""You are FinalOrchestratorAgent — the Chief Investment Officer of a Warren Buffett-style hedge fund.
You have just received a compressed summary of the full multi-iteration debate between the Portfolio Manager, Monitor, and What-If Agent.

YOUR GOAL: Make the single FINAL, definitive trading decision that will be executed on the user's account.

DECISION FRAMEWORK:
1. CONSENSUS CHECK: Which trade proposals appeared consistently across multiple iterations (stable conviction)?
   Prefer stable proposals over one-off suggestions that changed every iteration.
2. MONITOR COMPLIANCE: Only consider proposals that passed (or would pass) Monitor validation.
   If all iterations had violations, synthesise a conservative valid plan.
3. WHAT-IF SYNTHESIS: Did the What-If Agent raise a point that was never addressed?
   If yes, incorporate it. If the PM consistently refuted it, side with the PM.
4. CAPITAL EFFICIENCY: Ensure the final trades deploy capital productively (avoid leaving >20% uninvested
   unless risk profile is Low 1-3).
5. ORIGINAL SIGNALS: Always cross-reference with Warren Buffett signals. BEARISH → do not buy. BULLISH + high confidence → reward with allocation.

SYNTHESIS vs. SELECTION:
- PREFER selecting the best single iteration's trades if one clearly dominated.
- SYNTHESISE a new plan only if no single iteration was satisfactory.

HARD CONSTRAINTS (same as Monitor):
- No shorting: sell_shares ≤ current holdings.
- Budget: Σ(buy × price) − Σ(sell × price) ≤ available_capital.
- Only trade tickers with a valid price in price_map.

Output JSON ONLY:
{
  "agent": "final_orchestrator",
  "final_decision_reasoning": "Which iteration/proposal was chosen and why, addressing PM vs What-If debate",
  "final_trades": [{"action": "buy|sell", "ticker": "XXX", "shares": int}],
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
        content = _text(response.content).strip()
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
    table = Table(title="Trade Proposals Over Iterations (PM vs What-If)", box=box.ROUNDED)
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