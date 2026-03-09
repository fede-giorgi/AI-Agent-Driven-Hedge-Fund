import json
from typing import Dict, List, Any
from llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage


def _text(content) -> str:
    """Return plain text from an LLM response content (handles str or list of blocks)."""
    if isinstance(content, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return content if isinstance(content, str) else str(content)

def run_portfolio_manager_agent(
    current_portfolio: Dict[str, int],
    available_capital: float,
    risk_profile: int,
    warren_signals: Dict[str, Any],
    price_map: Dict[str, float],
    current_iteration: int,
    total_iterations: int,
    history: List[Dict[str, Any]] = None # Standardized to history
    ) -> dict:
    """
    Runs the Portfolio Manager Agent to propose trades based on signals and risk profile.
    """
    llm = get_llm()
    
    # Get feedback from previous iteration if available
    previous_feedback = history[-1] if history else None
    
    system_message = SystemMessage(
        content=f"""You are PortfolioManagerAgent — the trading desk of a Warren Buffett-style hedge fund.
You receive analyst signals (bullish/bearish/neutral + confidence 0-100) from the Warren Buffett Agent and translate them into concrete, executable trade orders.

CONTEXT:
- Iteration {current_iteration} of {total_iterations}.
- A Monitor Agent validates every trade (hard rules: no shorting, no over-spending).
- A What-If Agent stress-tests your proposal. Adopt its counter only if it fixes a genuine flaw.
- The Final Orchestrator reviews all {total_iterations} iterations; be consistent and well-reasoned.

SIGNAL INTERPRETATION:
- BULLISH (confidence 70-100): High conviction. Scale position size with confidence.
- BULLISH (confidence 40-69): Moderate conviction. Smaller position; leave cash headroom.
- NEUTRAL: Do NOT add. Hold existing shares; sell only if significantly overweight.
- BEARISH: Reduce or exit the position.

RISK PROFILE {risk_profile}/10 — POSITION SIZING:
- Low (1-3): Capital preservation. Profile 1 = NO buys. Cash buffer ≥50%.
- Mid (4-7): Balanced. Cash buffer 5-15%.
  Per-ticker target ≈ (confidence/100) × (risk_profile/10) × total_capital.
- High (8-10): Aggressive. Cash buffer <5%. Concentrate in top bullish ideas.

VOLATILITY GUARD: Never put >30% of capital in one ticker, regardless of confidence.

REFINEMENT:
- If previous Monitor check was INVALID, you MUST fix that exact violation this iteration.
- Adopt the What-If counter ONLY if it reduces risk or fixes a Monitor violation.

HARD CONSTRAINTS (Monitor will reject violations):
- No shorting: sell_shares ≤ currently_held.
- Budget: Σ(buy × price) − Σ(sell × price) ≤ available_capital.
- Only trade tickers with price > 0 in price_map.

Output JSON ONLY:
{{
  "agent": "portfolio_manager",
  "proposed_trades": [{{"action": "buy|sell", "ticker": "XXX", "shares": int}}],
  "notes": ["brief reasoning per decision"],
  "errors": []
}}
"""
    )

    human_message = HumanMessage(
        content=f"""
        Inputs:
        - Current Portfolio: {json.dumps(current_portfolio)}
        - Available Capital: {available_capital}
        - Risk Profile: {risk_profile}
        - Warren Signals: {json.dumps(warren_signals)}
        - Price Map: {json.dumps(price_map)}
        - Feedback from Previous Iteration: {json.dumps(previous_feedback) if previous_feedback else "None (First Iteration)"}
        """
    )
    
    response = llm.invoke([system_message, human_message])
    try:
        # Clean up potential markdown code blocks
        content = _text(response.content).strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "agent": "portfolio_manager",
            "proposed_trades": [],
            "notes": ["Error parsing LLM response"],
            "errors": [str(response.content)]
        }
