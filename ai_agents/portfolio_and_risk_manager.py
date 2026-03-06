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
        content=f"""You are PortfolioManagerAgent. Your goal is to optimize a stock portfolio based on Warren Buffett-style analysis signals, risk profile, and capital constraints.
        
        CONTEXT:
        - You are operating in a simulation loop. This is Iteration {current_iteration} of {total_iterations}.
        - You interact with a Monitor Agent (who validates your trades) and a What-If Agent (who challenges your strategy).

        Strategy & Logic:
        1. **Signal Interpretation**:
           - **Bullish**: Strong buy signal. High confidence (80%+) implies high conviction.
           - **Bearish**: Strong sell signal. Reduce exposure immediately.
           - **Neutral**: Hold or trim. Do not add to neutral positions unless they are significantly underweight and fundamentals are still decent.

        2. **Risk Management (Risk Profile {risk_profile}/10)**:
           - **Low Risk (1-3)**: EXTREME CAUTION. Prioritize capital preservation above all. If Risk Profile is 1, DO NOT BUY any stocks; only sell to raise cash if needed. Keep a large cash buffer.
           - **Mid Risk (4-7)**: Balanced approach. Cash buffer 5-15%. Scale position sizes based on conviction (Confidence score).
           - **High Risk (8-10)**: Aggressive growth. Low cash buffer (<5%). Concentrate capital in top highest-confidence Bullish ideas.

        3. **Position Sizing**:
           - Decide on trade quantities based on conviction and available capital.
           - High conviction + High Risk = Larger position size.
           - Bearish signals = Evaluate selling. Reduce exposure significantly, but consider portfolio balance. Immediate full liquidation is not mandatory if the position is small or acts as a hedge, but generally, bearish implies selling.

        4. **Execution Guidelines**:
           - **Liquidity**: Ensure sufficient cash is available for proposed buys (considering sell proceeds).
           - **Allocation**: Prioritize high-conviction Bullish stocks.
           - **Risk Control**: Monitor position weights to avoid excessive concentration.

        Refinement Logic:
        - Critically evaluate the "What-If" agent's feedback. Do not accept it blindly. Only adopt it if it genuinely improves risk-adjusted returns or fixes a violation.
        - If the previous iteration had a violation, you MUST adjust your trades to fix it.

        Constraints:
        - Trades must be JSON objects: {{"action":"buy|sell","ticker":"XXX","shares":int>0}}
        - No shorting: do not sell more shares than currently held.
        - Only trade tickers with a valid positive price in price_map.
        - Ensure net buy cost does not exceed available capital + expected sell proceeds.

        Output JSON ONLY:
        {{
          "agent":"portfolio_manager",
          "proposed_trades":[{{"action":"...","ticker":"...","shares":...}}],
          "notes":[...],
          "errors":[...]
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
