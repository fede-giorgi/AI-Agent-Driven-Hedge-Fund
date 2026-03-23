"""Deterministic position-sizing tool for the Portfolio Manager Agent."""

from langchain.tools import tool


@tool(description="Compute the target dollar allocation and share count for a position.")
def calculate_position_size(
    confidence: int,
    risk_profile: int,
    total_capital: float,
    price: float,
    current_shares: int = 0,
    max_position_pct: float = 0.30,
) -> dict:
    """
    Calculates target dollar allocation and share count for a single position.

    Offloads arithmetic to Python so the LLM does not need to perform
    multiplication or division to derive share counts.

    Position formula:
        target_allocation = (confidence / 100) × (risk_profile / 10) × total_capital
        capped at max_position_pct × total_capital

    Args:
        confidence: Warren Buffett signal confidence (0-100).
        risk_profile: Portfolio risk level (1-10).
        total_capital: Total portfolio value (cash + holdings market value).
        price: Current share price for this ticker.
        current_shares: Shares already held (default 0).
        max_position_pct: Maximum single-position allocation as a fraction (default 0.30).

    Returns:
        dict with target_allocation_usd, target_shares, shares_to_buy,
        shares_to_sell, and details.
    """
    if price <= 0:
        return {
            "target_allocation_usd": 0.0,
            "target_shares": 0,
            "shares_to_buy": 0,
            "shares_to_sell": 0,
            "details": "Price must be > 0.",
        }

    raw_allocation = (confidence / 100) * (risk_profile / 10) * total_capital
    cap = max_position_pct * total_capital
    target_allocation = min(raw_allocation, cap)
    target_shares = int(target_allocation / price)
    delta = target_shares - current_shares

    return {
        "target_allocation_usd": round(target_allocation, 2),
        "target_shares": target_shares,
        "shares_to_buy": max(delta, 0),
        "shares_to_sell": max(-delta, 0),
        "details": (
            f"confidence={confidence}%, risk={risk_profile}/10, "
            f"raw=${raw_allocation:,.0f}, cap=${cap:,.0f} → "
            f"target ${target_allocation:,.0f} = {target_shares} shares @ ${price}. "
            f"Currently hold {current_shares} → "
            f"{'buy' if delta > 0 else 'sell'} {abs(delta)} shares."
        ),
    }
