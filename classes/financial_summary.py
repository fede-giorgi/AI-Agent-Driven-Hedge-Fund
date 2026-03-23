from typing import Literal

from pydantic import BaseModel, Field


class FinancialSummary(BaseModel):
    """Aggregated financial data for a single ticker, compiled by the Research Agent."""

    ticker: str
    price: float | None = None

    # ── Valuation & efficiency metrics (from get_metrics) ────────────────────
    market_cap: float | None = None
    enterprise_value: float | None = None
    price_to_earnings_ratio: float | None = None
    price_to_book_ratio: float | None = None
    price_to_sales_ratio: float | None = None
    enterprise_value_to_ebitda_ratio: float | None = None
    enterprise_value_to_revenue_ratio: float | None = None
    free_cash_flow_yield: float | None = None
    peg_ratio: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    return_on_equity: float | None = None
    return_on_assets: float | None = None
    return_on_invested_capital: float | None = None
    asset_turnover: float | None = None
    inventory_turnover: float | None = None
    receivables_turnover: float | None = None
    days_sales_outstanding: float | None = None
    operating_cycle: float | None = None
    working_capital_turnover: float | None = None
    current_ratio: float | None = None
    quick_ratio: float | None = None
    cash_ratio: float | None = None
    operating_cash_flow_ratio: float | None = None
    debt_to_equity: float | None = None
    debt_to_assets: float | None = None
    interest_coverage: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    book_value_growth: float | None = None
    earnings_per_share_growth: float | None = None
    free_cash_flow_growth: float | None = None
    operating_income_growth: float | None = None
    ebitda_growth: float | None = None
    payout_ratio: float | None = None
    earnings_per_share: float | None = None
    book_value_per_share: float | None = None
    free_cash_flow_per_share: float | None = None

    # ── Income statement / balance sheet line items (from get_financial_line_items) ──
    capital_expenditure: float | None = None
    depreciation_and_amortization: float | None = None
    net_income: float | None = None
    outstanding_shares: float | None = None
    total_assets: float | None = None
    total_liabilities: float | None = None
    shareholders_equity: float | None = None
    dividends_and_other_cash_distributions: float | None = None
    issuance_or_purchase_of_equity_shares: float | None = None
    gross_profit: float | None = None
    revenue: float | None = None
    free_cash_flow: float | None = None
    current_assets: float | None = None
    current_liabilities: float | None = None

    # ── Historical time-series (most-recent first, up to 8 periods) ──────────
    historical_net_income: list[float] | None = None
    historical_revenue: list[float] | None = None
    historical_gross_profit: list[float] | None = None
    historical_return_on_equity: list[float] | None = None
    historical_operating_margin: list[float] | None = None
    historical_shareholders_equity: list[float] | None = None
    historical_outstanding_shares: list[float] | None = None
    historical_issuance_or_purchase_of_equity_shares: list[float] | None = None

    # ── Qualitative signals ───────────────────────────────────────────────────

    # News summary from FinancialDatasets.ai; formatted as "[date] headline (source)\n..."
    recent_news: str | None = None

    # Most-recent period's business-segment breakdown, e.g. {"iPhone": 200.5e9}
    segmented_revenue: dict[str, float] | None = None

    # Derived from recent Form 4 filings; positive = net buying, negative = net selling
    net_insider_buying: float | None = None
    insider_buy_count: int | None = None
    insider_sell_count: int | None = None

    # Analyst consensus for the next annual period
    analyst_revenue_estimate: float | None = None
    analyst_eps_estimate: float | None = None
    analyst_estimate_period: str | None = None


class ToolStatus(BaseModel):
    """Tracks which data-gathering tools succeeded or were skipped for a ticker."""

    get_financials: Literal["ok", "error"]
    get_metrics: Literal["ok", "error"]
    get_financial_line_items: Literal["ok", "error"]
    get_stock_prices: Literal["ok", "error"]
    get_company_news: Literal["ok", "error", "skipped"] = "skipped"
    get_segmented_revenues: Literal["ok", "error", "skipped"] = "skipped"
    get_insider_trades: Literal["ok", "error", "skipped"] = "skipped"
    get_analyst_estimates: Literal["ok", "error", "skipped"] = "skipped"


class Error(BaseModel):
    """Represents a tool or processing failure for a specific ticker."""

    tool: str
    message: str
    ticker: str


class Result(BaseModel):
    """Full research output for a single ticker, ready for downstream agents."""

    ticker: str
    financial_summary: FinancialSummary
    tool_status: ToolStatus
    extra_fields: dict = Field(default_factory=dict)
    data_quality_notes: list[str] = Field(
        default_factory=list,
        description="Free-text notes on missing data or unusual values found during structuring.",
    )
    errors: list[Error] = Field(
        default_factory=list,
        description="Non-fatal tool errors encountered while gathering data for this ticker.",
    )


class ResearchAgentOutput(BaseModel):
    """Aggregated output of the Research Agent across all requested tickers."""

    agent: str = "research_agent"
    period: str = "annual"
    requested_tickers: list[str]
    results: list[Result] = Field(default_factory=list)
    errors: list[Error] = Field(default_factory=list)


class WarrenBuffettSignal(BaseModel):
    """Investment signal produced by the Warren Buffett analysis agent."""

    signal: Literal["bullish", "bearish", "neutral"] = Field(
        description="Directional conviction: bullish = buy, bearish = reduce/avoid, neutral = hold.",
    )
    confidence: int = Field(
        description="Conviction level 0–100. Drives position sizing in the Portfolio Manager.",
    )
    reasoning: str = Field(
        description="Concise rationale referencing the specific tool outputs that drove the signal.",
    )
