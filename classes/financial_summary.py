from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Optional

class FinancialSummary(BaseModel):
    ticker: str
    price: Optional[float] = None

    # Metrics from get_metrics
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    price_to_earnings_ratio: Optional[float] = None
    price_to_book_ratio: Optional[float] = None
    price_to_sales_ratio: Optional[float] = None
    enterprise_value_to_ebitda_ratio: Optional[float] = None
    enterprise_value_to_revenue_ratio: Optional[float] = None
    free_cash_flow_yield: Optional[float] = None
    peg_ratio: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None
    return_on_invested_capital: Optional[float] = None
    asset_turnover: Optional[float] = None
    inventory_turnover: Optional[float] = None
    receivables_turnover: Optional[float] = None
    days_sales_outstanding: Optional[float] = None
    operating_cycle: Optional[float] = None
    working_capital_turnover: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    cash_ratio: Optional[float] = None
    operating_cash_flow_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    debt_to_assets: Optional[float] = None
    interest_coverage: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    book_value_growth: Optional[float] = None
    earnings_per_share_growth: Optional[float] = None
    free_cash_flow_growth: Optional[float] = None
    operating_income_growth: Optional[float] = None
    ebitda_growth: Optional[float] = None
    payout_ratio: Optional[float] = None
    earnings_per_share: Optional[float] = None
    book_value_per_share: Optional[float] = None
    free_cash_flow_per_share: Optional[float] = None

    # Line items from get_financial_line_items
    capital_expenditure: Optional[float] = None
    depreciation_and_amortization: Optional[float] = None
    net_income: Optional[float] = None
    outstanding_shares: Optional[float] = None
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    shareholders_equity: Optional[float] = None
    dividends_and_other_cash_distributions: Optional[float] = None
    issuance_or_purchase_of_equity_shares: Optional[float] = None
    gross_profit: Optional[float] = None
    revenue: Optional[float] = None
    free_cash_flow: Optional[float] = None
    current_assets: Optional[float] = None
    current_liabilities: Optional[float] = None

class ToolStatus(BaseModel):
    get_financials: Literal["ok", "error"]
    get_metrics: Literal["ok", "error"]
    get_financial_line_items: Literal["ok", "error"]
    get_stock_prices: Literal["ok", "error"]

class Error(BaseModel):
    tool: str
    message: str
    ticker: str

class Result(BaseModel):
    ticker: str
    financial_summary: FinancialSummary
    extra_fields: dict = Field(default_factory=dict)
    tool_status: ToolStatus
    data_quality_notes: List[str] = Field(default_factory=list)
    errors: List[Error] = Field(default_factory=list)

class ResearchAgentOutput(BaseModel):
    agent: str = "research_agent"
    period: str = "annual"
    requested_tickers: List[str]
    results: List[Result] = Field(default_factory=list)
    errors: List[Error] = Field(default_factory=list)



class WarrenBuffettSignal(BaseModel):
    """The final output of the Warren Buffett agent."""
    signal: Literal["bullish", "bearish", "neutral"] = Field(description="The investment signal for the stock.")
    confidence: int = Field(description="The confidence level of the signal, from 0 to 100.")
    reasoning: str = Field(description="A brief reasoning for the signal.")