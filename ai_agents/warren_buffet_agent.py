"""
This script defines an investment agent that analyzes stocks according to Warren Buffett's value investing principles.
"""
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

import math
import json

from models.financial_summary import FinancialSummary, WarrenBuffettSignal
from llm import get_llm

from tools.analyze_book_value_growth import analyze_book_value_growth
from tools.analyze_consistency import analyze_consistency
from tools.analyze_fundamentals import analyze_fundamentals
from tools.analyze_management_quality import analyze_management_quality
from tools.analyze_moat import analyze_moat
from tools.analyze_pricing_power import analyze_pricing_power
from tools.calculate_intrinsic_value import calculate_intrinsic_value

# Define the Research Strategy/Briefing
def get_research_brief():
    """
    Generates the Research Brief that defines exactly what the Research Agent should look for.
    This ensures the data gathering is intent-driven based on Warren Buffett's analysis needs.
    """
    return {
        "focus_areas": [
            "Current Stock Price",
            "Economic Moat (Competitive Advantage)",
            "Management Quality & Integrity",
            "Financial Strength & Health",
            "Earnings Consistency & Growth",
            "Intrinsic Value Calculation"
        ],
        "required_metrics": [
            "return_on_invested_capital", 
            "gross_margin", 
            "operating_margin",
            "debt_to_equity", 
            "return_on_equity", 
            "current_ratio", 
            "interest_coverage", 
            "revenue_growth", 
            "earnings_growth", 
            "book_value_growth", 
            "payout_ratio", 
            "free_cash_flow_per_share", 
            "earnings_per_share"
        ],
        "required_line_items": [
            "capital_expenditure",
            "depreciation_and_amortization",
            "net_income",
            "outstanding_shares",
            "total_assets",
            "total_liabilities",
            "shareholders_equity",
            "dividends_and_other_cash_distributions",
            "issuance_or_purchase_of_equity_shares",
            "gross_profit",
            "revenue",
            "free_cash_flow",
            "current_assets",
            "current_liabilities"
        ],
        "search_queries": [
            "competitive advantage",
            "economic moat",
            "management integrity",
            "capital allocation strategy",
            "regulatory risks",
            "antitrust issues",
            "market share trends"
        ]
    }

def warren_buffett_agent(summary: FinancialSummary) -> dict:
    """
    Runs the Warren Buffett agent to analyze a stock.
    """
    print(f"Analyzing {summary.ticker} with Warren Buffett agent...")
    
    llm = get_llm()
    
    # 1. Define tools that are bound to the specific FinancialSummary of this stock.
    # This allows the LLM to "call" the analysis functions without needing to pass the complex object.
    
    @tool
    def check_fundamentals():
        """Analyzes key financial health metrics like ROE, debt, margins, and liquidity."""
        return analyze_fundamentals.func(summary=summary)

    @tool
    def check_consistency():
        """Checks for a track record of consistent and growing earnings."""
        return analyze_consistency.func(summary=summary)

    @tool
    def check_moat():
        """Evaluates the company's durable competitive advantage (moat)."""
        return analyze_moat.func(summary=summary)

    @tool
    def check_management():
        """Assesses management's shareholder-friendliness (buybacks, dividends)."""
        return analyze_management_quality.func(summary=summary)

    @tool
    def check_book_value_growth():
        """Analyzes the growth of book value per share over time."""
        return analyze_book_value_growth.func(summary=summary)

    @tool
    def check_intrinsic_value():
        """Estimates the company's intrinsic value using a DCF model."""
        return calculate_intrinsic_value.func(summary=summary)

    @tool
    def check_pricing_power():
        """Assesses the company's ability to raise prices (gross margins)."""
        return analyze_pricing_power.func(summary=summary)

    tools = [
        check_fundamentals, 
        check_consistency, 
        check_moat, 
        check_management, 
        check_book_value_growth, 
        check_intrinsic_value, 
        check_pricing_power
    ]
    
    llm_with_tools = llm.bind_tools(tools)

    # 2. Agent Loop
    messages = [
        SystemMessage(content=f"""You are a virtual Warren Buffett. Your goal is to evaluate the company {summary.ticker} based on value investing principles.
    
    You have access to specific analysis tools. You can call them in any order to gather the insights you need.
    Once you have enough information, you will provide a final investment signal.
    
    Key Principles:
    - Circle of Competence
    - Durable Moat
    - Rational Management
    - Financial Strength
    - Margin of Safety (Discount to Intrinsic Value)
    """),
        HumanMessage(content=f"Please analyze {summary.ticker} and provide an investment signal.")
    ]

    while True:
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            break
            
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            # Map tool names to functions
            tool_map = {t.name: t for t in tools}
            
            if tool_name in tool_map:
                try:
                    # Execute the tool (no args needed as they are bound to summary)
                    result = tool_map[tool_name].invoke({})
                except Exception as e:
                    result = f"Error executing {tool_name}: {e}"
            else:
                result = f"Unknown tool: {tool_name}"
            
            messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tool_call["id"]))

    # 3. Final Decision
    # We use a structured output LLM to parse the final conversation into the signal format
    structured_llm = llm.with_structured_output(WarrenBuffettSignal)
    
    final_instruction = HumanMessage(content="""Based on the analysis you performed above, determine a bullish, bearish, or neutral signal.
    - Assign a confidence score (0-100).
    - Provide a brief, decisive reasoning.""")
    
    final_signal = structured_llm.invoke(messages + [final_instruction])
    
    return {summary.ticker: final_signal.model_dump()}
