import json
from pydantic import BaseModel, Field
import asyncio
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from llm import get_llm
from models.financial_summary import FinancialSummary, ToolStatus, Error, Result, ResearchAgentOutput
from tools.get_financials import get_financials
from tools.get_metrics import get_metrics
from tools.get_financial_line_items import get_financial_line_items
from tools.get_stock_prices import get_stock_prices
from tools.mcp_client import MCPClient

REQUIRED_LIST = [
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
    "current_liabilities",
]


async def run_research_agent(
        tickers: List[str],
        backtesting_date: str = None
        ) -> ResearchAgentOutput:
    """
    Runs the research agent to gather and structure financial data for a list of tickers.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(Result)

    agent_output = ResearchAgentOutput(requested_tickers=tickers)

    mcp_client = MCPClient(
        command="npx", 
        args=["-y", "@brave/brave-search-mcp-server"]
    )
    
    # We use 'async with' to manage the server process lifecycle
    # If connection fails (e.g. no API key), we proceed with standard tools only.
    mcp_tools = []
    try:
        async with mcp_client as client:
            mcp_tools = await client.get_tools()
            print(f"Loaded MCP Tools: {[t.name for t in mcp_tools]}")
            await _process_tickers(tickers, backtesting_date, llm, structured_llm, agent_output, mcp_tools)
    except Exception as e:
        print(f"MCP Connection Failed (proceeding without web search): {e}")
        # Fallback: Run without MCP tools
        await _process_tickers(tickers, backtesting_date, llm, structured_llm, agent_output, [])

    return agent_output

async def _process_tickers(tickers, backtesting_date, llm, structured_llm, agent_output, mcp_tools):
    """Helper function to process tickers with a given set of tools."""
    
    for ticker in tickers:
        print(f"Researching {ticker}...")
        
        # 1. Define the tools available to the agent
        tools = [get_financials, get_metrics, get_financial_line_items, get_stock_prices] + mcp_tools
        llm_with_tools = llm.bind_tools(tools)

        # 2. Initialize conversation history with a directive System Message
        messages = [
            SystemMessage(content=f"""You are a Research Agent. Your goal is to gather all necessary financial data for the ticker '{ticker}' to populate a comprehensive `FinancialSummary`.
            
            You have access to the following tools:
            - `get_financials`: Fetch income statement, balance sheet, and cash flow.
            - `get_metrics`: Fetch calculated financial ratios and metrics.
            - `get_financial_line_items`: Fetch specific line items. IMPORTANT: You MUST request exactly these line items: {json.dumps(REQUIRED_LIST)}
            - `get_stock_prices`: Fetch recent price history.
            - `brave_web_search` (if available): Search the web for general context.
            - `brave_news_search` (if available): Search specifically for news articles to explain price movements or sentiment.

            Instructions:
            1. Call the tools to gather data. You can call multiple tools in parallel or sequentially.
            2. If a tool fails, you can retry or proceed.
            3. Ensure you pass `end_date='{backtesting_date}'` to all tools if a backtesting date is provided.
            4. If you have access to search tools, use `brave_news_search` to look for recent news (relative to the end_date) to explain major price movements or financial anomalies. Use `brave_web_search` for broader context if needed.
            5. Do not call the same tool twice for the same ticker. Once you have the data, proceed.
            6. Once you have gathered sufficient data (or exhausted attempts), stop calling tools.
            """),
            HumanMessage(content=f"Start research for {ticker}.")
        ]

        # 3. Agent Loop (Reasoning + Acting)
        # Track tools to prevent redundant calls
        tools_called_successfully = set()
        
        while True:
            # Use ainvoke for async support (needed for MCP tools)
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)

            # If the LLM didn't make any tool calls, we are done gathering data
            if not response.tool_calls:
                break
            
            # Execute the tool calls
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                args = tool_call["args"]
                
                # Create a map for easy lookup
                tool_map = {t.name: t for t in tools}
                
                # Prevent redundant calls for financial tools
                if tool_name in ["get_financials", "get_metrics", "get_financial_line_items", "get_stock_prices"] and tool_name in tools_called_successfully:
                    print(f"  --> Skipping redundant tool call: {tool_name}")
                    tool_result = {"error": f"Tool '{tool_name}' was already called successfully. Do not call it again."}
                else:
                    print(f"  --> Agent calling tool: {tool_name}")
                    try:
                        if tool_name in tool_map:
                            tool_instance = tool_map[tool_name]
                            
                            # Check if the tool is async (MCP tools are async)
                            if tool_instance.coroutine:
                                tool_result = await tool_instance.ainvoke(args)
                            else:
                                # Fallback for sync tools (your existing tools)
                                tool_result = tool_instance.invoke(args)
                            
                            # Mark as successful if no error in result
                            if isinstance(tool_result, dict) and "error" in tool_result:
                                pass 
                            else:
                                tools_called_successfully.add(tool_name)
                        else:
                            tool_result = {"error": f"Unknown tool: {tool_name}"}
                    except Exception as e:
                        tool_result = {"error": str(e)}
                
                messages.append(ToolMessage(content=json.dumps(tool_result), tool_call_id=tool_call["id"]))

        try:
            # 4. Final Structuring: Ask the LLM to compile the conversation history into the Result model
            system_message = SystemMessage(content="""You are a Data Structuring Expert. Your task is to parse the conversation history (which contains tool outputs) and populate the `Result` model.
            Rules:
            - You will be given the raw JSON output from each of the three tools.
            - Populate the `financial_summary` field using the provided data. All fields in `FinancialSummary` must be present; use null if a value is not available.
            - Specifically for the `price` field in `FinancialSummary`, extract the latest closing price from the `get_stock_prices` output.
            - Any keys from the raw tool output that are not part of the `FinancialSummary` model should be placed in the `extra_fields` dictionary.
            - If a tool failed (indicated by an error message instead of JSON), reflect this in the `tool_status` and `errors` fields.
            - Analyze the provided data for any potential inconsistencies or quality issues and add notes to `data_quality_notes`. For example, if `revenue` from one tool is drastically different from another.
            - If news/sentiment data was gathered via search, summarize it in `data_quality_notes` or `extra_fields` under a 'market_sentiment' key.
            - Use numbers when the source data is a number. If it's a string that looks like a number, try to convert it. If unsure, keep the original value and add a note to `data_quality_notes`. Do not use `NaN` or `Infinity`; use `null` instead.
            - Output a valid JSON object matching the `Result` model only. Do not add any extra prose or markdown.
            """)
            
            # We pass the entire history of tool calls and results to the structured LLM
            final_messages = [system_message] + messages
            result = structured_llm.invoke(final_messages)
            
            agent_output.results.append(result)
            print(f"Research result for {ticker}: {result.model_dump_json(indent=2)}")
            
        except Exception as e:
            agent_output.errors.append(Error(tool="processing_chain", message=str(e), ticker=ticker))