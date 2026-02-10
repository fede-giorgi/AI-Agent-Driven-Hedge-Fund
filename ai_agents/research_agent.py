import json
from pydantic import BaseModel, Field
import asyncio
from typing import List, Dict, Any
import os

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from llm import get_llm
from classes.financial_summary import FinancialSummary, ToolStatus, Error, Result, ResearchAgentOutput
from tools.get_financials import get_financials
from tools.get_metrics import get_metrics
from tools.get_financial_line_items import get_financial_line_items
from tools.get_stock_prices import get_stock_prices
from tools.mcp import MCPClient


async def run_research_agent(
        tickers: List[str],
        research_config: Dict[str, Any],
        backtesting_date: str = None
        ) -> ResearchAgentOutput:
    """
    Runs the research agent to gather and structure financial data for a list of tickers.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(Result)

    agent_output = ResearchAgentOutput(requested_tickers=tickers)

    if not os.getenv("BRAVE_API_KEY"):
        print("WARNING: BRAVE_API_KEY not found in environment variables. Brave Search tools will likely fail.")

    mcp_client = MCPClient(
        command="npx", 
        args=["-y", "@brave/brave-search-mcp-server@latest", "--enabled-tools", "brave_news_search"]
    )
    
    # We use 'async with' to manage the server process lifecycle
    # If connection fails (e.g. no API key), we proceed with standard tools only.
    mcp_tools = []
    try:
        async with mcp_client as client:
            all_mcp_tools = await client.get_tools()
            print(f"All available MCP tools: {[t.name for t in all_mcp_tools]}")
            
            # Filter to allow only brave_news_search as requested
            mcp_tools = [t for t in all_mcp_tools if t.name == "brave_news_search"]
            
            if not mcp_tools and all_mcp_tools:
                print("Warning: 'brave_news_search' not found in available tools.")
                
            await _process_tickers(tickers, backtesting_date, llm, structured_llm, agent_output, mcp_tools, research_config)
    except Exception as e:
        print(f"MCP Connection Failed (proceeding without web search): {e}")
        # Fallback: Run without MCP tools
        await _process_tickers(tickers, backtesting_date, llm, structured_llm, agent_output, [], research_config)

    return agent_output

async def _process_tickers(tickers, backtesting_date, llm, structured_llm, agent_output, mcp_tools, research_config):
    """Helper function to process tickers with a given set of tools."""
    
    for ticker in tickers:
        print(f"Researching {ticker}...")
        
        # 1. Define the tools available to the agent
        tools = [get_financials, get_metrics, get_financial_line_items, get_stock_prices] + mcp_tools
        llm_with_tools = llm.bind_tools(tools)
        tool_map = {t.name: t for t in tools}

        # Dynamically build prompt based on available tools
        search_tool_desc = ""
        search_instruction = ""
        
        # Check if brave_news_search is actually available in the passed mcp_tools
        if any(t.name == "brave_news_search" for t in mcp_tools):
            search_tool_desc = f'- `brave_news_search`: News search. Use this tool to investigate qualitative aspects related to the focus areas, specifically looking for information on: {json.dumps(research_config.get("search_queries", []))}.'
            search_instruction = "4. Use `brave_news_search` to find qualitative data (news, sentiment) to complement the quantitative data."
        

        # 2. Initialize conversation history with a directive System Message
        messages = [
            SystemMessage(content=f"""You are a Research Agent. Your goal is to gather financial data for the ticker '{ticker}' to populate a `FinancialSummary`.
            
            You have been given a specific RESEARCH BRIEF. Focus your efforts on these areas:
            {json.dumps(research_config.get('focus_areas', []), indent=2)}

            You have access to a suite of financial tools and search capabilities.
            
            Tools available:
            - `get_financials`: Income statement, balance sheet, cash flow.
            - `get_metrics`: Key ratios and metrics. Prioritize fetching: {json.dumps(research_config.get('required_metrics', []))}
            - `get_financial_line_items`: Specific line items. You MUST fetch these: {json.dumps(research_config.get('required_line_items', []))}
            - `get_stock_prices`: Price history.
            {search_tool_desc}

            Instructions:
            1. Always fetch the current stock price using `get_stock_prices`.
            2. Decide which other tools are necessary to fulfill the RESEARCH BRIEF.
            2. If a tool fails, use your judgment to retry or find alternative data sources.
            3. Ensure you pass `end_date='{backtesting_date}'` to all tools if a backtesting date is provided.
            {search_instruction}
            5. Avoid redundant calls.
            6. Stop when you have sufficient information to satisfy the brief.
            """),
            HumanMessage(content=f"Start research for {ticker}.")
        ]

        # 3. Agent Loop (Reasoning + Acting)
        # Track tools to prevent redundant calls
        tools_called_successfully = set()
        search_call_count = 0
        
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
                
                # Circuit breaker for search tools to prevent infinite loops or excessive costs
                if tool_name == "brave_news_search":
                    if "goggles" not in args or args["goggles"] is None:
                        args["goggles"] = []

                    if search_call_count >= 10:
                        print(f"  --> Skipping search tool call (limit reached): {tool_name}")
                        tool_result = {"error": "Max search limit (10) reached. Please proceed with the data you have."}
                        messages.append(ToolMessage(content=json.dumps(tool_result), tool_call_id=tool_call["id"]))
                        continue
                    else:
                        search_call_count += 1

                # Prevent redundant calls for financial tools
                if tool_name in ["get_financials", "get_metrics", "get_financial_line_items", "get_stock_prices"] and tool_name in tools_called_successfully:
                    print(f"  --> Skipping redundant tool call: {tool_name}")
                    tool_result = {"error": f"Tool '{tool_name}' was already called successfully. Do not call it again."}
                else:
                    print(f"  --> Agent calling tool: {tool_name} (Args: {json.dumps(args)})")
                    try:
                        if tool_name in tool_map:
                            tool_instance = tool_map[tool_name]
                            
                            tool_result = await tool_instance.ainvoke(args)
                            
                            if tool_name == "brave_news_search":
                                print(f"  --> [MCP] Output for {tool_name}: {str(tool_result)[:1000]}...")
                            
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
            system_message = SystemMessage(content="""You are a Data Structuring Expert. Your task is to parse the conversation history and populate the `Result` model.
            
            Guidelines:
            - Extract financial data from the tool outputs to populate the `financial_summary` field.
            - Use your best judgment to map data to the correct fields. If a field is missing, use null.
            - Extract the latest closing price for the `price` field.
            - Place any additional interesting data or context in `extra_fields`.
            - Note any data quality issues or tool failures in `data_quality_notes` or `tool_status`.
            - If news data was gathered via search, summarize it in `extra_fields` under a 'news_related_to_the_ticker' key.
            - Ensure numeric values are formatted correctly (no NaN/Infinity).
            - Output valid JSON matching the `Result` model.
            """)
            
            # We pass the entire history of tool calls and results to the structured LLM
            final_messages = [system_message] + messages
            result = structured_llm.invoke(final_messages)
            
            agent_output.results.append(result)
            print(f"Research result for {ticker}: {result.model_dump_json(indent=2)}")
            
        except Exception as e:
            agent_output.errors.append(Error(tool="processing_chain", message=str(e), ticker=ticker))