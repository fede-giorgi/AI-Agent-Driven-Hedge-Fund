import asyncio
import os
from typing import List, Dict
from contextlib import AsyncExitStack

from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPClient:
    """
    A client to connect to an MCP server and adapt its tools for LangChain.
    """
    def __init__(self, command: str, args: List[str], env: Dict[str, str] = None):
        self.command = command
        self.args = args
        self.env = env or os.environ.copy()
        self.exit_stack = AsyncExitStack()
        self.session = None

    async def __aenter__(self):
        """Initializes the connection to the MCP server."""
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env
        )
        
        # Start the server process and connect stdio
        read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
        
        # Initialize the MCP session
        self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Closes the connection."""
        await self.exit_stack.aclose()

    async def get_tools(self) -> List[StructuredTool]:
        """Fetches tools from the MCP server and converts them to LangChain tools."""
        if not self.session:
            raise RuntimeError("MCPClient not initialized. Use 'async with'.")
            
        mcp_tools = await self.session.list_tools()
        langchain_tools = []

        for tool in mcp_tools.tools:
            # Define the async wrapper for the tool execution
            async def _tool_wrapper(**kwargs):
                result = await self.session.call_tool(tool.name, arguments=kwargs)
                # Extract and join text content from the result
                return "\n".join([c.text for c in result.content if c.type == 'text'])

            # Create the LangChain tool
            lc_tool = StructuredTool.from_function(
                func=None, # We only provide the async coroutine
                coroutine=_tool_wrapper,
                name=tool.name,
                description=tool.description or f"MCP Tool: {tool.name}"
            )
            langchain_tools.append(lc_tool)
            
        return langchain_tools