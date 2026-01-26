import os
from typing import List, Dict
from langchain_core.tools import BaseTool
from mcp_use import MCPClient as McpUseClient
from mcp_use.agents.adapters import LangChainAdapter

class MCPClient:
    """
    A client to connect to an MCP server and adapt its tools for LangChain using mcp-use.
    """
    def __init__(self, command: str, args: List[str], env: Dict[str, str] = None):
        self.config = {
            "mcpServers": {
                "default": {
                    "command": command,
                    "args": args,
                    "env": env or os.environ.copy()
                }
            }
        }
        self.client = None

    async def __aenter__(self):
        """Initializes the connection to the MCP server."""
        self.client = McpUseClient(config=self.config)
        await self.client.create_all_sessions()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Closes the connection."""
        if self.client:
            await self.client.close_all_sessions()

    async def get_tools(self) -> List[BaseTool]:
        """Fetches tools from the MCP server and converts them to LangChain tools."""
        if not self.client:
            raise RuntimeError("MCPClient not initialized. Use 'async with'.")
            
        adapter = LangChainAdapter()
        # Convert tools from active connectors to the LangChain's format
        await adapter.create_tools(self.client)
        
        return adapter.tools