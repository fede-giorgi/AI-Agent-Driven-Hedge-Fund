import asyncio
import os
from typing import List, Dict, Any, Optional
from contextlib import AsyncExitStack

from langchain_core.tools import StructuredTool
from pydantic import create_model, Field
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
        print(f"[MCP] Connecting to server: {self.command} {' '.join(self.args)}")
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
        print("[MCP] Session initialized successfully.")
        
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Closes the connection."""
        await self.exit_stack.aclose()
        print("[MCP] Connection closed.")

    async def get_tools(self) -> List[StructuredTool]:
        """Fetches tools from the MCP server and converts them to LangChain tools."""
        if not self.session:
            raise RuntimeError("MCPClient not initialized. Use 'async with'.")
            
        print("[MCP] Fetching tools list...")
        mcp_tools = await self.session.list_tools()
        print(f"[MCP] Discovered {len(mcp_tools.tools)} tools: {[t.name for t in mcp_tools.tools]}")
        langchain_tools = []

        for tool in mcp_tools.tools:
            # Define the async wrapper for the tool execution
            # Use a factory to capture the tool name properly without polluting the signature
            def create_wrapper(tool_name):
                async def _tool_wrapper(**kwargs):
                    print(f"[MCP] Executing tool '{tool_name}' with args: {kwargs}")
                    # Filter out None values to avoid validation errors on the MCP server side
                    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                    await asyncio.sleep(1.5)
                    result = await self.session.call_tool(tool_name, arguments=clean_kwargs)
                    # Extract and join text content from the result
                    text_content = "\n".join([c.text for c in result.content if c.type == 'text'])
                    print(f"[MCP] Tool '{tool_name}' execution complete. Result length: {len(text_content)} chars.")
                    return text_content
                return _tool_wrapper

            # Create a dynamic Pydantic model for the tool arguments based on MCP schema
            # This ensures the LLM knows exactly what arguments (e.g., 'query') to pass
            input_schema = tool.inputSchema
            fields = {}
            if input_schema and isinstance(input_schema, dict) and "properties" in input_schema:
                for prop, prop_def in input_schema["properties"].items():
                    # Map JSON schema types to Python types (simplified)
                    py_type = Any
                    if prop_def.get("type") == "string":
                        py_type = str
                    elif prop_def.get("type") == "integer":
                        py_type = int
                    elif prop_def.get("type") == "number":
                        py_type = float
                    elif prop_def.get("type") == "boolean":
                        py_type = bool
                    
                    # Check if required
                    if prop in input_schema.get("required", []):
                        fields[prop] = (py_type, Field(description=prop_def.get("description", "")))
                    else:
                        fields[prop] = (Optional[py_type], Field(default=None, description=prop_def.get("description", "")))
            
            SchemaModel = create_model(f"{tool.name.replace('-', '_')}Input", **fields)

            # Create the LangChain tool
            lc_tool = StructuredTool.from_function(
                func=None, # We only provide the async coroutine
                coroutine=create_wrapper(tool.name),
                name=tool.name,
                description=tool.description or f"MCP Tool: {tool.name}",
                args_schema=SchemaModel
            )
            langchain_tools.append(lc_tool)
            
        return langchain_tools