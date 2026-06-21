"""
src/agent/mcp_client_tools.py
------------------------------
Wraps the DocSage MCP server's tools as LangChain StructuredTools so they
can be passed to LangGraph's create_react_agent.

KEY ENGINEERING NOTE — args_schema bug fix:
  If you wrap an MCP tool with StructuredTool.from_function() and let LangChain
  infer the schema from a bare **kwargs function, it produces an EMPTY schema.
  When the agent calls .run({"query": "..."}), LangChain passes an empty dict
  to the tool, Pydantic rejects it as missing required fields, and the tool fails.

  Fix: dynamically build a Pydantic model from the MCP tool's inputSchema
  (standard JSON Schema dict) and pass it explicitly as `args_schema=`.
  This was validated in sandbox testing before being included here.

Architecture note — per-call subprocess:
  Each tool call opens a fresh asyncio event loop, starts the MCP server as a
  new subprocess, makes one call, and shuts down. This is simple and avoids
  cross-event-loop bugs (the persistent-session alternative requires careful
  lifecycle management).

  Tradeoff: ~200-400ms subprocess startup overhead per tool call.
  For an interactive demo this is acceptable. For production, replace with a
  persistent MCP session managed at application startup.
"""

import asyncio
import json
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Type

from pydantic import BaseModel, create_model
from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

# Path to the MCP server module entry point.
MCP_SERVER_MODULE = "src.agent.mcp_server"


# ---------------------------------------------------------------------------
# JSON Schema → Pydantic model conversion
# ---------------------------------------------------------------------------

_JSON_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _schema_to_model(tool_name: str, input_schema: Dict[str, Any]) -> Type[BaseModel]:
    """
    Convert an MCP tool's JSON Schema inputSchema into a Pydantic BaseModel class.

    Args:
        tool_name: Used as the model class name (for readable error messages).
        input_schema: The JSON Schema dict from the MCP tool definition.
                      Expected keys: "properties" (dict), "required" (list, optional).

    Returns:
        A dynamically created Pydantic BaseModel subclass.
    """
    properties = input_schema.get("properties", {})
    required_fields = set(input_schema.get("required", []))

    field_definitions: Dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        json_type = field_schema.get("type", "string")
        python_type = _JSON_TYPE_MAP.get(json_type, Any)

        if field_name in required_fields:
            # Required field — no default
            field_definitions[field_name] = (python_type, ...)
        else:
            # Optional field — use None as default
            default = field_schema.get("default", None)
            field_definitions[field_name] = (python_type, default)

    return create_model(f"{tool_name}Args", **field_definitions)


# ---------------------------------------------------------------------------
# Async MCP call helper
# ---------------------------------------------------------------------------

async def _call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Open a fresh MCP session, call one tool, return the result string.
    """
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", MCP_SERVER_MODULE],
        env=None,  # Inherit current process environment (includes GROQ_API_KEY)
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)

    # result.content is a list of content blocks. For our text-returning tools,
    # the first block's text field is the answer.
    if result.content and hasattr(result.content[0], "text"):
        return result.content[0].text
    return str(result.content)


def _run_mcp_tool(tool_name: str, **kwargs: Any) -> str:
    """
    Synchronous wrapper around _call_mcp_tool for use in StructuredTool.
    Opens a fresh event loop per call.
    """
    return asyncio.run(_call_mcp_tool(tool_name, kwargs))


# ---------------------------------------------------------------------------
# Public function: get all MCP tools as LangChain StructuredTools
# ---------------------------------------------------------------------------

async def _list_mcp_tools() -> List[Any]:
    """List all tools registered on the MCP server."""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", MCP_SERVER_MODULE],
        env=None,
    )
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_response = await session.list_tools()
    return tools_response.tools


def get_mcp_tools() -> List[StructuredTool]:
    """
    Connect to the DocSage MCP server, discover all tools, and return them
    as LangChain StructuredTool objects ready for use with create_react_agent.

    Returns:
        List of StructuredTool objects. Pass directly to create_react_agent as
        the `tools` argument.
    """
    raw_tools = asyncio.run(_list_mcp_tools())
    langchain_tools = []

    for mcp_tool in raw_tools:
        tool_name = mcp_tool.name
        input_schema = mcp_tool.inputSchema or {"properties": {}, "required": []}

        # Build typed Pydantic schema from the MCP tool's JSON Schema.
        args_model = _schema_to_model(tool_name, input_schema)

        # Capture tool_name in closure correctly.
        def make_runner(name: str):
            def runner(**kwargs: Any) -> str:
                return _run_mcp_tool(name, **kwargs)
            return runner

        langchain_tool = StructuredTool.from_function(
            func=make_runner(tool_name),
            name=tool_name,
            description=mcp_tool.description or f"MCP tool: {tool_name}",
            args_schema=args_model,  # CRITICAL: must be set explicitly
        )
        langchain_tools.append(langchain_tool)
        logger.info(f"Registered MCP tool as StructuredTool: {tool_name}")

    return langchain_tools
