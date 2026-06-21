"""
src/agent/mcp_server.py
-----------------------
FastMCP server that exposes DocSage capabilities as MCP tools.

MCP (Model Context Protocol) is an open protocol that standardises how AI
models connect to external tools and data sources. Think of it as USB-C for
AI integrations: one standard interface, many compatible tools.

This server exposes two tools:
  1. search_documentation — runs the full retrieval pipeline
  2. estimate_monthly_cost — simple cost calculator (avoids LLM arithmetic errors)

The server communicates over stdio (stdin/stdout). The client (mcp_client_tools.py)
starts this script as a subprocess and exchanges JSON-RPC messages through its pipes.

Usage (standalone test):
    python -m src.agent.mcp_server

Usage (normal — started automatically by mcp_client_tools.py):
    The client launches this as a subprocess. You don't run it manually in production.
"""

import json
import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from src.retrieval.search import search_documents

logger = logging.getLogger(__name__)

# Create the FastMCP server instance.
mcp = FastMCP("DocSage")


@mcp.tool()
def search_documentation(query: str, top_k: int = 5) -> str:
    """
    Search the AWS documentation index and return relevant passages.

    Args:
        query: The question or search string.
        top_k: Number of result chunks to return (default: 5).

    Returns:
        A JSON string containing a list of relevant chunks with their text,
        source document name, and page number.
    """
    chunks = search_documents(query, top_k=top_k)

    # Return serialisable summary — omit internal score fields from the tool output.
    results = [
        {
            "text": c["text"],
            "source_doc": c.get("source_doc", ""),
            "page_num": c.get("page_num", 0),
            "chunk_type": c.get("chunk_type", "text"),
        }
        for c in chunks
    ]
    return json.dumps(results, ensure_ascii=False)


@mcp.tool()
def estimate_monthly_cost(hourly_rate_usd: float, hours_per_month: float = 730.0) -> str:
    """
    Estimate the monthly cost of an AWS resource given an hourly rate.

    This tool exists to avoid asking the LLM to do arithmetic — LLMs are
    unreliable at exact multiplication. A deterministic tool is always correct.

    Args:
        hourly_rate_usd: The per-hour cost in USD (e.g., 0.0416 for t3.micro).
        hours_per_month: Hours in a month. Default 730 (365 * 24 / 12).

    Returns:
        A plain-text cost estimate string.
    """
    monthly = hourly_rate_usd * hours_per_month
    annual = monthly * 12
    return (
        f"Estimated cost:\n"
        f"  Hourly:  ${hourly_rate_usd:.4f}\n"
        f"  Monthly: ${monthly:.2f} ({hours_per_month:.0f} hours)\n"
        f"  Annual:  ${annual:.2f}\n"
        f"Note: This is a rough estimate. Check the AWS Pricing Calculator at "
        f"https://calculator.aws for exact figures including data transfer and storage."
    )


if __name__ == "__main__":
    # Run with stdio transport (default for subprocess-based MCP).
    mcp.run(transport="stdio")
