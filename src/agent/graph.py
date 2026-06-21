"""
src/agent/graph.py
------------------
Builds the LangGraph ReAct agent that orchestrates DocSage.

The agent decides:
  - Should I search the documentation? → call search_documentation tool
  - Is this a cost calculation? → call estimate_monthly_cost tool
  - Can I answer directly? → generate without tool use

DEPRECATION NOTE:
  langgraph.prebuilt.create_react_agent was deprecated in LangGraph v1.x with
  a target removal in v2.0. The replacement is from langchain.agents import
  create_agent (new API). However, as of the build date, the new API is gated
  by a langchain_core version that has a breaking import incompatibility with
  langchain-community. The prebuilt function works correctly — it only emits
  a deprecation warning. Migrate when LangGraph v2 stabilises.
  Track: https://github.com/langchain-ai/langgraph/releases

create_react_agent API note:
  The system prompt parameter is named `prompt` (not `state_modifier`, which
  was the older name). Both a plain string and a SystemMessage are accepted.
"""

import logging
from functools import lru_cache

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent  # See deprecation note above

from src.agent.mcp_client_tools import get_mcp_tools
from src.config import GROQ_API_KEY, GENERATION_MODEL

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """You are DocSage, an AI assistant specialised in AWS technical documentation.

You have access to two tools:
  - search_documentation: search the indexed AWS documentation for relevant passages
  - estimate_monthly_cost: calculate an exact monthly/annual cost from an hourly rate

Decision rules:
  1. For any question about AWS services, concepts, or best practices → call search_documentation first.
  2. For cost/pricing questions that involve arithmetic → call estimate_monthly_cost.
  3. For greetings or meta questions about yourself → answer directly without tools.
  4. If search_documentation returns relevant context, base your answer ONLY on that context.
  5. If the context is insufficient, say so clearly rather than guessing.
  6. Always cite your sources as [source_doc, page X] when answering from documentation.
"""


@lru_cache(maxsize=1)
def get_agent():
    """
    Build and return the compiled LangGraph ReAct agent.

    Cached with lru_cache so the agent (and its tool list, which requires an
    MCP subprocess round-trip) is only initialised once per process.

    Returns:
        CompiledStateGraph — call .invoke({"messages": [...]}) to use it.
    """
    logger.info("Initialising DocSage agent ...")

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=GENERATION_MODEL,
        temperature=0.1,
    )

    tools = get_mcp_tools()
    logger.info(f"Agent loaded {len(tools)} MCP tools: {[t.name for t in tools]}")

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=AGENT_SYSTEM_PROMPT,  # 'prompt' param, not 'state_modifier'
    )

    logger.info("DocSage agent ready.")
    return agent


def run_agent(question: str) -> str:
    """
    Convenience function: run the agent on a question and return the answer text.

    Args:
        question: User's natural language question.

    Returns:
        The agent's final answer as a plain string.
    """
    agent = get_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    # The last message in the output is the agent's final response.
    return result["messages"][-1].content
