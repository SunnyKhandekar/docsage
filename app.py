"""
app.py
------
DocSage Streamlit frontend.

This is the entry point for the HF Spaces deployment and local demo.
Run with: streamlit run app.py

Features:
  - Chat interface with message history
  - Toggle between direct RAG mode and agent mode (MCP tools enabled)
  - Shows source citations in an expandable section under each answer
  - Displays a warning banner when the model refuses to answer
  - Shows latency for each response
"""

import os
import time
import logging

import streamlit as st
from dotenv import load_dotenv

# Load .env for local dev. On HF Spaces, env vars are set via Space secrets
# and load_dotenv() will simply find no .env file — that's fine.
load_dotenv()

# Check for API key early so the error is clear.
if not os.environ.get("GROQ_API_KEY"):
    st.error(
        "GROQ_API_KEY environment variable is not set. "
        "Add it to your .env file (local) or Space secrets (HF Spaces)."
    )
    st.stop()

from src.retrieval.search import search_documents
from src.generation.generate import generate_answer, REFUSAL_PHRASE
from src.agent.graph import run_agent

logging.basicConfig(level=logging.WARNING)  # Keep logs quiet in the UI

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DocSage — AWS Documentation Assistant",
    page_icon="📚",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    agent_mode = st.toggle(
        "Agent Mode",
        value=False,
        help="When enabled, a LangGraph ReAct agent decides whether to search docs, "
             "use tools (e.g. cost calculator), or answer directly. "
             "When disabled, runs the RAG pipeline directly (faster).",
    )
    top_k = st.slider(
        "Context chunks (top_k)",
        min_value=1,
        max_value=10,
        value=5,
        help="How many document chunks to retrieve and pass to the model.",
    )
    st.divider()
    st.markdown("**Mode:** " + ("🤖 Agent (MCP tools)" if agent_mode else "📄 Direct RAG"))
    st.divider()
    st.markdown(
        "**DocSage** is a multimodal RAG agent for AWS technical documentation. "
        "It answers questions using only the indexed docs and cites every source."
    )
    st.markdown(
        "Built with: Groq · LangGraph · FastMCP · Chroma · RAGAS\n\n"
        "[View source on GitHub](https://github.com/YOUR_USERNAME/docsage)"
    )

# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------
st.title("📚 DocSage")
st.caption("Ask questions about AWS documentation. Sources and page numbers included.")

# Initialise chat history in session state.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    st.markdown(f"- **{src['source_doc']}** — page {src['page_num']}")
        if msg.get("latency_ms"):
            st.caption(f"⏱ {msg['latency_ms']:.0f} ms")

# Chat input.
if prompt := st.chat_input("Ask something about AWS..."):
    # Show user message immediately.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response.
    with st.chat_message("assistant"):
        with st.spinner("Searching documentation..."):
            start = time.perf_counter()

            if agent_mode:
                # Agent mode: LangGraph ReAct agent with MCP tools.
                answer_text = run_agent(prompt)
                sources = []
                refused = False
            else:
                # Direct RAG mode: hybrid search → rerank → generate.
                chunks = search_documents(prompt, top_k=top_k)
                result = generate_answer(prompt, chunks)
                answer_text = result["answer"]
                sources = result["sources"]
                refused = result["refused"]

            elapsed_ms = (time.perf_counter() - start) * 1000

        # Render response.
        if refused:
            st.warning(
                "⚠️ " + REFUSAL_PHRASE,
                icon="🚫",
            )
        else:
            st.markdown(answer_text)

        if sources:
            with st.expander(f"📎 Sources ({len(sources)})"):
                for src in sources:
                    st.markdown(f"- **{src['source_doc']}** — page {src['page_num']}")

        st.caption(
            f"⏱ {elapsed_ms:.0f} ms · "
            f"{'Agent mode' if agent_mode else 'Direct RAG'} · "
            f"top_k={top_k}"
        )

    # Save to history.
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": sources,
        "latency_ms": elapsed_ms,
    })
