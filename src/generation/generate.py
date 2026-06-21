"""
src/generation/generate.py
--------------------------
Generates answers using the Groq LLM, grounded strictly in retrieved context.

Two hard rules enforced via the system prompt:
  1. Every factual claim must cite its source as [source_doc, page X].
  2. If the context doesn't support a confident answer, respond with the exact
     refusal phrase so downstream code (and the UI) can detect it cleanly.

Why a strict system prompt matters:
  Without explicit instructions, LLMs will "helpfully" fill gaps with
  plausible-sounding but ungrounded statements — hallucination. Forcing
  citation format + an explicit refusal phrase closes this loop.
"""

import logging
from typing import List, Dict, Any

from groq import Groq

from src.config import GROQ_API_KEY, GENERATION_MODEL

logger = logging.getLogger(__name__)

# Exact phrase the model must use when it cannot answer from context.
# Keep this consistent — the UI checks for this string to style the response.
REFUSAL_PHRASE = "I don't have enough information in the documentation to answer that."

SYSTEM_PROMPT = f"""You are DocSage, an expert assistant for AWS technical documentation.

RULES — follow these without exception:
1. Answer ONLY using the context passages provided below. Do not use any outside knowledge.
2. Every factual claim in your answer must be followed by a citation in this exact format: [source_doc, page X]
   Example: "The six pillars are Operational Excellence, Security, Reliability, Performance Efficiency, Cost Optimisation, and Sustainability [AWS Well-Architected Framework.pdf, page 3]."
3. If the provided context does not contain enough information to answer the question confidently, respond with EXACTLY this sentence and nothing else:
   {REFUSAL_PHRASE}
4. Do not make up information. Do not speculate. Do not combine context with outside knowledge.
5. Be concise. Answer the question directly before expanding with detail.
"""


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source_doc", "unknown")
        page = chunk.get("page_num", "?")
        text = chunk.get("text", "")
        parts.append(f"[{i}] Source: {source}, page {page}\n{text}")
    return "\n\n---\n\n".join(parts)


def generate_answer(
    query: str,
    context_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate a grounded, cited answer from retrieved context.

    Args:
        query: The user's question.
        context_chunks: Ranked list of chunk dicts from search_documents().

    Returns:
        Dict with:
          - "answer": str — the model's response (or the refusal phrase)
          - "refused": bool — True if the model triggered the refusal
          - "sources": list of {source_doc, page_num} dicts from used chunks
    """
    client = Groq(api_key=GROQ_API_KEY)

    context_text = _format_context(context_chunks)

    user_message = f"""Context from documentation:

{context_text}

---

Question: {query}

Answer (citing sources):"""

    logger.info(f"Calling Groq {GENERATION_MODEL} for query: {query!r}")

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,   # Low temperature = factual, consistent answers
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()
    refused = answer.strip() == REFUSAL_PHRASE

    sources = [
        {"source_doc": c.get("source_doc", ""), "page_num": c.get("page_num", 0)}
        for c in context_chunks
    ]

    return {
        "answer": answer,
        "refused": refused,
        "sources": sources,
    }
