"""
src/evaluation/run_eval.py
--------------------------
Runs the DocSage evaluation harness.

For each question in eval_set.json, this script:
  1. Runs the full retrieval pipeline (hybrid search → rerank)
  2. Generates an answer using the Groq LLM
  3. Records the answer, context, ground truth, and latency

Then passes everything to RAGAS for scoring:
  - Faithfulness: are all claims grounded in the retrieved context?
  - Answer Relevancy: does the answer address the actual question?
  - Context Precision: what fraction of retrieved chunks were useful?
  - Context Recall: did retrieval find all the context needed?

Plus: p50 (median) and p95 (95th percentile) latency across the eval set.

TROUBLESHOOTING — ragas ImportError:
  If you see:
      ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
  This is a version mismatch between ragas and langchain-community.
  Fix by pinning compatible versions:
      pip install "ragas==0.1.21" "langchain-community==0.2.16"
  Check https://github.com/explodinggradients/ragas/issues for the current
  recommended version pair.
"""

import json
import time
import logging
import statistics
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src.retrieval.search import search_documents
from src.generation.generate import generate_answer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"


def run_evaluation():
    """Run the full evaluation pipeline and print results."""

    # Load evaluation questions.
    with open(EVAL_SET_PATH) as f:
        eval_items = json.load(f)

    print(f"\nRunning DocSage evaluation on {len(eval_items)} questions ...\n")

    questions = []
    answers = []
    contexts = []
    ground_truths = []
    latencies = []

    for i, item in enumerate(eval_items, start=1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"[{i}/{len(eval_items)}] {question[:80]}...")

        start = time.perf_counter()

        # Retrieve context
        chunks = search_documents(question)
        context_texts = [c["text"] for c in chunks]

        # Generate answer
        result = generate_answer(question, chunks)
        answer = result["answer"]

        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        questions.append(question)
        answers.append(answer)
        contexts.append(context_texts)
        ground_truths.append(ground_truth)

        print(f"    → {elapsed:.2f}s | refused={result['refused']}")

    # Build RAGAS dataset.
    # RAGAS expects a HuggingFace Dataset with these exact column names.
    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    print("\nRunning RAGAS scoring ...\n")

    scores = evaluate(
        ragas_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    # Latency statistics.
    p50 = statistics.median(latencies)
    latencies_sorted = sorted(latencies)
    p95_idx = int(len(latencies_sorted) * 0.95)
    p95 = latencies_sorted[min(p95_idx, len(latencies_sorted) - 1)]

    # Print report.
    print("=" * 60)
    print("DOCSAGE EVALUATION REPORT")
    print("=" * 60)
    print(f"\nRAGAS Metrics:")
    print(f"  Faithfulness:       {scores['faithfulness']:.4f}")
    print(f"  Answer Relevancy:   {scores['answer_relevancy']:.4f}")
    print(f"  Context Precision:  {scores['context_precision']:.4f}")
    print(f"  Context Recall:     {scores['context_recall']:.4f}")
    print(f"\nLatency (n={len(latencies)}):")
    print(f"  p50 (median): {p50:.2f}s")
    print(f"  p95:          {p95:.2f}s")
    print(f"  min:          {min(latencies):.2f}s")
    print(f"  max:          {max(latencies):.2f}s")
    print("=" * 60)

    return scores


if __name__ == "__main__":
    run_evaluation()
