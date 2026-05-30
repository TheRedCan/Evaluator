"""Run the Evaluator harness against the engineering-codes RAG system.

Usage:
    python -m integrations.run_rag_eval --dataset data/rag_subset.jsonl --out reports/rag/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from evaluator.dataset import load_jsonl
from evaluator.metrics import DEFAULT_METRICS
from evaluator.report import write_reports
from evaluator.runner import RowResult, EvalReport

from integrations.rag_provider import RAGProvider


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", default="reports/rag/")
    args = p.parse_args()

    examples = load_jsonl(args.dataset)
    provider = RAGProvider()
    metrics = list(DEFAULT_METRICS)

    print(f"Running {len(examples)} examples through the RAG system...")
    print("(each query takes ~60-90s — retrieval + rerank + 7B local LLM)\n")

    rows: list[RowResult] = []
    for i, ex in enumerate(examples, start=1):
        print(f"[{i}/{len(examples)}] {ex.input[:80]}")
        t0 = time.perf_counter()
        err: str | None = None
        try:
            pred = provider.generate(ex.input)
        except Exception as e:
            pred = ""
            err = f"{type(e).__name__}: {e}"
        latency = time.perf_counter() - t0
        # Use the RAG's cited-chunk text as grounding for the Hallucination
        # metric. This is the actual point of the integration: the harness
        # scores faithfulness against the RAG's own retrieved evidence.
        rag_context = provider.last_context
        scores = {m.name: m.score(pred, ex.expected, rag_context) for m in metrics}
        print(f"    -> {latency:.1f}s, {provider.last_n_chunks} chunks cited, "
              f"scores: {', '.join(f'{k}={v:.2f}' for k, v in scores.items())}")
        if err:
            print(f"    ERROR: {err}")
        rows.append(RowResult(
            input=ex.input, expected=ex.expected, prediction=pred,
            latency_s=latency, scores=scores, error=err,
        ))

    n = len(rows) or 1
    aggs = {"latency_s_mean": sum(r.latency_s for r in rows) / n}
    for m in metrics:
        aggs[m.name + "_mean"] = sum(r.scores[m.name] for r in rows) / n
    report = EvalReport(provider=provider.name, n=len(rows), aggregates=aggs, rows=rows)

    json_path, md_path = write_reports(report, args.out)
    print(f"\nWrote {json_path}\nWrote {md_path}\n\nAggregates:")
    for k, v in aggs.items():
        print(f"  {k}: {v:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
