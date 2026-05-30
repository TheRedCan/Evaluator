"""CLI entrypoints.

Single-run mode:
    python -m evaluator.run --dataset data/sample.jsonl --provider mock --out reports/

A/B comparison mode (runs both providers on the same dataset):
    python -m evaluator.run compare \\
        --dataset data/sample.jsonl \\
        --provider-a mock --provider-b anthropic --model-b claude-haiku-4-5-20251001 \\
        --out reports/compare/

Add optional metrics:
    --with-embedding   # cosine similarity (sentence-transformers)
    --with-llm-judge   # LLM-as-judge (Anthropic)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .compare import compare, render_markdown
from .dataset import load_jsonl
from .metrics import DEFAULT_METRICS, EmbeddingSimilarity, LLMJudge
from .providers import get_provider
from .report import write_reports
from .runner import run_eval


def _build_metrics(with_embedding: bool, with_llm_judge: bool, judge_model: str | None):
    metrics = list(DEFAULT_METRICS)
    if with_embedding:
        metrics.append(EmbeddingSimilarity())
    if with_llm_judge:
        metrics.append(LLMJudge(model=judge_model) if judge_model else LLMJudge())
    return metrics


def _add_metric_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--with-embedding", action="store_true",
                   help="Add embedding cosine similarity (requires sentence-transformers)")
    p.add_argument("--with-llm-judge", action="store_true",
                   help="Add LLM-as-judge metric (requires ANTHROPIC_API_KEY)")
    p.add_argument("--judge-model", default=None, help="Judge model id (defaults to Haiku)")


def _cmd_run(args: argparse.Namespace) -> int:
    examples = load_jsonl(args.dataset)
    provider = get_provider(args.provider, args.model)
    metrics = _build_metrics(args.with_embedding, args.with_llm_judge, args.judge_model)

    print(f"Running {len(examples)} examples through provider={provider.name}...")
    report = run_eval(provider, examples, metrics)
    json_path, md_path = write_reports(report, args.out)
    print(f"Wrote {json_path}\nWrote {md_path}\n\nAggregates:")
    for k, v in report.aggregates.items():
        print(f"  {k}: {v:.4f}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    examples = load_jsonl(args.dataset)
    provider_a = get_provider(args.provider_a, args.model_a)
    provider_b = get_provider(args.provider_b, args.model_b)
    metrics = _build_metrics(args.with_embedding, args.with_llm_judge, args.judge_model)

    print(f"Running {len(examples)} examples through A={provider_a.name}...")
    report_a = run_eval(provider_a, examples, metrics)
    print(f"Running {len(examples)} examples through B={provider_b.name}...")
    report_b = run_eval(provider_b, examples, metrics)

    cmp = compare(report_a, report_b)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_reports(report_a, out / "a")
    write_reports(report_b, out / "b")
    (out / "compare.json").write_text(json.dumps(cmp.to_dict(), indent=2), encoding="utf-8")
    (out / "compare.md").write_text(render_markdown(cmp), encoding="utf-8")
    print(f"\nWrote {out / 'compare.md'}")
    print("\nDeltas (B - A):")
    for m in cmp.metrics:
        print(f"  {m.metric:24s}  delta {m.delta:+.4f}   B win-rate {m.win_rate_b:.1%}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evaluator", description="LLM evaluation harness.")
    sub = p.add_subparsers(dest="cmd")

    # Default single-run flags live on the root parser so the original
    # invocation `python -m evaluator.run --dataset ...` still works.
    p.add_argument("--dataset", help="Path to JSONL dataset")
    p.add_argument("--provider", default="mock", choices=["mock", "anthropic"])
    p.add_argument("--model", default=None)
    p.add_argument("--out", default="reports/")
    _add_metric_flags(p)

    pc = sub.add_parser("compare", help="A/B compare two providers on the same dataset")
    pc.add_argument("--dataset", required=True)
    pc.add_argument("--provider-a", default="mock", choices=["mock", "anthropic"])
    pc.add_argument("--provider-b", default="anthropic", choices=["mock", "anthropic"])
    pc.add_argument("--model-a", default=None)
    pc.add_argument("--model-b", default=None)
    pc.add_argument("--out", default="reports/compare/")
    _add_metric_flags(pc)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "compare":
        return _cmd_compare(args)
    if not args.dataset:
        parser.error("--dataset is required")
    return _cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
