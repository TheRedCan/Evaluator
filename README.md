# Evaluator — A Minimal LLM Evaluation Harness

A from-scratch LLM evaluation framework that benchmarks model outputs against
labeled datasets, reporting **accuracy**, **hallucination rate**, **latency**,
**embedding similarity**, and **LLM-as-judge** scores — with first-class
**A/B comparison** between providers, prompts, or models.

Built to demonstrate end-to-end understanding of how LLM evaluation actually
works under the hood. See [How this relates to DeepEval / RAGAS](#how-this-relates-to-deepeval--ragas) below.

[![tests](https://github.com/TheRedCan/Evaluator/actions/workflows/test.yml/badge.svg)](https://github.com/TheRedCan/Evaluator/actions/workflows/test.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## What it does

1. Loads a dataset of `{input, expected, context?}` rows from JSONL.
2. Runs each row through a pluggable **Provider** (mock by default, or Anthropic).
3. Scores each output with a set of pluggable **Metrics**.
4. Writes a `report.json` + `report.md` — and in **compare mode**, a paired
   `compare.json` + `compare.md` with per-metric deltas, win rates, and flipped
   rows.

## Metrics

| Metric                  | Type        | What it measures                                            | Dependencies |
|-------------------------|-------------|--------------------------------------------------------------|--------------|
| `ExactMatch`            | Accuracy    | Normalized string equality                                   | stdlib |
| `TokenF1`               | Accuracy    | Token-overlap F1 vs. reference                               | stdlib |
| `Hallucination`         | Faithfulness| Fraction of answer sentences unsupported by the context     | stdlib |
| `EmbeddingSimilarity`   | Semantic    | Cosine similarity between sentence embeddings                | `sentence-transformers` |
| `LLMJudge`              | Holistic    | 1–5 rating from a second model, normalized to [0,1]          | `anthropic` + `ANTHROPIC_API_KEY` |
| `latency_s`             | Performance | Wall-clock seconds per call                                  | stdlib |

The core harness has **zero required dependencies**. Embedding and LLM-judge
metrics are opt-in via CLI flags, so the framework stays runnable offline by
default.

## Quick start

```bash
pip install -e .

# Offline, zero-dep run with the mock provider
python -m evaluator.run --dataset data/sample.jsonl --provider mock --out reports/

# Against Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
python -m evaluator.run --dataset data/sample.jsonl --provider anthropic --out reports/

# Add the heavyweight metrics (need optional deps)
pip install -e ".[anthropic,embedding]"
python -m evaluator.run --dataset data/sample.jsonl --provider anthropic \
    --with-embedding --with-llm-judge --out reports/
```

## A/B comparison

The point of an eval harness is making changes measurable. `compare` runs two
providers on the same dataset and produces:

```bash
python -m evaluator.run compare \
    --dataset data/sample.jsonl \
    --provider-a mock \
    --provider-b anthropic --model-b claude-haiku-4-5-20251001 \
    --with-embedding \
    --out reports/compare/
```

```
reports/compare/
  a/report.{json,md}      # full per-row results for provider A
  b/report.{json,md}      # full per-row results for provider B
  compare.{json,md}       # paired deltas, win rates, flipped rows
```

Sample output:

```
| Metric             | A mean | B mean | Δ (B−A)   | A wins | B wins | Ties | B win-rate |
| exact_match        | 0.0000 | 0.8000 | ↑ +0.8000 |   0    |   4    |   1  |   100.00%  |
| token_f1           | 0.0986 | 0.7321 | ↑ +0.6335 |   0    |   5    |   0  |   100.00%  |
| hallucination_rate | 0.0000 | 0.0000 | · +0.0000 |   0    |   0    |   5  |     0.00%  |
```

## Case study: evaluating a local engineering-codes RAG system

The harness was tested end-to-end against a separate project — a local
bilingual (Arabic/English) RAG system for building-code documents (BGE-M3 +
Qdrant + Qwen-7B via Ollama). The integration code lives in
[`integrations/`](integrations/).

The RAG was run on three representative queries:

| # | Question | Latency | Hallucination | Notes |
|---|---|---|---|---|
| 1 | "What are the load combinations for seismic design in ASCE 7-22?" | 96s | **0.00** | Fully grounded in retrieved chunks; quoted real ASCE 7-22 equation `0.9D − Ev + Emh`. |
| 2 | "What is soil-structure interaction?" | 75s | **0.00** | Substantive definition with internal source quotes; every claim cited. |
| 3 | "What does ASCE 7-22 §99.99.99 require?" *(made-up section)* | 70s | **0.50** | RAG correctly refused, but the metric flagged half the sentences as ungrounded — see [Known limitations](#known-limitations). |

**Aggregate hallucination rate: 16.7%**, entirely concentrated in Q3.

This is the result that made the project actually useful: it surfaced a real
distinction between *true hallucination* (inventing facts) and
*meta-commentary about absence* ("Section X doesn't exist") that a lexical
grounding metric can't tell apart. See the
[Hallucination metric limitation](#known-limitations) for the writeup.

## Dataset format

One JSON object per line:

```json
{"input": "What is the capital of France?", "expected": "Paris", "context": "France is a country in Europe. Its capital is Paris."}
```

`context` is optional; the `Hallucination` metric is skipped (returns 0) when
absent.

## Architecture

```
evaluator/
  providers.py   # MockProvider, AnthropicProvider
  metrics.py     # ExactMatch, TokenF1, Hallucination, EmbeddingSimilarity, LLMJudge
  dataset.py     # JSONL loader
  runner.py      # eval loop + latency timing
  report.py      # report.md + report.json
  compare.py     # A/B comparison: deltas, win-rates, flipped rows
  run.py         # CLI (run + compare subcommands)
integrations/
  rag_provider.py    # Example integration: wraps an external RAG system as a Provider
  run_rag_eval.py    # RAG-aware runner that feeds cited chunks to Hallucination
data/
  sample.jsonl       # 5-row demo dataset
  rag_subset.jsonl   # 3-row RAG case study
tests/                 # 26 tests across metrics, runner, report, dataset, CLI
```

### Extension points

| Want to... | Do this |
|---|---|
| Add a new model/API | Implement `generate(prompt, context) -> str` in `providers.py` and register it in `get_provider`. |
| Add a new metric | Add a class with `name: str` and `score(prediction, expected, context) -> float` to `metrics.py`. |
| Customize the LLM judge | `LLMJudge(criteria="Conciseness. A 5 means ...")` |
| Evaluate a RAG / agent system | Mirror [`integrations/rag_provider.py`](integrations/rag_provider.py) — capture the retrieved context per call so `Hallucination` can ground against it. |

## Running tests

```bash
pip install -e ".[dev]"
pytest                          # 26 tests, ~0.2s
pytest --cov=evaluator          # with coverage
ruff check evaluator tests      # lint
```

CI runs the full suite on push against Python 3.10, 3.11, and 3.12 —
see [`.github/workflows/test.yml`](.github/workflows/test.yml).

## Known limitations

**1. `Hallucination` cannot distinguish true fabrications from refusal language.**
The metric scores an answer sentence as "grounded" if a sufficient share of
its content tokens appears in the provided context. This works well for
substantive answers, but a *correct refusal* like "Section 99.99.99 does not
exist in the provided sources" uses vocabulary that — by definition — isn't
in the context. The metric flagged Q3 of the case study at 50% hallucination
even though the RAG's behavior was correct.

A more sophisticated grounding metric would distinguish:
- **True hallucination** = inventing facts that contradict or aren't in source.
- **Meta-commentary about absence** = stating that something isn't in source.

In practice this means: don't read `hallucination_rate_mean` as a single
quality score. Read it alongside `exact_match` / `LLMJudge` so refusals don't
inflate the rate.

**2. `TokenF1` undervalues paraphrased correct answers.**
A model that says "the capital city is Paris" scores ~0.4 against the gold
"Paris". For datasets where wording matters less than meaning, prefer
`EmbeddingSimilarity` or `LLMJudge`. Token-F1 is most useful as a *floor*
metric — answers that score 0 on token-F1 are almost certainly wrong.

**3. The default `Hallucination` is lexical, not semantic.**
A paraphrase that preserves meaning but uses different words may be flagged
as ungrounded. The metric is meant as a fast, deterministic, offline check —
swap in an embedding-based or LLM-judged grounding metric when you need
phrasing tolerance.

## How this relates to DeepEval / RAGAS

This is **intentionally a from-scratch implementation**, not a competitor.

- [DeepEval](https://github.com/confident-ai/deepeval) gives you pytest-style
  assertions over LLM outputs with ~40 built-in metrics and tight CI/CD
  integration.
- [RAGAS](https://github.com/explodinggradients/ragas) is RAG-specific:
  faithfulness, answer relevancy, context precision, context recall.

Both are excellent and production-grade. **Use them for real work.**

This project exists to show the mechanics underneath:
- What "lexical vs. semantic vs. holistic" actually means at the code level.
- How LLM-as-judge is wired (system prompt, output parsing, failure modes).
- Why a/b comparison — not single-run scoring — is the real value of an
  eval harness.
- The non-obvious limitations of common metrics (see the case study above).

If you're learning the space, this is small enough to read in one sitting
(~400 lines of metric/runner/compare code). For production, reach for
DeepEval or RAGAS.

## License

MIT — see [LICENSE](LICENSE).
