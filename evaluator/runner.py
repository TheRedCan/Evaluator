from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .dataset import Example
from .providers import Provider


@dataclass
class RowResult:
    input: str
    expected: str
    prediction: str
    latency_s: float
    scores: dict[str, float]
    error: str | None = None


@dataclass
class EvalReport:
    provider: str
    n: int
    aggregates: dict[str, float]
    rows: list[RowResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "n": self.n,
            "aggregates": self.aggregates,
            "rows": [asdict(r) for r in self.rows],
        }


def run_eval(provider: Provider, examples: list[Example], metrics: list) -> EvalReport:
    rows: list[RowResult] = []
    for ex in examples:
        t0 = time.perf_counter()
        err: str | None = None
        try:
            pred = provider.generate(ex.input, ex.context)
        except Exception as e:
            pred = ""
            err = f"{type(e).__name__}: {e}"
        latency = time.perf_counter() - t0
        scores = {m.name: m.score(pred, ex.expected, ex.context) for m in metrics}
        rows.append(RowResult(
            input=ex.input, expected=ex.expected, prediction=pred,
            latency_s=latency, scores=scores, error=err,
        ))

    n = len(rows) or 1
    aggregates: dict[str, float] = {"latency_s_mean": sum(r.latency_s for r in rows) / n}
    for m in metrics:
        aggregates[m.name + "_mean"] = sum(r.scores[m.name] for r in rows) / n

    return EvalReport(provider=provider.name, n=len(rows), aggregates=aggregates, rows=rows)
