"""A/B comparison between two eval runs over the same dataset.

The point of an eval harness is to make changes measurable. This module pairs
two `EvalReport`s row-by-row and reports:

  - aggregate delta per metric (B minus A)
  - win/tie/loss counts per metric (which side scored higher on each row)
  - rows where the verdict flipped (useful for qualitative review)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .runner import EvalReport


@dataclass
class MetricComparison:
    metric: str
    mean_a: float
    mean_b: float
    delta: float           # B - A
    wins_b: int            # rows where B scored strictly higher
    wins_a: int
    ties: int
    win_rate_b: float      # wins_b / decided_rows (excludes ties)


@dataclass
class RowComparison:
    input: str
    expected: str
    prediction_a: str
    prediction_b: str
    scores_a: dict[str, float]
    scores_b: dict[str, float]


@dataclass
class CompareReport:
    provider_a: str
    provider_b: str
    n: int
    metrics: list[MetricComparison]
    flipped: list[RowComparison]   # rows where any metric's winner differs
    rows: list[RowComparison]

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_a": self.provider_a,
            "provider_b": self.provider_b,
            "n": self.n,
            "metrics": [asdict(m) for m in self.metrics],
            "flipped": [asdict(r) for r in self.flipped],
            "rows": [asdict(r) for r in self.rows],
        }


def compare(report_a: EvalReport, report_b: EvalReport) -> CompareReport:
    if report_a.n != report_b.n:
        raise ValueError(
            f"row count mismatch: A has {report_a.n}, B has {report_b.n}. "
            "Run both providers on the same dataset."
        )

    rows: list[RowComparison] = []
    flipped: list[RowComparison] = []
    for ra, rb in zip(report_a.rows, report_b.rows, strict=False):
        if ra.input != rb.input:
            raise ValueError(f"input mismatch on paired row: {ra.input!r} vs {rb.input!r}")
        row = RowComparison(
            input=ra.input,
            expected=ra.expected,
            prediction_a=ra.prediction,
            prediction_b=rb.prediction,
            scores_a=dict(ra.scores),
            scores_b=dict(rb.scores),
        )
        rows.append(row)
        if any(ra.scores[k] != rb.scores[k] for k in ra.scores):
            flipped.append(row)

    metric_names = list(report_a.rows[0].scores.keys()) if report_a.rows else []
    metrics: list[MetricComparison] = []
    for name in metric_names:
        a_vals = [r.scores[name] for r in report_a.rows]
        b_vals = [r.scores[name] for r in report_b.rows]
        mean_a = sum(a_vals) / len(a_vals)
        mean_b = sum(b_vals) / len(b_vals)
        wins_b = sum(1 for a, b in zip(a_vals, b_vals, strict=False) if b > a)
        wins_a = sum(1 for a, b in zip(a_vals, b_vals, strict=False) if a > b)
        ties = len(a_vals) - wins_a - wins_b
        decided = wins_a + wins_b
        metrics.append(MetricComparison(
            metric=name,
            mean_a=mean_a, mean_b=mean_b, delta=mean_b - mean_a,
            wins_a=wins_a, wins_b=wins_b, ties=ties,
            win_rate_b=(wins_b / decided) if decided else 0.0,
        ))

    return CompareReport(
        provider_a=report_a.provider, provider_b=report_b.provider,
        n=report_a.n, metrics=metrics, flipped=flipped, rows=rows,
    )


def render_markdown(cmp: CompareReport) -> str:
    lines: list[str] = []
    lines.append(f"# A/B Comparison — `{cmp.provider_a}` vs `{cmp.provider_b}`")
    lines.append("")
    lines.append(f"**Examples:** {cmp.n}  ·  **Rows with score differences:** {len(cmp.flipped)}")
    lines.append("")
    lines.append("## Per-metric comparison")
    lines.append("")
    lines.append("| Metric | A mean | B mean | Δ (B−A) | A wins | B wins | Ties | B win-rate |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for m in cmp.metrics:
        arrow = "↑" if m.delta > 0 else ("↓" if m.delta < 0 else "·")
        lines.append(
            f"| {m.metric} | {m.mean_a:.4f} | {m.mean_b:.4f} | {arrow} {m.delta:+.4f} "
            f"| {m.wins_a} | {m.wins_b} | {m.ties} | {m.win_rate_b:.2%} |"
        )
    lines.append("")
    if cmp.flipped:
        lines.append("## Rows where scores differ")
        lines.append("")
        lines.append("| # | Input | A prediction | B prediction |")
        lines.append("|---|---|---|---|")
        for i, r in enumerate(cmp.flipped, start=1):
            lines.append(
                f"| {i} | {_clip(r.input)} | {_clip(r.prediction_a)} | {_clip(r.prediction_b)} |"
            )
    return "\n".join(lines) + "\n"


def _clip(s: str, n: int = 60) -> str:
    s = s.replace("|", "\\|").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"
