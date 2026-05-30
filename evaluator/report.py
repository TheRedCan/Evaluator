from __future__ import annotations

import json
from pathlib import Path

from .runner import EvalReport


def write_reports(report: EvalReport, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "report.json"
    md_path = out / "report.md"

    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _render_markdown(report: EvalReport) -> str:
    lines: list[str] = []
    lines.append(f"# Eval Report — provider: `{report.provider}`")
    lines.append("")
    lines.append(f"**Examples:** {report.n}")
    lines.append("")
    lines.append("## Aggregates")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for k, v in report.aggregates.items():
        lines.append(f"| {k} | {v:.4f} |")
    lines.append("")
    lines.append("## Per-row")
    lines.append("")
    lines.append("| # | Input | Expected | Prediction | Latency (s) | Scores |")
    lines.append("|---|---|---|---|---|---|")
    for i, r in enumerate(report.rows, start=1):
        scores = ", ".join(f"{k}={v:.2f}" for k, v in r.scores.items())
        lines.append(
            f"| {i} | {_clip(r.input)} | {_clip(r.expected)} | {_clip(r.prediction)} "
            f"| {r.latency_s:.3f} | {scores} |"
        )
    return "\n".join(lines) + "\n"


def _clip(s: str, n: int = 80) -> str:
    s = s.replace("|", "\\|").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"
