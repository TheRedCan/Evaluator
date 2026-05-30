import json

from evaluator.report import write_reports
from evaluator.runner import EvalReport, RowResult


def _make_report():
    rows = [
        RowResult(input="Q1", expected="A1", prediction="A1",
                  latency_s=0.05, scores={"exact_match": 1.0, "token_f1": 1.0}),
        RowResult(input="Q2", expected="A2", prediction="wrong",
                  latency_s=0.07, scores={"exact_match": 0.0, "token_f1": 0.0}),
    ]
    return EvalReport(
        provider="test", n=2,
        aggregates={"latency_s_mean": 0.06, "exact_match_mean": 0.5, "token_f1_mean": 0.5},
        rows=rows,
    )


def test_write_reports_produces_both_files(tmp_path):
    json_path, md_path = write_reports(_make_report(), tmp_path)
    assert json_path.exists() and md_path.exists()
    assert json_path.name == "report.json"
    assert md_path.name == "report.md"


def test_report_json_round_trip(tmp_path):
    rep = _make_report()
    json_path, _ = write_reports(rep, tmp_path)
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["provider"] == "test"
    assert loaded["n"] == 2
    assert len(loaded["rows"]) == 2
    assert loaded["aggregates"]["exact_match_mean"] == 0.5


def test_report_markdown_contains_aggregates_and_rows(tmp_path):
    _, md_path = write_reports(_make_report(), tmp_path)
    md = md_path.read_text(encoding="utf-8")
    assert "exact_match_mean" in md
    assert "Q1" in md and "Q2" in md
    assert "0.0500" in md or "0.050" in md or "0.06" in md  # latency rendered


def test_report_markdown_escapes_pipes(tmp_path):
    rep = _make_report()
    rep.rows[0].prediction = "value | with | pipes"
    _, md_path = write_reports(rep, tmp_path)
    md = md_path.read_text(encoding="utf-8")
    # The escaping must keep the table well-formed — count of pipes per row
    # should match the column count (6 separators + 2 edges = 7).
    table_lines = [ln for ln in md.splitlines() if ln.startswith("| 1 |")]
    assert table_lines, "row 1 missing from markdown"
    assert table_lines[0].count("|") - table_lines[0].count("\\|") == 7
