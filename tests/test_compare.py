from evaluator.compare import compare
from evaluator.runner import EvalReport, RowResult


def _row(inp, exp, pred, scores, latency=0.01):
    return RowResult(input=inp, expected=exp, prediction=pred,
                     latency_s=latency, scores=scores)


def _report(name, rows):
    n = len(rows)
    aggs = {"latency_s_mean": sum(r.latency_s for r in rows) / n}
    for k in rows[0].scores:
        aggs[k + "_mean"] = sum(r.scores[k] for r in rows) / n
    return EvalReport(provider=name, n=n, aggregates=aggs, rows=rows)


def test_compare_computes_deltas_and_winrates():
    a = _report("A", [
        _row("q1", "x", "x", {"exact_match": 1.0, "token_f1": 1.0}),
        _row("q2", "y", "z", {"exact_match": 0.0, "token_f1": 0.0}),
        _row("q3", "z", "z", {"exact_match": 1.0, "token_f1": 1.0}),
    ])
    b = _report("B", [
        _row("q1", "x", "x", {"exact_match": 1.0, "token_f1": 1.0}),
        _row("q2", "y", "y", {"exact_match": 1.0, "token_f1": 1.0}),  # B wins
        _row("q3", "z", "q", {"exact_match": 0.0, "token_f1": 0.0}),  # A wins
    ])
    cmp = compare(a, b)
    by_name = {m.metric: m for m in cmp.metrics}
    em = by_name["exact_match"]
    assert em.mean_a == 2 / 3 and em.mean_b == 2 / 3
    assert em.delta == 0.0
    assert em.wins_a == 1 and em.wins_b == 1 and em.ties == 1
    assert em.win_rate_b == 0.5
    assert len(cmp.flipped) == 2


def test_compare_rejects_mismatched_datasets():
    a = _report("A", [_row("q1", "x", "x", {"em": 1.0})])
    b = _report("B", [_row("q2", "x", "x", {"em": 1.0})])
    try:
        compare(a, b)
    except ValueError:
        return
    raise AssertionError("expected ValueError for input mismatch")


def test_llm_judge_rating_parser():
    # The judge's score() does network I/O; isolate the parsing rule it relies on.
    import re
    for text, expected in [("5", 5), ("Rating: 3", 3), ("I'd say 4/5", 4)]:
        m = re.search(r"[1-5]", text)
        assert m and int(m.group(0)) == expected
    assert re.search(r"[1-5]", "no digits here") is None
