from evaluator.dataset import Example
from evaluator.metrics import ExactMatch, TokenF1
from evaluator.runner import run_eval


class _EchoProvider:
    name = "echo"
    def generate(self, prompt, context=None):
        return prompt


class _FixedProvider:
    name = "fixed"
    def __init__(self, output):
        self._output = output
    def generate(self, prompt, context=None):
        return self._output


class _RaisingProvider:
    name = "broken"
    def generate(self, prompt, context=None):
        raise RuntimeError("provider exploded")


def test_run_eval_scores_each_row_with_each_metric():
    examples = [Example(input="Paris", expected="Paris"),
                Example(input="London", expected="Tokyo")]
    rep = run_eval(_EchoProvider(), examples, [ExactMatch(), TokenF1()])
    assert rep.n == 2
    assert rep.rows[0].scores["exact_match"] == 1.0
    assert rep.rows[1].scores["exact_match"] == 0.0


def test_run_eval_aggregates_are_means():
    examples = [Example(input="a", expected="a"), Example(input="b", expected="x")]
    rep = run_eval(_EchoProvider(), examples, [ExactMatch()])
    assert rep.aggregates["exact_match_mean"] == 0.5
    assert "latency_s_mean" in rep.aggregates


def test_run_eval_captures_provider_errors_without_crashing():
    rep = run_eval(_RaisingProvider(), [Example(input="x", expected="y")], [ExactMatch()])
    assert rep.rows[0].error and "RuntimeError" in rep.rows[0].error
    assert rep.rows[0].prediction == ""
    assert rep.rows[0].scores["exact_match"] == 0.0


def test_run_eval_records_latency_per_row():
    rep = run_eval(_FixedProvider("ok"), [Example(input="x", expected="ok")], [ExactMatch()])
    assert rep.rows[0].latency_s >= 0.0
