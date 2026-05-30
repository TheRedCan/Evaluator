from evaluator.metrics import ExactMatch, Hallucination, TokenF1


def test_exact_match_normalizes():
    assert ExactMatch().score("Paris.", "paris") == 1.0
    assert ExactMatch().score("London", "Paris") == 0.0


def test_token_f1_partial_overlap():
    f1 = TokenF1().score("the capital is Paris", "Paris")
    assert 0 < f1 < 1


def test_hallucination_unsupported_claim():
    ctx = "Mars is the fourth planet from the Sun."
    pred = "Mars is the fourth planet. It has seventeen moons made of cheese."
    rate = Hallucination().score(pred, "Mars", context=ctx)
    assert rate > 0


def test_hallucination_fully_grounded():
    ctx = "Paris is the capital of France."
    rate = Hallucination().score("Paris is the capital of France.", "Paris", context=ctx)
    assert rate == 0.0
