import json

import pytest

from evaluator.providers import MockProvider, get_provider
from evaluator.run import main as cli_main


def test_mock_provider_uses_context_when_present():
    out = MockProvider().generate("ignored?", context="First sentence. Second.")
    assert out == "First sentence"


def test_mock_provider_falls_back_to_prompt_without_context():
    out = MockProvider().generate("Hello world?")
    assert out == "Hello world."


def test_get_provider_returns_mock():
    assert get_provider("mock").name == "mock"


def test_get_provider_rejects_unknown():
    with pytest.raises(ValueError, match="unknown provider"):
        get_provider("not-a-real-provider")


def _write_dataset(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"input": "q", "expected": "a"}) + "\n", encoding="utf-8")
    return p


def test_cli_run_writes_report(tmp_path):
    dataset = _write_dataset(tmp_path)
    out = tmp_path / "out"
    rc = cli_main(["--dataset", str(dataset), "--provider", "mock", "--out", str(out)])
    assert rc == 0
    assert (out / "report.json").exists()
    assert (out / "report.md").exists()


def test_cli_compare_writes_paired_report(tmp_path):
    dataset = _write_dataset(tmp_path)
    out = tmp_path / "cmp"
    rc = cli_main([
        "compare",
        "--dataset", str(dataset),
        "--provider-a", "mock", "--provider-b", "mock",
        "--out", str(out),
    ])
    assert rc == 0
    assert (out / "compare.json").exists()
    assert (out / "compare.md").exists()
    assert (out / "a" / "report.json").exists()
    assert (out / "b" / "report.json").exists()
