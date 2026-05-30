import json
from pathlib import Path

import pytest

from evaluator.dataset import Example, load_jsonl


def _write(tmp_path: Path, lines: list[str]) -> Path:
    p = tmp_path / "data.jsonl"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def test_load_jsonl_parses_required_fields(tmp_path):
    p = _write(tmp_path, [json.dumps({"input": "Q", "expected": "A"})])
    rows = load_jsonl(p)
    assert rows == [Example(input="Q", expected="A", context=None)]


def test_load_jsonl_keeps_optional_context(tmp_path):
    p = _write(tmp_path, [json.dumps({"input": "Q", "expected": "A", "context": "C"})])
    assert load_jsonl(p)[0].context == "C"


def test_load_jsonl_skips_blank_lines(tmp_path):
    p = _write(tmp_path, [
        json.dumps({"input": "Q1", "expected": "A1"}),
        "",
        "   ",
        json.dumps({"input": "Q2", "expected": "A2"}),
    ])
    rows = load_jsonl(p)
    assert [r.input for r in rows] == ["Q1", "Q2"]


def test_load_jsonl_rejects_missing_keys(tmp_path):
    p = _write(tmp_path, [json.dumps({"input": "Q"})])  # no expected
    with pytest.raises(ValueError, match="missing required keys"):
        load_jsonl(p)


def test_load_jsonl_rejects_malformed_json(tmp_path):
    p = _write(tmp_path, ["{not valid json}"])
    with pytest.raises(json.JSONDecodeError):
        load_jsonl(p)
