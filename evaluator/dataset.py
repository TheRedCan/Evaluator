import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Example:
    input: str
    expected: str
    context: str | None = None


def load_jsonl(path: str | Path) -> list[Example]:
    rows: list[Example] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "input" not in obj or "expected" not in obj:
                raise ValueError(f"line {i}: missing required keys 'input'/'expected'")
            rows.append(Example(
                input=obj["input"],
                expected=obj["expected"],
                context=obj.get("context"),
            ))
    return rows
