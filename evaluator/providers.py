"""Pluggable LLM providers. Add a new one by implementing `generate(prompt, context)`."""
from __future__ import annotations

import os
from typing import Protocol


class Provider(Protocol):
    name: str
    def generate(self, prompt: str, context: str | None = None) -> str: ...


class MockProvider:
    """Deterministic offline provider.

    If `context` is given, returns the first sentence of the context (a crude
    extractive 'answer') — good enough to exercise the harness end-to-end and
    to produce non-trivial metric values on the sample dataset.
    """
    name = "mock"

    def generate(self, prompt: str, context: str | None = None) -> str:
        if context:
            first = context.split(".")[0].strip()
            return first if first else context.strip()
        return prompt.strip().rstrip("?.") + "."


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise RuntimeError("Install `anthropic` to use AnthropicProvider") from e
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = Anthropic(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, context: str | None = None) -> str:
        user = prompt if not context else f"Context:\n{context}\n\nQuestion: {prompt}"
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            system="Answer concisely using only the provided context when given.",
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text").strip()


def get_provider(name: str, model: str | None = None) -> Provider:
    if name == "mock":
        return MockProvider()
    if name == "anthropic":
        return AnthropicProvider(model=model) if model else AnthropicProvider()
    raise ValueError(f"unknown provider: {name}")
