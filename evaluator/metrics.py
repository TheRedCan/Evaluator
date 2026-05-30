"""Scoring metrics. Each metric returns a float in [0, 1] except Latency (seconds)."""
from __future__ import annotations

import re
import string
from dataclasses import dataclass

_WORD_RE = re.compile(r"\w+")


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s)


def _tokens(s: str) -> list[str]:
    return _WORD_RE.findall(s.lower())


@dataclass
class MetricResult:
    name: str
    value: float


class ExactMatch:
    name = "exact_match"
    def score(self, prediction: str, expected: str, context: str | None = None) -> float:
        return 1.0 if _normalize(prediction) == _normalize(expected) else 0.0


class TokenF1:
    """Token-overlap F1 — tolerant of phrasing differences."""
    name = "token_f1"
    def score(self, prediction: str, expected: str, context: str | None = None) -> float:
        pred, gold = _tokens(prediction), _tokens(expected)
        if not pred and not gold:
            return 1.0
        if not pred or not gold:
            return 0.0
        common: dict[str, int] = {}
        gold_counts: dict[str, int] = {}
        for t in gold:
            gold_counts[t] = gold_counts.get(t, 0) + 1
        for t in pred:
            if gold_counts.get(t, 0) > 0:
                common[t] = common.get(t, 0) + 1
                gold_counts[t] -= 1
        overlap = sum(common.values())
        if overlap == 0:
            return 0.0
        precision = overlap / len(pred)
        recall = overlap / len(gold)
        return 2 * precision * recall / (precision + recall)


class Hallucination:
    """Fraction of answer sentences NOT supported by the provided context.

    Lexical grounding: a sentence is 'supported' if a sufficient share of its
    content tokens appear in the context. Crude vs. an LLM judge, but fast,
    free, and deterministic — and good enough to flag obvious fabrications.
    """
    name = "hallucination_rate"

    def __init__(self, support_threshold: float = 0.6, min_content_tokens: int = 2):
        self.support_threshold = support_threshold
        self.min_content_tokens = min_content_tokens

    def score(self, prediction: str, expected: str, context: str | None = None) -> float:
        if not context:
            return 0.0  # nothing to ground against — skip
        ctx_tokens = set(_tokens(context))
        sentences = [s.strip() for s in re.split(r"[.!?]+", prediction) if s.strip()]
        if not sentences:
            return 0.0
        unsupported = 0
        scored = 0
        for sent in sentences:
            content = [t for t in _tokens(sent) if len(t) > 2]
            if len(content) < self.min_content_tokens:
                continue
            overlap = sum(1 for t in content if t in ctx_tokens) / len(content)
            scored += 1
            if overlap < self.support_threshold:
                unsupported += 1
        return unsupported / scored if scored else 0.0


class EmbeddingSimilarity:
    """Cosine similarity between sentence embeddings of prediction and expected.

    Uses `sentence-transformers` lazily so the harness keeps zero hard deps.
    Model is loaded once and reused across calls.
    """
    name = "embedding_similarity"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError(
                    "EmbeddingSimilarity requires `sentence-transformers`. "
                    "Install with: pip install sentence-transformers"
                ) from e
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def score(self, prediction: str, expected: str, context: str | None = None) -> float:
        if not prediction.strip() or not expected.strip():
            return 0.0
        model = self._load()
        vecs = model.encode([prediction, expected], normalize_embeddings=True)
        # Normalized vectors → dot product == cosine similarity, in [-1, 1].
        sim = float((vecs[0] * vecs[1]).sum())
        # Clamp to [0, 1] so it composes with the other metrics.
        return max(0.0, min(1.0, (sim + 1) / 2)) if sim < 0 else min(1.0, sim)


class LLMJudge:
    """LLM-as-judge: a second model rates the answer on a 1-5 scale, normalized to [0, 1].

    Defaults to grading *faithfulness* (is the prediction supported by the
    context and consistent with the expected answer). Pass a custom `criteria`
    string to grade something else (helpfulness, conciseness, etc.).

    Requires `ANTHROPIC_API_KEY`. Failures degrade gracefully to 0.0 with the
    error captured on the metric instance for inspection.
    """
    name = "llm_judge"

    DEFAULT_CRITERIA = (
        "Faithfulness and correctness. A 5 means the answer is fully correct "
        "and entirely supported by the context; a 1 means it is wrong or "
        "fabricated. Penalize unsupported claims even if they sound plausible."
    )

    JUDGE_PROMPT = """You are a strict evaluator. Rate the ANSWER on a 1-5 integer scale.

Criteria: {criteria}

QUESTION:
{question}

CONTEXT (ground truth source, may be empty):
{context}

REFERENCE ANSWER:
{expected}

ANSWER TO GRADE:
{prediction}

Respond with ONLY a single integer 1-5. No explanation, no other text."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", criteria: str | None = None):
        self.model = model
        self.criteria = criteria or self.DEFAULT_CRITERIA
        self._client = None
        self.last_error: str | None = None

    def _get_client(self):
        if self._client is None:
            import os
            try:
                from anthropic import Anthropic
            except ImportError as e:
                raise RuntimeError("Install `anthropic` to use LLMJudge") from e
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            self._client = Anthropic()
        return self._client

    def score(self, prediction: str, expected: str, context: str | None = None) -> float:
        try:
            client = self._get_client()
            prompt = self.JUDGE_PROMPT.format(
                criteria=self.criteria,
                question="(see reference)",
                context=context or "(none provided)",
                expected=expected,
                prediction=prediction,
            )
            resp = client.messages.create(
                model=self.model,
                max_tokens=8,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            # Extract first digit 1-5 from the response.
            m = re.search(r"[1-5]", text)
            if not m:
                self.last_error = f"unparseable judge response: {text!r}"
                return 0.0
            rating = int(m.group(0))
            return (rating - 1) / 4  # 1→0.0, 5→1.0
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return 0.0


DEFAULT_METRICS = [ExactMatch(), TokenF1(), Hallucination()]
