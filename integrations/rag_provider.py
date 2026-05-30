"""Provider that wraps the engineering-codes RAG system as an Evaluator provider.

Why this re-implements the RAG's `answer_question` loop instead of calling it:

The RAG's `Answer` model only carries `used_chunks: list[str]` — just chunk
IDs, no text. The Hallucination metric needs the *actual text* the LLM was
shown so it can check the answer's claims against it. So we mirror the same
pipeline (retrieve → rerank → LLM) here and capture the reranked chunks'
text before flattening into a prediction string.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

RAG_PROJECT_ROOT = Path(r"E:\imp\GitHub Portfolio\RAGS project")


class RAGProvider:
    name = "engineering-codes-rag"

    def __init__(self) -> None:
        if str(RAG_PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(RAG_PROJECT_ROOT))

        # Lazy imports — pulling these at module import would force the RAG
        # dep graph on anyone who imports `integrations`.
        from common.errors import LlmOutputError  # type: ignore
        from common.models import Answer, Language  # type: ignore
        from generation.llm import chat, health_check  # type: ignore
        from generation.prompt import (  # type: ignore
            SYSTEM_PROMPT,
            build_user_prompt,
            llm_response_schema,
            parse_llm_response,
        )
        from generation.translate import translate_query_for_retrieval  # type: ignore
        from retrieval.multihop import multihop_search  # type: ignore
        from retrieval.rerank import rerank as rerank_chunks  # type: ignore

        self._LlmOutputError = LlmOutputError
        self._Answer = Answer
        self._Language = Language
        self._chat = chat
        self._health_check = health_check
        self._SYSTEM_PROMPT = SYSTEM_PROMPT
        self._build_user_prompt = build_user_prompt
        self._llm_response_schema = llm_response_schema
        self._parse_llm_response = parse_llm_response
        self._translate = translate_query_for_retrieval
        self._multihop_search = multihop_search
        self._rerank = rerank_chunks

        self.last_context: str = ""
        self.last_n_chunks: int = 0
        self.last_answer = None  # type: ignore

    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        # The dataset's `context` field is ignored: the RAG retrieves its own.
        self._health_check()

        # Translation for retrieval; original question goes to the LLM so it
        # answers in the user's language (matches answer_question's contract).
        try:
            retrieval_query, detected_lang = self._translate(prompt)
        except self._LlmOutputError:
            return self._empty_answer_string()

        candidates = self._multihop_search(retrieval_query)
        hop_count = max((c.source_hop for c in candidates), default=0) + 1

        if not candidates:
            self.last_context = ""
            self.last_n_chunks = 0
            return self._empty_answer_string()

        final_candidates = self._rerank(retrieval_query, candidates)

        # Capture the exact text the LLM will see — this is what the
        # Hallucination metric will ground against.
        chunk_texts = [c.chunk.text for c in final_candidates if c.chunk.text]
        self.last_context = "\n\n".join(chunk_texts)
        self.last_n_chunks = len(chunk_texts)

        user_prompt = self._build_user_prompt(prompt, final_candidates)
        raw = self._chat(
            system=self._SYSTEM_PROMPT,
            user=user_prompt,
            json_schema=self._llm_response_schema(),
        )

        try:
            answer = self._parse_llm_response(
                raw, prompt, final_candidates, hop_count=hop_count
            )
        except self._LlmOutputError:
            return self._empty_answer_string()

        self.last_answer = answer
        if not answer.claims:
            return self._empty_answer_string()
        return " ".join(claim.text for claim in answer.claims)

    @staticmethod
    def _empty_answer_string() -> str:
        return "No answer. The indexed sources do not contain enough information."
