"""Answer generation behind an `LLMProvider` protocol.

* ExtractiveProvider - offline default. Selects the passages' most relevant sentences verbatim
  and cites them. No generation, so it cannot hallucinate (but it cannot paraphrase either).
* AnthropicProvider / OpenAIProvider - hosted models, opt-in via LLM_PROVIDER. SDKs are imported
  lazily and clients can be injected, so tests never touch the network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from docanswer.models import ScoredChunk
from docanswer.text import detect_lang, split_sentences, tokenize

REFUSAL = "I can't find this in the documents."

SYSTEM_PROMPT = f"""You are DocAnswer, an internal assistant that answers employee questions using ONLY \
the company documents provided in <context>.
Rules:
1. Use only facts stated in the context. Never use outside knowledge, never guess.
2. After every sentence that states a fact, add its citation tag exactly as written in the context, \
e.g. [Travel Policy p.3]. Do not invent tags.
3. If the context does not contain the answer, reply exactly: "{REFUSAL}"
4. Answer in the language of the question. Be concise: at most 4 sentences."""


def render_context(contexts: list[ScoredChunk]) -> str:
    blocks = [
        f'<passage tag="{s.chunk.citation_tag}" section="{s.chunk.heading}">\n{s.chunk.text}\n</passage>'
        for s in contexts
    ]
    return "<context>\n" + "\n".join(blocks) + "\n</context>"


def render_user_prompt(question: str, contexts: list[ScoredChunk]) -> str:
    return f"{render_context(contexts)}\n\nQuestion: {question}"


@dataclass(frozen=True)
class GenerationRequest:
    question: str
    contexts: list[ScoredChunk]

    @property
    def system(self) -> str:
        return SYSTEM_PROMPT

    @property
    def user(self) -> str:
        return render_user_prompt(self.question, self.contexts)


class LLMProvider(Protocol):
    name: str

    def generate(self, request: GenerationRequest) -> str: ...


class ExtractiveProvider:
    name = "extractive"

    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    def generate(self, request: GenerationRequest) -> str:
        q_terms = set(tokenize(request.question, detect_lang(request.question)))
        candidates: list[tuple[float, int, str, str]] = []
        for s in request.contexts:
            heading_terms = set(tokenize(s.chunk.heading, s.chunk.lang))
            for i, sentence in enumerate(split_sentences(s.chunk.text)):
                # The section heading is part of each sentence's context ("1. Your first day").
                overlap = len(q_terms & (set(tokenize(sentence, s.chunk.lang)) | heading_terms))
                if overlap:
                    # Prefer term overlap, then the retriever's ranking, then document order.
                    candidates.append((overlap - 0.1 * (s.rank - 1), -i, sentence, s.chunk.citation_tag))
        if not candidates:
            return REFUSAL
        candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
        best = candidates[0][0]
        # Extra sentences must be at least half as relevant as the best one (precision over recall).
        picked = [c for c in candidates[: self.max_sentences] if c[0] >= best / 2]
        return " ".join(f"{sentence} {tag}" for _, _, sentence, tag in picked)


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str | None = None, client: object | None = None) -> None:
        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY")
            import anthropic  # optional dependency: pip install '.[llm]'

            client = anthropic.Anthropic()
        self._client = client
        self.model = model or os.environ.get("LLM_MODEL") or "claude-sonnet-4-5"

    def generate(self, request: GenerationRequest) -> str:
        message = self._client.messages.create(  # type: ignore[attr-defined]
            model=self.model,
            max_tokens=600,
            temperature=0,
            system=request.system,
            messages=[{"role": "user", "content": request.user}],
        )
        return "".join(block.text for block in message.content if getattr(block, "type", "") == "text").strip()


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str | None = None, client: object | None = None) -> None:
        if client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("LLM_PROVIDER=openai requires OPENAI_API_KEY")
            from openai import OpenAI  # optional dependency: pip install '.[llm]'

            client = OpenAI()
        self._client = client
        self.model = model or os.environ.get("LLM_MODEL") or "gpt-4o-mini"

    def generate(self, request: GenerationRequest) -> str:
        response = self._client.chat.completions.create(  # type: ignore[attr-defined]
            model=self.model,
            temperature=0,
            messages=[{"role": "system", "content": request.system}, {"role": "user", "content": request.user}],
        )
        return (response.choices[0].message.content or "").strip()


def make_provider(name: str, model: str = "") -> LLMProvider:
    providers = {"extractive": ExtractiveProvider, "anthropic": AnthropicProvider, "openai": OpenAIProvider}
    if name not in providers:
        raise ValueError(f"unknown LLM provider: {name!r}")
    if name == "extractive":
        return ExtractiveProvider()
    return providers[name](model=model or None)  # type: ignore[call-arg]
