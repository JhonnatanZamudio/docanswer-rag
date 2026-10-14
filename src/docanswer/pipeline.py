"""Question answering pipeline: retrieve -> confidence gate -> generate -> verify citations."""

from __future__ import annotations

import re
from dataclasses import dataclass

from docanswer.config import Settings
from docanswer.ingest import Index, build_index
from docanswer.llm import REFUSAL, GenerationRequest, LLMProvider, make_provider
from docanswer.models import Answer, Citation, ScoredChunk
from docanswer.retrieval import HybridRetriever, Mode
from docanswer.text import detect_lang, fold

CITATION_RE = re.compile(r"\[([^\[\]]+?) p\.\s?(\d+)\]")
REFUSAL_ES = "No encuentro esta información en los documentos."


def refusal_for(question: str) -> str:
    return REFUSAL_ES if detect_lang(question) == "es" else REFUSAL


def is_refusal(text: str) -> bool:
    folded = fold(text)
    return any(fold(r)[:20] in folded for r in (REFUSAL, REFUSAL_ES))


@dataclass(frozen=True)
class CitationCheck:
    valid: list[ScoredChunk]
    invalid: list[str]


def verify_citations(text: str, contexts: list[ScoredChunk]) -> CitationCheck:
    """Every [Title p.N] tag must point to a passage that was actually retrieved and shown."""
    by_tag: dict[tuple[str, int], ScoredChunk] = {}
    for s in contexts:
        by_tag.setdefault((fold(s.chunk.short_title), s.chunk.page), s)
    valid: list[ScoredChunk] = []
    invalid: list[str] = []
    for match in CITATION_RE.finditer(text):
        hit = by_tag.get((fold(match.group(1).strip()), int(match.group(2))))
        if hit is None:
            invalid.append(match.group(0))
        elif hit not in valid:
            valid.append(hit)
    return CitationCheck(valid, invalid)


def snippet(text: str, limit: int = 240) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


class DocAnswer:
    def __init__(self, index: Index, provider: LLMProvider, settings: Settings) -> None:
        self.settings = settings
        self.provider = provider
        self.retriever = HybridRetriever(index)

    @classmethod
    def from_settings(cls, settings: Settings, rebuild: bool = False) -> DocAnswer:
        if rebuild or not (settings.index_dir / "chunks.jsonl").exists():
            index = build_index(settings)
            index.save(settings.index_dir)
        else:
            index = Index.load(settings.index_dir)
        return cls(index, make_provider(settings.llm_provider, settings.llm_model), settings)

    def ask(self, question: str, role: str, mode: Mode = "hybrid") -> Answer:
        retrieval = self.retriever.search(question, role, k=self.settings.top_k, mode=mode)
        refusal = Answer(
            refusal_for(question), confidence=retrieval.confidence, refused=True, retrieved=retrieval.results
        )
        if not retrieval.results or retrieval.confidence < self.settings.refusal_threshold:
            return refusal  # nothing reliable to ground on: do not even call the model
        text = self.provider.generate(GenerationRequest(question, retrieval.results))
        if is_refusal(text):
            return refusal
        check = verify_citations(text, retrieval.results)
        for bad in check.invalid:  # drop fabricated citations rather than show them
            text = text.replace(bad, "").replace("  ", " ")
        if not check.valid:
            return refusal  # an answer that cites nothing verifiable is not shown
        citations = [Citation(s.chunk.title, s.chunk.doc_id, s.chunk.page, snippet(s.chunk.text)) for s in check.valid]
        return Answer(text.strip(), citations, retrieval.confidence, False, retrieval.results)
