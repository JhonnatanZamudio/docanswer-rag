"""Hybrid retrieval: BM25 + dense vectors fused with Reciprocal Rank Fusion.

Access control is applied BEFORE ranking: each clearance level gets its own BM25 index built
only from chunks it may read, so restricted text never influences a lower-clearance query,
not even through term statistics. Dense scores are only computed over permitted rows.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi

from docanswer.ingest import Index
from docanswer.models import AccessLevel, ScoredChunk, clearance_for
from docanswer.text import detect_lang, tokenize

Mode = Literal["bm25", "dense", "hybrid"]
RRF_K = 60
CANDIDATES = 20


@dataclass(frozen=True)
class RetrievalResult:
    results: list[ScoredChunk]
    confidence: float


class _Partition:
    """Everything a single clearance level is allowed to search."""

    def __init__(self, rows: list[int], tokens: list[list[str]]) -> None:
        self.rows = np.array(rows, dtype=int)
        self.tokens = [set(t) for t in tokens]
        self.bm25 = BM25Okapi(tokens) if tokens else None
        df = Counter(term for doc in self.tokens for term in doc)
        n = len(tokens)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.max_idf = math.log(1 + (n + 0.5) / 0.5) if n else 1.0


class HybridRetriever:
    def __init__(self, index: Index) -> None:
        self.index = index
        tokens = [tokenize(c.index_text, c.lang) for c in index.chunks]
        self._partitions: dict[AccessLevel, _Partition] = {}
        for level in AccessLevel:
            rows = [i for i, c in enumerate(index.chunks) if c.access_level <= level]
            self._partitions[level] = _Partition(rows, [tokens[i] for i in rows])

    @property
    def has_dense(self) -> bool:
        return self.index.embedder is not None and self.index.embeddings is not None

    def search(self, query: str, role: str, k: int = 5, mode: Mode = "hybrid") -> RetrievalResult:
        part = self._partitions[clearance_for(role)]
        q_tokens = tokenize(query, detect_lang(query))
        if part.bm25 is None or not q_tokens:
            return RetrievalResult([], 0.0)
        rankings: list[list[int]] = []  # positions within the partition, best first
        if mode in ("bm25", "hybrid"):
            scores = part.bm25.get_scores(q_tokens)
            rankings.append([int(i) for i in np.argsort(-scores)[:CANDIDATES] if scores[i] > 0])
        if mode in ("dense", "hybrid") and self.has_dense:
            q_vec = self.index.embedder.embed([query])[0]  # type: ignore[union-attr]
            sims = self.index.embeddings[part.rows] @ q_vec  # type: ignore[index]
            rankings.append([int(i) for i in np.argsort(-sims)[:CANDIDATES]])
        fused = reciprocal_rank_fusion(rankings)[:k]
        results = [
            ScoredChunk(self.index.chunks[part.rows[pos]], score, rank)
            for rank, (pos, score) in enumerate(fused, start=1)
        ]
        confidence = max((self._coverage(q_tokens, part, pos) for pos, _ in fused[:3]), default=0.0)
        return RetrievalResult(results, round(confidence, 4))

    @staticmethod
    def _coverage(q_tokens: list[str], part: _Partition, pos: int) -> float:
        """IDF-weighted share of the query's terms present in a chunk (0..1).

        Terms unknown to the permitted corpus get the maximum IDF, so a question about
        something the documents never mention ("stock options") scores low.
        """
        weights = {t: part.idf.get(t, part.max_idf) for t in set(q_tokens)}
        total = sum(weights.values())
        hit = sum(w for t, w in weights.items() if t in part.tokens[pos])
        return hit / total if total else 0.0


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
