"""Dense-vector embedders behind a small protocol.

`TfidfEmbedder` is the offline default: character n-gram TF-IDF vectors (robust to accents,
plurals and Spanish/English morphology). `OpenAIEmbedder` is a drop-in replacement that only
activates when explicitly selected AND an API key is present.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from docanswer.text import fold


@runtime_checkable
class Embedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> None: ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) float32 matrix of L2-normalised vectors."""
        ...

    def state(self) -> dict: ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.where(norms == 0, 1, norms)).astype(np.float32)


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self, vocabulary: dict[str, int] | None = None, idf: list[float] | None = None) -> None:
        params = dict(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, preprocessor=fold)
        if vocabulary is not None and idf is not None:
            self._vec = TfidfVectorizer(vocabulary=vocabulary, **params)
            self._vec.idf_ = np.asarray(idf)
            self._fitted = True
        else:
            self._vec = TfidfVectorizer(min_df=1, **params)
            self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self._vec.fit(texts)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder must be fitted on the corpus before embedding")
        return _normalise(self._vec.transform(texts).toarray())

    def state(self) -> dict:
        vocab = {term: int(i) for term, i in self._vec.vocabulary_.items()}
        return {"name": self.name, "vocabulary": vocab, "idf": self._vec.idf_.tolist()}

    @classmethod
    def from_state(cls, state: dict) -> TfidfEmbedder:
        return cls(vocabulary=state["vocabulary"], idf=state["idf"])


class OpenAIEmbedder:
    """Hosted embeddings. Never constructed unless EMBEDDER=openai and OPENAI_API_KEY is set."""

    name = "openai"

    def __init__(self, model: str | None = None, client: object | None = None) -> None:
        if client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("EMBEDDER=openai requires OPENAI_API_KEY")
            from openai import OpenAI  # optional dependency: pip install '.[llm]'

            client = OpenAI()
        self._client = client
        self.model = model or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

    def fit(self, texts: list[str]) -> None:  # stateless
        return None

    def embed(self, texts: list[str]) -> np.ndarray:
        response = self._client.embeddings.create(model=self.model, input=texts)  # type: ignore[attr-defined]
        return _normalise(np.array([item.embedding for item in response.data], dtype=np.float32))

    def state(self) -> dict:
        return {"name": self.name, "model": self.model}


def make_embedder(name: str, state: dict | None = None) -> Embedder | None:
    """Build an embedder by name; `state` restores a fitted one from the persisted index."""
    if name == "none":
        return None
    if name == "tfidf":
        return TfidfEmbedder.from_state(state) if state else TfidfEmbedder()
    if name == "openai":
        return OpenAIEmbedder(model=(state or {}).get("model"))
    raise ValueError(f"unknown embedder: {name!r}")
