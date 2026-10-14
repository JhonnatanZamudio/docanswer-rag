"""Language-agnostic text utilities shared by ingestion, retrieval and the extractive answerer."""

from __future__ import annotations

import re
import unicodedata

import snowballstemmer

# Small EN+ES stopword list: enough for BM25 and coverage scoring without extra dependencies.
STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i if in is it its many may much
    me my of on or our per should so than that the their there this to was we what when where which
    who why will with within you your after before about any all also each must
    el la los las un una unos unas de del al y o u en por para con sin que qué cual cuál cuales cuáles
    cuanto cuánto cuanta cuánta cuantos cuántos cuantas cuántas como cómo cuando cuándo donde dónde es son se su sus
    lo le les mi me yo tengo puedo debo hay ser mas más muy ya este esta estos estas""".split()
)

_TOKEN_RE = re.compile(r"[^\W_]+(?:[.,][0-9]+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡0-9])")
_STEMMERS = {"en": snowballstemmer.stemmer("english"), "es": snowballstemmer.stemmer("spanish")}
_ES_HINTS = frozenset("cual cuales cuantos cuantas cuanto cuanta como donde puedo debo tengo que las los del".split())


def fold(text: str) -> str:
    """Lowercase and strip accents so that 'días' and 'dias' match."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


_FOLDED_STOPWORDS = frozenset(fold(w) for w in STOPWORDS)


def detect_lang(text: str) -> str:
    """Tiny EN/ES detector, sufficient for routing stemmers and refusal messages."""
    words = set(re.findall(r"[a-z]+", fold(text)))
    return "es" if "¿" in text or len(words & _ES_HINTS) >= 2 else "en"


def tokenize(text: str, lang: str = "en", *, drop_stopwords: bool = True) -> list[str]:
    """Lowercase -> stopwords -> Snowball stem (per language) -> accent folding."""
    stemmer = _STEMMERS.get(lang, _STEMMERS["en"])
    tokens = [t for t in _TOKEN_RE.findall(text.lower()) if not (drop_stopwords and fold(t) in _FOLDED_STOPWORDS)]
    return [fold(t) if t[0].isdigit() else fold(stemmer.stemWord(t)) for t in tokens]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def normalize_whitespace(text: str) -> str:
    text = text.replace("­", "")  # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate line breaks
    return re.sub(r"[ \t]+", " ", text).strip()
