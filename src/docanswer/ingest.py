"""Ingestion: PDF -> per-page text -> cleaned sections -> overlapping chunks -> persisted index.

Chunks never cross a page boundary, so every chunk has exactly one citable page number.
The most recent section heading is carried across pages so continuation text keeps its context.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pypdf import PdfReader

from docanswer.config import Settings
from docanswer.embeddings import Embedder, make_embedder
from docanswer.models import AccessLevel, Chunk
from docanswer.text import normalize_whitespace, split_sentences

HEADING_RE = re.compile(r"^\d{1,2}(?:\.\d{1,2})*\.?\s+[A-ZÁÉÍÓÚÑ][^.:]{2,70}$")


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    title: str
    short_title: str
    access_level: AccessLevel
    lang: str


def load_manifest(docs_dir: Path) -> dict[str, DocMeta]:
    raw = json.loads((docs_dir / "manifest.json").read_text(encoding="utf-8"))
    return {
        name: DocMeta(
            Path(name).stem, m["title"], m.get("short", m["title"]), AccessLevel.parse(m["access"]), m.get("lang", "en")
        )
        for name, m in raw.items()
    }


def extract_pages(pdf_path: Path) -> list[str]:
    return [page.extract_text() or "" for page in PdfReader(pdf_path).pages]


def boilerplate_lines(pages: list[str]) -> set[str]:
    """Lines (digits masked) repeated on most pages are headers/footers, e.g. 'Page 3'."""
    if len(pages) < 2:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({_mask(line) for line in page.splitlines() if line.strip()})
    return {line for line, n in counts.items() if n >= max(2, len(pages) // 2 + 1)}


def _mask(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip())


def clean_page(text: str, boilerplate: set[str]) -> list[str]:
    lines = normalize_whitespace(text).splitlines()
    return [ln.strip() for ln in lines if ln.strip() and _mask(ln) not in boilerplate]


def sections_from_lines(lines: list[str], carry_heading: str) -> list[tuple[str, str]]:
    """Group lines under headings. Returns [(heading, body_text)] in reading order."""
    sections: list[tuple[str, list[str]]] = [(carry_heading, [])]
    for line in lines:
        if HEADING_RE.match(line):
            sections.append((line, []))
        else:
            sections[-1][1].append(line)
    return [(h, " ".join(body)) for h, body in sections if body]


def chunk_page(
    meta: DocMeta, page_no: int, sections: list[tuple[str, str]], max_words: int, overlap: int
) -> list[Chunk]:
    """Pack sentences into chunks of <= max_words. Overlap applies only inside a section."""
    chunks: list[Chunk] = []
    for heading, body in sections:
        sentences = split_sentences(body)
        start = 0
        while start < len(sentences):
            end, words = start, 0
            while end < len(sentences) and (end == start or words + len(sentences[end].split()) <= max_words):
                words += len(sentences[end].split())
                end += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{meta.doc_id}:p{page_no}:c{len(chunks)}",
                    doc_id=meta.doc_id,
                    title=meta.title,
                    short_title=meta.short_title,
                    page=page_no,
                    heading=heading,
                    text=" ".join(sentences[start:end]),
                    access_level=meta.access_level,
                    lang=meta.lang,
                )
            )
            if end >= len(sentences):
                break
            start = max(end - overlap, start + 1)
    return chunks


def chunk_document(pdf_path: Path, meta: DocMeta, max_words: int = 90, overlap: int = 1) -> list[Chunk]:
    pages = extract_pages(pdf_path)
    boilerplate = boilerplate_lines(pages)
    chunks: list[Chunk] = []
    heading = meta.title
    for page_no, raw in enumerate(pages, start=1):
        sections = sections_from_lines(clean_page(raw, boilerplate), heading)
        chunks.extend(chunk_page(meta, page_no, sections, max_words, overlap))
        heading = sections[-1][0] if sections else heading
    return chunks


@dataclass
class Index:
    chunks: list[Chunk]
    embeddings: np.ndarray | None
    embedder: Embedder | None

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        meta = {"n_chunks": len(self.chunks), "embedder": self.embedder.state() if self.embedder else None}
        (index_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        if self.embeddings is not None:
            np.save(index_dir / "embeddings.npy", self.embeddings)

    @classmethod
    def load(cls, index_dir: Path) -> Index:
        lines = (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        chunks = [Chunk.from_dict(json.loads(line)) for line in lines if line]
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        state = meta.get("embedder")
        embedder = make_embedder(state["name"], state) if state else None
        emb_path = index_dir / "embeddings.npy"
        embeddings = np.load(emb_path) if embedder and emb_path.exists() else None
        return cls(chunks, embeddings, embedder)


def build_index(settings: Settings) -> Index:
    manifest = load_manifest(settings.docs_dir)
    chunks: list[Chunk] = []
    for name, meta in sorted(manifest.items()):
        chunks.extend(
            chunk_document(settings.docs_dir / name, meta, settings.chunk_max_words, settings.chunk_overlap_sentences)
        )
    embedder = make_embedder(settings.embedder)
    embeddings = None
    if embedder is not None:
        texts = [c.index_text for c in chunks]
        embedder.fit(texts)
        embeddings = embedder.embed(texts)
    return Index(chunks, embeddings, embedder)


def main() -> None:
    settings = Settings.from_env()
    index = build_index(settings)
    index.save(settings.index_dir)
    per_doc = Counter(c.doc_id for c in index.chunks)
    print(f"indexed {len(index.chunks)} chunks from {len(per_doc)} documents -> {settings.index_dir}")
    for doc_id, n in sorted(per_doc.items()):
        print(f"  {doc_id:<34} {n:>3} chunks")


if __name__ == "__main__":
    main()
