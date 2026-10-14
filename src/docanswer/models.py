"""Domain types. Kept free of framework imports so every layer can depend on them."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum


class AccessLevel(IntEnum):
    """Ordered access levels: a role may read any document at or below its clearance."""

    PUBLIC = 0
    EMPLOYEES = 1
    MANAGERS = 2

    @classmethod
    def parse(cls, value: str) -> AccessLevel:
        return cls[value.strip().upper()]


ROLE_CLEARANCE: dict[str, AccessLevel] = {
    "guest": AccessLevel.PUBLIC,
    "employee": AccessLevel.EMPLOYEES,
    "manager": AccessLevel.MANAGERS,
}


def clearance_for(role: str) -> AccessLevel:
    """Unknown roles get the lowest clearance (fail closed)."""
    return ROLE_CLEARANCE.get(role.strip().lower(), AccessLevel.PUBLIC)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    short_title: str
    page: int
    heading: str
    text: str
    access_level: AccessLevel
    lang: str = "en"

    @property
    def citation_tag(self) -> str:
        return f"[{self.short_title} p.{self.page}]"

    @property
    def index_text(self) -> str:
        """Text used for retrieval: headings carry a lot of signal in policy documents."""
        return f"{self.title}. {self.heading}. {self.text}"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["access_level"] = self.access_level.name.lower()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(**{**data, "access_level": AccessLevel.parse(data["access_level"])})


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    rank: int


@dataclass(frozen=True)
class Citation:
    doc: str
    doc_id: str
    page: int
    snippet: str


@dataclass
class Answer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    refused: bool = False
    retrieved: list[ScoredChunk] = field(default_factory=list)
