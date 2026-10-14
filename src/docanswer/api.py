"""HTTP API and single-page UI.

NOTE: `role` comes from the request body to keep the demo self-contained. In production it must
come from the authenticated identity (SSO/JWT claims), never from the client.

Run with: uvicorn docanswer.api:create_app --factory
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from docanswer.config import Settings
from docanswer.pipeline import DocAnswer

STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    role: Literal["guest", "employee", "manager"] = "employee"


class CitationOut(BaseModel):
    doc: str
    doc_id: str
    page: int
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    confidence: float
    refused: bool


class IngestResponse(BaseModel):
    chunks: int
    documents: int


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="DocAnswer", version="0.1.0", description="Answers from company documents, with citations.")
    state: dict[str, DocAnswer] = {"engine": DocAnswer.from_settings(settings)}

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health() -> dict:
        engine = state["engine"]
        return {"status": "ok", "chunks": len(engine.retriever.index.chunks), "llm": engine.provider.name}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        result = state["engine"].ask(req.question, req.role)
        return AskResponse(
            answer=result.answer,
            confidence=result.confidence,
            refused=result.refused,
            citations=[CitationOut(**vars(c)) for c in result.citations],
        )

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(x_ingest_token: str = Header(default="")) -> IngestResponse:
        if settings.ingest_token and not secrets.compare_digest(x_ingest_token, settings.ingest_token):
            raise HTTPException(status_code=401, detail="invalid ingest token")
        state["engine"] = DocAnswer.from_settings(settings, rebuild=True)
        chunks = state["engine"].retriever.index.chunks
        return IngestResponse(chunks=len(chunks), documents=len({c.doc_id for c in chunks}))

    return app
