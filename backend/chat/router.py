"""Grounded chat endpoint over the local knowledge base."""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from backend import config
from backend.chat.retriever import search
from backend.chat.small_talk import small_talk
from backend.security import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

_MAX_CITED = 3


class ChatRequest(BaseModel):
    """Incoming chat question."""

    question: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """Reject whitespace-only questions before they reach the retriever."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must contain non-whitespace text")
        return cleaned


class Citation(BaseModel):
    """One retrieved knowledge-base chunk."""

    id: str
    source: str
    section: str
    text: str
    score: float


class ChatResponse(BaseModel):
    """Grounded answer and its source citations."""

    answer: str
    citations: list[Citation]
    grounded: bool
    status: Literal["grounded", "small_talk", "out_of_scope"] = "out_of_scope"


_NO_ANSWER = (
    "I don't have that in the knowledge base. "
    "Try rephrasing, or add a document under backend/docs/kb/."
)


def _score(hit: dict) -> float:
    """Retriever similarity for a hit, 0.0 when missing or not finite."""
    try:
        value = float(hit.get("cosine", hit.get("score", 0.0)))
    except (TypeError, ValueError):
        return 0.0
    return value if value == value else 0.0  # drop NaN


def _select(hits: list[dict]) -> list[dict]:
    """Keep hits above the threshold, best first, capped."""
    kept = [h for h in hits if _score(h) >= config.KB_MIN_COSINE and h.get("text")]
    kept.sort(key=_score, reverse=True)
    return kept[:_MAX_CITED]


def _compose(hits: list[dict]) -> str:
    """Stitch retrieved chunks into a cited, extractive answer."""
    return "\n\n".join(
        f"[{h.get('id', '')}] {h.get('section', '')}\n{str(h['text']).strip()}"
        for h in hits
    )


def _no_answer() -> ChatResponse:
    """Uniform refusal when nothing in the KB supports an answer."""
    return ChatResponse(
        answer=_NO_ANSWER,
        citations=[],
        grounded=False,
        status="out_of_scope",
    )


@router.post("", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    _: str = Depends(require_api_key),
) -> ChatResponse:
    """Answer using only retrieved local-KB content."""
    # Small‑talk gate – works even if the index isn't built yet
    reply = small_talk(req.question)
    if reply is not None:
        return ChatResponse(
            answer=reply,
            citations=[],
            grounded=False,
            status="small_talk",
        )

    if not config.kb_is_built():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KB index not built. Run: python -m backend.tools.build_kb",
        )

    try:
        hits = search(req.question, top_k=req.top_k)
    except Exception:
        logger.exception("Knowledge-base search failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Knowledge-base search failed.",
        ) from None

    kept = _select(hits)
    if not kept:
        return _no_answer()

    citations = [
        Citation(
            id=str(h.get("id", "")),
            source=str(h.get("source", "")),
            section=str(h.get("section", "")),
            text=str(h.get("text", "")),
            score=_score(h),
        )
        for h in kept
    ]

    return ChatResponse(
        answer=_compose(kept),
        citations=citations,
        grounded=True,
        status="grounded",
    )