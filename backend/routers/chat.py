"""Chat endpoint exposing read-only analytics to the frontend."""

from fastapi import APIRouter
from pydantic import BaseModel

from backend.analytics import intent

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Answer a natural-language analytics question offline."""
    return ChatResponse(reply=intent.answer(req.message))
