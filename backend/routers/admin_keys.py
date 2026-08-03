"""Admin-only API key management: mint, list, revoke."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend import keys
from backend.security import require_admin

router = APIRouter(prefix="/admin/keys", tags=["admin"])


class MintRequest(BaseModel):
    """Payload for minting a shift key."""

    label: str = Field(min_length=1, max_length=40)
    hours: int = Field(default=8, ge=1, le=720)


class MintResponse(BaseModel):
    """The raw key is returned once and never stored."""

    api_key: str
    label: str
    expires_at: str


class KeyInfo(BaseModel):
    """Key metadata; never contains the raw key."""

    id: int
    preview: str
    label: str
    created_at: str
    expires_at: str
    revoked: bool
    last_used_at: str | None = None


@router.post("", response_model=MintResponse, status_code=status.HTTP_201_CREATED)
def mint(payload: MintRequest, _: str = Depends(require_admin)) -> MintResponse:
    """Mint a shift key. The raw value is shown once and cannot be recovered."""
    raw, expires_at = keys.issue_key(payload.label, payload.hours)
    return MintResponse(api_key=raw, label=payload.label, expires_at=expires_at)


@router.get("", response_model=list[KeyInfo])
def index(active_only: bool = False, _: str = Depends(require_admin)) -> list[KeyInfo]:
    """List issued keys, metadata only."""
    return [KeyInfo(**dict(row)) for row in keys.list_keys(active_only)]


# FIX: add response_model=None to satisfy 204 No Content
@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
def revoke(key_id: int, _: str = Depends(require_admin)) -> None:
    """Revoke a key by id."""
    if not keys.revoke_key(key_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Key not found.")
    # No return body – 204 response.