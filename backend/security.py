"""API key dependency: static admin key from env + shift keys from the DB."""
from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from backend import config, keys

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid, expired or revoked API key.",
    headers={"WWW-Authenticate": "ApiKey"},
)


def require_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> str:
    """Return the caller identity ("admin" or the shift label), else 401."""
    if not x_api_key:
        raise _UNAUTHORIZED
    if config.API_KEY and secrets.compare_digest(x_api_key, config.API_KEY):
        return "admin"
    row = keys.verify_key(x_api_key)
    if row is None:
        raise _UNAUTHORIZED
    return row["label"]


def require_admin(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> str:
    """Allow only the static admin key from config; shift keys are rejected."""
    if not x_api_key:
        raise _UNAUTHORIZED
    if config.API_KEY and secrets.compare_digest(x_api_key, config.API_KEY):
        return "admin"
    raise _UNAUTHORIZED
