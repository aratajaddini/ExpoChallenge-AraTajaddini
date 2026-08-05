from __future__ import annotations

import logging
import secrets

from fastapi import Header, HTTPException, status

from backend import config, keys

logger = logging.getLogger(__name__)


_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid, expired or revoked API key.",
    headers={"WWW-Authenticate": "ApiKey"},
)


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str:
    logger.info("AUTH dependency called")

    try:
        if not x_api_key:
            logger.info("AUTH: no API key supplied")
            raise _UNAUTHORIZED

        logger.info("AUTH: checking configured admin key")

        if config.API_KEY and secrets.compare_digest(x_api_key, config.API_KEY):
            logger.info("AUTH: admin key accepted")
            return "admin"

        logger.info("AUTH: calling keys.verify_key(...)")
        row = keys.verify_key(x_api_key)

        logger.info(
            "AUTH: verify_key returned: %s",
            "a row" if row is not None else "None",
        )

        if row is None:
            raise _UNAUTHORIZED

        logger.info("AUTH: row keys = %s", list(row.keys()))
        logger.info("AUTH: label = %r", row["label"])

        return row["label"]

    except HTTPException:
        raise

    except Exception:
        logger.exception("Unexpected error in require_api_key")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during authentication.",
        )


def require_admin(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> str:
    """Allow only the static admin key from config; shift keys are rejected."""
    logger.info("ADMIN AUTH dependency called")

    try:
        if not x_api_key:
            raise _UNAUTHORIZED

        if config.API_KEY and secrets.compare_digest(x_api_key, config.API_KEY):
            return "admin"

        raise _UNAUTHORIZED

    except HTTPException:
        raise

    except Exception:
        logger.exception("Unexpected error in require_admin")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during authentication.",
        )
