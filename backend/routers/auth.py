"""Auth router: lets a client verify its API key before doing real work."""

from fastapi import APIRouter, Depends

from backend.security import require_api_key

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/verify")
def verify(identity: str = Depends(require_api_key)) -> dict[str, object]:
    """Return the caller's identity if the X-API-Key header is valid."""
    return {"valid": True, "identity": identity}
