from fastapi import APIRouter

from backend.inference import get_class_names

router = APIRouter(tags=["classes"])


@router.get("/classes")
def list_classes() -> dict[str, list[str]]:
    """Class labels read from the loaded model."""
    return {"classes": get_class_names()}
