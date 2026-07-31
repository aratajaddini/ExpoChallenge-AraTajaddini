"""Read and clear prediction history."""
from fastapi import APIRouter, Depends, Query

from backend.models.database import get_conn
from backend.schemas.prediction import HistoryItem
from backend.security import require_api_key

router = APIRouter(
    prefix="/history",
    tags=["history"],
    dependencies=[Depends(require_api_key)],
)


@router.get("", response_model=list[HistoryItem])
def history(limit: int = Query(50, ge=1, le=500)) -> list[HistoryItem]:
    """Most recent predictions, newest first."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, filename, top_class AS predicted_class, confidence, "
            "source, frames_analyzed, created_at "
            "FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [HistoryItem(**dict(row)) for row in rows]


@router.delete("")
def clear_history() -> dict[str, object]:
    """Delete every prediction. Feedback rows cascade."""
    with get_conn() as conn:
        deleted = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        conn.execute("DELETE FROM predictions")
    return {"status": "ok", "deleted": deleted}
