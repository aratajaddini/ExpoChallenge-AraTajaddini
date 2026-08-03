"""Image and video classification endpoint."""
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from backend.config import IMAGE_EXTS, MAX_UPLOAD_BYTES, UPLOAD_DIR, VIDEO_EXTS
from backend.inference import run_inference, run_video_inference
from backend.models.database import get_conn
from backend.schemas.prediction import PredictionResponse
from backend.security import require_api_key

router = APIRouter(
    prefix="/predict",
    tags=["predict"],
    dependencies=[Depends(require_api_key)],
)

_CHUNK = 1 << 20  # 1 MB
Mode = Literal["image", "video"]


def _resolve_mode(filename: str | None, requested: str) -> tuple[Mode, str]:
    """Pick the inference mode and a safe extension from the client's filename."""
    ext = Path(filename or "").suffix.lower()
    is_image, is_video = ext in IMAGE_EXTS, ext in VIDEO_EXTS

    if requested == "image":
        if not is_image:
            raise HTTPException(415, f"not an image extension: {ext or 'missing'}")
        return "image", ext
    if requested == "video":
        if not is_video:
            raise HTTPException(415, f"not a video extension: {ext or 'missing'}")
        return "video", ext
    if is_video:
        return "video", ext
    if is_image:
        return "image", ext
    raise HTTPException(415, f"unsupported extension: {ext or 'missing'}")


def _save_upload(file: UploadFile, ext: str) -> Path:
    """Stream the upload to a generated name, aborting past MAX_UPLOAD_BYTES."""
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    written = 0
    try:
        with dest.open("wb") as out:
            while chunk := file.file.read(_CHUNK):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "file exceeds the upload limit")
                out.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, "empty file")
    return dest


@router.post("", response_model=PredictionResponse)
def predict(
    file: UploadFile = File(...),
    mode: Literal["auto", "image", "video"] = Query("auto"),
) -> dict:
    """Classify an uploaded image or video. Sync on purpose: runs in a threadpool."""
    resolved, ext = _resolve_mode(file.filename, mode)
    dest = _save_upload(file, ext)
    try:
        if resolved == "video":
            result = run_video_inference(dest)
        else:
            result = run_inference(dest.read_bytes())
    except (ValueError, OSError) as exc:
        raise HTTPException(422, f"could not decode {resolved}: {exc}") from exc
    finally:
        dest.unlink(missing_ok=True)

    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO predictions "
            "(filename, top_class, confidence, source, frames_analyzed) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                file.filename,
                result["top_class"],
                result["confidence"],
                resolved,
                result["frames_analyzed"],
            ),
        )
        pred_id = cur.lastrowid

    return {**result, "id": pred_id, "filename": file.filename, "source": resolved}


# ===== GET /classes =====
# This endpoint is protected by the router‑level require_api_key dependency.
@router.get("/classes", tags=["predict"])
def get_classes(_: str = Depends(require_api_key)) -> dict[str, list[str]]:
    """Return the list of supported waste categories."""
    return {
        "classes": [
            "paper",
            "plastic",
            "metal",
            "glass",
            "organic",
            "other",
        ]
    }