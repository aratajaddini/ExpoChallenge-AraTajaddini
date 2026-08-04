"""Model inference. Independent of FastAPI; model loaded once via lru_cache."""
import io
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from backend.config import MAX_VIDEO_FRAMES, MIN_FRAME_CONFIDENCE, MODEL_PATH


@lru_cache(maxsize=1)
def _get_model():
    """Load the YOLO classifier once per process."""
    from ultralytics import YOLO

    return YOLO(str(MODEL_PATH), task="classify")


def get_class_names() -> list[str]:
    """Class names from the model, ordered by class index."""
    names = _get_model().names
    return [names[i] for i in sorted(names)]


def _classify(img: np.ndarray) -> dict:
    """Classify one RGB frame."""
    model = _get_model()
    probs = model(img, verbose=False)[0].probs
    names = model.names
    return {
        "top_class": names[int(probs.top1)],
        "confidence": round(float(probs.top1conf), 4),
        "scores": {names[i]: round(float(probs.data[i]), 4) for i in names},
    }


def run_inference(image_bytes: bytes) -> dict:
    """Classify a single image given its raw bytes."""
    try:
        img = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("could not decode image") from exc

    return {
        **_classify(img),
        "source": "image",
        "frames_analyzed": 1,
        "frames_used": 1,
    }


def iter_video_frames(
    path: Path, max_frames: int = MAX_VIDEO_FRAMES
) -> Iterator[np.ndarray]:
    """Yield up to max_frames RGB frames, evenly spaced across the clip.

    Decodes sequentially and keeps every Nth frame; no seeking, so it stays
    correct on containers where random access is unreliable.
    """
    if max_frames < 1:
        raise ValueError("max_frames must be >= 1")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError("could not open video")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // max_frames) if total > 0 else 1

        idx = yielded = 0
        while yielded < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                yielded += 1
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            idx += 1
    finally:
        cap.release()


def run_video_inference(path: Path) -> dict:
    """Classify a video by averaging per-frame scores over confident frames."""
    totals: dict[str, float] = {name: 0.0 for name in get_class_names()}
    seen = used = 0

    for frame in iter_video_frames(path):
        seen += 1
        result = _classify(frame)
        if result["confidence"] < MIN_FRAME_CONFIDENCE:
            continue
        used += 1
        for name, score in result["scores"].items():
            totals[name] += score

    if seen == 0:
        raise ValueError("no readable frames in video")
    if used == 0:
        raise ValueError(
            f"no frame reached the {MIN_FRAME_CONFIDENCE} confidence threshold"
        )

    scores = {name: round(total / used, 4) for name, total in totals.items()}
    top = max(scores, key=scores.__getitem__)
    return {
        "top_class": top,
        "confidence": scores[top],
        "scores": scores,
        "source": "video",
        "frames_analyzed": seen,
        "frames_used": used,
    }
