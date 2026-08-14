"""TACO detector — analytics only. Never called from the delivery path."""

from __future__ import annotations

import io
import logging
from functools import lru_cache

import numpy as np
from PIL import Image, UnidentifiedImageError

from backend.config import DETECTION_MIN_CONFIDENCE, DETECTION_MODEL_PATH

log = logging.getLogger(__name__)

FALLBACK_CLASS = "Waste"

# Keys are official TACO category names, normalised at lookup time.
TACO_TO_5: dict[str, str] = {
    # --- Metal ---
    "Aluminium foil": "Metal",
    "Aluminium blister pack": "Metal",
    "Metal bottle cap": "Metal",
    "Food Can": "Metal",
    "Drink can": "Metal",
    "Aerosol": "Metal",
    "Metal lid": "Metal",
    "Pop tab": "Metal",
    "Scrap metal": "Metal",
    # --- Plastic ---
    "Clear plastic bottle": "Plastic",
    "Other plastic bottle": "Plastic",
    "Plastic bottle cap": "Plastic",
    "Disposable plastic cup": "Plastic",
    "Foam cup": "Plastic",
    "Other plastic cup": "Plastic",
    "Plastic lid": "Plastic",
    "Other plastic": "Plastic",
    "Plastic film": "Plastic",
    "Six pack rings": "Plastic",
    "Garbage bag": "Plastic",
    "Other plastic wrapper": "Plastic",
    "Single-use carrier bag": "Plastic",
    "Polypropylene bag": "Plastic",
    "Crisp packet": "Plastic",
    "Spread tub": "Plastic",
    "Tupperware": "Plastic",
    "Disposable food container": "Plastic",
    "Foam food container": "Plastic",
    "Other plastic container": "Plastic",
    "Plastic glooves": "Plastic",  # sic — TACO ships this typo
    "Plastic gloves": "Plastic",
    "Plastic utensils": "Plastic",
    "Plastic straw": "Plastic",
    "Squeezable tube": "Plastic",
    "Styrofoam piece": "Plastic",
    # --- Paper ---
    "Toilet tube": "Paper",
    "Other carton": "Paper",
    "Egg carton": "Paper",
    "Drink carton": "Paper",  # composite (Tetra Pak) — revisit with Abbas
    "Corrugated carton": "Paper",
    "Meal carton": "Paper",
    "Pizza box": "Paper",
    "Paper cup": "Paper",
    "Magazine paper": "Paper",
    "Normal paper": "Paper",
    "Wrapping paper": "Paper",
    "Tissues": "Paper",
    "Paper bag": "Paper",
    "Plastified paper bag": "Paper",
    "Paper straw": "Paper",
    # --- Glass ---
    "Glass bottle": "Glass",
    "Broken glass": "Glass",
    "Glass cup": "Glass",
    "Glass jar": "Glass",
    # --- Waste ---
    "Battery": "Waste",  # hazardous in reality; analytics-only bucket
    "Carded blister pack": "Waste",
    "Food waste": "Waste",
    "Cigarette": "Waste",
    "Rope & strings": "Waste",
    "Shoe": "Waste",
    "Unlabeled litter": "Waste",
}


def _norm(name: str) -> str:
    """Normalise a class label so casing/spacing differences don't break lookup."""
    return " ".join(name.strip().lower().replace("&", "and").split())


_NORM_MAP = {_norm(k): v for k, v in TACO_TO_5.items()}


@lru_cache(maxsize=1)
def _get_detector():
    """Load the TACO detector once per process."""
    from ultralytics import YOLO

    if not DETECTION_MODEL_PATH.exists():
        raise FileNotFoundError(f"detector weights missing: {DETECTION_MODEL_PATH}")
    return YOLO(str(DETECTION_MODEL_PATH), task="detect")


@lru_cache(maxsize=1)
def _get_index_map() -> dict[int, str]:
    """Map model class indices to the 5 official classes, reading model.names."""
    names: dict[int, str] = _get_detector().names
    mapping: dict[int, str] = {}
    unmapped: list[str] = []
    for idx, raw in names.items():
        mapped = _NORM_MAP.get(_norm(raw))
        if mapped is None:
            unmapped.append(raw)
            mapped = FALLBACK_CLASS
        mapping[int(idx)] = mapped
    if unmapped:
        log.warning(
            "detection: %d TACO labels not in TACO_TO_5, forced to %s: %s",
            len(unmapped),
            FALLBACK_CLASS,
            sorted(unmapped),
        )
    return mapping


def get_detection_class_names() -> list[str]:
    """Raw detector labels, ordered by class index."""
    names: dict[int, str] = _get_detector().names
    return [names[i] for i in sorted(names)]


def run_detection(image_bytes: bytes) -> list[dict]:
    """Detect litter items and map each to one of the 5 official classes.

    Analytics only — the output must never drive an Arduino signal.
    Each item: {"class", "raw_class", "conf", "bbox": [x1, y1, x2, y2]}
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("could not decode image") from exc

    model = _get_detector()
    index_map = _get_index_map()
    raw_names: dict[int, str] = model.names

    results = model.predict(
        np.asarray(image), conf=DETECTION_MIN_CONFIDENCE, verbose=False
    )
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    items: list[dict] = []
    for cls_id, conf, xyxy in zip(
        boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
    ):
        idx = int(cls_id)
        items.append(
            {
                "class": index_map.get(idx, FALLBACK_CLASS),
                "raw_class": raw_names.get(idx, "unknown"),
                "conf": round(float(conf), 4),
                "bbox": [round(float(v), 2) for v in xyxy],
            }
        )
    return items


def count_by_class(items: list[dict]) -> dict[str, int]:
    """Per-class item counts for the dashboard chart."""
    counts: dict[str, int] = {}
    for item in items:
        counts[item["class"]] = counts.get(item["class"], 0) + 1
    return counts
