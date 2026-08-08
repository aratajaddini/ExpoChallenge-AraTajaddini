"""Validate that the trained model checkpoint loads and exposes sane classes."""

from __future__ import annotations

import os
import sys

from ultralytics import YOLO

from backend.config import MODEL_PATH


def _expected_classes() -> set[str] | None:
    """Read optional expected class names from the EXPECTED_CLASSES env var."""
    raw = os.getenv("EXPECTED_CLASSES", "").strip()
    if not raw:
        return None
    return {name.strip().lower() for name in raw.split(",") if name.strip()}


def validate() -> int:
    """Check the checkpoint on disk and return a process exit code."""
    if not MODEL_PATH.is_file():
        print(f"FAIL: model file not found at {MODEL_PATH}")
        return 1

    try:
        model = YOLO(str(MODEL_PATH), task="classify")
    except Exception as exc:  # noqa: BLE001 - surface any load failure to CI
        print(f"FAIL: could not load {MODEL_PATH}: {exc}")
        return 1

    names = getattr(model, "names", None)
    if not names:
        print("FAIL: model exposes no class names")
        return 1

    found = {str(name).lower() for name in dict(names).values()}
    print(f"OK: loaded {MODEL_PATH.name} with {len(found)} classes")
    for index, name in sorted(dict(names).items()):
        print(f"  {index}: {name}")

    expected = _expected_classes()
    if expected and expected != found:
        print(f"FAIL: expected {sorted(expected)}, got {sorted(found)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(validate())
