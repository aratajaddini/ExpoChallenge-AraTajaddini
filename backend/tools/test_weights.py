"""Inspect metadata from a local YOLO checkpoint."""

from pathlib import Path

import pytest
import torch

CKPT = Path(__file__).resolve().parents[1] / "weights" / "best.pt"


@pytest.mark.skipif(
    not CKPT.is_file(),
    reason="Local model weights are not included in the repository.",
)
def test_checkpoint_metadata_can_be_loaded() -> None:
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = ckpt.get("model")
    args = ckpt.get("train_args") or getattr(model, "args", None) or {}

    print("path      :", CKPT)
    print("task      :", ckpt.get("task"))
    print("names     :", getattr(model, "names", None))
    print("date      :", ckpt.get("date"))
    print("version   :", ckpt.get("version"))
    print("epoch     :", ckpt.get("epoch"))
    print("imgsz     :", args.get("imgsz") if isinstance(args, dict) else None)
    print("args      :", args)


if __name__ == "__main__":
    if not CKPT.is_file():
        raise SystemExit(f"Model checkpoint not found: {CKPT}")

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = ckpt.get("model")
    args = ckpt.get("train_args") or getattr(model, "args", None) or {}

    print("path      :", CKPT)
    print("task      :", ckpt.get("task"))
    print("names     :", getattr(model, "names", None))
    print("date      :", ckpt.get("date"))
    print("version   :", ckpt.get("version"))
    print("epoch     :", ckpt.get("epoch"))
    print("imgsz     :", args.get("imgsz") if isinstance(args, dict) else None)
    print("args      :", args)
