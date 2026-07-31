"""Dump metadata from a YOLO checkpoint."""
from pathlib import Path

import torch

CKPT = Path(__file__).resolve().parents[1] / "weights" / "best.pt"

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
