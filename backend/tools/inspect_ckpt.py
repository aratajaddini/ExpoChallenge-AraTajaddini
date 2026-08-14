import sys
from pathlib import Path

import torch

path = Path(sys.argv[1])
if not path.is_file():
    sys.exit(f"not found: {path}")

ckpt = torch.load(path, map_location="cpu", weights_only=False)

print("path      :", path)
print("size MB   :", round(path.stat().st_size / 1e6, 3))
print("keys      :", sorted(ckpt.keys()))

m = ckpt.get("ema") or ckpt.get("model")
print("task      :", getattr(m, "task", None))
print("names     :", getattr(m, "names", None))
print("nc        :", len(getattr(m, "names", {}) or {}))
print("params    :", sum(p.numel() for p in m.parameters()))
print("optimizer :", ckpt.get("optimizer") is not None)
print("epoch     :", ckpt.get("epoch"))
print("train task:", (ckpt.get("train_args") or {}).get("task"))
