"""Central configuration. No hard-coded class names — see inference.get_class_names()."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "weights" / "best.pt"
UPLOAD_DIR = BASE_DIR / "uploads"

# DB location is overridable so the frozen (PyInstaller) mint_key.exe can point at
# the real waste.db instead of the temporary _MEIPASS extraction folder.
DB_PATH = Path(os.getenv("SWR_DB_PATH") or (BASE_DIR / "waste.db")).resolve()

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Uploads
MAX_UPLOAD_BYTES = 150 * 1024 * 1024          # 150 MB
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Video sampling
MAX_VIDEO_FRAMES = 120                         # ~1 fps for a 2-minute clip
MIN_FRAME_CONFIDENCE = 0.35                    # frames below this are counted as "uncertain"

# Analytics detector (TACO) — never used in the delivery path
DETECTION_MODEL_PATH = BASE_DIR / "weights" / "taco_det.pt"
DETECTION_MIN_CONFIDENCE = 0.40

# Security
API_KEY = os.getenv("API_KEY", "")             # no default secret in code
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5500").split(",")
    if o.strip()
]


def assert_configured() -> None:
    """Fail fast at startup if required secrets or model weights are missing."""
    if not API_KEY:
        raise RuntimeError("API_KEY is not set. Add it to .env before starting the server.")
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model weights not found: {MODEL_PATH}")
