"""Central configuration.

All values are overridable via environment variables (or a local .env file).
No hard-coded class names — see inference.get_class_names().
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Optional .env support: the file is read if python-dotenv is installed,
# otherwise plain environment variables are used.
try:  # pragma: no cover - depends on optional dependency
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR.parent / ".env")
except ImportError:
    pass


# --------------------------------------------------------------------------- #
# Env helpers
# --------------------------------------------------------------------------- #
def _env_str(name: str, default: str = "") -> str:
    """Return a stripped string from the environment."""
    return (os.getenv(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    """Return an int from the environment, falling back to default if invalid."""
    raw = _env_str(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Return a float from the environment, falling back to default if invalid."""
    raw = _env_str(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_list(name: str, default: str) -> list[str]:
    """Parse a comma-separated environment variable into a list."""
    raw = _env_str(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_path(name: str, default: Path) -> Path:
    """
    Return a resolved path from the environment, or the given default.

    Relative paths are resolved from the project root (BASE_DIR.parent),
    which is safer and more portable than resolving from the current
    working directory.
    """
    raw = _env_str(name)
    if not raw:
        return default.resolve()

    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (BASE_DIR.parent / path).resolve()


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
MODEL_PATH: Path = _env_path("MODEL_PATH", BASE_DIR / "weights" / "best.pt")
UPLOAD_DIR: Path = _env_path("UPLOAD_DIR", BASE_DIR / "uploads")

# DB location is overridable so the frozen (PyInstaller) mint_key.exe can point at
# the real waste.db instead of the temporary _MEIPASS extraction folder.
DB_PATH: Path = _env_path("SWR_DB_PATH", BASE_DIR / "waste.db")

# --------------------------------------------------------------------------- #
# Uploads
# --------------------------------------------------------------------------- #
MAX_UPLOAD_BYTES: int = _env_int("MAX_UPLOAD_BYTES", 150 * 1024 * 1024)  # 150 MB
IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
VIDEO_EXTS: frozenset[str] = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})
ALLOWED_EXTS: frozenset[str] = IMAGE_EXTS | VIDEO_EXTS

# --------------------------------------------------------------------------- #
# Video sampling
# --------------------------------------------------------------------------- #
MAX_VIDEO_FRAMES: int = _env_int("MAX_VIDEO_FRAMES", 120)  # ~1 fps for a 2-minute clip
MIN_FRAME_CONFIDENCE: float = _env_float("MIN_FRAME_CONFIDENCE", 0.35)

# --------------------------------------------------------------------------- #
# Analytics detector (TACO) — never used in the delivery path
# --------------------------------------------------------------------------- #
DETECTION_MODEL_PATH: Path = _env_path(
    "DETECTION_MODEL_PATH", BASE_DIR / "weights" / "taco_det.pt"
)

DETECTION_MIN_CONFIDENCE: float = _env_float("DETECTION_MIN_CONFIDENCE", 0.40)

# --------------------------------------------------------------------------- #
# Security
# --------------------------------------------------------------------------- #
API_KEY: str = _env_str("API_KEY")
ALLOWED_ORIGINS: list[str] = _env_list(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5500"
)

# --------------------------------------------------------------------------- #
# Knowledge base (chatbot RAG)
# --------------------------------------------------------------------------- #
KB_DIR: Path = _env_path("KB_DIR", BASE_DIR / "docs" / "kb")
KB_INDEX: Path = _env_path("KB_INDEX", BASE_DIR / "data" / "kb_index.npz")

# Local directory path or a HuggingFace model ID.
# The HF default downloads on first use — vendor the model locally for offline demos.
EMBED_MODEL: str = _env_str("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

KB_CHUNK_WORDS: int = _env_int("KB_CHUNK_WORDS", 220)
KB_CHUNK_OVERLAP: int = _env_int("KB_CHUNK_OVERLAP", 40)
KB_TOP_K: int = _env_int("KB_TOP_K", 5)
KB_RRF_K: int = _env_int("KB_RRF_K", 60)

# Grounding gate: max cosine over retrieved hits must reach this.
# 0.35 is a placeholder — set it with tools/calibrate_kb.py.
KB_MIN_COSINE: float = _env_float("KB_MIN_COSINE", 0.35)

# --------------------------------------------------------------------------- #
# Model requirement (optional soft check)
# --------------------------------------------------------------------------- #
REQUIRE_MODEL: bool = _env_str("REQUIRE_MODEL", "1").lower() not in {"0", "false", "no"}


# --------------------------------------------------------------------------- #
# Runtime setup / validation
# --------------------------------------------------------------------------- #
def ensure_dirs() -> None:
    """Create the directories the app writes to. Source dirs are never created."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    KB_INDEX.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def has_detector() -> bool:
    """True if the optional analytics detector weights are present."""
    return DETECTION_MODEL_PATH.exists()


def kb_is_built() -> bool:
    """True if the chatbot KB index artifact exists and is not empty."""
    return KB_INDEX.is_file() and KB_INDEX.stat().st_size > 0


def assert_configured() -> None:
    """Fail fast at startup if required secrets, weights or settings are invalid."""
    if not API_KEY:
        raise RuntimeError(
            "API_KEY is not set. Add it to .env (or the environment) before starting."
        )
    if not MODEL_PATH.is_file():
        if REQUIRE_MODEL:
            raise RuntimeError(f"Model weights not found: {MODEL_PATH}")
        warnings.warn(
            f"Model weights not found: {MODEL_PATH}; /predict will fail.",
            stacklevel=2,
        )
    if not ALLOWED_ORIGINS:
        raise RuntimeError(
            "ALLOWED_ORIGINS is empty; CORS would block every browser call."
        )
    if "*" in ALLOWED_ORIGINS:
        raise RuntimeError("ALLOWED_ORIGINS='*' is not allowed with API-key auth.")
    if KB_CHUNK_WORDS < 1:
        raise RuntimeError("KB_CHUNK_WORDS must be >= 1.")
    if KB_CHUNK_OVERLAP < 0 or KB_CHUNK_OVERLAP >= KB_CHUNK_WORDS:
        raise RuntimeError(
            f"KB_CHUNK_OVERLAP ({KB_CHUNK_OVERLAP}) must be in [0, KB_CHUNK_WORDS)."
        )
    if KB_TOP_K < 1:
        raise RuntimeError("KB_TOP_K must be >= 1.")
    if KB_RRF_K < 1:
        raise RuntimeError("KB_RRF_K must be >= 1.")
    if not 0 <= KB_MIN_COSINE <= 1:
        raise RuntimeError("KB_MIN_COSINE must be between 0 and 1.")
    # Additional range checks for video/analytics settings
    if MAX_VIDEO_FRAMES < 1:
        raise RuntimeError("MAX_VIDEO_FRAMES must be >= 1.")
    if not 0 <= MIN_FRAME_CONFIDENCE <= 1:
        raise RuntimeError("MIN_FRAME_CONFIDENCE must be between 0 and 1.")
    if not 0 <= DETECTION_MIN_CONFIDENCE <= 1:
        raise RuntimeError("DETECTION_MIN_CONFIDENCE must be between 0 and 1.")
