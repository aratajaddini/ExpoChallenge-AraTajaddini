"""FastAPI application entrypoint for the Smart Waste Robot API."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend import config, keys
from backend.analytics import store as detection_store
from backend.chat import router as chat_router
from backend.models.database import init_db
# ✅ Only import routers that actually exist.
# classes_router is NOT defined, so it's removed.
from backend.routers import admin_keys, auth, feedback, history, predict

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Validate configuration, then initialise all database schemas."""
    config.assert_configured()
    config.ensure_dirs()
    init_db()
    keys.init_schema()
    detection_store.init_schema()
    yield


app = FastAPI(title="Smart Waste Robot API", lifespan=lifespan)

# Add CORS middleware with the configured allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=False,          # authentication uses header, not cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers – order matters: keep these before the static fallback.
app.include_router(auth.router)         # prefix: /auth
app.include_router(predict.router)      # prefix: /predict
app.include_router(history.router)      # prefix: /history
app.include_router(feedback.router)     # prefix: /feedback
app.include_router(admin_keys.router)   # prefix: /admin/keys
app.include_router(chat_router.router)  # prefix: /chat


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness probe for monitoring."""
    return {"status": "ok"}


# Serve the frontend static files ONLY if the folder exists.
# This must be the LAST mount: it catches any path not matched by routers.
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")