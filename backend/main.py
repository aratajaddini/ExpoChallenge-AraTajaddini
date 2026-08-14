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

# CORS: origins come from config.ALLOWED_ORIGINS (never "*", see
# config.assert_configured). Auth is header-based (X-API-Key), so
# credentialed requests are not needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routers first — the static mount below is a catch-all.
app.include_router(auth.router)  # /auth
app.include_router(predict.router)  # /predict
app.include_router(history.router)  # /history
app.include_router(feedback.router)  # /feedback
app.include_router(admin_keys.router)  # /admin/keys
app.include_router(chat_router.router)  # /chat


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness probe for monitoring."""
    return {"status": "ok"}


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
