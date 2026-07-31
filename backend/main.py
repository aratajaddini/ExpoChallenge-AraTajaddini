"""FastAPI application entrypoint for the Smart Waste Robot API."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend import keys
from backend.analytics import store as detection_store
from backend.models.database import init_db
from backend.routers import predict, history, feedback, admin_keys

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise all database schemas on startup."""
    # Startup
    init_db()                      # core tables (predictions, history, feedback)
    keys.init_schema()             # api_keys table
    detection_store.init_schema()  # detections table (analytics)
    yield
    # Shutdown: nothing to clean up yet


app = FastAPI(title="Smart Waste Robot API", lifespan=lifespan)

# Register API routers first (they have priority)
app.include_router(predict.router)
app.include_router(history.router)
app.include_router(feedback.router)
app.include_router(admin_keys.router)   # prefix: /admin/keys


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness probe for monitoring."""
    return {"status": "ok"}


# Serve the frontend static files ONLY if the folder exists.
# This must be the LAST mount: it catches any path not matched by routers.
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
