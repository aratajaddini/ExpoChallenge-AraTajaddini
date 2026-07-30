from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.models.database import init_db
from backend.routers import predict, history, feedback

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup logic
    init_db()
    yield
    # shutdown logic (nothing to clean up yet)


app = FastAPI(title="Smart Waste Robot API", lifespan=lifespan)

app.include_router(predict.router)
app.include_router(history.router)
app.include_router(feedback.router)


@app.get("/")
def root() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
