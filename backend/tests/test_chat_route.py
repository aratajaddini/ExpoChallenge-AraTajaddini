"""/chat route contract: auth, validation, and grounded responses."""

import pytest
from fastapi.testclient import TestClient

from backend import config
from backend.chat import router as chat_router
from backend.main import app
from backend.security import require_api_key

client = TestClient(app)

_HITS = [
    {
        "id": "02-classes.md#0",
        "source": "02-classes.md",
        "section": "Detected classes",
        "text": "The model detects plastic, metal, paper, and glass.",
        "cosine": 0.90,
    }
]


@pytest.fixture(autouse=True)
def _authed(monkeypatch):
    """Bypass the API-key check and the on-disk KB index."""
    app.dependency_overrides[require_api_key] = lambda: "test"
    monkeypatch.setattr(config, "kb_is_built", lambda: True)
    monkeypatch.setattr(chat_router, "search", lambda q, top_k=None: _HITS)
    yield
    app.dependency_overrides.clear()


def test_chat_returns_grounded_answer():
    res = client.post("/chat", json={"question": "what classes does the model detect?"})

    assert res.status_code == 200
    body = res.json()
    assert body["grounded"] is True
    assert body["citations"][0]["source"] == "02-classes.md"
    assert "plastic" in body["answer"]


def test_chat_rejects_missing_question():
    assert client.post("/chat", json={}).status_code == 422


def test_chat_requires_api_key():
    app.dependency_overrides.clear()
    res = client.post("/chat", json={"question": "what classes are detected?"})

    assert res.status_code == 401