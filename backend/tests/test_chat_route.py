"""Integration test for the /chat endpoint."""

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_chat_returns_reply():
    res = client.post("/chat", json={"message": "how many plastic?"})
    assert res.status_code == 200
    assert isinstance(res.json()["reply"], str)


def test_chat_rejects_missing_message():
    assert client.post("/chat", json={}).status_code == 422
