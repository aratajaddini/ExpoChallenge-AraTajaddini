"""Tests for the /auth/verify endpoint."""


def test_verify_accepts_static_key(client):
    """The configured API_KEY authenticates as the admin identity."""
    response = client.get("/auth/verify", headers={"X-API-Key": "testing_key"})
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["identity"] == "admin"


def test_verify_rejects_missing_key(client):
    """No X-API-Key header means 401."""
    assert client.get("/auth/verify").status_code == 401


def test_verify_rejects_bad_key(client):
    """A wrong key means 401, not 500."""
    response = client.get("/auth/verify", headers={"X-API-Key": "nope"})
    assert response.status_code == 401
