"""CORS and auth-router regression tests."""

from backend import config

ALLOWED = config.ALLOWED_ORIGINS[0]
BLOCKED = "https://evil.example.com"


def test_cors_allows_configured_origin(client):
    """A GET from an allowed Origin echoes the CORS header back."""
    response = client.get("/health", headers={"Origin": ALLOWED})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED


def test_cors_rejects_unknown_origin(client):
    """An Origin outside ALLOWED_ORIGINS gets no CORS header."""
    response = client.get("/health", headers={"Origin": BLOCKED})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_cors_preflight_allows_configured_origin(client):
    """OPTIONS preflight succeeds and permits the X-API-Key header."""
    response = client.options(
        "/predict",
        headers={
            "Origin": ALLOWED,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-api-key",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED


def test_cors_never_wildcard():
    """Guards against a regression to allow_origins=['*']."""
    assert "*" not in config.ALLOWED_ORIGINS
