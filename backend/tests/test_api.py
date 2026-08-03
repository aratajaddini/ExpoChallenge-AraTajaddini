import io
from fastapi.testclient import TestClient

# Define the authentication header for the test environment
HEADERS = {"X-API-Key": "testing_key"}


def _fake_image():
    return io.BytesIO(b"fakeimagebytes")


def test_root(client: TestClient):
    # Root endpoint is usually public, so no headers needed here
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict(client: TestClient):
    files = {"file": ("test.jpg", _fake_image(), "image/jpeg")}
    # Added headers=HEADERS
    response = client.post("/predict", files=files, headers=HEADERS)
    
    assert response.status_code == 200
    data = response.json()

    assert data["top_class"] == "Glass"
    assert set(data["scores"].keys()) == {"Glass", "Metal", "Paper", "Plastic", "Waste"}
    assert data["id"] >= 1


def test_history(client: TestClient):
    files = {"file": ("test.jpg", _fake_image(), "image/jpeg")}
    # Predict must also send the header to create the history entry
    client.post("/predict", files=files, headers=HEADERS)
    
    # Added headers=HEADERS
    response = client.get("/history", headers=HEADERS)
    assert response.status_code == 200
    assert len(response.json()) >= 1


def test_feedback(client: TestClient):
    files = {"file": ("test.jpg", _fake_image(), "image/jpeg")}
    prediction = client.post("/predict", files=files, headers=HEADERS).json()
    
    # Added headers=HEADERS
    response = client.post(
        "/feedback",
        json={"prediction_id": prediction["id"], "correct_class": "Metal"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["correct_class"] == "Metal"


def test_feedback_not_found(client: TestClient):
    # Added headers=HEADERS
    response = client.post(
        "/feedback",
        json={"prediction_id": 9999, "correct_class": "Metal"},
        headers=HEADERS,
    )
    assert response.status_code == 404
