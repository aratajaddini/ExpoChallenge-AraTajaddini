import os

import pytest
from fastapi.testclient import TestClient

# Set environment variables before importing the app so config picks them up
os.environ["API_KEY"] = "testing_key"
os.environ["REQUIRE_MODEL"] = "0"   # CI has no best.pt; model is mocked

from backend.main import app
import backend.inference as inference_module


class DummyProbs:
    top1 = 0
    top1conf = 0.95
    data = [0.95, 0.02, 0.01, 0.01, 0.01]


class DummyResult:
    probs = DummyProbs()


class DummyModel:
    names = {
        0: "Glass",
        1: "Metal",
        2: "Paper",
        3: "Plastic",
        4: "Waste",
    }

    def __call__(self, image, verbose=False):
        return [DummyResult()]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        inference_module,
        "_get_model",
        lambda: DummyModel(),
    )

    with TestClient(app) as test_client:
        yield test_client