"""Unit tests for the offline intent layer (no database required)."""

import pytest

from backend.analytics import intent


@pytest.fixture
def fake_counts(monkeypatch):
    """Stub the data layer with deterministic counts."""
    counts = {"plastic": 12, "metal": 3, "paper": 7}
    monkeypatch.setattr(intent.data, "count_by_class", lambda: counts)
    monkeypatch.setattr(intent.data, "total_count", lambda: sum(counts.values()))
    monkeypatch.setattr(intent.data, "most_common_class", lambda: "plastic")
    return counts


def test_class_specific_count(fake_counts):
    assert "12 plastic" in intent.answer("how many plastic items?")


def test_unknown_class_returns_zero(fake_counts):
    assert "0 glass" in intent.answer("how much glass?")


def test_total(fake_counts):
    assert "22" in intent.answer("what is the total?")


def test_most_common(fake_counts):
    assert "plastic" in intent.answer("which is the most common class?")


def test_fallback():
    assert "counts per class" in intent.answer("what is the weather")


def test_empty_database(monkeypatch):
    monkeypatch.setattr(intent.data, "count_by_class", dict)
    monkeypatch.setattr(intent.data, "most_common_class", lambda: None)
    assert "No detections yet" in intent.answer("most common class")
