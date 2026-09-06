"""Smoke tests. These must not require a database."""

from fastapi.testclient import TestClient

from app.enums import OUTCOME_LABEL_VI, Outcome
from app.main import app

client = TestClient(app)


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_index_renders():
    response = client.get("/")
    assert response.status_code == 200
    assert "trích xuất thông tin" in response.text


def test_all_five_outcomes_have_vietnamese_labels():
    """The five outcome classes are fixed by the registered project description."""
    assert len(Outcome) == 5
    for outcome in Outcome:
        assert OUTCOME_LABEL_VI[outcome]
