import pytest
from fastapi.testclient import TestClient

from db.session import get_db
from main import app


def override_get_db():
    """Avoid opening a real database connection during validation tests."""

    yield None


@pytest.fixture
def client():
    """Create a test client with the database dependency overridden."""

    app.dependency_overrides[get_db] = override_get_db

    yield TestClient(app)

    app.dependency_overrides.clear()


# Missing contract_text should be rejected by FastAPI validation.
def test_analyze_rejects_missing_contract_text(client):
    response = client.post("/api/analyze", json={})

    assert response.status_code == 422


# Empty contract_text should be rejected by the minimum-length rule.
def test_analyze_rejects_empty_contract_text(client):
    response = client.post(
        "/api/analyze",
        json={"contract_text": ""},
    )

    assert response.status_code == 422


# Invalid contract_text types should be rejected.
def test_analyze_rejects_non_string_contract_text(client):
    response = client.post(
        "/api/analyze",
        json={"contract_text": 12345},
    )

    assert response.status_code == 422