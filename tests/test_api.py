import pytest
from fastapi.testclient import TestClient

from eris_ai.api import create_app


SERVICE_KEY = "test-service-key-at-least-24-characters"


@pytest.fixture
def client():
    with TestClient(create_app(service_key=SERVICE_KEY, demo_enabled=False)) as test_client:
        yield test_client


def test_health_is_available_without_model(client):
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("path", ["/ready", "/v1/model-info"])
def test_protected_status_endpoints_require_authentication(client, path):
    assert client.get(path).status_code == 401


@pytest.mark.parametrize("path", ["/ready", "/v1/model-info"])
def test_status_endpoints_report_unconfigured_model(client, path):
    response = client.get(path, headers={"X-ERIS-Service-Key": SERVICE_KEY})
    assert response.status_code == 503
    assert response.json()["detail"] == "Model is not configured"


@pytest.mark.parametrize(
    ("path", "body"),
    [
        (
            "/v1/predict",
            {"schema_version": "unconfigured", "employee": {"employee_id": "1", "features": {}}},
        ),
        (
            "/v1/explain",
            {"schema_version": "unconfigured", "employee": {"employee_id": "1", "features": {}}},
        ),
        (
            "/v1/predict-batch",
            {
                "schema_version": "unconfigured",
                "employees": [{"employee_id": "1", "features": {}}],
            },
        ),
    ],
)
def test_prediction_endpoints_report_unconfigured_model(client, path, body):
    response = client.post(path, json=body, headers={"X-ERIS-Service-Key": SERVICE_KEY})
    assert response.status_code == 503
    assert response.json()["detail"] == "Model is not configured"


def test_service_key_is_required():
    with pytest.raises(RuntimeError, match="ERIS_SERVICE_KEY"):
        with TestClient(create_app(service_key="")):
            pass
