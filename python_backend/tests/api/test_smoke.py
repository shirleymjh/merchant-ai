import os

import pytest
from fastapi.testclient import TestClient

os.environ["OPENAI_API_KEY"] = ""
os.environ["YSHOPPING_LLM_API_KEY"] = ""

from app.main import create_app
from merchant_ai.config import Settings


@pytest.fixture
def client(tmp_path):
    return TestClient(
        create_app(
            Settings(
                merchant_id="100",
                allowed_merchant_ids="100",
                identity_auth_required=False,
                llm_api_key="",
                harness_workspace_path=str(tmp_path),
            )
        )
    )


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "UP"


def test_greeting_chat_is_fail_closed_when_grounded_models_are_unavailable(client):
    response = client.post("/api/chat", json={"message": "你好", "merchantId": "100"})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "GROUNDED_ONLINE_RUNTIME_UNAVAILABLE"


def test_query_is_fail_closed_when_grounded_models_are_unavailable(client):
    response = client.post(
        "/api/chat",
        json={"message": "最近30天GMV最高的前5天，同时看退款金额、赔付金额、工单量", "merchantId": "100"},
    )
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "GROUNDED_ONLINE_RUNTIME_UNAVAILABLE"
