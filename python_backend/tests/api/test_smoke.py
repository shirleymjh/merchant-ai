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


def test_greeting_chat_uses_preflight_fast_path_when_grounded_models_are_unavailable(client):
    response = client.post("/api/chat", json={"message": "你好", "merchantId": "100"})
    assert response.status_code == 200
    payload = response.json()
    assert "商家 AI 助手" in payload["answer"]
    assert payload["categoryName"] == "GREETING"
    assert payload["dataRows"] == []
    assert payload["dorisTables"] == []


def test_assistant_capability_chat_uses_preflight_fast_path(client):
    response = client.post(
        "/api/chat",
        json={"message": "你能做什么", "merchantId": "100"},
    )

    assert response.status_code == 200
    assert response.json()["categoryName"] == "GREETING"


def test_empty_and_write_requests_are_stopped_before_grounded_core(client):
    empty = client.post(
        "/api/chat",
        json={"message": "", "merchantId": "100"},
    )
    write = client.post(
        "/api/chat",
        json={"message": "删除昨天的订单", "merchantId": "100"},
    )

    assert empty.status_code == 200
    assert empty.json()["clarification"]["type"] == "business_scope"
    assert write.status_code == 200
    assert write.json()["categoryName"] == "UNSUPPORTED_WRITE"
    assert write.json()["clarification"]["type"] == "write_operation"


def test_pending_clarification_reply_still_enters_grounded_core(client):
    response = client.post(
        "/api/chat",
        json={
            "message": "最近7天",
            "merchantId": "100",
            "context": {
                "pendingClarificationStage": "BUSINESS_SCOPE",
                "pendingClarificationType": "business_scope",
                "pendingQuestion": "看订单量",
            },
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == (
        "GROUNDED_ONLINE_RUNTIME_UNAVAILABLE"
    )


def test_query_is_fail_closed_when_grounded_models_are_unavailable(client):
    response = client.post(
        "/api/chat",
        json={"message": "最近30天GMV最高的前5天，同时看退款金额、赔付金额、工单量", "merchantId": "100"},
    )
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "GROUNDED_ONLINE_RUNTIME_UNAVAILABLE"
