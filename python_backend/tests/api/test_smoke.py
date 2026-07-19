import os

from fastapi.testclient import TestClient

os.environ["OPENAI_API_KEY"] = ""
os.environ["YSHOPPING_LLM_API_KEY"] = ""

from app.main import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "UP"


def test_greeting_chat_contract():
    client = TestClient(app)
    response = client.post("/api/chat", json={"message": "你好", "merchantId": "100"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["thinkingSteps"]
    assert body["context"]["answerMode"] == "CHAT"


def test_query_graph_smoke_contract():
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={"message": "最近30天GMV最高的前5天，同时看退款金额、赔付金额、工单量", "merchantId": "100"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert "debugTrace" not in body
    assert "debug_trace" not in body
