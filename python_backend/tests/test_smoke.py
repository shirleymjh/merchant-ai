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
    assert body["debugTrace"]["harness"]["mode"] == "harness"
    assert body["debugTrace"]["planningAssetPack"]["skills"]
    agent_trace = body["debugTrace"]["agentTrace"]
    assert agent_trace
    if any("planner.no_llm_configured" in item for item in agent_trace):
        assert body["debugTrace"]["queryGraphValidation"]["gaps"]
        assert not body["debugTrace"]["taskResults"]
        assert not body["debugTrace"]["planIntents"]
    assert "PY_FALLBACK:EXPLICIT_ORDER_LOOKUP" not in str(body["debugTrace"])
    assert body["debugTrace"].get("planIntents") or body["debugTrace"]["queryGraphValidation"]["gaps"] or body["debugTrace"].get("taskResults")
