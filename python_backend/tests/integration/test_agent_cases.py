import os

import pytest
from fastapi.testclient import TestClient


AGENT_CASES = [
    "查询子订单 sub_order_id_100 的订单、退款和商品发布信息。",
    "最近 90 天下单最多的前 5 个 SPU，同时看它们的退款量和退款金额。",
    "最近 30 天退款金额最高的商品，看看对应下单量是否也高。",
    "最近 90 天有退款的订单，关联看一下对应商品发布时间。",
    "最近 30 天客服工单里涉及退款的订单，同时看这些订单是否发生了赔付。",
    "最近 90 天赔付金额较高的订单，关联看一下订单金额和退款金额。",
    "最近 90 天优惠券相关订单表现怎么样，是否带来了更多下单？",
    "最近 90 天供应链入库较多的商品，同时看下这些商品的下单情况。",
    "最近 30 天商品审核被拒的 SPU，后续有没有产生订单或退款？",
]


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_AGENT_INTEGRATION") != "1", reason="set RUN_AGENT_INTEGRATION=1 to run real LLM/Doris cases")
@pytest.mark.parametrize("question", AGENT_CASES)
def test_real_agent_cases_do_not_crash(question):
    from app.main import app

    client = TestClient(app)
    response = client.post("/api/chat", json={"message": question, "merchantId": "100"}, timeout=240)
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["debugTrace"]["planIntents"] or body["debugTrace"]["queryGraphValidation"]["gaps"]
    assert "top_spu" not in str(body["debugTrace"].get("queryGraphValidation", {})).lower()
