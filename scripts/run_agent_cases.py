#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue as queue_module
import sys
import time
from pathlib import Path
from typing import Any


CORE_CASES = [
    "最近 7 天查询订单 order_id_100 的订单明细，并查看退货时间、退货用户、退款金额，再看下对应 SPU 什么时候发布的。",
    "最近 7 天查询子订单 sub_order_id_100 的订单、退款和商品发布信息。",
    "最近 7 天下单最多的前 5 个 SPU，同时看它们的退款量和退款金额。",
    "最近 7 天退款金额最高的商品，看看对应下单量是否也高。",
    "最近 7 天有退款的订单，关联看一下对应商品发布时间。",
    "最近 7 天客服工单里涉及退款的订单，同时看这些订单是否发生了赔付。",
    "最近 7 天赔付金额较高的订单，关联看一下订单金额和退款金额。",
    "最近 7 天优惠券相关订单表现怎么样，是否带来了更多下单？",
    "最近 7 天供应链入库较多的商品，同时看下这些商品的下单情况。",
    "最近 7 天商品审核被拒的 SPU，后续有没有产生订单或退款？",
]

EXTENDED_CASES = [
    "最近30天退款率最高的前10个商品，同时看下单数、退款金额和商品发布时间，帮我判断哪些是高风险新品。",
    "最近60天赔付金额最高的前5个订单，关联看订单金额、退款金额、退款状态和对应客服工单情况。",
    "最近30天有客服工单的订单里，哪些后来发生了退款或赔付？分别占多少。",
    "最近90天下单量前20的SPU里，哪些退款率明显高于店铺平均水平？",
    "最近30天优惠券带来的订单里，退款率最高的商品有哪些，同时看券金额投入是否过高。",
    "最近45天供应链入库量前10的商品，后续下单表现和退款表现怎么样。",
    "最近30天审核被拒后又重新发布成功的商品，后续有没有产生订单、退款或赔付。",
    "最近30天退款金额最高的几天，对应主要是哪几个商品、哪些订单、有没有赔付。",
    "最近60天赔付单量较高的商品，关联看退款量、退款金额和商品发布时间。",
    "最近30天GMV最高的前5天，同时看退款金额、赔付金额、工单量，判断是否存在异常波动。",
    "最近90天哪些商品下单数不高，但退款率和赔付率都偏高。",
    "最近30天有退款的订单里，哪些对应的是最近15天新发布商品。",
    "最近60天客服工单量最高的前10个商品，同时看这些商品的退款量和赔付金额。",
    "最近30天退款状态为处理中或异常的订单，关联看商品发布时间和订单金额。",
    "最近90天高销量商品里，哪些是“下单多、退款多、赔付也多”的三高商品。",
    "最近30天优惠券活动覆盖的商品中，哪些商品虽然下单多，但退款金额也高。",
    "最近60天审核被拒商品中，哪些后来仍然产生了较多退款订单。",
    "最近30天赔付金额高的订单，对应商品是否也是退款高发商品。",
    "最近90天店铺整体退款率、赔付率、工单率走势是否同步上升，帮我分析可能原因。",
    "最近30天哪些商品最值得优先处理？请结合下单量、退款率、赔付金额、工单量一起判断。",
]


def run_case_worker(python_backend_path: str, question: str, merchant_id: str, case_index: int, queue: Any) -> None:
    started = time.time()
    sys.path.insert(0, python_backend_path)
    try:
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        response = client.post("/api/chat", json={"message": question, "merchantId": merchant_id}, timeout=240)
        body = response.json()
        debug = body.get("debugTrace") or {}
        harness_debug = debug.get("harness") or {}
        queue.put(
            {
                "case": case_index,
                "question": question,
                "statusCode": response.status_code,
                "elapsedSeconds": round(time.time() - started, 2),
                "answer": body.get("answer", ""),
                "route": (body.get("context") or {}).get("category", ""),
                "topic": body.get("categoryName", ""),
                "assetPack": debug.get("planningAssetPack", {}),
                "harness": {
                    "actions": harness_debug.get("actions", []),
                    "actionHistory": harness_debug.get("actionHistory", []),
                    "leadDecisions": harness_debug.get("leadDecisions", []),
                    "decisionReason": harness_debug.get("decisionReason", ""),
                    "performance": harness_debug.get("performance", {}),
                    "traceReplay": harness_debug.get("traceReplay", {}),
                },
                "plannerReflection": debug.get("plannerReflection", {}),
                "queryGraph": {
                    "intents": debug.get("planIntents", []),
                    "dependencies": debug.get("dependencies", []),
                    "validation": debug.get("queryGraphValidation", {}),
                },
                "sql": [
                    ((task.get("queryBundle") or {}).get("sql") or "")
                    for task in debug.get("taskResults", [])
                ],
                "taskResults": debug.get("taskResults", []),
                "nodeToolTraces": debug.get("nodeToolTraces", []),
                "freshnessReports": debug.get("freshnessReports", []),
                "evidenceGaps": debug.get("evidenceGaps", []),
                "verifiedEvidence": debug.get("verifiedEvidence", {}),
                "partialAnswerReason": debug.get("partialAnswerReason", ""),
                "performance": harness_debug.get("performance", {}),
                "traceReplay": harness_debug.get("traceReplay", {}),
            }
        )
    except Exception as exc:
        queue.put({"case": case_index, "question": question, "elapsedSeconds": round(time.time() - started, 2), "error": str(exc)})


def run_case_with_timeout(python_backend: Path, question: str, merchant_id: str, case_index: int, timeout_seconds: int) -> dict:
    started = time.time()
    queue = mp.Queue()
    process = mp.Process(
        target=run_case_worker,
        args=(str(python_backend), question, merchant_id, case_index, queue),
    )
    process.start()
    # Read before join: large debug traces may exceed the multiprocessing pipe
    # buffer, in which case the worker cannot finish queue.put() until the
    # parent drains it. Joining first therefore creates a false case timeout.
    try:
        result = queue.get(timeout=timeout_seconds)
    except queue_module.Empty:
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        return {
            "case": case_index,
            "question": question,
            "elapsedSeconds": round(time.time() - started, 2),
            "error": "CASE_TIMEOUT after %s seconds" % timeout_seconds,
        }
    process.join(5)
    if process.is_alive():
        process.terminate()
        process.join(5)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Merchant AI harness regression cases against the local FastAPI app.")
    parser.add_argument("--merchant-id", default="100")
    parser.add_argument("--output", default="python_backend/.merchant-ai/agent_case_results.json")
    parser.add_argument("--suite", choices=["core", "extended", "all"], default="core")
    parser.add_argument("--questions-file", default="", help="Optional JSON array of questions; overrides --suite.")
    parser.add_argument("--case-timeout-seconds", type=int, default=240)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    python_backend = repo_root / "python_backend"
    cases = CORE_CASES if args.suite == "core" else EXTENDED_CASES if args.suite == "extended" else CORE_CASES + EXTENDED_CASES
    if args.questions_file:
        questions_path = Path(args.questions_file).expanduser().resolve()
        payload = json.loads(questions_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(item, str) and item.strip() for item in payload):
            raise ValueError("--questions-file must contain a non-empty JSON string array")
        cases = [item.strip() for item in payload]
    results = []
    output = repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    for index, question in enumerate(cases, start=1):
        result = run_case_with_timeout(python_backend, question, args.merchant_id, index, args.case_timeout_seconds)
        results.append(result)
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(
            json.dumps(
                {
                    "case": index,
                    "elapsedSeconds": result.get("elapsedSeconds"),
                    "statusCode": result.get("statusCode"),
                    "error": result.get("error", ""),
                    "taskCount": len(((result.get("queryGraph") or {}).get("intents") or [])),
                    "gapCount": len(result.get("evidenceGaps") or []),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"success": True, "cases": len(results), "output": str(output), "performanceSummary": summarize_performance(results)}, ensure_ascii=False))
    return 0


def summarize_performance(results: list[dict[str, Any]]) -> dict[str, Any]:
    slow_cases = sorted(
        [
            {
                "case": item.get("case"),
                "elapsedSeconds": item.get("elapsedSeconds", 0),
                "question": str(item.get("question") or "")[:80],
                "totalDurationMs": ((item.get("performance") or {}).get("totalDurationMs") or 0),
            }
            for item in results
        ],
        key=lambda item: float(item.get("elapsedSeconds") or 0),
        reverse=True,
    )[:8]
    timeout_count = 0
    sql_errors = 0
    row_count = 0
    for item in results:
        perf = item.get("performance") or {}
        timeout_count += int(((perf.get("llm") or {}).get("timeouts") or 0))
        sql_errors += int(((perf.get("sql") or {}).get("errors") or 0))
        row_count += int(((perf.get("sql") or {}).get("rows") or 0))
    return {
        "slowCases": slow_cases,
        "llmTimeouts": timeout_count,
        "sqlErrors": sql_errors,
        "sqlRows": row_count,
    }


if __name__ == "__main__":
    raise SystemExit(main())
