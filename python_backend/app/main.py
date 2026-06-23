from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Path as ApiPath, Query
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from merchant_ai.config import get_settings
from merchant_ai.graph.workflow import create_workflow
from merchant_ai.models import (
    ChatRequest,
    FeedbackRequest,
    KnowledgeSuggestionReviewRequest,
    RunCreateRequest,
    TopicBuildRequest,
    TopicReviewRequest,
    WikiCompressRequest,
)
from merchant_ai.services.answer import DailyReportService, FeedbackService
from merchant_ai.services.assets import TopicBuilderWorkflow
from merchant_ai.services.repositories import write_json
from merchant_ai.services.runs import AgentRunManager, AgentRunStreamService

settings = get_settings()
workflow = create_workflow(settings)

run_manager = AgentRunManager()
stream_service = AgentRunStreamService(run_manager, workflow.run, settings.merchant_id)
topic_assets = workflow.recall_service.topic_assets
doris_repository = workflow.node_worker.doris_repository
daily_report_service = DailyReportService(doris_repository)
feedback_service = FeedbackService(workflow.answer_repository, workflow.pending_store)
topic_builder_workflow = TopicBuilderWorkflow(settings, doris_repository, topic_assets)

app = FastAPI(title="yshopping Merchant AI Python", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "UP", "service": "yshopping-merchant-ai-python"}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> Dict[str, Any]:
    response = await workflow.run_async(request.message, request.merchant_id or settings.merchant_id, request.context)
    return response.model_dump(by_alias=True)


@app.post("/api/chat/stream")
def stream_chat(request: RunCreateRequest):
    return StreamingResponse(stream_service.stream(request), media_type="text/event-stream")


@app.post("/api/threads")
def create_thread(request: Optional[RunCreateRequest] = None) -> Dict[str, Any]:
    safe = request or RunCreateRequest()
    merchant_id = safe.merchant_id or settings.merchant_id
    thread = run_manager.create_thread(merchant_id, safe.context.topic if safe.context else "", safe.context)
    return {"success": True, "thread": jsonable_encoder(thread, by_alias=True)}


@app.get("/api/threads/{thread_id}")
def get_thread(thread_id: str = ApiPath(...)) -> Dict[str, Any]:
    thread = run_manager.get_thread(thread_id)
    if not thread:
        return {"success": False, "message": "thread not found", "threadId": thread_id}
    return {"success": True, "thread": jsonable_encoder(thread, by_alias=True)}


@app.get("/api/threads/{thread_id}/runs/{run_id}")
def get_run(thread_id: str, run_id: str) -> Dict[str, Any]:
    run = run_manager.get_run(run_id)
    if not run or run.thread_id != thread_id:
        return {"success": False, "message": "run not found", "threadId": thread_id, "runId": run_id}
    return {"success": True, "run": jsonable_encoder(run, by_alias=True)}


@app.get("/api/threads/{thread_id}/runs/{run_id}/events")
def get_run_events(thread_id: str, run_id: str) -> Dict[str, Any]:
    run = run_manager.get_run(run_id)
    if not run or run.thread_id != thread_id:
        return {"success": False, "message": "run not found", "threadId": thread_id, "runId": run_id}
    return {
        "success": True,
        "runId": run_id,
        "threadId": thread_id,
        "events": jsonable_encoder(run_manager.events(run_id), by_alias=True),
    }


@app.post("/api/threads/{thread_id}/runs/{run_id}/cancel")
def cancel_run(thread_id: str, run_id: str) -> Dict[str, Any]:
    run = run_manager.get_run(run_id)
    if not run or run.thread_id != thread_id:
        return {"success": False, "message": "run not found", "threadId": thread_id, "runId": run_id}
    return {"success": True, "run": jsonable_encoder(run_manager.cancel_run(run_id), by_alias=True)}


@app.post("/api/answers/{answer_id}/feedback")
def feedback(answer_id: str, request: FeedbackRequest) -> Dict[str, Any]:
    persisted = feedback_service.apply_feedback(answer_id, request.adopted, request.liked, request.disliked)
    return {"success": True, "persisted": persisted}


@app.get("/api/ops/knowledge-suggestions")
def operator_knowledge_suggestions(status: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    items = load_knowledge_suggestions()
    if status:
        items = [item for item in items if str(item.get("status", "")).lower() == status.lower()]
    return {"success": True, "items": items}


@app.post("/api/ops/knowledge-suggestions/{item_id}/review")
def review_operator_knowledge_suggestion(item_id: str, request: KnowledgeSuggestionReviewRequest) -> Dict[str, Any]:
    items = load_knowledge_suggestions()
    target = None
    for item in items:
        if item.get("id") == item_id:
            target = item
            break
    if target is None:
        target = {"id": item_id}
        items.append(target)
    target.update(
        {
            "status": "APPROVED" if request.approved else "REJECTED",
            "reviewer": request.reviewer,
            "reviewNote": request.review_note,
        }
    )
    save_knowledge_suggestions(items)
    return {"success": True, "item": target}


@app.get("/api/daily-report")
def daily_report(merchant_id: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    return daily_report_service.report(merchant_id or settings.merchant_id).model_dump(by_alias=True)


@app.post("/api/wiki/compress")
def compress_wiki(request: WikiCompressRequest) -> Dict[str, Any]:
    if not request.category_name:
        paths: Dict[str, str] = {}
        for name in ["平台商家规则", "电商交易", "电商退货", "电商客服工单", "电商理赔/赔付", "商品管理"]:
            rows = workflow.answer_repository.recent_answers_by_category(settings.merchant_id, name, 200)
            path = workflow.wiki_memory.compress_to_wiki(name, rows, request.manual_markdown)
            paths[name] = str(path)
        return {"success": True, "paths": paths}
    rows = workflow.answer_repository.recent_answers_by_category(settings.merchant_id, request.category_name, 200)
    path = workflow.wiki_memory.compress_to_wiki(request.category_name, rows, request.manual_markdown)
    return {"success": True, "path": str(path)}


@app.post("/api/es/rebuild-recall-index")
def rebuild_recall_index(merchant_id: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    docs = workflow.recall_service._load_documents()
    return {
        "success": True,
        "merchantId": merchant_id or settings.merchant_id,
        "mode": "local_recall",
        "indexedDocuments": len(docs),
        "message": "Python 版当前复用本地 wiki/runtime topic 召回；ES 配置可在该服务中继续扩展。",
    }


@app.get("/api/es/recall-mapping")
def recall_mapping() -> Dict[str, Any]:
    return {
        "success": True,
        "index": settings.es_index,
        "mapping": {
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "title": {"type": "text"},
                    "content": {"type": "text"},
                    "source_type": {"type": "keyword"},
                    "topic": {"type": "keyword"},
                    "table": {"type": "keyword"},
                }
            }
        },
    }


@app.post("/api/topics/build")
def build_topic_asset(request: TopicBuildRequest) -> Dict[str, Any]:
    if not request.merchant_id:
        request.merchant_id = settings.merchant_id
    return topic_builder_workflow.build(request)


@app.post("/api/topics/{topic}/tables/{table_name}/publish")
def publish_topic_asset(topic: str, table_name: str, request: TopicReviewRequest) -> Dict[str, Any]:
    result = topic_assets.publish(topic, table_name, request.approved, request.reviewer, request.review_note)
    if request.approved and result.get("status") == "PUBLISHED":
        result["esUpsert"] = {"success": True, "mode": "local_recall", "topic": topic, "tableName": table_name}
    return result


@app.post("/api/topics/{topic}/tables/{table_name}/schema-diff")
def diff_topic_table_schema(topic: str, table_name: str, request: Optional[TopicBuildRequest] = None) -> Dict[str, Any]:
    target = request or TopicBuildRequest()
    target.topic = topic
    target.table_name = table_name
    if not target.merchant_id:
        target.merchant_id = settings.merchant_id
    return topic_builder_workflow.diff_schema(target)


@app.post("/api/topics/{topic}/tables/{table_name}/refresh-incremental")
def refresh_topic_table_incrementally(topic: str, table_name: str, request: Optional[TopicBuildRequest] = None) -> Dict[str, Any]:
    target = request or TopicBuildRequest()
    target.topic = topic
    target.table_name = table_name
    if not target.merchant_id:
        target.merchant_id = settings.merchant_id
    return topic_builder_workflow.refresh_incremental(target)


@app.post("/api/topics/{topic}/tables/{table_name}/es-upsert")
def upsert_topic_table_recall_index(topic: str, table_name: str) -> Dict[str, Any]:
    return {"success": True, "mode": "local_recall", "topic": topic, "tableName": table_name}


@app.get("/api/topics/{topic}/assets")
def topic_assets_endpoint(topic: str) -> Dict[str, Any]:
    return topic_assets.list_topic(topic)


def knowledge_suggestions_path() -> Path:
    return settings.resolved_ops_path / "knowledge_suggestions.json"


def load_knowledge_suggestions() -> Any:
    path = knowledge_suggestions_path()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else data.get("items", [])
    except Exception:
        return []
    return []


def save_knowledge_suggestions(items: Any) -> None:
    write_json(knowledge_suggestions_path(), {"items": items})
