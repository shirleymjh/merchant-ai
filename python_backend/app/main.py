from __future__ import annotations

import json
import hmac
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Path as ApiPath, Query, status
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from merchant_ai.config import Settings, get_settings
from merchant_ai.graph.workflow import create_workflow
from merchant_ai.models import (
    ChatRequest,
    FeedbackRequest,
    GoldenEvaluationRequest,
    KnowledgeSuggestionReviewRequest,
    MemoryCleanupRequest,
    MemoryItemPatchRequest,
    MemoryRecallEvaluationRequest,
    RunCreateRequest,
    SkillDraftReviewRequest,
    SkillEvaluationRequest,
    TopicBuildRequest,
    TopicReviewRequest,
    WikiCompressRequest,
)
from merchant_ai.services.answer import DailyReportService, FeedbackService
from merchant_ai.services.assets import SemanticAssetGovernanceService, TopicBuilderWorkflow
from merchant_ai.services.evaluation import GoldenEvaluationService
from merchant_ai.services.memory import MemoryManagementService
from merchant_ai.services.recall_index import RecallIndexManager
from merchant_ai.services.repositories import write_json
from merchant_ai.services.runs import AgentAsyncRunService, AgentRunManager, AgentRunStreamService, run_duration_ms, run_summary_payload
from merchant_ai.services.security import Permission, authorize_merchant_access, merchant_principal, ops_principal
from merchant_ai.services.skill_drafts import SkillDraftService
from merchant_ai.services.skill_evaluation import SkillEvaluationService

settings: Settings
workflow: Any
run_manager: AgentRunManager
stream_service: AgentRunStreamService
async_run_service: AgentAsyncRunService
topic_assets: Any
doris_repository: Any
daily_report_service: DailyReportService
feedback_service: FeedbackService
memory_management_service: MemoryManagementService
skill_draft_service: SkillDraftService
skill_evaluation_service: SkillEvaluationService
golden_evaluation_service: GoldenEvaluationService
topic_builder_workflow: TopicBuilderWorkflow
semantic_governance: SemanticAssetGovernanceService
recall_index_manager: RecallIndexManager


def _init_services(runtime_settings: Optional[Settings] = None) -> None:
    global settings
    global workflow
    global run_manager
    global stream_service
    global async_run_service
    global topic_assets
    global doris_repository
    global daily_report_service
    global feedback_service
    global memory_management_service
    global skill_draft_service
    global skill_evaluation_service
    global golden_evaluation_service
    global topic_builder_workflow
    global semantic_governance
    global recall_index_manager

    settings = runtime_settings or get_settings()
    workflow = create_workflow(settings)
    run_manager = AgentRunManager(settings)
    stream_service = AgentRunStreamService(run_manager, workflow.run, settings.merchant_id)
    async_run_service = AgentAsyncRunService(
        run_manager,
        workflow.run,
        settings.merchant_id,
        max_workers=settings.max_concurrent_sub_agents,
    )
    topic_assets = workflow.recall_service.topic_assets
    doris_repository = workflow.node_worker.doris_repository
    daily_report_service = DailyReportService(doris_repository)
    feedback_service = FeedbackService(workflow.answer_repository, workflow.pending_store, workflow.memory_store)
    memory_management_service = MemoryManagementService(settings, workflow.memory_store)
    skill_draft_service = SkillDraftService(settings)
    skill_evaluation_service = SkillEvaluationService(settings, workflow.answer_service)
    golden_evaluation_service = GoldenEvaluationService(settings)
    topic_builder_workflow = TopicBuilderWorkflow(settings, doris_repository, topic_assets)
    semantic_governance = SemanticAssetGovernanceService(settings, doris_repository, topic_assets)
    recall_index_manager = RecallIndexManager(
        settings,
        workflow.recall_service,
        cache_clearers=[
            workflow.asset_builder.clear_cache,
            workflow.node_worker.doris_repository.clear_cache,
        ],
    )


def require_ops_token(
    authorization: Optional[str] = Header(default=None),
    x_ops_token: Optional[str] = Header(default=None, alias="X-Ops-Token"),
) -> None:
    expected = str(settings.ops_token or "").strip()
    if not expected:
        return
    bearer_prefix = "Bearer "
    provided = str(x_ops_token or "").strip()
    if not provided and authorization and authorization.startswith(bearer_prefix):
        provided = authorization[len(bearer_prefix) :].strip()
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid ops token",
            headers={"WWW-Authenticate": "Bearer"},
        )


OpsAuth = Depends(require_ops_token)


def require_merchant_access(merchant_id: str) -> str:
    target = str(merchant_id or settings.merchant_id).strip()
    return authorize_merchant_access(settings, merchant_principal(target), target, Permission.CHAT_RUN)


def require_ops_merchant_access(merchant_id: str, permission: Permission = Permission.OPS_READ) -> str:
    return authorize_merchant_access(settings, ops_principal(), merchant_id or settings.merchant_id, permission)


def create_app(runtime_settings: Optional[Settings] = None) -> FastAPI:
    _init_services(runtime_settings)
    application = FastAPI(title="yshopping Merchant AI Python", version="0.1.0")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=bool(settings.cors_allow_credentials),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(router)
    return application


router = APIRouter()


@router.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "UP", "service": "yshopping-merchant-ai-python"}


@router.post("/api/chat")
async def chat(request: ChatRequest) -> Dict[str, Any]:
    merchant_id = require_merchant_access(request.merchant_id or settings.merchant_id)
    thread = run_manager.create_thread(merchant_id, request.context.topic if request.context else "", request.context)
    run = run_manager.create_run(thread.thread_id, merchant_id, request.message)

    def listener(event_type: str, node: str, payload: Dict[str, Any]) -> None:
        run_manager.append_event(run.run_id, thread.thread_id, event_type, node, payload)

    try:
        response = await workflow.run_async(
            request.message,
            merchant_id,
            request.context,
            listener,
            thread.thread_id,
            run.run_id,
            message_history=request.message_history,
        )
        run_manager.complete_run(run.run_id, response)
        return response.model_dump(by_alias=True)
    except Exception as exc:
        run_manager.fail_run(run.run_id, str(exc))
        raise


@router.post("/api/chat/stream")
def stream_chat(request: RunCreateRequest):
    request.merchant_id = require_merchant_access(request.merchant_id or settings.merchant_id)
    return StreamingResponse(stream_service.stream(request), media_type="text/event-stream")


@router.post("/api/runs/async")
def create_async_run(request: RunCreateRequest) -> Dict[str, Any]:
    request.merchant_id = require_merchant_access(request.merchant_id or settings.merchant_id)
    run = async_run_service.submit(request)
    return {
        "success": True,
        "mode": "async",
        "threadId": run.thread_id,
        "runId": run.run_id,
        "run": jsonable_encoder(run_summary_payload(run, run_duration_ms(run)), by_alias=True),
        "links": {
            "run": "/api/threads/%s/runs/%s" % (run.thread_id, run.run_id),
            "events": "/api/threads/%s/runs/%s/events" % (run.thread_id, run.run_id),
            "trace": "/api/threads/%s/runs/%s/trace" % (run.thread_id, run.run_id),
            "checkpoint": "/api/threads/%s/runs/%s/checkpoint" % (run.thread_id, run.run_id),
        },
    }


@router.get("/api/runs")
def list_runs(
    limit: int = Query(default=50, ge=1, le=200),
    status: Optional[str] = Query(default=None),
    merchant_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    effective_merchant_id = require_merchant_access(merchant_id or settings.merchant_id)
    runs = run_manager.list_runs(limit=limit, status=status or "", merchant_id=effective_merchant_id)
    summaries = [run_summary_payload(run, run_duration_ms(run)) for run in runs]
    return {"success": True, "runs": jsonable_encoder(summaries, by_alias=True)}


@router.get("/api/runs/dashboard")
def runs_dashboard(
    limit: int = Query(default=50, ge=1, le=200),
    status: Optional[str] = Query(default=None),
    merchant_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    effective_merchant_id = require_merchant_access(merchant_id or settings.merchant_id)
    dashboard = run_manager.dashboard(limit=limit, status=status or "", merchant_id=effective_merchant_id)
    return {"success": True, "dashboard": jsonable_encoder(dashboard, by_alias=True)}


def _runtime_trace() -> Dict[str, Any]:
    traces: Dict[str, Any] = {}
    for name, owner in [("planner", workflow.planner), ("node", workflow.node_worker)]:
        service = getattr(owner, "tool_runtime_service", None)
        if service is not None:
            traces[name] = service.trace()
    metrics = []
    alerts = []
    events = []
    rate_limits: Dict[str, Any] = {}
    load_balancer: Dict[str, Any] = {}
    for name, trace in traces.items():
        for item in trace.get("metrics", {}).get("tools", []):
            next_item = dict(item)
            next_item["runtime"] = name
            metrics.append(next_item)
        alerts.extend(trace.get("alerts", []))
        for event in trace.get("events", []):
            next_event = dict(event)
            next_event["runtime"] = name
            events.append(next_event)
        rate_limits[name] = trace.get("rateLimits", {})
        load_balancer[name] = trace.get("loadBalancer", {})
    return {
        "metrics": {"tools": metrics},
        "alerts": alerts,
        "events": events[-200:],
        "rateLimits": rate_limits,
        "loadBalancer": load_balancer,
        "degraded": {
            "answerRepository": workflow.answer_repository.trace()
            if hasattr(workflow.answer_repository, "trace")
            else {},
            "merchantService": workflow.merchant_service.trace()
            if hasattr(workflow.merchant_service, "trace")
            else {},
            "doris": workflow.node_worker.doris_repository.cache_trace()
            if hasattr(workflow.node_worker.doris_repository, "cache_trace")
            else {},
        },
        "config": settings.grouped_summary(),
    }


@router.get("/api/runtime/metrics")
def runtime_metrics(_auth: None = OpsAuth) -> Dict[str, Any]:
    trace = _runtime_trace()
    return {
        "success": True,
        "metrics": jsonable_encoder(trace["metrics"], by_alias=True),
        "rateLimits": trace["rateLimits"],
        "loadBalancer": trace["loadBalancer"],
        "degraded": jsonable_encoder(trace["degraded"], by_alias=True),
        "config": jsonable_encoder(trace["config"], by_alias=True),
    }


@router.get("/api/runtime/alerts")
def runtime_alerts(_auth: None = OpsAuth) -> Dict[str, Any]:
    trace = _runtime_trace()
    return {"success": True, "alerts": jsonable_encoder(trace["alerts"], by_alias=True)}


@router.get("/ops/runs", response_class=HTMLResponse)
def ops_runs_dashboard(_auth: None = OpsAuth) -> HTMLResponse:
    return HTMLResponse(RUNS_DASHBOARD_HTML)


@router.post("/api/threads")
def create_thread(request: Optional[RunCreateRequest] = None) -> Dict[str, Any]:
    safe = request or RunCreateRequest()
    merchant_id = require_merchant_access(safe.merchant_id or settings.merchant_id)
    thread = run_manager.create_thread(merchant_id, safe.context.topic if safe.context else "", safe.context)
    return {"success": True, "thread": jsonable_encoder(thread, by_alias=True)}


@router.get("/api/threads/{thread_id}")
def get_thread(thread_id: str = ApiPath(...)) -> Dict[str, Any]:
    thread = run_manager.get_thread(thread_id)
    if not thread:
        return {"success": False, "message": "thread not found", "threadId": thread_id}
    return {"success": True, "thread": jsonable_encoder(thread, by_alias=True)}


@router.get("/api/threads/{thread_id}/runs/{run_id}")
def get_run(thread_id: str, run_id: str) -> Dict[str, Any]:
    run = run_manager.get_run(run_id)
    if not run or run.thread_id != thread_id:
        return {"success": False, "message": "run not found", "threadId": thread_id, "runId": run_id}
    return {"success": True, "run": jsonable_encoder(run, by_alias=True)}


@router.get("/api/threads/{thread_id}/runs/{run_id}/events")
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


@router.get("/api/threads/{thread_id}/runs/{run_id}/trace")
def get_run_trace(thread_id: str, run_id: str, _auth: None = OpsAuth) -> Dict[str, Any]:
    run = run_manager.get_run(run_id)
    if not run or run.thread_id != thread_id:
        return {"success": False, "message": "run not found", "threadId": thread_id, "runId": run_id}
    trace = run_manager.trace(run_id)
    if trace is None:
        return {"success": False, "message": "trace not found", "threadId": thread_id, "runId": run_id}
    return {"success": True, "runId": run_id, "threadId": thread_id, "trace": trace}


@router.get("/api/threads/{thread_id}/runs/{run_id}/checkpoint")
def get_run_checkpoint(thread_id: str, run_id: str, _auth: None = OpsAuth) -> Dict[str, Any]:
    run = run_manager.get_run(run_id)
    if not run or run.thread_id != thread_id:
        return {"success": False, "message": "run not found", "threadId": thread_id, "runId": run_id}
    try:
        checkpoint = workflow.checkpoint_state_summary(thread_id, run_id)
    except Exception as exc:
        checkpoint = {"checkpointRef": run.checkpoint_ref, "error": str(exc)[:500], "hasValues": False}
    return {"success": True, "runId": run_id, "threadId": thread_id, "checkpoint": jsonable_encoder(checkpoint, by_alias=True)}


@router.post("/api/threads/{thread_id}/runs/{run_id}/cancel")
def cancel_run(thread_id: str, run_id: str) -> Dict[str, Any]:
    run = run_manager.get_run(run_id)
    if not run or run.thread_id != thread_id:
        return {"success": False, "message": "run not found", "threadId": thread_id, "runId": run_id}
    return {"success": True, "run": jsonable_encoder(async_run_service.cancel(run_id), by_alias=True)}


@router.post("/api/answers/{answer_id}/feedback")
def feedback(answer_id: str, request: FeedbackRequest) -> Dict[str, Any]:
    persisted = feedback_service.apply_feedback(answer_id, request.adopted, request.liked, request.disliked)
    return {"success": True, "persisted": persisted}


@router.get("/api/memory/{merchant_id}")
def get_memory(merchant_id: str, include_inactive: bool = Query(default=True), _auth: None = OpsAuth) -> Dict[str, Any]:
    require_merchant_access(merchant_id)
    return memory_management_service.get_memory(merchant_id, include_inactive=include_inactive)


@router.patch("/api/memory/{merchant_id}/items/{memory_id}")
def patch_memory_item(
    merchant_id: str,
    memory_id: str,
    request: MemoryItemPatchRequest,
    _auth: None = OpsAuth,
) -> Dict[str, Any]:
    require_merchant_access(merchant_id)
    return memory_management_service.patch_item(merchant_id, memory_id, request)


@router.delete("/api/memory/{merchant_id}/items/{memory_id}")
def delete_memory_item(
    merchant_id: str,
    memory_id: str,
    hard_delete: bool = Query(default=False),
    _auth: None = OpsAuth,
) -> Dict[str, Any]:
    require_merchant_access(merchant_id)
    return memory_management_service.delete_item(merchant_id, memory_id, hard_delete=hard_delete)


@router.post("/api/memory/{merchant_id}/cleanup")
def cleanup_memory(merchant_id: str, request: MemoryCleanupRequest, _auth: None = OpsAuth) -> Dict[str, Any]:
    require_merchant_access(merchant_id)
    return memory_management_service.cleanup_expired(merchant_id, hard_delete=request.hard_delete, dry_run=request.dry_run)


@router.post("/api/memory/{merchant_id}/recall-eval")
def evaluate_memory_recall(
    merchant_id: str,
    request: MemoryRecallEvaluationRequest,
    _auth: None = OpsAuth,
) -> Dict[str, Any]:
    require_merchant_access(merchant_id)
    return memory_management_service.evaluate_recall(
        merchant_id,
        request.cases,
        budget_tokens=request.budget_tokens,
        budget_chars=request.budget_chars,
    )


@router.get("/api/ops/knowledge-suggestions")
def operator_knowledge_suggestions(status: Optional[str] = Query(default=None), _auth: None = OpsAuth) -> Dict[str, Any]:
    items = load_knowledge_suggestions()
    if status:
        items = [item for item in items if str(item.get("status", "")).lower() == status.lower()]
    return {"success": True, "items": items}


@router.post("/api/ops/knowledge-suggestions/{item_id}/review")
def review_operator_knowledge_suggestion(
    item_id: str,
    request: KnowledgeSuggestionReviewRequest,
    _auth: None = OpsAuth,
) -> Dict[str, Any]:
    items = load_knowledge_suggestions()
    target = None
    for item in items:
        if item.get("id") == item_id or item.get("suggestionId") == item_id:
            target = item
            break
    if target is None:
        target = {"id": item_id, "suggestionId": item_id}
        items.append(target)
    target.setdefault("id", item_id)
    target.setdefault("suggestionId", item_id)
    target.update(
        {
            "status": "approved" if request.approved else "rejected",
            "reviewer": request.reviewer,
            "reviewNote": request.review_note,
            "approvedBy": request.reviewer if request.approved else "",
            "reviewedAt": datetime.now().isoformat(),
        }
    )
    save_knowledge_suggestions(items)
    return {"success": True, "item": target}


@router.get("/api/ops/skill-drafts")
def list_skill_drafts(status: Optional[str] = Query(default=None), _auth: None = OpsAuth) -> Dict[str, Any]:
    return {"success": True, "items": skill_draft_service.list_drafts(status or "")}


@router.post("/api/ops/skill-drafts/{draft_id}/review")
def review_skill_draft(draft_id: str, request: SkillDraftReviewRequest, _auth: None = OpsAuth) -> Dict[str, Any]:
    return skill_draft_service.review_draft(draft_id, request)


@router.post("/api/ops/skill-evaluations")
def evaluate_skill_triggers(request: SkillEvaluationRequest, _auth: None = OpsAuth) -> Dict[str, Any]:
    return skill_evaluation_service.evaluate(request)


@router.post("/api/ops/golden-evaluations")
def evaluate_golden_cases(request: GoldenEvaluationRequest, _auth: None = OpsAuth) -> Dict[str, Any]:
    if bool(request.partition_date_anchor_enabled) == bool(settings.agent_partition_date_anchor_enabled):
        return golden_evaluation_service.evaluate(request, workflow.run)
    evaluation_settings = settings.model_copy(
        update={"agent_partition_date_anchor_enabled": bool(request.partition_date_anchor_enabled)}
    )
    evaluation_workflow = create_workflow(evaluation_settings)
    evaluation_service = GoldenEvaluationService(evaluation_settings)
    try:
        return evaluation_service.evaluate(request, evaluation_workflow.run)
    finally:
        evaluation_workflow.checkpoint_manager.close()


@router.get("/api/daily-report")
def daily_report(merchant_id: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    return daily_report_service.report(require_merchant_access(merchant_id or settings.merchant_id)).model_dump(by_alias=True)


@router.post("/api/wiki/compress")
def compress_wiki(request: WikiCompressRequest, _auth: None = OpsAuth) -> Dict[str, Any]:
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


@router.post("/api/es/rebuild-recall-index")
def rebuild_recall_index(merchant_id: Optional[str] = Query(default=None), _auth: None = OpsAuth) -> Dict[str, Any]:
    effective_merchant_id = require_merchant_access(merchant_id or settings.merchant_id)
    result = recall_index_manager.rebuild(changed_only=True)
    return {
        "success": True,
        "merchantId": effective_merchant_id,
        **result,
    }


@router.get("/api/es/recall-mapping")
def recall_mapping(_auth: None = OpsAuth) -> Dict[str, Any]:
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


@router.post("/api/topics/build")
def build_topic_asset(request: TopicBuildRequest, _auth: None = OpsAuth) -> Dict[str, Any]:
    if not request.merchant_id:
        request.merchant_id = settings.merchant_id
    require_merchant_access(request.merchant_id)
    return topic_builder_workflow.build(request)


@router.post("/api/topics/{topic}/tables/{table_name}/publish")
def publish_topic_asset(
    topic: str,
    table_name: str,
    request: TopicReviewRequest,
    _auth: None = OpsAuth,
) -> Dict[str, Any]:
    preflight = semantic_governance.preflight_publish(topic, table_name) if request.approved else {}
    if request.approved and not preflight.get("publishable", False):
        return {
            "success": False,
            "status": "PREFLIGHT_FAILED",
            "topic": topic,
            "tableName": table_name,
            "preflight": preflight,
        }
    result = topic_assets.publish(topic, table_name, request.approved, request.reviewer, request.review_note)
    if preflight:
        result["preflight"] = preflight
    if request.approved and result.get("status") == "PUBLISHED":
        governance_result = semantic_governance.after_publish(topic, table_name, request.reviewer, request.review_note)
        result["semanticGovernance"] = governance_result
        index_result = recall_index_manager.rebuild(changed_only=True, topic=topic, table_name=table_name)
        result["recallIndex"] = index_result
        result["esUpsert"] = index_result.get("es", {})
        result["cacheInvalidated"] = bool(index_result.get("cacheInvalidated"))
    return result


@router.post("/api/topics/{topic}/tables/{table_name}/schema-diff")
def diff_topic_table_schema(
    topic: str,
    table_name: str,
    request: Optional[TopicBuildRequest] = None,
    _auth: None = OpsAuth,
) -> Dict[str, Any]:
    target = request or TopicBuildRequest()
    target.topic = topic
    target.table_name = table_name
    if not target.merchant_id:
        target.merchant_id = settings.merchant_id
    require_merchant_access(target.merchant_id)
    return topic_builder_workflow.diff_schema(target)


@router.get("/api/topics/{topic}/tables/{table_name}/impact")
def semantic_asset_impact_analysis(topic: str, table_name: str, _auth: None = OpsAuth) -> Dict[str, Any]:
    return semantic_governance.impact_analysis(topic, table_name)


@router.post("/api/topics/{topic}/tables/{table_name}/rollback")
def rollback_semantic_asset(
    topic: str,
    table_name: str,
    version: str = Query(default=""),
    reviewer: str = Query(default=""),
    reason: str = Query(default=""),
    _auth: None = OpsAuth,
) -> Dict[str, Any]:
    result = semantic_governance.rollback(topic, table_name, version=version, reviewer=reviewer, reason=reason)
    if result.get("success"):
        index_result = recall_index_manager.rebuild(changed_only=True, topic=topic, table_name=table_name)
        result["recallIndex"] = index_result
        result["cacheInvalidated"] = bool(index_result.get("cacheInvalidated"))
    return result


@router.post("/api/topics/{topic}/tables/{table_name}/refresh-incremental")
def refresh_topic_table_incrementally(
    topic: str,
    table_name: str,
    request: Optional[TopicBuildRequest] = None,
    _auth: None = OpsAuth,
) -> Dict[str, Any]:
    target = request or TopicBuildRequest()
    target.topic = topic
    target.table_name = table_name
    if not target.merchant_id:
        target.merchant_id = settings.merchant_id
    require_merchant_access(target.merchant_id)
    return topic_builder_workflow.refresh_incremental(target)


@router.post("/api/topics/{topic}/tables/{table_name}/es-upsert")
def upsert_topic_table_recall_index(topic: str, table_name: str, _auth: None = OpsAuth) -> Dict[str, Any]:
    result = recall_index_manager.rebuild(changed_only=True, topic=topic, table_name=table_name)
    return {"success": bool(result.get("success", True)), "topic": topic, "tableName": table_name, "recallIndex": result}


@router.get("/api/topics/{topic}/assets")
def topic_assets_endpoint(topic: str, _auth: None = OpsAuth) -> Dict[str, Any]:
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


RUNS_DASHBOARD_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Merchant AI Runs</title>
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172033; background: #f6f7f9; }
    main { max-width: 1180px; margin: 0 auto; padding: 24px; }
    header { display: flex; align-items: end; justify-content: space-between; gap: 16px; margin-bottom: 18px; }
    h1 { margin: 0; font-size: 24px; }
    button { border: 1px solid #cfd6e4; background: white; border-radius: 6px; padding: 8px 12px; cursor: pointer; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 16px; }
    .card { background: white; border: 1px solid #e2e7f0; border-radius: 8px; padding: 14px; }
    .label { color: #667085; font-size: 12px; }
    .value { margin-top: 6px; font-size: 22px; font-weight: 700; }
    table { width: 100%; border-collapse: collapse; background: white; border: 1px solid #e2e7f0; border-radius: 8px; overflow: hidden; }
    th, td { padding: 10px 12px; border-bottom: 1px solid #edf1f7; text-align: left; vertical-align: top; font-size: 13px; }
    th { color: #667085; background: #fbfcfe; font-weight: 600; }
    tr:last-child td { border-bottom: 0; }
    a { color: #245dc1; text-decoration: none; }
    .status { display: inline-block; min-width: 78px; font-weight: 700; }
    .error { color: #b42318; }
    .muted { color: #667085; }
    @media (max-width: 760px) { .grid { grid-template-columns: 1fr 1fr; } main { padding: 14px; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Merchant AI Runs</h1>
        <div class="muted">Run lifecycle, latency, errors, trace replay</div>
      </div>
      <button onclick="loadDashboard()">Refresh</button>
    </header>
    <section class="grid" id="cards"></section>
    <table>
      <thead>
        <tr>
          <th>Time</th>
          <th>Status</th>
          <th>Duration</th>
          <th>Question</th>
          <th>Links</th>
        </tr>
      </thead>
      <tbody id="runs"></tbody>
    </table>
  </main>
  <script>
    function fmtMs(value) {
      if (value === null || value === undefined) return "-";
      if (value >= 1000) return (value / 1000).toFixed(2) + "s";
      return Math.round(value) + "ms";
    }
    function esc(value) {
      return String(value || "").replace(/[&<>"']/g, function(ch) {
        return {"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}[ch];
      });
    }
    async function loadDashboard() {
      const res = await fetch("/api/runs/dashboard?limit=100");
      const payload = await res.json();
      const data = payload.dashboard || {};
      const counts = data.statusCounts || {};
      document.getElementById("cards").innerHTML = [
        ["Total", data.totalRuns || 0],
        ["Queued", counts.QUEUED || 0],
        ["Running", counts.RUNNING || 0],
        ["Completed", counts.COMPLETED || 0],
        ["Failed", counts.FAILED || 0],
        ["Avg Duration", fmtMs(data.avgDurationMs || 0)]
      ].map(([label, value]) => `<div class="card"><div class="label">${label}</div><div class="value">${value}</div></div>`).join("");
      document.getElementById("runs").innerHTML = (data.runs || []).map(function(run) {
        const detail = `/api/threads/${encodeURIComponent(run.threadId)}/runs/${encodeURIComponent(run.runId)}`;
        const events = `${detail}/events`;
        const trace = `${detail}/trace`;
        const checkpoint = `${detail}/checkpoint`;
        const statusClass = run.status === "FAILED" ? "status error" : "status";
        return `<tr>
          <td>${esc(run.startTime)}</td>
          <td><span class="${statusClass}">${esc(run.status)}</span><div class="error">${esc(run.error)}</div></td>
          <td>${fmtMs(run.durationMs)}</td>
          <td>${esc(run.question)}<div class="muted">${esc(run.answerPreview)}</div></td>
          <td><a href="${detail}">run</a> · <a href="${events}">events</a> · <a href="${trace}">trace</a> · <a href="${checkpoint}">checkpoint</a></td>
        </tr>`;
      }).join("");
    }
    loadDashboard().catch(function(err) {
      document.getElementById("runs").innerHTML = `<tr><td colspan="5" class="error">${esc(err.message)}</td></tr>`;
    });
  </script>
</body>
</html>
"""


app = create_app()
