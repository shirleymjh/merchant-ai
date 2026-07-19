from __future__ import annotations

import json
import hmac
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, FastAPI, Header, HTTPException, Path as ApiPath, Query, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from merchant_ai.config import Settings, get_settings
from merchant_ai.services.runtime_factory import create_runtime
from merchant_ai.services.runtime_factory import GroundedOnlineRuntimeUnavailable
from merchant_ai.models import (
    ChatRequest,
    FeedbackRequest,
    GoldenEvaluationRequest,
    ChatContext,
    KnowledgeSuggestionActionRequest,
    KnowledgeSuggestionPublishRequest,
    KnowledgeSuggestionReviewRequest,
    MemoryCleanupRequest,
    MemoryItemPatchRequest,
    MemoryRecallEvaluationRequest,
    MetricDefinitionPreferenceRequest,
    RunCreateRequest,
    SkillDraftReviewRequest,
    SkillEvaluationRequest,
    TopicBuildRequest,
    TopicReviewRequest,
)
from merchant_ai.services.answer import DailyReportService, FeedbackService
from merchant_ai.services.attachments import AttachmentStore
from merchant_ai.services.assets import SemanticAssetGovernanceService, TopicBuilderWorkflow
from merchant_ai.services.evaluation import GoldenEvaluationService
from merchant_ai.services.memory import MemoryGovernanceService, MemoryManagementService
from merchant_ai.services.recall_index import RecallIndexManager
from merchant_ai.services.repositories import write_json
from merchant_ai.services.semantic_publish import SemanticPublishCoordinator
from merchant_ai.services.runs import (
    AgentAsyncRunService,
    AgentRunManager,
    AgentRunStreamService,
    public_response_payload,
    run_duration_ms,
    run_summary_payload,
    valid_run_id,
    valid_thread_id,
)
from merchant_ai.services.security import (
    Permission,
    authorize_authenticated_merchant_access,
    authorize_merchant_access,
    identity_scope_hash,
    merchant_principal,
    ops_principal,
    principal_from_authenticated_identity,
    resolve_authenticated_identity,
)
from merchant_ai.services.skill_drafts import SkillDraftService
from merchant_ai.services.skill_evaluation import SkillEvaluationService

settings: Settings
runtime: Any
run_manager: AgentRunManager
stream_service: AgentRunStreamService
async_run_service: AgentAsyncRunService
topic_assets: Any
doris_repository: Any
daily_report_service: DailyReportService
feedback_service: FeedbackService
memory_management_service: MemoryManagementService
memory_governance_service: MemoryGovernanceService
skill_draft_service: SkillDraftService
skill_evaluation_service: SkillEvaluationService
golden_evaluation_service: GoldenEvaluationService
topic_builder_workflow: TopicBuilderWorkflow
semantic_governance: SemanticAssetGovernanceService
recall_index_manager: RecallIndexManager
semantic_publish_coordinator: SemanticPublishCoordinator
attachment_store: AttachmentStore


def _init_services(runtime_settings: Optional[Settings] = None) -> None:
    global settings
    global runtime
    global run_manager
    global stream_service
    global async_run_service
    global topic_assets
    global doris_repository
    global daily_report_service
    global feedback_service
    global memory_management_service
    global memory_governance_service
    global skill_draft_service
    global skill_evaluation_service
    global golden_evaluation_service
    global topic_builder_workflow
    global semantic_governance
    global recall_index_manager
    global semantic_publish_coordinator
    global attachment_store

    settings = runtime_settings or get_settings()
    runtime = create_runtime(settings)
    run_manager = AgentRunManager(settings)
    stream_service = AgentRunStreamService(run_manager, runtime.run, settings.merchant_id)
    async_run_service = AgentAsyncRunService(
        run_manager,
        runtime.run,
        settings.merchant_id,
        max_workers=settings.max_concurrent_sub_agents,
    )
    services = runtime.services
    topic_assets = services.topic_assets
    doris_repository = services.doris_repository
    daily_report_service = DailyReportService(doris_repository, settings.merchant_id, topic_assets)
    feedback_service = FeedbackService(
        services.answer_repository,
        services.pending_store,
        services.memory_store,
    )
    memory_management_service = MemoryManagementService(settings, services.memory_store)
    skill_draft_service = SkillDraftService(settings)
    skill_evaluation_service = SkillEvaluationService(settings, services.answer_service)
    golden_evaluation_service = GoldenEvaluationService(settings)
    topic_builder_workflow = TopicBuilderWorkflow(settings, doris_repository, topic_assets)
    semantic_governance = SemanticAssetGovernanceService(settings, doris_repository, topic_assets)
    recall_index_manager = RecallIndexManager(
        settings,
        services.recall_service,
        cache_clearers=list(services.recall_cache_clearers),
    )
    semantic_publish_coordinator = SemanticPublishCoordinator(
        settings,
        topic_assets,
        semantic_governance,
        recall_index_manager,
    )
    memory_governance_service = MemoryGovernanceService(
        settings,
        services.memory_store,
        topic_assets,
        semantic_governance,
        doris_repository,
        semantic_publish_coordinator,
    )
    attachment_store = AttachmentStore(settings)


def require_ops_token(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_ops_token: Optional[str] = Header(default=None, alias="X-Ops-Token"),
) -> None:
    expected = str(settings.ops_token or "").strip()
    if not expected:
        client_host = str(request.client.host if request.client else "")
        if not settings.identity_auth_required and client_host in {"127.0.0.1", "::1", "localhost"}:
            return
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ops token is not configured")
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


def require_merchant_access(
    merchant_id: str,
    authorization: Optional[str] = None,
    requested_identity: Any = None,
    permission: Permission = Permission.CHAT_RUN,
) -> str:
    """Authorize a merchant API target without trusting its request merchantId.

    Production identity mode is deliberately fail-closed: only a verified JWT
    identity may establish merchant scope.  The synthetic same-merchant
    principal remains available solely when identity authentication has been
    explicitly disabled for local development.
    """

    requested_target = str(merchant_id or "").strip()
    should_resolve_identity = bool(
        settings.identity_auth_required
        or str(authorization or "").strip()
    )
    identity = (
        resolve_authenticated_identity(settings, authorization or "", requested_identity)
        if should_resolve_identity
        else None
    )
    if identity is not None:
        target = requested_target or str(identity.merchant_id or "").strip()
        return authorize_authenticated_merchant_access(settings, identity, target, permission)
    if settings.identity_auth_required:
        raise HTTPException(status_code=401, detail="authenticated merchant identity is required")
    target = requested_target or str(settings.merchant_id or "").strip()
    return authorize_merchant_access(settings, merchant_principal(target), target, permission)


def require_ops_merchant_access(merchant_id: str, permission: Permission = Permission.OPS_READ) -> str:
    return authorize_merchant_access(settings, ops_principal(), merchant_id or settings.merchant_id, permission)


def require_memory_write_identity(merchant_id: str, authorization: Optional[str]) -> Any:
    identity = resolve_authenticated_identity(settings, authorization or "", None)
    if identity is None:
        raise HTTPException(status_code=401, detail="authenticated identity is required for memory writes")
    target = str(merchant_id or settings.merchant_id).strip()
    if identity.merchant_id and identity.merchant_id != target:
        raise HTTPException(status_code=403, detail="authenticated identity cannot write requested merchant memory")
    if not principal_from_authenticated_identity(identity).has_permission(Permission.MEMORY_WRITE):
        raise HTTPException(status_code=403, detail="permission denied: memory.write")
    return identity


def require_thread_access(
    thread_id: str,
    merchant_id: str = "",
    authorization: Optional[str] = None,
    requested_identity: Any = None,
) -> Any:
    if not valid_thread_id(thread_id):
        raise HTTPException(status_code=400, detail="invalid threadId")
    thread = run_manager.get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="thread not found")
    requested_merchant = str(merchant_id or thread.merchant_id).strip()
    if requested_merchant != thread.merchant_id:
        raise HTTPException(status_code=403, detail="thread does not belong to requested merchantId")
    identity = resolve_authenticated_identity(settings, authorization or "", requested_identity)
    if identity and identity.merchant_id and identity.merchant_id != thread.merchant_id:
        raise HTTPException(status_code=403, detail="authenticated identity cannot access thread")
    if identity and thread.owner_scope_hash:
        if identity_scope_hash(identity, thread.merchant_id) != thread.owner_scope_hash:
            raise HTTPException(status_code=403, detail="authenticated identity scope cannot access thread")
    require_merchant_access(
        thread.merchant_id,
        authorization=authorization,
        requested_identity=identity,
        permission=Permission.RUN_READ,
    )
    return thread


def require_thread_run_access(
    thread_id: str,
    run_id: str,
    authorization: Optional[str] = None,
) -> Any:
    thread = require_thread_access(thread_id, authorization=authorization)
    if not valid_run_id(run_id):
        raise HTTPException(status_code=400, detail="invalid runId")
    run = run_manager.get_run(run_id)
    if not run or run.thread_id != thread.thread_id or run.merchant_id != thread.merchant_id:
        raise HTTPException(status_code=404, detail="run not found")
    return run


def require_ops_thread_run_access(thread_id: str, run_id: str) -> Any:
    if not valid_thread_id(thread_id):
        raise HTTPException(status_code=400, detail="invalid threadId")
    if not valid_run_id(run_id):
        raise HTTPException(status_code=400, detail="invalid runId")
    thread = run_manager.get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="thread not found")
    require_ops_merchant_access(thread.merchant_id, Permission.OPS_READ)
    run = run_manager.get_run(run_id)
    if (
        not run
        or run.thread_id != thread.thread_id
        or run.merchant_id != thread.merchant_id
    ):
        raise HTTPException(status_code=404, detail="run not found")
    return run


def create_app(runtime_settings: Optional[Settings] = None) -> FastAPI:
    _init_services(runtime_settings)
    application = FastAPI(title="yshopping Merchant AI Python", version="0.1.0")
    application.state.runtime = runtime

    @application.exception_handler(GroundedOnlineRuntimeUnavailable)
    async def grounded_runtime_unavailable(
        _request: Request,
        exc: GroundedOnlineRuntimeUnavailable,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "detail": {
                    "code": exc.code,
                    "message": exc.reason,
                }
            },
        )

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
async def chat(request: ChatRequest, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    apply_request_identity(request, authorization)
    merchant_id = require_merchant_access(
        request.merchant_id or settings.merchant_id,
        authorization=authorization,
        requested_identity=request.user_identity,
    )
    request.message = message_with_attachments(request.message, request.attachments, merchant_id)
    thread = run_manager.create_thread(
        merchant_id,
        request.context.topic if request.context else "",
        request.context,
        identity=request.user_identity,
    )
    run = run_manager.create_run(thread.thread_id, merchant_id, request.message, identity=request.user_identity)
    message_history = run_manager.effective_message_history(
        thread.thread_id,
        merchant_id,
        run.run_id,
        request.message_history,
    )

    def listener(event_type: str, node: str, payload: Dict[str, Any]) -> None:
        run_manager.append_event(run.run_id, thread.thread_id, event_type, node, payload)

    try:
        response = await runtime.run_async(
            request.message,
            merchant_id,
            request.context,
            listener,
            thread.thread_id,
            run.run_id,
            message_history=message_history,
        )
        run_manager.complete_run(run.run_id, response)
        return public_response_payload(response)
    except Exception as exc:
        run_manager.fail_run(run.run_id, str(exc))
        raise


@router.post("/api/chat/stream")
def stream_chat(request: RunCreateRequest, authorization: Optional[str] = Header(default=None)):
    apply_request_identity(request, authorization)
    request.merchant_id = require_merchant_access(
        request.merchant_id or settings.merchant_id,
        authorization=authorization,
        requested_identity=request.user_identity,
    )
    if request.thread_id:
        require_thread_access(request.thread_id, request.merchant_id, authorization, request.user_identity)
    request.message = message_with_attachments(request.message, request.attachments, request.merchant_id)
    return StreamingResponse(stream_service.stream(request), media_type="text/event-stream")


@router.post("/api/runs/async")
def create_async_run(request: RunCreateRequest, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    apply_request_identity(request, authorization)
    request.merchant_id = require_merchant_access(
        request.merchant_id or settings.merchant_id,
        authorization=authorization,
        requested_identity=request.user_identity,
    )
    if request.thread_id:
        require_thread_access(request.thread_id, request.merchant_id, authorization, request.user_identity)
    request.message = message_with_attachments(request.message, request.attachments, request.merchant_id)
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


@router.post("/api/chat/resume")
def resume_chat(request: RunCreateRequest, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    apply_request_identity(request, authorization)
    request.merchant_id = require_merchant_access(
        request.merchant_id or settings.merchant_id,
        authorization=authorization,
        requested_identity=request.user_identity,
    )
    request.message = message_with_attachments(request.message, request.attachments, request.merchant_id)
    if not request.thread_id:
        raise HTTPException(status_code=400, detail="threadId is required for resume")
    require_thread_access(request.thread_id, request.merchant_id, authorization, request.user_identity)
    run = async_run_service.submit(request)
    return {
        "success": True,
        "mode": "resume",
        "threadId": run.thread_id,
        "runId": run.run_id,
        "run": jsonable_encoder(run_summary_payload(run, run_duration_ms(run)), by_alias=True),
        "links": {
            "run": "/api/threads/%s/runs/%s" % (run.thread_id, run.run_id),
            "events": "/api/threads/%s/runs/%s/events" % (run.thread_id, run.run_id),
            "checkpoint": "/api/threads/%s/runs/%s/checkpoint" % (run.thread_id, run.run_id),
        },
    }


@router.post("/api/attachments")
async def upload_attachment(
    request: Request,
    name: str = Query("attachment"),
    content_type: str = Query("application/octet-stream", alias="type"),
    merchant_id: str = Query("", alias="merchantId"),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    effective_merchant_id = require_merchant_access(
        merchant_id or settings.merchant_id,
        authorization=authorization,
    )
    allowed_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".pdf", ".xls", ".xlsx", ".csv", ".txt", ".json", ".md"}
    if Path(name).suffix.lower() not in allowed_suffixes:
        raise HTTPException(status_code=415, detail="unsupported attachment type")
    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=400, detail="empty attachment")
    if len(payload) > int(settings.attachment_max_bytes or 20 * 1024 * 1024):
        raise HTTPException(status_code=413, detail="attachment exceeds configured size limit")
    try:
        metadata = attachment_store.save(name, content_type, payload, effective_merchant_id)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="attachment parsing failed: %s" % type(exc).__name__) from exc
    return {"success": True, **metadata}


def message_with_attachments(message: str, attachments: List[Any], merchant_id: str = "") -> str:
    context = attachment_store.context_for(attachments, merchant_id)
    return str(message or "") + (("\n\n[用户附件上下文]\n" + context) if context else "")


@router.get("/api/runs")
def list_runs(
    limit: int = Query(default=50, ge=1, le=200),
    status: Optional[str] = Query(default=None),
    merchant_id: Optional[str] = Query(default=None, alias="merchantId"),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    effective_merchant_id = require_merchant_access(
        merchant_id or settings.merchant_id,
        authorization=authorization,
        permission=Permission.RUN_READ,
    )
    runs = run_manager.list_runs(limit=limit, status=status or "", merchant_id=effective_merchant_id)
    summaries = [run_summary_payload(run, run_duration_ms(run)) for run in runs]
    return {"success": True, "runs": jsonable_encoder(summaries, by_alias=True)}


@router.get("/api/runs/dashboard")
def runs_dashboard(
    limit: int = Query(default=50, ge=1, le=200),
    status: Optional[str] = Query(default=None),
    merchant_id: Optional[str] = Query(default=None, alias="merchantId"),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    effective_merchant_id = require_merchant_access(
        merchant_id or settings.merchant_id,
        authorization=authorization,
        permission=Permission.RUN_READ,
    )
    dashboard = run_manager.dashboard(limit=limit, status=status or "", merchant_id=effective_merchant_id)
    return {"success": True, "dashboard": jsonable_encoder(dashboard, by_alias=True)}


def _runtime_trace() -> Dict[str, Any]:
    trace = runtime.runtime_trace()
    return {
        "metrics": trace.get("metrics") or {"tools": []},
        "alerts": trace.get("alerts") or [],
        "events": trace.get("events") or [],
        "rateLimits": trace.get("rateLimits") or {},
        "loadBalancer": trace.get("loadBalancer") or {},
        "degraded": trace.get("degraded") or {},
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
def create_thread(
    request: Optional[RunCreateRequest] = None,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    safe = request or RunCreateRequest()
    apply_request_identity(safe, authorization)
    merchant_id = require_merchant_access(
        safe.merchant_id or settings.merchant_id,
        authorization=authorization,
        requested_identity=safe.user_identity,
    )
    thread = run_manager.create_thread(
        merchant_id,
        safe.context.topic if safe.context else "",
        safe.context,
        identity=safe.user_identity,
    )
    return {"success": True, "thread": jsonable_encoder(thread, by_alias=True)}


@router.get("/api/threads/{thread_id}")
def get_thread(thread_id: str = ApiPath(...), authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    thread = require_thread_access(thread_id, authorization=authorization)
    return {"success": True, "thread": jsonable_encoder(thread, by_alias=True)}


@router.get("/api/threads/{thread_id}/runs/{run_id}")
def get_run(thread_id: str, run_id: str, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    run = require_thread_run_access(thread_id, run_id, authorization)
    return {"success": True, "run": jsonable_encoder(run, by_alias=True)}


@router.get("/api/threads/{thread_id}/runs/{run_id}/events")
def get_run_events(thread_id: str, run_id: str, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    require_thread_run_access(thread_id, run_id, authorization)
    return {
        "success": True,
        "runId": run_id,
        "threadId": thread_id,
        "events": jsonable_encoder(run_manager.events(run_id), by_alias=True),
    }


@router.get("/api/threads/{thread_id}/runs/{run_id}/trace")
def get_run_trace(
    thread_id: str,
    run_id: str,
    _auth: None = OpsAuth,
) -> Dict[str, Any]:
    require_ops_thread_run_access(thread_id, run_id)
    trace = run_manager.trace(run_id)
    if trace is None:
        return {"success": False, "message": "trace not found", "threadId": thread_id, "runId": run_id}
    return {"success": True, "runId": run_id, "threadId": thread_id, "trace": trace}


@router.get("/api/threads/{thread_id}/runs/{run_id}/checkpoint")
def get_run_checkpoint(
    thread_id: str,
    run_id: str,
    _auth: None = OpsAuth,
) -> Dict[str, Any]:
    run = require_ops_thread_run_access(thread_id, run_id)
    try:
        checkpoint = runtime.checkpoint_state_summary(thread_id, run_id)
    except Exception as exc:
        checkpoint = {"checkpointRef": run.checkpoint_ref, "error": str(exc)[:500], "hasValues": False}
    return {"success": True, "runId": run_id, "threadId": thread_id, "checkpoint": jsonable_encoder(checkpoint, by_alias=True)}


@router.post("/api/threads/{thread_id}/runs/{run_id}/cancel")
def cancel_run(thread_id: str, run_id: str, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    require_thread_run_access(thread_id, run_id, authorization)
    return {"success": True, "run": jsonable_encoder(async_run_service.cancel(run_id), by_alias=True)}


@router.post("/api/answers/{answer_id}/feedback")
def feedback(
    answer_id: str,
    request: FeedbackRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    identity = resolve_authenticated_identity(settings, authorization or "", None)
    persisted = feedback_service.apply_feedback(
        answer_id,
        request.adopted,
        request.liked,
        request.disliked,
        identity=identity,
    )
    return {"success": True, "persisted": persisted}


@router.post("/api/merchant-preferences/metric-definition")
def record_metric_definition_preference(
    request: MetricDefinitionPreferenceRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    target = require_merchant_access(
        request.merchant_id or settings.merchant_id,
        authorization=authorization,
    )
    require_memory_write_identity(target, authorization)
    return memory_management_service.record_metric_definition_preference(target, request)


def merchant_public_knowledge_action_result(result: Dict[str, Any], requested_action: str = "") -> Dict[str, Any]:
    suggestion = result.get("suggestion") if isinstance(result.get("suggestion"), dict) else {}
    payload = suggestion.get("payload") if isinstance(suggestion.get("payload"), dict) else {}
    scope = str(suggestion.get("scopeType") or payload.get("scopeType") or payload.get("proposedScope") or "merchant").lower()
    rule_text = str(payload.get("correctionText") or payload.get("question") or "")[:240]
    title = str(payload.get("title") or suggestion.get("metricName") or "业务知识")
    status_value = str(result.get("status") or "")
    if status_value == "MERCHANT_ACTIVE":
        message = "已用于本商家后续分析。"
        notice_type = "merchant_rule_confirmed"
    elif status_value == "PLATFORM_SUGGESTED" or scope == "platform":
        message = "已提交反馈，平台审核通过后才会进入公共知识。"
        notice_type = "platform_feedback_submitted"
    elif status_value in {"DISMISSED", "EXISTING_KNOWLEDGE_REUSED"}:
        message = "已取消本次知识更新。"
        notice_type = "knowledge_suggestion_dismissed"
    elif status_value == "CONFLICT_CONFIRMATION_REQUIRED":
        message = "已有相似或冲突规则，需要先确认处理方式。"
        notice_type = "knowledge_conflict_confirmation"
    else:
        message = "知识建议已处理。"
        notice_type = "knowledge_action_processed"
    public_result = {
        "success": bool(result.get("success")),
        "status": status_value,
        "suggestionId": str(result.get("suggestionId") or suggestion.get("suggestionId") or ""),
        "noticeType": notice_type,
        "title": title,
        "message": message,
        "ruleText": rule_text,
        "requestedAction": str(requested_action or ""),
    }
    if result.get("conflictCheck"):
        public_result["conflictCheck"] = result.get("conflictCheck")
    if result.get("allowedResolutions"):
        public_result["resolutionActions"] = [
            {
                "actionId": str(item),
                "label": {"use_existing": "沿用已有规则", "keep_both": "保留两条", "replace": "替换旧规则", "cancel": "取消"}.get(str(item), str(item)),
            }
            for item in result.get("allowedResolutions") or []
        ]
    if not result.get("success") and result.get("allowedActions"):
        public_result["userActions"] = [
            {"actionId": "submit_feedback", "label": "提交反馈"},
            {"actionId": "dismiss", "label": "取消"},
        ]
    return public_result


@router.post("/api/merchant/knowledge-suggestions/{item_id}/action")
def merchant_knowledge_suggestion_action(
    item_id: str,
    request: KnowledgeSuggestionActionRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    target = require_merchant_access(
        request.merchant_id or settings.merchant_id,
        authorization=authorization,
    )
    require_memory_write_identity(target, authorization)
    result = memory_governance_service.apply_merchant_action(
        target,
        item_id,
        request.action,
        actor=request.actor,
        note=request.note,
        conflict_resolution=request.conflict_resolution,
    )
    if not result.get("success"):
        status_code = 404 if result.get("status") == "NOT_FOUND" else 422
        raise HTTPException(status_code=status_code, detail=merchant_public_knowledge_action_result(result, request.action))
    return merchant_public_knowledge_action_result(result, request.action)


@router.get("/api/memory/{merchant_id}")
def get_memory(merchant_id: str, include_inactive: bool = Query(default=True), _auth: None = OpsAuth) -> Dict[str, Any]:
    require_ops_merchant_access(merchant_id)
    return memory_management_service.get_memory(merchant_id, include_inactive=include_inactive)


@router.patch("/api/memory/{merchant_id}/items/{memory_id}")
def patch_memory_item(
    merchant_id: str,
    memory_id: str,
    request: MemoryItemPatchRequest,
    _auth: None = OpsAuth,
) -> Dict[str, Any]:
    require_ops_merchant_access(merchant_id, Permission.OPS_WRITE)
    return memory_management_service.patch_item(merchant_id, memory_id, request)


@router.delete("/api/memory/{merchant_id}/items/{memory_id}")
def delete_memory_item(
    merchant_id: str,
    memory_id: str,
    hard_delete: bool = Query(default=False),
    _auth: None = OpsAuth,
) -> Dict[str, Any]:
    require_ops_merchant_access(merchant_id, Permission.OPS_WRITE)
    return memory_management_service.delete_item(merchant_id, memory_id, hard_delete=hard_delete)


@router.post("/api/memory/{merchant_id}/cleanup")
def cleanup_memory(merchant_id: str, request: MemoryCleanupRequest, _auth: None = OpsAuth) -> Dict[str, Any]:
    require_ops_merchant_access(merchant_id, Permission.OPS_WRITE)
    return memory_management_service.cleanup_expired(merchant_id, hard_delete=request.hard_delete, dry_run=request.dry_run)


@router.post("/api/memory/{merchant_id}/recall-eval")
def evaluate_memory_recall(
    merchant_id: str,
    request: MemoryRecallEvaluationRequest,
    _auth: None = OpsAuth,
) -> Dict[str, Any]:
    require_ops_merchant_access(merchant_id)
    return memory_management_service.evaluate_recall(
        merchant_id,
        request.cases,
        budget_tokens=request.budget_tokens,
        budget_chars=request.budget_chars,
    )


@router.get("/api/ops/knowledge-suggestions")
def operator_knowledge_suggestions(
    status: Optional[str] = Query(default=None),
    merchant_id: str = Query(default=""),
    _auth: None = OpsAuth,
) -> Dict[str, Any]:
    target = require_ops_merchant_access(merchant_id or settings.merchant_id)
    memory = memory_management_service.get_memory(target).get("memory") or {}
    items = list(memory.get("knowledgeSuggestions") or [])
    if status:
        items = [item for item in items if str(item.get("status", "")).lower() == status.lower()]
    return {"success": True, "items": items}


@router.post("/api/ops/knowledge-suggestions/{item_id}/review")
def review_operator_knowledge_suggestion(
    item_id: str,
    request: KnowledgeSuggestionReviewRequest,
    merchant_id: str = Query(default=""),
    _auth: None = OpsAuth,
) -> Dict[str, Any]:
    target = require_ops_merchant_access(merchant_id or settings.merchant_id)
    return memory_governance_service.review_suggestion(target, item_id, request)


@router.post("/api/ops/knowledge-suggestions/{item_id}/request-publish")
def request_operator_knowledge_publish(
    item_id: str,
    request: KnowledgeSuggestionPublishRequest,
    merchant_id: str = Query(default=""),
    _auth: None = OpsAuth,
) -> Dict[str, Any]:
    target = require_ops_merchant_access(merchant_id or settings.merchant_id)
    return memory_governance_service.request_publish_suggestion(target, item_id, request.reviewer, request.review_note)


@router.post("/api/ops/knowledge-suggestions/{item_id}/publish")
def publish_operator_knowledge_suggestion(
    item_id: str,
    request: KnowledgeSuggestionPublishRequest,
    merchant_id: str = Query(default=""),
    _auth: None = OpsAuth,
) -> Dict[str, Any]:
    target = require_ops_merchant_access(merchant_id or settings.merchant_id)
    result = memory_governance_service.publish_suggestion(
        target,
        item_id,
        reviewer=request.reviewer,
        review_note=request.review_note,
        topic=request.topic,
        table_name=request.table_name,
    )
    if result.get("success") and request.auto_index:
        result["indexed"] = memory_governance_service.mark_suggestion_indexed(
            target,
            item_id,
            result.get("suggestion", {}).get("publishedRefId", ""),
        )
    return result


@router.get("/api/ops/skill-drafts")
def list_skill_drafts(status: Optional[str] = Query(default=None), _auth: None = OpsAuth) -> Dict[str, Any]:
    return {"success": True, "items": skill_draft_service.list_drafts(status or "")}


@router.post("/api/ops/skill-drafts/{draft_id}/review")
def review_skill_draft(draft_id: str, request: SkillDraftReviewRequest, _auth: None = OpsAuth) -> Dict[str, Any]:
    return skill_draft_service.review_draft(draft_id, request)


@router.post("/api/ops/skill-evaluations")
def evaluate_skill_triggers(request: SkillEvaluationRequest, _auth: None = OpsAuth) -> Dict[str, Any]:
    return skill_evaluation_service.evaluate(request)


@router.get("/api/ops/skill-market")
def skill_market(_auth: None = OpsAuth) -> Dict[str, Any]:
    return skill_draft_service.market()


@router.post("/api/ops/skill-market/{skill_name}/install")
def install_market_skill(skill_name: str, payload: Dict[str, Any], _auth: None = OpsAuth) -> Dict[str, Any]:
    return skill_draft_service.install_skill(
        skill_name,
        scope=str(payload.get("scope") or "merchant"),
        merchant_ids=payload.get("merchantIds") if isinstance(payload.get("merchantIds"), list) else [],
        industry_tags=payload.get("industryTags") if isinstance(payload.get("industryTags"), list) else [],
        traffic_percent=int(payload.get("trafficPercent") or 100),
    )


@router.post("/api/ops/golden-evaluations")
def evaluate_golden_cases(request: GoldenEvaluationRequest, _auth: None = OpsAuth) -> Dict[str, Any]:
    if bool(request.partition_date_anchor_enabled) == bool(settings.agent_partition_date_anchor_enabled):
        return golden_evaluation_service.evaluate(request, runtime.run, query_executor=doris_repository.query)
    evaluation_settings = settings.model_copy(
        update={"agent_partition_date_anchor_enabled": bool(request.partition_date_anchor_enabled)}
    )
    evaluation_runtime = create_runtime(evaluation_settings)
    evaluation_service = GoldenEvaluationService(evaluation_settings)
    try:
        return evaluation_service.evaluate(
            request,
            evaluation_runtime.run,
            query_executor=evaluation_runtime.services.doris_repository.query,
        )
    finally:
        evaluation_runtime.close()


@router.get("/api/daily-report")
def daily_report(
    merchant_id: Optional[str] = Query(default=None, alias="merchantId"),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    effective_merchant_id = require_merchant_access(
        merchant_id or settings.merchant_id,
        authorization=authorization,
    )
    return daily_report_service.report(effective_merchant_id).model_dump(by_alias=True)


@router.get("/api/merchant-profile")
def get_current_merchant_profile(
    merchant_id: Optional[str] = Query(default=None, alias="merchantId"),
    include_expired: bool = Query(default=False),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    effective_merchant_id = require_merchant_access(
        merchant_id or "",
        authorization=authorization,
    )
    return _merchant_profile_payload(effective_merchant_id, include_expired)


@router.get("/api/merchant-profile/{merchant_id}")
def get_merchant_profile(
    merchant_id: str,
    include_expired: bool = Query(default=False),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    effective_merchant_id = require_merchant_access(
        merchant_id,
        authorization=authorization,
    )
    return _merchant_profile_payload(effective_merchant_id, include_expired)


def _merchant_profile_payload(merchant_id: str, include_expired: bool) -> Dict[str, Any]:
    return {
        "success": True,
        "profile": runtime.services.merchant_profile_store.get_profile(
            merchant_id,
            include_expired=include_expired,
        ),
    }


@router.patch("/api/merchant-profile/{merchant_id}")
def update_merchant_profile(
    merchant_id: str,
    patch: Dict[str, Any],
    reviewer: str = Query(default="ops"),
    _auth: None = OpsAuth,
) -> Dict[str, Any]:
    effective_merchant_id = require_ops_merchant_access(merchant_id, Permission.OPS_WRITE)
    profile = runtime.services.merchant_profile_store.upsert_profile(
        effective_merchant_id,
        patch,
        reviewer=reviewer,
        review_status="reviewed",
    )
    return {"success": True, "profile": profile}


@router.post("/api/merchant-profile/{merchant_id}/review")
def review_merchant_profile(
    merchant_id: str,
    approved: bool = Query(default=True),
    reviewer: str = Query(default="ops"),
    note: str = Query(default=""),
    _auth: None = OpsAuth,
) -> Dict[str, Any]:
    effective_merchant_id = require_ops_merchant_access(merchant_id, Permission.OPS_WRITE)
    profile = runtime.services.merchant_profile_store.review_profile(
        effective_merchant_id,
        approved=approved,
        reviewer=reviewer,
        note=note,
    )
    return {"success": True, "profile": profile}


@router.get("/api/ops/access-control/policy")
def get_access_control_policy(_auth: None = OpsAuth) -> Dict[str, Any]:
    service = runtime.services.access_control
    return {
        "success": True,
        "path": str(service.policy_path),
        "policy": service._load_policy(),
    }


@router.put("/api/ops/access-control/policy")
def update_access_control_policy(policy: Dict[str, Any], _auth: None = OpsAuth) -> Dict[str, Any]:
    service = runtime.services.access_control
    service.policy_path.parent.mkdir(parents=True, exist_ok=True)
    service.policy_path.write_text(json.dumps(policy or {}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"success": True, "path": str(service.policy_path), "policy": service._load_policy()}


@router.get("/api/ops/access-control/audit")
def get_access_control_audit(limit: int = Query(default=50, ge=1, le=500), _auth: None = OpsAuth) -> Dict[str, Any]:
    service = runtime.services.access_control
    return {"success": True, **service.audit_summary(limit=limit)}


@router.post("/api/es/rebuild-recall-index")
def rebuild_recall_index(merchant_id: Optional[str] = Query(default=None), _auth: None = OpsAuth) -> Dict[str, Any]:
    effective_merchant_id = require_ops_merchant_access(merchant_id or settings.merchant_id, Permission.OPS_WRITE)
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
    require_ops_merchant_access(request.merchant_id, Permission.OPS_WRITE)
    return topic_builder_workflow.build(request)


@router.post("/api/topics/build-batch")
def build_topic_assets_batch(requests: List[TopicBuildRequest], _auth: None = OpsAuth) -> Dict[str, Any]:
    for request in requests:
        if not request.merchant_id:
            request.merchant_id = settings.merchant_id
        require_ops_merchant_access(request.merchant_id, Permission.OPS_WRITE)
    return topic_builder_workflow.build_batch(requests)


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
    if request.approved:
        return semantic_publish_coordinator.publish_approved(
            topic,
            table_name,
            request.reviewer,
            request.review_note,
            preflight,
        )
    return topic_assets.publish(topic, table_name, False, request.reviewer, request.review_note)


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
    require_ops_merchant_access(target.merchant_id, Permission.OPS_WRITE)
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
    require_ops_merchant_access(target.merchant_id, Permission.OPS_WRITE)
    return topic_builder_workflow.refresh_incremental(target)


@router.post("/api/topics/{topic}/tables/{table_name}/es-upsert")
def upsert_topic_table_recall_index(topic: str, table_name: str, _auth: None = OpsAuth) -> Dict[str, Any]:
    result = recall_index_manager.rebuild(changed_only=True, topic=topic, table_name=table_name)
    return {"success": bool(result.get("success", True)), "topic": topic, "tableName": table_name, "recallIndex": result}


@router.get("/api/topics/{topic}/assets")
def topic_assets_endpoint(topic: str, _auth: None = OpsAuth) -> Dict[str, Any]:
    return topic_assets.list_topic(topic)


@router.get("/api/topics/{topic}/tables/{table_name}/governance")
def topic_table_governance(topic: str, table_name: str, _auth: None = OpsAuth) -> Dict[str, Any]:
    target = topic_assets.table_asset_dir(topic, table_name)
    pending = topic_assets.root / topic / "pending" / table_name
    return {
        "success": target.exists(),
        "topic": topic,
        "tableName": table_name,
        "asset": topic_assets.load_table_asset(topic, table_name),
        "pendingAsset": read_json_file(pending / "asset.json"),
        "pendingPatch": read_json_file(pending / "knowledge_suggestion_patch.json"),
        "publishHistory": read_json_file(target / "semantic_publish_history.json"),
        "impact": semantic_governance.impact_analysis(topic, table_name),
    }


@router.post("/api/topics/{topic}/tables/{table_name}/draft")
def stage_topic_table_draft(
    topic: str,
    table_name: str,
    payload: Dict[str, Any] = Body(default_factory=dict),
    _auth: None = OpsAuth,
) -> Dict[str, Any]:
    current = topic_assets.load_table_asset(topic, table_name)
    allowed = {"description", "schemaColumns", "semanticColumns", "metrics", "terms", "knowledgeRules"}
    draft = {**current, **{key: payload[key] for key in allowed if key in payload}, "topic": topic, "tableName": table_name}
    pending = topic_assets.root / topic / "pending" / table_name
    pending.mkdir(parents=True, exist_ok=True)
    write_json(pending / "asset.json", draft)
    for file_name in topic_assets.SEMANTIC_SIDECAR_FILES:
        sidecar = pending / file_name
        if sidecar.exists() and sidecar.is_file():
            sidecar.unlink()
    write_json(
        pending / "editor_metadata.json",
        {"editor": str(payload.get("editor") or "merchant_ops"), "updatedAt": datetime_now_iso(), "fields": sorted(set(payload) & allowed)},
    )
    return {
        "success": True,
        "status": "DRAFT_STAGED",
        "topic": topic,
        "tableName": table_name,
        "pendingPath": str(pending),
        "preflight": semantic_governance.preflight_publish(topic, table_name),
    }


@router.get("/api/topics")
def list_topics(_auth: None = OpsAuth) -> Dict[str, Any]:
    return {"success": True, "items": topic_assets.all_topic_names()}


def apply_request_identity(request: Any, authorization: Optional[str] = None) -> None:
    context = request.context or ChatContext()
    identity = resolve_authenticated_identity(settings, authorization or "", getattr(request, "user_identity", None))
    if identity is not None:
        requested_merchant = str(getattr(request, "merchant_id", "") or "").strip()
        if identity.merchant_id and requested_merchant and identity.merchant_id != requested_merchant:
            raise HTTPException(status_code=403, detail="authenticated identity cannot access requested merchantId")
        if identity.merchant_id:
            request.merchant_id = identity.merchant_id
        request.user_identity = identity
        context.user_identity = identity
    request.context = context


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


def read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        return {}


def datetime_now_iso() -> str:
    from datetime import datetime

    return datetime.utcnow().isoformat() + "Z"


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
