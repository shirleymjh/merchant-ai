from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Protocol

import requests

from merchant_ai.config import Settings
from merchant_ai.models import (
    ExtractedKeywords,
    KnowledgeBundle,
    KnowledgeRetrievalRequest,
    QuestionCategory,
    RecallBundle,
    RecallItem,
    RecallRoundTrace,
    RetrievalIssue,
    category_display,
)
from merchant_ai.services.assets import (
    HybridRecallService,
    TopicAssetService,
    compact_metric_for_recall,
    semantic_metric_path,
)
from merchant_ai.services.authorization_policy import load_authorization_policy
from merchant_ai.services.cache import build_ttl_cache, stable_cache_key
from merchant_ai.services.goal_recall_coverage import (
    attach_goal_recall_capabilities,
    filter_and_tag_goal_recall_items,
    load_goal_recall_capability_protocol,
)
from merchant_ai.services.language_policy import load_language_policy
from merchant_ai.services.semantic_request import semantic_request_cache_key
from merchant_ai.services.text_parsing import collapse_whitespace, safe_ascii_component
from merchant_ai.services.time_semantics import resolve_time_range
from merchant_ai.services.tool_runtime import ToolFailureRegistry


class KnowledgeRetrievalService(Protocol):
    """Unified knowledge retrieval boundary used by the agent harness."""

    backend_name: str

    def retrieve(self, request: KnowledgeRetrievalRequest) -> KnowledgeBundle: ...


class RetrievalLaneFailure(RuntimeError):
    """A supplemental retrieval lane failed after another lane completed."""

    def __init__(self, code: str, lane: str, message: str, partial_items: list[RecallItem]):
        super().__init__(message)
        self.code = code
        self.lane = lane
        self.partial_items = list(partial_items or [])


class HybridKnowledgeRetrievalService:
    """Adapter that exposes the local hybrid recall backend through the unified API."""

    backend_name = "hybrid"

    def __init__(self, recall_service: HybridRecallService):
        self.recall_service = recall_service

    def retrieve(self, request: KnowledgeRetrievalRequest) -> KnowledgeBundle:
        rewritten_query = rewrite_retrieval_query(request)
        recall_bundle = self.recall_service.recall(
            rewritten_query,
            ExtractedKeywords(keywords=request.keywords),
            request.history_rows,
            request.knowledge_context,
            request.merchant_id,
            request.topic_categories,
        )
        if request.topic_categories and not request.knowledge_request and not request.strict_topic_scope:
            broad_bundle = self.recall_service.recall(
                rewritten_query,
                ExtractedKeywords(keywords=request.keywords),
                request.history_rows,
                request.knowledge_context,
                request.merchant_id,
                [],
            )
            recall_bundle = merge_recall_bundles(recall_bundle, broad_bundle)
        governed_items, filtered = filter_recall_items_by_governance(recall_bundle.items, request)
        if request.target_goal_ids or request.required_capabilities:
            governed_items = filter_and_tag_goal_recall_items(
                governed_items,
                target_goal_ids=request.target_goal_ids,
                required_capabilities=request.required_capabilities,
                coverage_receipt_id=request.coverage_receipt_id,
            )
        reranked_items = business_rerank_recall_items(governed_items, rewritten_query, request)
        source_caps = source_type_top_k_policy(
            include_rules=route_is_rule_sensitive(request),
            query_text=rewritten_query,
            topics=[str(item.value if hasattr(item, "value") else item) for item in request.topic_categories],
        )
        reranked_items = limit_recall_items_by_source_type(reranked_items, source_caps, limit=24)
        recall_bundle = RecallBundle(
            items=reranked_items,
            top_score=reranked_items[0].fusion_score if reranked_items else 0.0,
            merged_context="\n\n".join(
                "召回片段 [%s] %s\n%s" % (item.source_type, item.title, item.content[:1200]) for item in reranked_items
            ),
        )
        source_refs = unique_source_refs(recall_bundle.items)
        retrieval_status = "success" if recall_bundle.items else "empty"
        request_key = request.knowledge_request.request_key if request.knowledge_request else ""
        trace = RecallRoundTrace(
            request_key=str(request_key or ""),
            query=request.query,
            topics=[str(item.value if hasattr(item, "value") else item) for item in request.topic_categories],
            backend=self.backend_name,
            recall_queries=recall_queries_from_items(recall_bundle.items),
            source_refs=source_refs,
            item_count=len(recall_bundle.items),
            rewritten_query=rewritten_query,
            governance_filtered=filtered,
            rerank_applied=bool(reranked_items),
            source_type_top_k=source_caps,
            retrieval_status=retrieval_status,
        )
        return KnowledgeBundle(
            recall_bundle=recall_bundle,
            source_refs=source_refs,
            recall_rounds=[trace],
            backend=self.backend_name,
            index_version=self._index_version(),
            semantic_source_hash=semantic_hash_for_items(recall_bundle.items),
            retrieval_status=retrieval_status,
        )

    def _index_version(self) -> str:
        manifest_path = self.recall_service.settings.resolved_workspace_path / "recall_index_manifest.json"
        if not manifest_path.exists():
            return ""
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        return str(payload.get("indexVersion") or "")

    def cache_trace(self) -> dict[str, object]:
        return self.recall_service.cache_trace() if hasattr(self.recall_service, "cache_trace") else {}


class EsKnowledgeRetrievalService:
    """Elasticsearch-backed knowledge retrieval adapter.

    The rest of the harness still consumes KnowledgeBundle/RecallItem, so ES is
    a backend choice, not a second recall path.
    """

    backend_name = "es"

    def __init__(self, settings: Settings, topic_assets: TopicAssetService):
        self.settings = settings
        self.topic_assets = topic_assets
        self._recall_cache = build_ttl_cache("es_recall", settings, settings.cache_recall_ttl_seconds)
        self._embedding_cache = build_ttl_cache("es_embedding", settings, settings.cache_recall_ttl_seconds)
        self._active_retrieval_profile: dict[str, Any] | None = None
        self._active_directory_scope: dict[str, Any] | None = None
        self._active_required_capabilities: list[str] = []

    def retrieve(self, request: KnowledgeRetrievalRequest) -> KnowledgeBundle:
        request_key = request.knowledge_request.request_key if request.knowledge_request else ""
        rewritten_query = rewrite_retrieval_query(request)
        query_text = retrieval_query_text(request, rewritten_query=rewritten_query)
        normalized_categories = [
            category
            for category in [normalize_question_category(item) for item in request.topic_categories]
            if category
        ]
        topics = self._allowed_topics(normalized_categories)
        include_rules = topic_categories_support_knowledge_capability(
            self.topic_assets,
            normalized_categories,
            "rule",
        ) or route_is_rule_sensitive(request)
        # Initial hybrid recall is goal-agnostic. Metric resolution is applied
        # only to candidates that BM25/vector retrieval has already returned.
        metric_candidates: list[dict[str, Any]] = []
        retrieval_profile = build_retrieval_profile(
            query_text=query_text,
            topics=topics,
            include_rules=include_rules,
            metric_candidates=metric_candidates,
            intent_kind=request.intent_kind,
            complexity=request.complexity,
            settings=self.settings,
        )
        retrieval_query_plan = build_retrieval_query_plan(
            query_text=query_text,
            request=request,
            retrieval_profile=retrieval_profile,
            metric_candidates=metric_candidates,
            include_rules=include_rules,
            max_queries=int(self.settings.es_multi_query_max_queries or 5),
            enabled=bool(self.settings.es_multi_query_enabled),
        )
        source_type_top_k = source_type_top_k_policy(
            include_rules=include_rules,
            query_text=query_text,
            topics=topics,
            metric_candidates=metric_candidates,
            retrieval_profile=retrieval_profile,
        )
        route_slots = request.route_slots if isinstance(request.route_slots, dict) else {}
        object_filters = list(route_slots.get("objectRefs") or route_slots.get("object_refs") or [])
        knowledge_filter = (
            request.knowledge_request.model_dump(by_alias=True) if request.knowledge_request is not None else {}
        )
        metric_contracts = [
            {
                "metricKey": str(item.get("canonicalMetricKey") or item.get("metricKey") or ""),
                "ownerTable": str(item.get("ownerTable") or item.get("tableName") or ""),
            }
            for item in metric_candidates
            if str(item.get("canonicalMetricKey") or item.get("metricKey") or "")
        ]
        normalized_retrieval_question = collapse_whitespace(query_text).casefold()
        retrieval_question_hash = hashlib.sha256(
            normalized_retrieval_question.encode("utf-8")
        ).hexdigest()
        cache_key = semantic_request_cache_key(
            "es_recall",
            topics=topics,
            metrics=metric_contracts,
            dimensions=list(route_slots.get("dimensions") or []),
            filters=[
                *object_filters,
                {
                    "retrievalQuestionHash": retrieval_question_hash,
                    "includeRules": include_rules,
                    "intentKind": request.intent_kind,
                    "complexity": request.complexity,
                    "strictTopicScope": request.strict_topic_scope,
                    "targetGoalIds": sorted(request.target_goal_ids),
                    "requiredCapabilities": sorted(request.required_capabilities),
                    "coverageReceiptId": request.coverage_receipt_id,
                    "retrievalQueryPlan": retrieval_query_plan,
                },
                *([knowledge_filter] if knowledge_filter else []),
            ],
            time_range=resolve_time_range(query_text, self.settings.business_timezone),
            asset_version={
                "indexVersion": self._index_version(),
                "vectorEnabled": self._vector_enabled(),
                "embeddingModel": self.settings.embedding_model if self._vector_enabled() else "",
                "embeddingDims": int(self.settings.embedding_dims or 0) if self._vector_enabled() else 0,
                "vectorField": self.settings.es_vector_field if self._vector_enabled() else "",
                "retrievalPolicy": {
                    "cacheIdentityVersion": "es_recall_cache_v2",
                    "profile": retrieval_profile,
                    "version": "hierarchical_v1",
                    "rrfK": self.settings.es_rrf_k,
                    "sourceTypeTopK": source_type_top_k,
                    "multiQueryEnabled": bool(self.settings.es_multi_query_enabled),
                    "multiQueryMaxQueries": int(self.settings.es_multi_query_max_queries or 5),
                    "hierarchicalEnabled": bool(self.settings.es_hierarchical_retrieval_enabled),
                    "hierarchicalMaxDirectories": int(self.settings.es_hierarchical_max_directories or 2),
                    "hierarchicalMaxLeafItems": int(self.settings.es_hierarchical_max_leaf_items or 16),
                },
            },
            scope={
                "merchantId": request.merchant_id,
                "accessRole": request.access_role,
                "permissions": sorted(request.permissions),
            },
        )
        cached = self._recall_cache.get(cache_key)
        if cached is not None:
            return normalize_knowledge_bundle_status(KnowledgeBundle.model_validate(cached))
        self._active_retrieval_profile = retrieval_profile
        self._active_required_capabilities = [
            str(item or "").strip().upper() for item in request.required_capabilities if str(item or "").strip()
        ]
        retrieval_issues: list[RetrievalIssue] = []
        directory_retrieval_trace: list[dict[str, Any]] = []
        retrieval_stop_reason = ""
        hierarchical_retrieval_applied = False
        try:
            try:
                try:
                    items = self._search(query_text, topics, include_rules=include_rules)
                    record_retrieval_query_result(
                        retrieval_query_plan,
                        query_id="base",
                        status="SUCCESS" if items else "EMPTY",
                        items=items,
                    )
                except RetrievalLaneFailure as exc:
                    items = list(exc.partial_items)
                    record_retrieval_query_result(
                        retrieval_query_plan,
                        query_id="base",
                        status="DEGRADED",
                        items=items,
                        error_code=exc.code,
                    )
                    retrieval_issues.append(
                        RetrievalIssue(
                            code=exc.code,
                            message=str(exc)[:500],
                            backend=self.backend_name,
                            lane=exc.lane,
                            stage="topic_scope",
                            severity="warning",
                            request_key=str(request_key or ""),
                            details={"partialItemCount": len(items)},
                        )
                    )
                query_ranked_groups: list[tuple[str, list[RecallItem]]] = [("query:base", items)]
                coverage_items = list(items)
                for subquery in retrieval_query_plan[1:]:
                    subquery_id = str(subquery.get("id") or "supplemental")
                    subquery_text = str(subquery.get("query") or "").strip()
                    if not subquery_text:
                        continue
                    if not uncovered_retrieval_queries([subquery], coverage_items):
                        record_retrieval_query_result(
                            retrieval_query_plan,
                            query_id=subquery_id,
                            status="SKIPPED",
                            items=[],
                            stop_reason="BASE_OR_PRIOR_QUERY_COVERAGE",
                        )
                        continue
                    try:
                        subquery_items = self._search(
                            subquery_text,
                            topics,
                            include_rules=include_rules,
                        )
                    except RetrievalLaneFailure as exc:
                        subquery_items = list(exc.partial_items)
                        record_retrieval_query_result(
                            retrieval_query_plan,
                            query_id=subquery_id,
                            status="DEGRADED",
                            items=subquery_items,
                            error_code=exc.code,
                        )
                        retrieval_issues.append(
                            RetrievalIssue(
                                code=exc.code,
                                message=str(exc)[:500],
                                backend=self.backend_name,
                                lane=exc.lane,
                                stage="multi_query",
                                severity="warning",
                                request_key=str(request_key or ""),
                                details={
                                    "queryId": subquery_id,
                                    "partialItemCount": len(subquery_items),
                                },
                            )
                        )
                    except Exception as exc:
                        record_retrieval_query_result(
                            retrieval_query_plan,
                            query_id=subquery_id,
                            status="FAILED",
                            items=[],
                            error_code="ES_MULTI_QUERY_FAILED",
                        )
                        retrieval_issues.append(
                            RetrievalIssue(
                                code="ES_MULTI_QUERY_FAILED",
                                message=str(exc)[:500],
                                backend=self.backend_name,
                                lane=subquery_id,
                                stage="multi_query",
                                severity="warning",
                                request_key=str(request_key or ""),
                                details={"queryId": subquery_id},
                            )
                        )
                        continue
                    else:
                        record_retrieval_query_result(
                            retrieval_query_plan,
                            query_id=subquery_id,
                            status="SUCCESS" if subquery_items else "EMPTY",
                            items=subquery_items,
                        )
                    query_ranked_groups.append(("query:%s" % subquery_id, subquery_items))
                    coverage_items = merge_recall_items(coverage_items, subquery_items)
                if len(query_ranked_groups) > 1:
                    items = rrf_fuse_recall_items(
                        query_ranked_groups,
                        rrf_k=self.settings.es_rrf_k,
                        score_scale=self.settings.es_rrf_score_scale,
                        limit=max(
                            1,
                            int(retrieval_profile.get("hybridTopK") or self.settings.es_hybrid_top_k or 24),
                        ),
                    )
                if (
                    topics
                    and not request.knowledge_request
                    and not request.strict_topic_scope
                    and bool(retrieval_profile.get("broadSearchEnabled", True))
                ):
                    try:
                        broad_items = self._search(query_text, [], include_rules=False)
                        items = rrf_fuse_recall_items(
                            [("topic_scope", items), ("broad_scope", broad_items)],
                            rrf_k=self.settings.es_rrf_k,
                            score_scale=self.settings.es_rrf_score_scale,
                            limit=max(
                                1, int(retrieval_profile.get("hybridTopK") or self.settings.es_hybrid_top_k or 24)
                            ),
                        )
                    except RetrievalLaneFailure as exc:
                        broad_items = list(exc.partial_items)
                        retrieval_issues.append(
                            RetrievalIssue(
                                code=exc.code,
                                message=str(exc)[:500],
                                backend=self.backend_name,
                                lane=exc.lane,
                                stage="broad_scope",
                                severity="warning",
                                request_key=str(request_key or ""),
                                details={"partialItemCount": len(broad_items)},
                            )
                        )
                        items = rrf_fuse_recall_items(
                            [("topic_scope", items), ("broad_scope", broad_items)],
                            rrf_k=self.settings.es_rrf_k,
                            score_scale=self.settings.es_rrf_score_scale,
                            limit=max(
                                1, int(retrieval_profile.get("hybridTopK") or self.settings.es_hybrid_top_k or 24)
                            ),
                        )
                    except Exception as exc:
                        retrieval_issues.append(
                            RetrievalIssue(
                                code="ES_RETRIEVAL_SCOPE_FAILED",
                                message=str(exc)[:500],
                                backend=self.backend_name,
                                lane="broad",
                                stage="broad_scope",
                                severity="warning",
                                request_key=str(request_key or ""),
                                details={"topicScopedItemCount": len(items)},
                            )
                        )
                (
                    hierarchical_items,
                    directory_retrieval_trace,
                    retrieval_stop_reason,
                    hierarchical_issues,
                ) = self._expand_hierarchical_retrieval(
                    request=request,
                    query_text=query_text,
                    topics=topics,
                    include_rules=include_rules,
                    initial_items=items,
                    retrieval_query_plan=retrieval_query_plan,
                    retrieval_profile=retrieval_profile,
                    request_key=str(request_key or ""),
                )
                retrieval_issues.extend(hierarchical_issues)
                hierarchical_retrieval_applied = any(
                    str(item.get("stage") or "") == "DIRECTORY_EXPANSION" for item in directory_retrieval_trace
                )
                items = merge_recall_items(items, hierarchical_items)
                items = self._attach_current_asset_governance(items)
                governance_decisions = recall_governance_decisions(items, request)
                items, governance_filtered = filter_recall_items_by_governance(items, request)
                if request.target_goal_ids or request.required_capabilities:
                    items = filter_and_tag_goal_recall_items(
                        items,
                        target_goal_ids=request.target_goal_ids,
                        required_capabilities=request.required_capabilities,
                        coverage_receipt_id=request.coverage_receipt_id,
                    )
                items, metric_candidates = self._validate_recalled_metric_candidates(
                    query_text,
                    topics,
                    items,
                )
                directory_retrieval_trace.append(
                    {
                        "stepId": "final:governance",
                        "parentStepId": (
                            str(directory_retrieval_trace[-1].get("stepId") or "") if directory_retrieval_trace else ""
                        ),
                        "stage": "FINAL_GOVERNANCE",
                        "depth": 3,
                        "candidateCount": len(items) + len(governance_decisions),
                        "selectedRefs": unique_source_refs(items)[:16],
                        "eliminatedCandidates": governance_decisions[:16],
                        "governanceFiltered": governance_filtered,
                    }
                )
                items = business_rerank_recall_items(items, query_text, request)
                pre_cap_items = list(items)
                items = limit_recall_items_by_source_type(
                    items,
                    source_type_top_k,
                    limit=max(
                        1, int(retrieval_profile.get("hybridTopK") or self.settings.es_hybrid_top_k or len(items) or 1)
                    ),
                )
                selected_refs = set(unique_source_refs(items))
                cap_eliminated = [
                    {
                        "refId": item.doc_id or str((item.metadata or {}).get("semanticRefId") or ""),
                        "sourceType": str(item.source_type or "UNKNOWN").upper(),
                        "reasonCode": "SOURCE_TYPE_OR_FINAL_TOP_K_CAP",
                    }
                    for item in pre_cap_items
                    if (item.doc_id or str((item.metadata or {}).get("semanticRefId") or "")) not in selected_refs
                ]
                directory_retrieval_trace.append(
                    {
                        "stepId": "final:selection",
                        "parentStepId": "final:governance",
                        "stage": "FINAL_SOURCE_TYPE_CAP",
                        "depth": 3,
                        "candidateCount": len(pre_cap_items),
                        "selectedRefs": unique_source_refs(items)[:16],
                        "eliminatedCandidates": cap_eliminated[:16],
                        "omittedCandidateCount": max(0, len(cap_eliminated) - 16),
                    }
                )
            except Exception as exc:
                items = []
                governance_filtered = {}
                retrieval_stop_reason = retrieval_stop_reason or "RETRIEVAL_FAILED"
                retrieval_issues.append(
                    RetrievalIssue(
                        code="ES_RETRIEVAL_FAILED",
                        message=str(exc)[:500],
                        backend=self.backend_name,
                        lane="primary",
                        stage="retrieve",
                        severity="blocking",
                        request_key=str(request_key or ""),
                    )
                )
        finally:
            self._active_retrieval_profile = None
            self._active_required_capabilities = []
        retrieval_status = retrieval_status_for(items, retrieval_issues)
        hierarchical_stop_reason = retrieval_stop_reason
        if retrieval_status == "failed":
            retrieval_stop_reason = "BACKEND_FAILED"
        elif not items:
            retrieval_stop_reason = "NO_CANDIDATES"
        elif retrieval_status == "degraded":
            retrieval_stop_reason = "FINAL_EVIDENCE_SELECTED_DEGRADED"
        else:
            retrieval_stop_reason = "FINAL_EVIDENCE_SELECTED"
        retrieval_stop_details = {
            "hierarchicalStopReason": hierarchical_stop_reason,
            "finalItemCount": len(items),
            "issueCount": len(retrieval_issues),
            "queryCount": len(retrieval_query_plan),
            "stepCount": len(directory_retrieval_trace),
        }
        blocked_reason = next(
            (
                "%s:%s" % (issue.code, issue.message[:240])
                for issue in retrieval_issues
                if issue.severity == "blocking" or retrieval_status == "failed"
            ),
            "",
        )
        source_refs = unique_source_refs(items)
        trace = RecallRoundTrace(
            request_key=str(request_key or ""),
            query=request.query,
            topics=[str(item.value if hasattr(item, "value") else item) for item in request.topic_categories],
            backend=self.backend_name,
            recall_queries=[
                str(item.get("query") or "") for item in retrieval_query_plan if str(item.get("query") or "").strip()
            ],
            source_refs=source_refs,
            item_count=len(items),
            blocked_reason=blocked_reason,
            recall_channels=recall_channels_for_items(items),
            source_type_top_k=source_type_top_k,
            vector_enabled=self._vector_enabled(),
            vector_disabled=not self._vector_enabled(),
            metric_candidates=metric_trace_payload(metric_candidates),
            retrieval_profile=retrieval_profile,
            query_type=str(retrieval_profile.get("queryType") or ""),
            intent_kind=str(request.intent_kind or ""),
            complexity=str(request.complexity or ""),
            retrieval_lanes=retrieval_lane_trace(
                retrieval_profile=retrieval_profile,
                vector_enabled=self._vector_enabled(),
                include_rules=include_rules,
                has_metric_candidates=bool(metric_candidates),
                broad_enabled=bool(topics and not request.strict_topic_scope),
                query_plan_size=len(retrieval_query_plan),
                hierarchical_enabled=bool(self.settings.es_hierarchical_retrieval_enabled),
            ),
            retrieval_query_plan=retrieval_query_plan,
            directory_retrieval_trace=directory_retrieval_trace,
            hierarchical_retrieval_applied=hierarchical_retrieval_applied,
            retrieval_stop_reason=retrieval_stop_reason,
            retrieval_stop_details=retrieval_stop_details,
            rewritten_query=rewritten_query,
            governance_filtered=governance_filtered,
            rerank_applied=bool(items),
            retrieval_status=retrieval_status,
            retrieval_issues=retrieval_issues,
        )
        merged = "\n\n".join(
            "召回片段 [%s] %s\n%s" % (item.source_type, item.title, item.content[:1200]) for item in items
        )
        bundle = KnowledgeBundle(
            recall_bundle=RecallBundle(
                items=items,
                top_score=items[0].fusion_score if items else 0.0,
                merged_context=merged,
            ),
            source_refs=source_refs,
            recall_rounds=[trace],
            backend=self.backend_name,
            index_version=self._index_version(),
            semantic_source_hash=semantic_hash_for_items(items),
            retrieval_status=retrieval_status,
            retrieval_issues=retrieval_issues,
        )
        if retrieval_status in {"success", "empty"}:
            self._recall_cache.set(cache_key, bundle.model_dump(by_alias=True))
        return bundle

    def _allowed_topics(self, topic_categories: list[QuestionCategory]) -> list[str]:
        topic_names = self.topic_assets.topic_names_for_categories(topic_categories)
        if topic_names:
            return topic_names
        names: list[str] = []
        for category in topic_categories:
            display = category_display(category)
            if display and display not in names:
                names.append(display)
        return names

    def _expand_hierarchical_retrieval(
        self,
        *,
        request: KnowledgeRetrievalRequest,
        query_text: str,
        topics: list[str],
        include_rules: bool,
        initial_items: list[RecallItem],
        retrieval_query_plan: list[dict[str, Any]],
        retrieval_profile: dict[str, Any],
        request_key: str,
    ) -> tuple[list[RecallItem], list[dict[str, Any]], str, list[RetrievalIssue]]:
        """Resolve a bounded Topic -> table -> exact-leaf retrieval path.

        The Core keeps orchestration authority.  This helper only turns the
        existing L0/L1/L2 semantic directory into a generic, observable search
        primitive and never treats a recall hit as execution authority.
        """

        trace: list[dict[str, Any]] = []
        issues: list[RetrievalIssue] = []
        if not bool(self.settings.es_hierarchical_retrieval_enabled):
            return [], trace, "HIERARCHICAL_RETRIEVAL_DISABLED", issues
        if not topics:
            return [], trace, "NO_TOPIC_DIRECTORY_SCOPE", issues

        governed_initial = self._attach_current_asset_governance(initial_items)
        governed_initial, initial_governance_filtered = filter_recall_items_by_governance(
            governed_initial,
            request,
        )
        target_queries = [item for item in retrieval_query_plan if item.get("targetSourceTypes")]
        if not target_queries:
            return [], trace, "SIMPLE_QUERY_SINGLE_PASS", issues

        uncovered = uncovered_retrieval_queries(target_queries, governed_initial)
        trace.append(
            {
                "stepId": "coverage:initial",
                "parentStepId": "",
                "stage": "INITIAL_LEAF_COVERAGE",
                "depth": 0,
                "queryIds": [str(item.get("id") or "") for item in target_queries],
                "coveredQueryIds": [str(item.get("id") or "") for item in target_queries if item not in uncovered],
                "uncoveredQueryIds": [str(item.get("id") or "") for item in uncovered],
                "candidateRefs": unique_source_refs(governed_initial)[:12],
                "governanceFiltered": initial_governance_filtered,
            }
        )
        if not uncovered:
            return [], trace, "TARGET_TYPES_ALREADY_COVERED", issues

        all_directories = select_retrieval_directories(
            governed_initial,
            allowed_topics=topics,
            limit=max(1, len(governed_initial)),
        )
        directory_limit = max(1, int(self.settings.es_hierarchical_max_directories or 2))
        directories = all_directories[:directory_limit]
        trace.append(
            {
                "stepId": "directory:selection",
                "parentStepId": "coverage:initial",
                "stage": "DIRECTORY_SELECTION",
                "depth": 1,
                "candidateCount": len(all_directories),
                "selectedDirectories": directories,
                "eliminatedByDirectoryCap": all_directories[directory_limit:][:8],
                "directoryLimit": directory_limit,
            }
        )
        if not directories:
            return [], trace, "NO_TABLE_DIRECTORY_CANDIDATE", issues

        selected_topics = sorted(
            {str(directory.get("topic") or "") for directory in directories if str(directory.get("topic") or "")}
        )
        selected_tables = [
            str(directory.get("table") or "") for directory in directories if str(directory.get("table") or "")
        ]
        selected_directory_ids = [
            "semantic:%s:%s:directory" % (str(directory.get("topic") or ""), str(directory.get("table") or ""))
            for directory in directories
        ]
        directory_query = build_directory_retrieval_query(
            query_text=query_text,
            directories=directories,
            uncovered_queries=uncovered,
        )
        previous_directory_scope = self._active_directory_scope
        self._active_directory_scope = {
            "topics": selected_topics,
            "tables": selected_tables,
            "directoryIds": selected_directory_ids,
        }
        try:
            hits = self._search(
                directory_query,
                selected_topics,
                include_rules=include_rules,
            )
        except RetrievalLaneFailure as exc:
            hits = list(exc.partial_items)
            issues.append(
                RetrievalIssue(
                    code=exc.code,
                    message=str(exc)[:500],
                    backend=self.backend_name,
                    lane=exc.lane,
                    stage="directory_expansion",
                    severity="warning",
                    request_key=request_key,
                    details={
                        "directories": selected_directory_ids,
                        "partialItemCount": len(hits),
                    },
                )
            )
        except Exception as exc:
            issues.append(
                RetrievalIssue(
                    code="ES_DIRECTORY_RETRIEVAL_FAILED",
                    message=str(exc)[:500],
                    backend=self.backend_name,
                    lane="directory_batch",
                    stage="directory_expansion",
                    severity="warning",
                    request_key=request_key,
                    details={"directories": selected_directory_ids},
                )
            )
            trace.append(
                {
                    "stepId": "directory:batch",
                    "parentStepId": "directory:selection",
                    "stage": "DIRECTORY_EXPANSION",
                    "depth": 2,
                    "directories": selected_directory_ids,
                    "query": directory_query,
                    "status": "FAILED",
                    "newRefs": [],
                }
            )
            return [], trace, "DIRECTORY_EXPANSION_DEGRADED", issues
        finally:
            self._active_directory_scope = previous_directory_scope

        governed_hits = self._attach_current_asset_governance(hits)
        governed_hits, leaf_governance_filtered = filter_recall_items_by_governance(
            governed_hits,
            request,
        )
        scoped_hits = [
            item
            for item in governed_hits
            if recall_item_in_directories(item, directories)
            and recall_item_is_exact_leaf(item)
            and recall_item_matches_uncovered_queries(item, uncovered)
        ]
        seen_refs = set(unique_source_refs(governed_initial))
        max_leaf_items = max(1, int(self.settings.es_hierarchical_max_leaf_items or 16))
        expanded: list[RecallItem] = []
        duplicate_refs: list[str] = []
        for item in scoped_hits:
            ref = item.doc_id or str((item.metadata or {}).get("semanticRefId") or "")
            if not ref or ref in seen_refs:
                if ref:
                    duplicate_refs.append(ref)
                continue
            seen_refs.add(ref)
            matched_directory = recall_item_matched_directory(item, directories)
            metadata = dict(item.metadata or {})
            metadata["directoryTraversal"] = {
                "topic": str(matched_directory.get("topic") or ""),
                "table": str(matched_directory.get("table") or ""),
                "depth": 2,
                "queryIds": [str(value.get("id") or "") for value in uncovered],
            }
            expanded.append(item.model_copy(update={"metadata": metadata}))
            if len(expanded) >= max_leaf_items:
                break
        trace.append(
            {
                "stepId": "directory:batch",
                "parentStepId": "directory:selection",
                "stage": "DIRECTORY_EXPANSION",
                "depth": 2,
                "directories": selected_directory_ids,
                "query": directory_query,
                "status": "EXPANDED",
                "candidateRefs": unique_source_refs(scoped_hits)[:12],
                "newRefs": unique_source_refs(expanded)[:12],
                "discardedOutsideDirectory": max(0, len(governed_hits) - len(scoped_hits)),
                "discardedDuplicateRefs": duplicate_refs[:12],
                "governanceFiltered": leaf_governance_filtered,
            }
        )

        if expanded:
            reason = "MAX_LEAF_BUDGET_REACHED" if len(expanded) >= max_leaf_items else "NEW_EXACT_LEAF_EVIDENCE_FOUND"
            return expanded[:max_leaf_items], trace, reason, issues
        return [], trace, "NO_NEW_EXACT_LEAF_EVIDENCE", issues

    def _attach_current_asset_governance(self, items: list[RecallItem]) -> list[RecallItem]:
        """Recheck ES hits against the live published semantic asset.

        This keeps role/status/version checks effective even before an older ES
        index has been rebuilt with the latest governance metadata.
        """
        governed: list[RecallItem] = []
        asset_cache: dict[tuple[str, str], dict[str, Any]] = {}
        for item in items or []:
            metadata = dict(item.metadata or {})
            topic = str(item.topic or metadata.get("topic") or "")
            table = str(item.table or metadata.get("tableName") or "")
            if not topic or not table:
                governed.append(item)
                continue
            key = (topic, table)
            if key not in asset_cache:
                asset = self.topic_assets.load_table_asset(topic, table)
                current = recall_governance_metadata(asset)
                current_version = (
                    str(asset.get("version") or asset.get("semanticVersion") or "") if isinstance(asset, dict) else ""
                )
                if current_version:
                    current["activeVersion"] = current_version
                asset_cache[key] = current
            current = asset_cache[key]
            merged = {
                **current,
                **metadata,
                "activeVersion": current.get("activeVersion") or metadata.get("activeVersion") or "",
                "assetStatus": current.get("status") or "",
                "assetMerchantId": current.get("merchantId") or "",
                "assetAllowedRoles": current.get("allowedRoles") or [],
                "assetRequiredPermissions": current.get("requiredPermissions") or [],
                "assetVisibilityPolicy": current.get("visibilityPolicy") or {},
                "assetExpiresAt": current.get("expiresAt") or "",
            }
            governed.append(item.model_copy(update={"metadata": merged}))
        return governed

    def _search(self, query_text: str, topics: list[str], include_rules: bool = False) -> list[RecallItem]:
        text_items = self._text_search(query_text, topics, include_rules=include_rules)
        vector_items: list[RecallItem] = []
        if not self._vector_enabled() or not query_text:
            return rrf_fuse_recall_items(
                [("bm25", text_items)],
                rrf_k=self.settings.es_rrf_k,
                score_scale=self.settings.es_rrf_score_scale,
                limit=self._hybrid_size(),
            )
        try:
            vector = self._embed_text(query_text)
            vector_items = (
                self._vector_search(query_text, vector, topics, include_rules=include_rules) if vector else []
            )
        except Exception as exc:
            raise RetrievalLaneFailure(
                "ES_RETRIEVAL_LANE_FAILED",
                "vector",
                str(exc)[:500],
                text_items,
            ) from exc
        if not vector_items:
            return rrf_fuse_recall_items(
                [("bm25", text_items)],
                rrf_k=self.settings.es_rrf_k,
                score_scale=self.settings.es_rrf_score_scale,
                limit=self._hybrid_size(),
            )
        return rrf_fuse_recall_items(
            [("bm25", text_items), ("vector", vector_items)],
            rrf_k=self.settings.es_rrf_k,
            score_scale=self.settings.es_rrf_score_scale,
            limit=self._hybrid_size(),
        )

    def _text_search(self, query_text: str, topics: list[str], include_rules: bool = False) -> list[RecallItem]:
        if not self.settings.es_base_url:
            raise RuntimeError("ES_BASE_URL_MISSING")
        size = self._text_size(topics)
        query = self._text_query(query_text, topics, include_rules=include_rules)
        response = requests.post(
            "%s/%s/_search" % (self.settings.es_base_url.rstrip("/"), self.settings.es_index),
            headers=self._headers(),
            auth=self._auth(),
            json={"size": size, "query": query},
            timeout=10,
        )
        response.raise_for_status()
        hits = ((response.json() or {}).get("hits") or {}).get("hits") or []
        return [es_hit_to_recall_item(hit, query_text, channel="bm25") for hit in hits]

    def _vector_search(
        self, query_text: str, query_vector: list[float], topics: list[str], include_rules: bool = False
    ) -> list[RecallItem]:
        if not self.settings.es_base_url:
            raise RuntimeError("ES_BASE_URL_MISSING")
        if not query_vector:
            return []
        size = self._vector_size(topics)
        filters = self._filters(topics, include_rules=include_rules)
        knn: dict[str, object] = {
            "field": self.settings.es_vector_field,
            "query_vector": query_vector,
            "k": size,
            "num_candidates": max(size, int(self.settings.es_vector_num_candidates or 0)),
        }
        if filters:
            knn["filter"] = filters if len(filters) > 1 else filters[0]
        response = requests.post(
            "%s/%s/_search" % (self.settings.es_base_url.rstrip("/"), self.settings.es_index),
            headers=self._headers(),
            auth=self._auth(),
            json={"size": size, "knn": knn},
            timeout=10,
        )
        response.raise_for_status()
        hits = ((response.json() or {}).get("hits") or {}).get("hits") or []
        return [es_hit_to_recall_item(hit, query_text, channel="vector") for hit in hits]

    def _text_query(self, query_text: str, topics: list[str], include_rules: bool = False) -> dict[str, object]:
        must: list[dict[str, object]] = []
        if query_text:
            must.append(
                {
                    "multi_match": {
                        "query": query_text,
                        "fields": [
                            "title^3",
                            "content^2",
                            "metadata.businessName^3",
                            "metadata.aliases^2",
                            "metadata.metricKey^3",
                            "metadata.columnName^3",
                            "metadata.term^3",
                            "metadata.ruleTitle^3",
                            "metadata.relationshipId^2",
                            "metadata.tableName^2",
                            "metadata.semanticKind^2",
                        ],
                    }
                }
            )
        filters = self._filters(topics, include_rules=include_rules)
        if must or filters:
            return {"bool": {"must": must or [{"match_all": {}}], "filter": filters}}
        return {"match_all": {}}

    def _filters(self, topics: list[str], include_rules: bool = False) -> list[dict[str, object]]:
        filters: list[dict[str, object]] = []
        if topics:
            topic_should: list[dict[str, object]] = [
                {"terms": {"topic": topics}},
                {"terms": {"topic.keyword": topics}},
                {"terms": {"metadata.topic": topics}},
                {"terms": {"metadata.topic.keyword": topics}},
            ]
            if include_rules:
                topic_should.append({"term": {"source_type": "GOVERNED_RULE"}})
            filters.append({"bool": {"should": topic_should, "minimum_should_match": 1}})
        elif include_rules:
            filters.append({"term": {"source_type": "GOVERNED_RULE"}})
        if self._active_required_capabilities:
            protocol = load_goal_recall_capability_protocol()
            source_types = protocol.source_types_for(self._active_required_capabilities)
            capability_should: list[dict[str, object]] = [
                {"terms": {"metadata.goalRecallCapabilities": (self._active_required_capabilities)}}
            ]
            if source_types:
                capability_should.append({"terms": {"source_type": source_types}})
            filters.append(
                {
                    "bool": {
                        "should": capability_should,
                        "minimum_should_match": 1,
                    }
                }
            )
        directory_scope = self._active_directory_scope or {}
        directory_ids = [str(item) for item in directory_scope.get("directoryIds") or [] if str(item or "").strip()]
        tables = [str(item) for item in directory_scope.get("tables") or [] if str(item or "").strip()]
        if directory_ids or tables:
            directory_should: list[dict[str, object]] = []
            if directory_ids:
                directory_should.extend(
                    [
                        {"terms": {"parent_directory_id": directory_ids}},
                        {"terms": {"directory_id": directory_ids}},
                    ]
                )
            if tables:
                directory_should.extend(
                    [
                        {"terms": {"table": tables}},
                        {"terms": {"metadata.tableName": tables}},
                        {"terms": {"metadata.leftTable": tables}},
                        {"terms": {"metadata.rightTable": tables}},
                    ]
                )
            filters.append(
                {
                    "bool": {
                        "should": directory_should,
                        "minimum_should_match": 1,
                    }
                }
            )
        return filters

    def _text_size(self, topics: list[str]) -> int:
        profile = self._active_retrieval_profile or {}
        key = "textTopK" if topics else "broadTextTopK"
        fallback = self.settings.es_text_top_k if topics else self.settings.es_broad_text_top_k
        return max(1, int(profile.get(key) or fallback))

    def _vector_size(self, topics: list[str]) -> int:
        profile = self._active_retrieval_profile or {}
        key = "vectorTopK" if topics else "broadVectorTopK"
        fallback = self.settings.es_vector_top_k if topics else self.settings.es_broad_vector_top_k
        return max(1, int(profile.get(key) or fallback))

    def _hybrid_size(self) -> int:
        profile = self._active_retrieval_profile or {}
        return max(1, int(profile.get("hybridTopK") or self.settings.es_hybrid_top_k or 24))

    def _vector_enabled(self) -> bool:
        return bool(
            self.settings.es_vector_enabled
            and self.settings.es_vector_field
            and self.settings.embedding_model
            and self._embedding_api_key()
        )

    def _embedding_api_key(self) -> str:
        return str(self.settings.embedding_api_key or self.settings.llm_api_key or "").strip()

    def _embed_text(self, text: str) -> list[float]:
        value = str(text or "").strip()
        if not value:
            return []
        cache_key = stable_cache_key(
            "embedding",
            {
                "baseUrl": self.settings.embedding_base_url,
                "model": self.settings.embedding_model,
                "dims": self.settings.embedding_dims,
                "text": value,
            },
        )
        cached = self._embedding_cache.get(cache_key)
        if isinstance(cached, list):
            return [float(item) for item in cached]
        payload: dict[str, object] = {"model": self.settings.embedding_model, "input": value}
        if int(self.settings.embedding_dims or 0) > 0:
            payload["dimensions"] = int(self.settings.embedding_dims)
        response = requests.post(
            "%s/embeddings" % self.settings.embedding_base_url.rstrip("/"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": "Bearer %s" % self._embedding_api_key(),
            },
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json() or {}
        vector = ((data.get("data") or [{}])[0] or {}).get("embedding") or []
        result = [float(item) for item in vector if isinstance(item, (int, float))]
        if result:
            self._embedding_cache.set(cache_key, result)
        return result

    def cache_trace(self) -> dict[str, object]:
        trace = {"esRecall": self._recall_cache.trace(), "esEmbedding": self._embedding_cache.trace()}
        trace["hybridRecall"] = {
            "vectorEnabled": self._vector_enabled(),
            "vectorField": self.settings.es_vector_field,
            "textTopK": self.settings.es_text_top_k,
            "vectorTopK": self.settings.es_vector_top_k,
            "broadTextTopK": self.settings.es_broad_text_top_k,
            "broadVectorTopK": self.settings.es_broad_vector_top_k,
            "rrfK": self.settings.es_rrf_k,
            "hybridTopK": self.settings.es_hybrid_top_k,
            "dynamicTopKEnabled": True,
            "multiQueryEnabled": bool(self.settings.es_multi_query_enabled),
            "multiQueryMaxQueries": int(self.settings.es_multi_query_max_queries or 5),
            "hierarchicalRetrievalEnabled": bool(self.settings.es_hierarchical_retrieval_enabled),
            "hierarchicalMaxDirectories": int(self.settings.es_hierarchical_max_directories or 2),
            "hierarchicalMaxLeafItems": int(self.settings.es_hierarchical_max_leaf_items or 16),
        }
        return trace

    def _validate_recalled_metric_candidates(
        self,
        query_text: str,
        topics: list[str],
        items: list[RecallItem],
    ) -> tuple[list[RecallItem], list[dict[str, Any]]]:
        """Attach exact-match metadata without adding unrecalled assets."""

        recalled_metric_refs = {
            str((item.metadata or {}).get("semanticRefId") or item.doc_id or "")
            for item in items
            if str(item.source_type or "").strip().upper()
            == "SEMANTIC_METRIC"
            and str(
                (item.metadata or {}).get("semanticRefId")
                or item.doc_id
                or ""
            ).strip()
        }
        if not recalled_metric_refs:
            return items, []
        resolved = {
            str(candidate.get("semanticRefId") or ""): candidate
            for candidate in self._resolve_metric_candidates(
                query_text,
                topics,
            )
            if str(candidate.get("semanticRefId") or "")
            in recalled_metric_refs
        }
        if not resolved:
            return items, []

        annotated: list[RecallItem] = []
        for item in items:
            ref_id = str(
                (item.metadata or {}).get("semanticRefId")
                or item.doc_id
                or ""
            ).strip()
            candidate = resolved.get(ref_id)
            if candidate is None:
                annotated.append(item)
                continue
            metadata = dict(item.metadata or {})
            metadata.update(
                {
                    "matchedMetricLabel": str(
                        candidate.get("matchedMetricLabel") or ""
                    ),
                    "metricResolutionType": str(
                        candidate.get("metricResolutionType") or ""
                    ),
                    "metricResolutionReason": str(
                        candidate.get("metricResolutionReason") or ""
                    ),
                    "metricResolutionConfidence": float(
                        candidate.get("metricResolutionConfidence") or 0.0
                    ),
                    "metricResolutionAmbiguous": bool(
                        candidate.get("metricResolutionAmbiguous") or False
                    ),
                    "metricResolutionStage": "POST_RECALL_VALIDATION",
                }
            )
            annotated.append(item.model_copy(update={"metadata": metadata}))
        selected = [
            resolved[ref_id]
            for ref_id in recalled_metric_refs
            if ref_id in resolved
        ]
        selected.sort(
            key=lambda item: (
                float(item.get("metricResolutionConfidence") or 0.0),
                int(item.get("matchLength") or 0),
                str(item.get("semanticRefId") or ""),
            ),
            reverse=True,
        )
        return annotated, selected

    def _resolve_metric_candidates(self, query_text: str, topics: list[str]) -> list[dict[str, Any]]:
        query = (query_text or "").strip()
        if not query:
            return []
        topic_names = topics or self.topic_assets.all_topic_names()
        candidates: list[dict[str, Any]] = []
        by_id: dict[str, dict[str, Any]] = {}
        metrics_by_scope: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
        governance_by_scope: dict[tuple[str, str], dict[str, Any]] = {}
        for topic in topic_names:
            for manifest_item in self.topic_assets.load_manifest(topic):
                table = str(manifest_item.get("tableName") or "")
                if not table:
                    continue
                metrics = [
                    metric for metric in self.topic_assets.load_table_metrics(topic, table) if isinstance(metric, dict)
                ]
                table_asset = self.topic_assets.load_table_asset(topic, table)
                table_governance = recall_governance_metadata(table_asset)
                metrics_by_key = {
                    str(metric.get("metricKey") or ""): metric
                    for metric in metrics
                    if str(metric.get("metricKey") or "")
                }
                metrics_by_scope[(topic, table)] = metrics_by_key
                governance_by_scope[(topic, table)] = table_governance
                for metric in metrics:
                    candidate = resolve_metric_candidate(metric, topic, table, query)
                    if candidate is None:
                        continue
                    candidate["governance"] = {
                        **table_governance,
                        **recall_governance_metadata(metric),
                        "assetStatus": str(table_governance.get("status") or ""),
                    }
                    semantic_ref_id = str(candidate["semanticRefId"])
                    current = by_id.get(semantic_ref_id)
                    if current is None or float(candidate.get("metricResolutionConfidence") or 0.0) > float(
                        current.get("metricResolutionConfidence") or 0.0
                    ):
                        by_id[semantic_ref_id] = candidate
                for term in self.topic_assets.load_table_terms(topic, table):
                    candidate = resolve_term_metric_candidate(term, metrics_by_key, topic, table, query)
                    if candidate is None:
                        continue
                    resolved_metric = candidate.get("metric") if isinstance(candidate.get("metric"), dict) else {}
                    candidate["governance"] = {
                        **table_governance,
                        **recall_governance_metadata(resolved_metric),
                        "assetStatus": str(table_governance.get("status") or ""),
                    }
                    semantic_ref_id = str(candidate["semanticRefId"])
                    current = by_id.get(semantic_ref_id)
                    if current is None or float(candidate.get("metricResolutionConfidence") or 0.0) > float(
                        current.get("metricResolutionConfidence") or 0.0
                    ):
                        by_id[semantic_ref_id] = candidate
        candidates = suppress_embedded_generic_metric_candidates(query, list(by_id.values()))
        by_id = {str(candidate.get("semanticRefId") or ""): candidate for candidate in candidates}
        for (topic, table), metrics_by_key in metrics_by_scope.items():
            scoped_matches = [
                candidate
                for candidate in candidates
                if str(candidate.get("topic") or "") == topic and str(candidate.get("tableName") or "") == table
            ]
            for linked_candidate in linked_metric_variant_candidates(
                scoped_matches,
                metrics_by_key,
                topic,
                table,
                governance_by_scope.get((topic, table), {}),
            ):
                semantic_ref_id = str(linked_candidate.get("semanticRefId") or "")
                current = by_id.get(semantic_ref_id)
                if current is None or compare_metric_candidate(linked_candidate, current) > 0:
                    by_id[semantic_ref_id] = linked_candidate
        candidates = list(by_id.values())
        label_groups: dict[str, list[dict[str, Any]]] = {}
        for candidate in candidates:
            label_key = normalize_recall_label(str(candidate.get("matchedMetricLabel") or ""))
            if label_key:
                label_groups.setdefault(label_key, []).append(candidate)
        suppressed_alias_candidates: set[str] = set()
        for label_key, group in label_groups.items():
            unique_metrics = {
                (str(item.get("topic") or ""), str(item.get("tableName") or ""), str(item.get("metricKey") or ""))
                for item in group
            }
            if len(unique_metrics) <= 1:
                continue
            canonical_owner, canonical_aliases = canonical_metric_family_owner(group)
            if canonical_owner is not None:
                canonical_key = str(canonical_owner.get("metricKey") or "")
                canonical_owner["metricResolutionAmbiguous"] = False
                canonical_owner["metricResolutionReason"] = "%s; canonical_family_owner=%s" % (
                    str(canonical_owner.get("metricResolutionReason") or ""),
                    canonical_key,
                )
                canonical_owner["metricResolutionConfidence"] = max(
                    0.9,
                    round(float(canonical_owner.get("metricResolutionConfidence") or 0.0), 3),
                )
                suppressed_alias_candidates.update(
                    str(item.get("semanticRefId") or "")
                    for item in canonical_aliases
                    if str(item.get("semanticRefId") or "")
                )
                continue
            for item in group:
                item["metricResolutionAmbiguous"] = True
                item["metricResolutionConfidence"] = max(
                    0.4, round(float(item.get("metricResolutionConfidence") or 0.0) - 0.18, 3)
                )
                item["metricResolutionReason"] = "%s; ambiguous_label=%s" % (
                    str(item.get("metricResolutionReason") or ""),
                    label_key,
                )
        if suppressed_alias_candidates:
            candidates = [
                candidate
                for candidate in candidates
                if str(candidate.get("semanticRefId") or "") not in suppressed_alias_candidates
            ]
        candidates.sort(
            key=lambda item: (
                float(item.get("metricResolutionConfidence") or 0.0),
                int(item.get("matchLength") or 0),
                float(item.get("fusionScore") or 0.0),
            ),
            reverse=True,
        )
        return candidates[:6]

    def _metric_candidate_items(self, query_text: str, candidates: list[dict[str, Any]]) -> list[RecallItem]:
        query = (query_text or "").strip()
        items: list[RecallItem] = []
        for rank, candidate in enumerate(candidates or [], start=1):
            semantic_ref_id = str(candidate.get("semanticRefId") or "")
            if not semantic_ref_id:
                continue
            metric = candidate.get("metric") or {}
            topic = str(candidate.get("topic") or "")
            table = str(candidate.get("tableName") or "")
            metric_key = str(candidate.get("metricKey") or "")
            confidence = float(candidate.get("metricResolutionConfidence") or 0.0)
            resolution_type = str(candidate.get("metricResolutionType") or "")
            score = metric_candidate_fusion_score(confidence, resolution_type, rank)
            metadata = {
                "semanticSource": "metrics",
                "semanticKind": "METRIC",
                "semanticRefId": semantic_ref_id,
                "semanticPath": semantic_metric_path(topic, table, metric_key),
                "metricKey": metric_key,
                "tableName": table,
                "topic": topic,
                "businessName": candidate.get("businessName") or metric_key,
                "canonicalMetricKey": candidate.get("canonicalMetricKey") or "",
                "aliasOf": candidate.get("aliasOf") or "",
                "metricLevel": candidate.get("metricLevel") or "",
                "metricGrain": candidate.get("metricGrain") or "",
                "metricIntent": candidate.get("metricIntent") or "",
                "aggregationPolicy": candidate.get("aggregationPolicy") or "",
                "applicableTimeGrain": candidate.get("applicableTimeGrain") or "",
                "timeColumn": candidate.get("timeColumn") or "",
                "timeSemantics": candidate.get("timeSemantics") or {},
                "missingValuePolicy": candidate.get("missingValuePolicy") or "",
                "zeroValueMeaning": candidate.get("zeroValueMeaning") or "",
                "selectionGuidance": candidate.get("selectionGuidance") or "",
                "preferredUseCases": candidate.get("preferredUseCases") or [],
                "notPreferredUseCases": candidate.get("notPreferredUseCases") or [],
                "temporalVariants": candidate.get("temporalVariants") or {},
                "linkedVariantOf": candidate.get("linkedVariantOf") or "",
                "linkedVariantPath": candidate.get("linkedVariantPath") or "",
                "formula": candidate.get("formula") or "",
                "sourceColumns": candidate.get("sourceColumns") or [],
                "aliases": candidate.get("aliases") or [],
                "recallQuery": query,
                "recallQueries": [query] if query else [],
                "recallChannel": "metric_resolver",
                "matchedMetricLabel": candidate.get("matchedMetricLabel") or "",
                "metricResolutionType": resolution_type,
                "metricResolutionReason": candidate.get("metricResolutionReason") or "",
                "metricResolutionConfidence": confidence,
                "metricResolutionAmbiguous": bool(candidate.get("metricResolutionAmbiguous") or False),
                "metricCandidateRank": rank,
                "metricResolverScore": score,
                "recallSupplement": "metric_candidate_resolution",
                **dict(candidate.get("governance") or {}),
            }
            items.append(
                RecallItem(
                    doc_id=semantic_ref_id,
                    title="%s/%s/%s metric" % (topic, table, metric_key),
                    content=compact_metric_for_recall(topic, table, metric if isinstance(metric, dict) else {}),
                    source_type="SEMANTIC_METRIC",
                    topic=topic,
                    table=table,
                    fusion_score=score,
                    metadata=metadata,
                )
            )
        return items

    def _exact_metric_evidence(self, query_text: str, topics: list[str]) -> list[RecallItem]:
        """Compatibility supplement for very high-confidence exact metric matches.

        The primary path is now metric candidate resolution before ranking. This
        adapter keeps an explicit exact-match lane so existing callers and
        diagnostics still have a stable high-confidence fallback.
        """
        resolved = self._resolve_metric_candidates(query_text, topics)
        exact_candidates = [
            candidate
            for candidate in resolved
            if str(candidate.get("metricResolutionType") or "").startswith("exact")
            and float(candidate.get("metricResolutionConfidence") or 0.0) >= 0.9
        ]
        items = self._metric_candidate_items(query_text, exact_candidates)
        for item in items:
            metadata = dict(item.metadata or {})
            metadata["recallChannel"] = "exact"
            metadata["matchedExactMetricLabel"] = metadata.get("matchedMetricLabel") or ""
            metadata["recallSupplement"] = "exact_metric_evidence"
            item.metadata = metadata
        return items

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.settings.es_api_key:
            headers["Authorization"] = "Bearer %s" % self.settings.es_api_key
        return headers

    def _auth(self) -> tuple[str, str] | None:
        if self.settings.es_api_key:
            return None
        if self.settings.es_username:
            return (self.settings.es_username, self.settings.es_password)
        return None

    def _index_version(self) -> str:
        manifest_path = self.settings.resolved_workspace_path / "recall_index_manifest.json"
        if not manifest_path.exists():
            return ""
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        return str(payload.get("indexVersion") or "")


def resolve_metric_candidate(metric: dict[str, Any], topic: str, table: str, query_text: str) -> dict[str, Any] | None:
    metric_key = str(metric.get("metricKey") or "").strip()
    if not metric_key:
        return None
    query = normalize_recall_label(query_text)
    if not query:
        return None
    labels = [
        ("exact_business_name", str(metric.get("businessName") or ""), 0.99, "businessName"),
        ("exact_metric_key", metric_key, 0.95, "metricKey"),
    ]
    labels.extend(("exact_alias", str(alias), 0.97, "alias") for alias in metric.get("aliases") or [])
    best: dict[str, Any] | None = None
    for resolution_type, raw_label, confidence, source in labels:
        label = str(raw_label or "").strip()
        normalized = normalize_recall_label(label)
        if not is_protective_metric_label(normalized) or normalized not in query:
            continue
        candidate = build_metric_candidate(metric, topic, table, label, resolution_type, confidence, source)
        if best is None or compare_metric_candidate(candidate, best) > 0:
            best = candidate
    return best


def linked_metric_variant_candidates(
    base_candidates: list[dict[str, Any]],
    metrics_by_key: dict[str, dict[str, Any]],
    topic: str,
    table: str,
    table_governance: dict[str, Any],
) -> list[dict[str, Any]]:
    """Expand only links explicitly published in a metric's variant contract.

    Retrieval does not decide which linked metric fits the question.  It exposes
    compact candidates with the asset's aggregation and selection metadata so a
    downstream semantic selector can make that decision.
    """

    linked: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for base_candidate in base_candidates:
        base_metric = base_candidate.get("metric") if isinstance(base_candidate.get("metric"), dict) else {}
        base_key = str(base_candidate.get("metricKey") or "")
        base_ref = str(base_candidate.get("semanticRefId") or "")
        base_confidence = max(0.0, min(1.0, float(base_candidate.get("metricResolutionConfidence") or 0.0)))
        for link_path, variant_key in metric_linked_variant_refs(base_metric):
            if not variant_key or variant_key == base_key or (base_ref, variant_key) in seen:
                continue
            seen.add((base_ref, variant_key))
            variant_metric = metrics_by_key.get(variant_key)
            if not isinstance(variant_metric, dict):
                continue
            linked_confidence = min(0.94, max(0.4, base_confidence - 0.03))
            candidate = build_metric_candidate(
                variant_metric,
                topic,
                table,
                str(base_candidate.get("matchedMetricLabel") or base_candidate.get("businessName") or variant_key),
                "linked_variant",
                linked_confidence,
                "temporalVariants.%s" % link_path,
            )
            candidate["metricResolutionReason"] = "%s; linked_variant_of=%s; link_path=%s" % (
                str(candidate.get("metricResolutionReason") or ""),
                base_key,
                link_path,
            )
            candidate["linkedVariantOf"] = base_ref
            candidate["linkedVariantPath"] = link_path
            candidate["governance"] = {
                **table_governance,
                **recall_governance_metadata(variant_metric),
            }
            linked.append(candidate)
    return linked


def metric_linked_variant_refs(metric: dict[str, Any]) -> list[tuple[str, str]]:
    variants = metric.get("temporalVariants") or metric.get("temporal_variants") or {}
    if not isinstance(variants, (dict, list, tuple)):
        return []
    refs: list[tuple[str, str]] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, str):
            metric_ref = value.strip()
            if metric_ref:
                refs.append((path, metric_ref))
            return
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = "%s.%s" % (path, key) if path else str(key)
                visit(child, child_path)
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                child_path = "%s[%d]" % (path, index)
                visit(child, child_path)

    visit(variants, "")
    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for path, metric_ref in refs:
        if metric_ref in seen:
            continue
        seen.add(metric_ref)
        deduped.append((path, metric_ref))
    return deduped


def suppress_embedded_generic_metric_candidates(
    query_text: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prefer an explicit qualified label over its embedded generic label.

    If a qualified label contains a shorter independent label, an occurrence
    embedded only inside the qualified phrase does not count as a second request.
    If the query separately contains both labels, both candidates are retained.
    """

    query = normalize_recall_label(query_text)
    if not query or len(candidates) <= 1:
        return candidates
    suppressed: set[str] = set()
    labelled = [
        (candidate, normalize_recall_label(str(candidate.get("matchedMetricLabel") or ""))) for candidate in candidates
    ]
    for qualified, long_label in labelled:
        if not long_label or long_label not in query:
            continue
        remainder = query.replace(long_label, " ")
        for generic, short_label in labelled:
            if generic is qualified or not short_label or short_label == long_label:
                continue
            if short_label not in long_label or short_label in remainder:
                continue
            qualified_ref = str(qualified.get("semanticRefId") or "")
            generic_ref = str(generic.get("semanticRefId") or "")
            if qualified_ref and generic_ref and qualified_ref != generic_ref:
                suppressed.add(generic_ref)
    if not suppressed:
        return candidates
    return [candidate for candidate in candidates if str(candidate.get("semanticRefId") or "") not in suppressed]


def resolve_term_metric_candidate(
    term: dict[str, Any], metrics_by_key: dict[str, dict[str, Any]], topic: str, table: str, query_text: str
) -> dict[str, Any] | None:
    if not isinstance(term, dict) or not metrics_by_key:
        return None
    query = normalize_recall_label(query_text)
    if not query:
        return None
    metric = resolve_term_metric_definition(term, metrics_by_key)
    if not metric:
        return None
    labels = [str(term.get("term") or ""), *[str(alias) for alias in term.get("aliases") or []]]
    best: dict[str, Any] | None = None
    for raw_label in labels:
        label = str(raw_label or "").strip()
        normalized = normalize_recall_label(label)
        if not is_protective_metric_label(normalized) or normalized not in query:
            continue
        candidate = build_metric_candidate(metric, topic, table, label, "exact_term", 0.96, "term")
        if best is None or compare_metric_candidate(candidate, best) > 0:
            best = candidate
    return best


def resolve_term_metric_definition(
    term: dict[str, Any], metrics_by_key: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    canonical = str(term.get("canonicalMetricKey") or "").strip()
    if canonical and canonical in metrics_by_key:
        return metrics_by_key[canonical]
    business_name = str(term.get("businessName") or "").strip()
    for metric in metrics_by_key.values():
        if business_name and business_name == str(metric.get("businessName") or "").strip():
            return metric
    return None


def build_metric_candidate(
    metric: dict[str, Any],
    topic: str,
    table: str,
    matched_label: str,
    resolution_type: str,
    confidence: float,
    reason_source: str,
) -> dict[str, Any]:
    metric_key = str(metric.get("metricKey") or "").strip()
    semantic_ref_id = "semantic:%s:%s:metric:%s" % (topic, table, metric_key)
    score = metric_candidate_fusion_score(confidence, resolution_type, 1)
    return {
        "semanticRefId": semantic_ref_id,
        "topic": topic,
        "tableName": table,
        "metricKey": metric_key,
        "businessName": str(metric.get("businessName") or metric_key),
        "canonicalMetricKey": str(metric.get("canonicalMetricKey") or ""),
        "aliasOf": str(metric.get("aliasOf") or ""),
        "metricLevel": str(metric.get("metricLevel") or ""),
        "metricGrain": str(metric.get("metricGrain") or metric.get("grainHint") or ""),
        "metricIntent": str(metric.get("metricIntent") or ""),
        "aggregationPolicy": str(metric.get("aggregationPolicy") or ""),
        "applicableTimeGrain": str(metric.get("applicableTimeGrain") or ""),
        "timeColumn": str(metric.get("timeColumn") or ""),
        "timeSemantics": metric.get("timeSemantics") or {},
        "missingValuePolicy": str(metric.get("missingValuePolicy") or ""),
        "zeroValueMeaning": str(metric.get("zeroValueMeaning") or ""),
        "selectionGuidance": str(metric.get("selectionGuidance") or ""),
        "preferredUseCases": metric.get("preferredUseCases") or [],
        "notPreferredUseCases": metric.get("notPreferredUseCases") or [],
        "temporalVariants": metric.get("temporalVariants") or {},
        "formula": str(metric.get("formula") or metric.get("metricFormula") or ""),
        "sourceColumns": metric.get("sourceColumns") or [],
        "aliases": metric.get("aliases") or [],
        "metric": metric,
        "matchedMetricLabel": matched_label,
        "matchLength": len(normalize_recall_label(matched_label)),
        "metricResolutionType": resolution_type,
        "metricResolutionReason": "matched_%s:%s" % (reason_source, matched_label),
        "metricResolutionConfidence": round(float(confidence or 0.0), 3),
        "metricResolutionAmbiguous": False,
        "fusionScore": score,
    }


def canonical_metric_family_owner(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Return the governed owner when every candidate belongs to one alias family.

    A shared user label is not a real ambiguity when the semantic layer explicitly
    declares every variant as an alias of one canonical metric and publishes that
    canonical metric in the same owner table.  Keeping this rule metadata-driven
    avoids teaching retrieval any business-specific metric names.
    """
    if len(candidates) <= 1:
        return None, []
    families: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        metric_key = str(candidate.get("metricKey") or "").strip()
        canonical_key = str(candidate.get("canonicalMetricKey") or candidate.get("aliasOf") or metric_key).strip()
        topic = str(candidate.get("topic") or "").strip()
        table = str(candidate.get("tableName") or "").strip()
        if not metric_key or not canonical_key or not table:
            return None, []
        families.add((topic, table, canonical_key))
    if len(families) != 1:
        return None, []
    _, _, canonical_key = next(iter(families))
    owners = [
        candidate
        for candidate in candidates
        if str(candidate.get("metricKey") or "").strip() == canonical_key
        and not str(candidate.get("aliasOf") or "").strip()
    ]
    if len(owners) != 1:
        return None, []
    owner = owners[0]
    aliases = [candidate for candidate in candidates if candidate is not owner]
    return owner, aliases


def compare_metric_candidate(left: dict[str, Any], right: dict[str, Any]) -> int:
    left_score = (
        float(left.get("metricResolutionConfidence") or 0.0),
        int(left.get("matchLength") or 0),
        float(left.get("fusionScore") or 0.0),
    )
    right_score = (
        float(right.get("metricResolutionConfidence") or 0.0),
        int(right.get("matchLength") or 0),
        float(right.get("fusionScore") or 0.0),
    )
    if left_score > right_score:
        return 1
    if left_score < right_score:
        return -1
    return 0


def metric_candidate_fusion_score(confidence: float, resolution_type: str, rank: int) -> float:
    type_score = {
        "exact_business_name": 1.0,
        "exact_alias": 0.98,
        "exact_term": 0.96,
        "exact_metric_key": 0.94,
    }.get(str(resolution_type or ""), 0.72)
    bounded_rank = max(1, int(rank or 1))
    confidence_score = max(0.0, min(float(confidence or 0.0), 1.0))
    rank_penalty = min(0.15, (bounded_rank - 1) * 0.02)
    return round(max(0.0, min(1.0, type_score * 0.55 + confidence_score * 0.45 - rank_penalty)), 6)


def metric_trace_payload(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in candidates or []:
        payload.append(
            {
                "semanticRefId": str(item.get("semanticRefId") or ""),
                "topic": str(item.get("topic") or ""),
                "tableName": str(item.get("tableName") or ""),
                "metricKey": str(item.get("metricKey") or ""),
                "businessName": str(item.get("businessName") or ""),
                "matchedMetricLabel": str(item.get("matchedMetricLabel") or ""),
                "metricResolutionType": str(item.get("metricResolutionType") or ""),
                "metricResolutionReason": str(item.get("metricResolutionReason") or ""),
                "metricResolutionConfidence": float(item.get("metricResolutionConfidence") or 0.0),
                "metricResolutionAmbiguous": bool(item.get("metricResolutionAmbiguous") or False),
                "aggregationPolicy": str(item.get("aggregationPolicy") or ""),
                "applicableTimeGrain": str(item.get("applicableTimeGrain") or ""),
                "timeColumn": str(item.get("timeColumn") or ""),
                "timeSemantics": item.get("timeSemantics") or {},
                "missingValuePolicy": str(item.get("missingValuePolicy") or ""),
                "zeroValueMeaning": str(item.get("zeroValueMeaning") or ""),
                "selectionGuidance": str(item.get("selectionGuidance") or ""),
                "temporalVariants": item.get("temporalVariants") or {},
                "linkedVariantOf": str(item.get("linkedVariantOf") or ""),
                "linkedVariantPath": str(item.get("linkedVariantPath") or ""),
            }
        )
    return payload


RETRIEVAL_LEAF_SOURCE_TYPES = {
    "SEMANTIC_METRIC",
    "SEMANTIC_COLUMN",
    "SEMANTIC_TERM",
    "SEMANTIC_BUSINESS_RULE",
    "SEMANTIC_RELATIONSHIP",
    "GOVERNED_RULE",
}


def build_retrieval_query_plan(
    *,
    query_text: str,
    request: KnowledgeRetrievalRequest,
    retrieval_profile: dict[str, Any],
    metric_candidates: list[dict[str, Any]],
    include_rules: bool,
    max_queries: int = 5,
    enabled: bool = True,
) -> list[dict[str, Any]]:
    """Build a bounded, deterministic search plan without inventing semantics."""

    base_query = collapse_whitespace(query_text)
    if not base_query:
        return []
    plan: list[dict[str, Any]] = [
        {
            "id": "base",
            "query": base_query,
            "purpose": "original_question",
            "priority": 100,
            "targetSourceTypes": [],
        }
    ]
    if request.target_goal_ids or request.required_capabilities:
        return plan
    if not enabled or max_queries <= 1:
        return plan

    query_type = str(retrieval_profile.get("queryType") or "")
    if query_type == "balanced":
        return plan
    complex_query = query_type in {
        "multi_metric",
        "multi_hop_analysis",
        "mixed_rule_data",
        "rule_qa",
    }
    if not complex_query and not include_rules:
        return plan

    candidates: list[dict[str, Any]] = []
    metric_labels: list[str] = []
    for candidate in metric_candidates[:3]:
        label = str(
            candidate.get("matchedMetricLabel") or candidate.get("businessName") or candidate.get("metricKey") or ""
        ).strip()
        if label and label not in metric_labels:
            metric_labels.append(label)
    if metric_labels or query_type in {"multi_metric", "multi_hop_analysis", "mixed_rule_data"}:
        focus = "、".join(metric_labels[:3]) or "问题涉及的指标"
        candidates.append(
            {
                "id": "metrics",
                "query": "%s；检索重点：%s的指标定义、计算公式、来源字段和时间口径" % (base_query, focus),
                "purpose": "metric_definition",
                "priority": 90,
                "targetSourceTypes": ["SEMANTIC_METRIC"],
            }
        )

    if query_type in {"multi_hop_analysis", "mixed_rule_data"}:
        candidates.append(
            {
                "id": "relationships",
                "query": "%s；检索重点：相关数据表之间的关联关系、关联字段、数据粒度和重复计算风险" % base_query,
                "purpose": "relationship_path",
                "priority": 85,
                "targetSourceTypes": ["SEMANTIC_RELATIONSHIP"],
            }
        )

    route_slots = request.route_slots if isinstance(request.route_slots, dict) else {}
    dimensions = retrieval_slot_labels(route_slots.get("dimensions") or [])
    if complex_query or dimensions:
        focus = "、".join(dimensions[:4]) or "问题涉及的维度、时间和筛选字段"
        candidates.append(
            {
                "id": "fields",
                "query": "%s；检索重点：%s对应的字段定义、业务术语和使用方式" % (base_query, focus),
                "purpose": "field_and_term_binding",
                "priority": 80,
                "targetSourceTypes": ["SEMANTIC_COLUMN", "SEMANTIC_TERM"],
            }
        )

    if include_rules or query_type in {"rule_qa", "mixed_rule_data"}:
        candidates.append(
            {
                "id": "rules",
                "query": "%s；检索重点：业务规则、指标限制、必选条件和风险提示" % base_query,
                "purpose": "business_rule",
                "priority": 95 if query_type == "rule_qa" else 75,
                "targetSourceTypes": ["SEMANTIC_BUSINESS_RULE", "GOVERNED_RULE"],
            }
        )

    seen_queries = {base_query}
    for candidate in sorted(candidates, key=lambda item: int(item.get("priority") or 0), reverse=True):
        query = collapse_whitespace(candidate.get("query"))
        if not query or query in seen_queries:
            continue
        seen_queries.add(query)
        plan.append({**candidate, "query": query})
        if len(plan) >= max(1, int(max_queries or 5)):
            break
    return plan


def retrieval_slot_labels(values: object) -> list[str]:
    candidates = values if isinstance(values, list) else [values]
    labels: list[str] = []
    for value in candidates:
        if isinstance(value, dict):
            label = str(
                value.get("businessName")
                or value.get("label")
                or value.get("name")
                or value.get("column")
                or value.get("key")
                or ""
            ).strip()
        else:
            label = str(value or "").strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def record_retrieval_query_result(
    query_plan: list[dict[str, Any]],
    *,
    query_id: str,
    status: str,
    items: list[RecallItem],
    error_code: str = "",
    stop_reason: str = "",
) -> None:
    for query in query_plan:
        if str(query.get("id") or "") != query_id:
            continue
        query["status"] = status
        query["itemCount"] = len(items or [])
        query["candidateRefs"] = unique_source_refs(items)[:12]
        query["candidateSourceTypes"] = sorted({str(item.source_type or "UNKNOWN").upper() for item in items or []})
        if error_code:
            query["errorCode"] = error_code
        if stop_reason:
            query["stopReason"] = stop_reason
        return


def uncovered_retrieval_queries(
    query_plan: list[dict[str, Any]],
    items: list[RecallItem],
) -> list[dict[str, Any]]:
    available = {str(item.source_type or "").upper() for item in items or []}
    uncovered: list[dict[str, Any]] = []
    for query in query_plan:
        targets = {str(value or "").upper() for value in query.get("targetSourceTypes") or [] if value}
        if targets and not targets.intersection(available):
            uncovered.append(query)
    return uncovered


def select_retrieval_directories(
    items: list[RecallItem],
    *,
    allowed_topics: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    allowed = {str(item) for item in allowed_topics if str(item or "").strip()}
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items or []:
        metadata = dict(item.metadata or {})
        topic = str(item.topic or metadata.get("topic") or "").strip()
        table = str(item.table or metadata.get("tableName") or "").strip()
        if not topic or not table or (allowed and topic not in allowed):
            continue
        key = (topic, table)
        entry = grouped.setdefault(
            key,
            {
                "topic": topic,
                "table": table,
                "score": 0.0,
                "hitCount": 0,
                "sourceTypes": [],
                "sourceRefs": [],
            },
        )
        entry["score"] = max(float(entry.get("score") or 0.0), float(item.fusion_score or 0.0))
        entry["hitCount"] = int(entry.get("hitCount") or 0) + 1
        source_type = str(item.source_type or "UNKNOWN").upper()
        if source_type not in entry["sourceTypes"]:
            entry["sourceTypes"].append(source_type)
        ref = item.doc_id or str(metadata.get("semanticRefId") or "")
        if ref and ref not in entry["sourceRefs"]:
            entry["sourceRefs"].append(ref)
    ranked = sorted(
        grouped.values(),
        key=lambda item: (
            float(item.get("score") or 0.0),
            int(item.get("hitCount") or 0),
            str(item.get("topic") or ""),
            str(item.get("table") or ""),
        ),
        reverse=True,
    )
    return [
        {
            **item,
            "score": round(float(item.get("score") or 0.0), 6),
            "sourceRefs": list(item.get("sourceRefs") or [])[:8],
        }
        for item in ranked[: max(1, int(limit or 1))]
    ]


def build_directory_retrieval_query(
    *,
    query_text: str,
    directories: list[dict[str, Any]],
    uncovered_queries: list[dict[str, Any]],
) -> str:
    purposes = [str(item.get("purpose") or "") for item in uncovered_queries if str(item.get("purpose") or "")]
    focus = "、".join(purposes[:4]) or "精确语义定义"
    paths = [
        "%s/%s" % (str(item.get("topic") or ""), str(item.get("table") or ""))
        for item in directories
        if str(item.get("topic") or "") and str(item.get("table") or "")
    ]
    return "%s；限定目录：%s；继续查找：%s" % (
        query_text,
        "、".join(paths[:4]),
        focus,
    )


def recall_item_in_directory(item: RecallItem, *, topic: str, table: str) -> bool:
    metadata = dict(item.metadata or {})
    item_topic = str(item.topic or metadata.get("topic") or "")
    item_table = str(item.table or metadata.get("tableName") or "")
    if item_topic != topic:
        return False
    if item_table == table:
        return True
    if str(item.source_type or "").upper() == "SEMANTIC_RELATIONSHIP":
        return table in {
            str(metadata.get("leftTable") or ""),
            str(metadata.get("rightTable") or ""),
        }
    return False


def recall_item_in_directories(
    item: RecallItem,
    directories: list[dict[str, Any]],
) -> bool:
    return bool(recall_item_matched_directory(item, directories))


def recall_item_matched_directory(
    item: RecallItem,
    directories: list[dict[str, Any]],
) -> dict[str, Any]:
    for directory in directories:
        topic = str(directory.get("topic") or "")
        table = str(directory.get("table") or "")
        if topic and table and recall_item_in_directory(item, topic=topic, table=table):
            return directory
    return {}


def recall_item_is_exact_leaf(item: RecallItem) -> bool:
    source_type = str(item.source_type or "").upper()
    metadata = dict(item.metadata or {})
    if source_type not in RETRIEVAL_LEAF_SOURCE_TYPES:
        return False
    if source_type == "SEMANTIC_RELATIONSHIP":
        kind = str(metadata.get("semanticKind") or "").upper()
        return kind == "RELATIONSHIP" or str(metadata.get("contextLayer") or "").upper() == "L2"
    return source_type != "GOVERNED_RULE" or bool(metadata.get("semanticPath"))


def recall_item_matches_uncovered_queries(
    item: RecallItem,
    uncovered_queries: list[dict[str, Any]],
) -> bool:
    source_type = str(item.source_type or "").upper()
    return any(
        source_type in {str(value or "").upper() for value in query.get("targetSourceTypes") or []}
        for query in uncovered_queries
    )


def build_retrieval_profile(
    query_text: str,
    topics: list[str],
    include_rules: bool,
    metric_candidates: list[dict[str, Any]],
    settings: Settings,
    intent_kind: str = "",
    complexity: str = "",
) -> dict[str, Any]:
    query = str(query_text or "").strip()
    lowered = query.lower()
    reasons: list[str] = []
    if not str(intent_kind or "").strip() and not str(
        complexity or ""
    ).strip() and not metric_candidates:
        query_type = "balanced"
        reasons.append("goal_agnostic_initial_recall")
    else:
        query_type = query_type_from_fast_understanding(
            intent_kind=intent_kind,
            complexity=complexity,
            include_rules=include_rules,
        )
        if query_type:
            reasons.append(
                "fast_understanding:%s/%s"
                % (intent_kind or "unknown", complexity or "unknown")
            )
        else:
            query_type = classify_query_type(
                query=query,
                topics=topics,
                metric_candidates=metric_candidates,
                include_rules=include_rules,
                reasons=reasons,
            )
    profile_templates = configured_retrieval_profiles(settings)
    selected = dict(profile_templates.get(query_type) or profile_templates.get("multi_hop_analysis") or {})
    profile_kind = str(selected.get("profileKind") or "balanced")
    text_top_k = int(selected.get("textTopK") or settings.es_text_top_k or 12)
    vector_top_k = int(selected.get("vectorTopK") or settings.es_vector_top_k or 12)
    broad_text_top_k = int(selected.get("broadTextTopK") or settings.es_broad_text_top_k or 4)
    broad_vector_top_k = int(selected.get("broadVectorTopK") or settings.es_broad_vector_top_k or 4)
    hybrid_top_k = int(selected.get("hybridTopK") or settings.es_hybrid_top_k or 24)
    configured_complexity = selected.get("complexityScore")
    complexity_score = (
        int(configured_complexity)
        if configured_complexity is not None
        else estimate_query_complexity(
            query,
            topics,
            metric_candidates,
            include_rules,
        )
    )
    if any(str(item.get("metricResolutionType") or "").startswith("exact") for item in metric_candidates):
        reasons.append("explicit_metric_candidate")
    return {
        "profileKind": profile_kind,
        "queryType": query_type,
        "intentKind": str(intent_kind or ""),
        "fastComplexity": str(complexity or ""),
        "complexity": complexity_score,
        "reasons": reasons,
        "textTopK": text_top_k,
        "vectorTopK": vector_top_k,
        "broadTextTopK": broad_text_top_k,
        "broadVectorTopK": broad_vector_top_k,
        "hybridTopK": hybrid_top_k,
        "broadSearchEnabled": bool(selected.get("broadSearchEnabled", True)),
        "sourceTypeCaps": selected.get("sourceTypeCaps") or {},
        "queryHash": hashlib.sha256(lowered.encode("utf-8")).hexdigest()[:12] if lowered else "",
    }


def query_type_from_fast_understanding(intent_kind: str, complexity: str, include_rules: bool) -> str:
    kind = str(intent_kind or "").strip().lower()
    level = str(complexity or "").strip().lower()
    if kind == "rule_only" or (include_rules and kind not in {"rule_data_mix", "mixed_rule_data"}):
        return "rule_qa"
    if kind in {"rule_data_mix", "mixed_rule_data"}:
        return "mixed_rule_data"
    if kind == "detail_lookup":
        return "detail_lookup"
    if kind == "multi_metric":
        return "multi_metric"
    if kind in {"multi_hop", "analysis"} or level == "complex":
        return "multi_hop_analysis"
    if kind == "metric_query" or level == "simple":
        return "simple_metric"
    return ""


def configured_retrieval_profiles(settings: Settings) -> dict[str, dict[str, Any]]:
    profiles = default_retrieval_profiles(settings)
    raw = str(getattr(settings, "es_retrieval_profiles_json", "") or "").strip()
    if not raw:
        return profiles
    try:
        payload = json.loads(raw)
    except Exception:
        return profiles
    if not isinstance(payload, dict):
        return profiles
    for query_type, override in payload.items():
        if not isinstance(override, dict):
            continue
        base = dict(profiles.get(str(query_type)) or {})
        source_type_caps = dict(base.get("sourceTypeCaps") or {})
        if isinstance(override.get("sourceTypeCaps"), dict):
            source_type_caps.update(
                {
                    str(key): int(value)
                    for key, value in override.get("sourceTypeCaps", {}).items()
                    if isinstance(value, (int, float))
                }
            )
        merged = {**base, **override}
        if source_type_caps:
            merged["sourceTypeCaps"] = source_type_caps
        profiles[str(query_type)] = merged
    return profiles


def default_retrieval_profiles(settings: Settings) -> dict[str, dict[str, Any]]:
    return {
        "balanced": {
            "profileKind": "balanced",
            "textTopK": int(settings.es_text_top_k or 12),
            "vectorTopK": int(settings.es_vector_top_k or 12),
            "broadTextTopK": int(settings.es_broad_text_top_k or 4),
            "broadVectorTopK": int(settings.es_broad_vector_top_k or 4),
            "hybridTopK": int(settings.es_hybrid_top_k or 24),
            "broadSearchEnabled": True,
            "complexityScore": 0,
            "sourceTypeCaps": {
                "SEMANTIC_METRIC": 12,
                "SEMANTIC_RELATIONSHIP": 8,
                "SEMANTIC_TABLE_ASSET": 6,
                "SEMANTIC_COLUMN": 10,
                "SEMANTIC_TERM": 6,
                "GOVERNED_RULE": 6,
            },
        },
        "simple_metric": {
            "profileKind": "focused",
            "textTopK": max(6, int(settings.es_text_top_k or 12) - 4),
            "vectorTopK": max(6, int(settings.es_vector_top_k or 12) - 4),
            "broadTextTopK": max(2, int(settings.es_broad_text_top_k or 4) - 1),
            "broadVectorTopK": max(2, int(settings.es_broad_vector_top_k or 4) - 1),
            "hybridTopK": max(12, min(int(settings.es_hybrid_top_k or 24), 16)),
            "broadSearchEnabled": True,
            "complexityScore": 1,
            "sourceTypeCaps": {
                "SEMANTIC_METRIC": 10,
                "SEMANTIC_RELATIONSHIP": 5,
                "SEMANTIC_TABLE_ASSET": 4,
                "GOVERNED_RULE": 2,
            },
        },
        "multi_metric": {
            "profileKind": "balanced",
            "textTopK": int(settings.es_text_top_k or 12),
            "vectorTopK": int(settings.es_vector_top_k or 12),
            "broadTextTopK": int(settings.es_broad_text_top_k or 4),
            "broadVectorTopK": int(settings.es_broad_vector_top_k or 4),
            "hybridTopK": int(settings.es_hybrid_top_k or 24),
            "broadSearchEnabled": True,
            "complexityScore": 2,
            "sourceTypeCaps": {
                "SEMANTIC_METRIC": 12,
                "SEMANTIC_RELATIONSHIP": 7,
                "SEMANTIC_TABLE_ASSET": 6,
                "GOVERNED_RULE": 3,
            },
        },
        "multi_hop_analysis": {
            "profileKind": "broad",
            "textTopK": min(max(int(settings.es_text_top_k or 12), 12) + 4, 18),
            "vectorTopK": min(max(int(settings.es_vector_top_k or 12), 12) + 4, 18),
            "broadTextTopK": min(max(int(settings.es_broad_text_top_k or 4), 4) + 2, 8),
            "broadVectorTopK": min(max(int(settings.es_broad_vector_top_k or 4), 4) + 2, 8),
            "hybridTopK": min(max(int(settings.es_hybrid_top_k or 24), 24) + 4, 32),
            "broadSearchEnabled": True,
            "complexityScore": 5,
            "sourceTypeCaps": {
                "SEMANTIC_METRIC": 14,
                "SEMANTIC_RELATIONSHIP": 10,
                "SEMANTIC_TABLE_ASSET": 8,
                "GOVERNED_RULE": 4,
            },
        },
        "rule_qa": {
            "profileKind": "balanced",
            "textTopK": max(8, int(settings.es_text_top_k or 12) - 2),
            "vectorTopK": max(6, int(settings.es_vector_top_k or 12) - 4),
            "broadTextTopK": max(2, int(settings.es_broad_text_top_k or 4) - 1),
            "broadVectorTopK": max(2, int(settings.es_broad_vector_top_k or 4) - 2),
            "hybridTopK": max(12, min(int(settings.es_hybrid_top_k or 24), 18)),
            "broadSearchEnabled": True,
            "complexityScore": 3,
            "sourceTypeCaps": {
                "SEMANTIC_METRIC": 8,
                "SEMANTIC_RELATIONSHIP": 4,
                "SEMANTIC_TABLE_ASSET": 4,
                "GOVERNED_RULE": 6,
            },
        },
        "mixed_rule_data": {
            "profileKind": "broad",
            "textTopK": min(max(int(settings.es_text_top_k or 12), 12) + 2, 16),
            "vectorTopK": min(max(int(settings.es_vector_top_k or 12), 12) + 2, 16),
            "broadTextTopK": min(max(int(settings.es_broad_text_top_k or 4), 4) + 1, 6),
            "broadVectorTopK": min(max(int(settings.es_broad_vector_top_k or 4), 4) + 1, 6),
            "hybridTopK": min(max(int(settings.es_hybrid_top_k or 24), 24) + 2, 28),
            "broadSearchEnabled": True,
            "complexityScore": 4,
            "sourceTypeCaps": {
                "SEMANTIC_METRIC": 12,
                "SEMANTIC_RELATIONSHIP": 9,
                "SEMANTIC_TABLE_ASSET": 7,
                "GOVERNED_RULE": 6,
            },
        },
        "detail_lookup": {
            "profileKind": "focused",
            "textTopK": max(6, int(settings.es_text_top_k or 12) - 3),
            "vectorTopK": max(4, int(settings.es_vector_top_k or 12) - 6),
            "broadTextTopK": max(2, int(settings.es_broad_text_top_k or 4) - 1),
            "broadVectorTopK": max(1, int(settings.es_broad_vector_top_k or 4) - 2),
            "hybridTopK": max(10, min(int(settings.es_hybrid_top_k or 24), 14)),
            "broadSearchEnabled": True,
            "complexityScore": 2,
            "sourceTypeCaps": {
                "SEMANTIC_METRIC": 8,
                "SEMANTIC_RELATIONSHIP": 6,
                "SEMANTIC_TABLE_ASSET": 5,
                "GOVERNED_RULE": 2,
            },
        },
    }


def classify_query_type(
    query: str,
    topics: list[str],
    metric_candidates: list[dict[str, Any]],
    include_rules: bool,
    reasons: list[str] | None = None,
) -> str:
    out = reasons if reasons is not None else []
    metric_count = len(metric_candidates)
    if include_rules and metric_count:
        out.append("mixed_rule_data")
        return "mixed_rule_data"
    if include_rules:
        out.append("rule_qa")
        return "rule_qa"
    if len(topics) >= 2:
        out.append("multi_hop_analysis")
        return "multi_hop_analysis"
    if metric_count >= 2:
        out.append("multi_metric")
        return "multi_metric"
    fallback = "simple_metric" if len(topics) <= 1 and metric_count <= 1 else "multi_hop_analysis"
    out.append(fallback)
    return fallback


def estimate_query_complexity(
    query: str,
    topics: list[str],
    metric_candidates: list[dict[str, Any]],
    include_rules: bool,
) -> int:
    score = 0
    if len(query or "") >= 24:
        score += 1
    if len(topics) >= 2:
        score += 1
    if len(metric_candidates) >= 2:
        score += 1
    if include_rules:
        score += 1
    return score


def merge_recall_bundles(primary: RecallBundle, secondary: RecallBundle) -> RecallBundle:
    items = merge_recall_items(primary.items, secondary.items)
    return RecallBundle(
        items=items,
        top_score=items[0].fusion_score if items else 0.0,
        merged_context="\n\n".join(
            "召回片段 [%s] %s\n%s" % (item.source_type, item.title, item.content[:1200]) for item in items
        ),
    )


def retrieval_status_for(items: list[RecallItem], issues: list[RetrievalIssue]) -> str:
    unresolved = [issue for issue in issues or [] if not issue.resolved]
    if unresolved and not items:
        return "failed"
    if issues:
        return "degraded"
    return "success" if items else "empty"


def normalize_knowledge_bundle_status(bundle: KnowledgeBundle) -> KnowledgeBundle:
    """Backfill status for providers/checkpoints created before the contract."""

    status = str(bundle.retrieval_status or "").strip().lower()
    if status not in {"success", "empty", "degraded", "failed"}:
        status = retrieval_status_for(bundle.recall_bundle.items, bundle.retrieval_issues)
    rounds = [
        trace.model_copy(
            update={
                "retrieval_status": (
                    trace.retrieval_status
                    if str(trace.retrieval_status or "").strip().lower() in {"success", "empty", "degraded", "failed"}
                    else status
                ),
                "retrieval_issues": list(trace.retrieval_issues or bundle.retrieval_issues),
            }
        )
        for trace in bundle.recall_rounds
    ]
    return bundle.model_copy(update={"retrieval_status": status, "recall_rounds": rounds})


def dedupe_retrieval_issues(issues: list[RetrievalIssue]) -> list[RetrievalIssue]:
    deduped: list[RetrievalIssue] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for issue in issues or []:
        identity = (
            str(issue.code or ""),
            str(issue.backend or ""),
            str(issue.lane or ""),
            str(issue.stage or ""),
            str(issue.request_key or ""),
        )
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(issue)
    return deduped


def failed_knowledge_bundle(
    request: KnowledgeRetrievalRequest,
    backend: str,
    code: str,
    message: str,
    *,
    stage: str = "retrieve",
) -> KnowledgeBundle:
    request_key = request.knowledge_request.request_key if request.knowledge_request else ""
    issue = RetrievalIssue(
        code=code,
        message=str(message or code)[:500],
        backend=str(backend or "unknown"),
        lane="primary",
        stage=stage,
        severity="blocking",
        request_key=str(request_key or ""),
    )
    trace = RecallRoundTrace(
        request_key=str(request_key or ""),
        query=request.query,
        topics=[str(item.value if hasattr(item, "value") else item) for item in request.topic_categories],
        backend=str(backend or "unknown"),
        blocked_reason="%s:%s" % (code, issue.message[:240]),
        retrieval_status="failed",
        retrieval_issues=[issue],
    )
    return KnowledgeBundle(
        backend=str(backend or "unknown"),
        recall_rounds=[trace],
        retrieval_status="failed",
        retrieval_issues=[issue],
    )


def merge_knowledge_fallback(primary: KnowledgeBundle, fallback: KnowledgeBundle) -> KnowledgeBundle:
    """Merge a backend fallback without erasing the primary operational state."""

    primary = normalize_knowledge_bundle_status(primary)
    fallback = normalize_knowledge_bundle_status(fallback)
    fallback_has_evidence = bool(fallback.recall_bundle.items)
    primary_issues = [
        issue.model_copy(
            update={
                "fallback_used": True,
                "resolved": fallback_has_evidence,
                "severity": "warning" if fallback_has_evidence else issue.severity,
                "details": {
                    **dict(issue.details or {}),
                    "fallbackBackend": fallback.backend,
                    "fallbackItemCount": len(fallback.recall_bundle.items),
                },
            }
        )
        for issue in primary.retrieval_issues
    ]
    issues = dedupe_retrieval_issues([*primary_issues, *fallback.retrieval_issues])
    items = merge_recall_items(primary.recall_bundle.items, fallback.recall_bundle.items)
    status = retrieval_status_for(items, issues)
    return KnowledgeBundle(
        recall_bundle=RecallBundle(
            items=items,
            top_score=items[0].fusion_score if items else 0.0,
            merged_context="\n\n".join(
                "召回片段 [%s] %s\n%s" % (item.source_type, item.title, item.content[:1200]) for item in items
            ),
        ),
        source_refs=unique_source_refs(items),
        recall_rounds=[*primary.recall_rounds, *fallback.recall_rounds],
        backend="%s_fallback_%s" % (primary.backend or "primary", fallback.backend or "secondary"),
        index_version=primary.index_version or fallback.index_version,
        semantic_source_hash=(fallback.semantic_source_hash if fallback_has_evidence else primary.semantic_source_hash),
        retrieval_status=status,
        retrieval_issues=issues,
    )


class ResilientKnowledgeRetrievalService:
    """Use local governed recall when Elasticsearch is unavailable.

    Empty ES recall is a valid result and does not trigger fallback. Only an
    operationally failed primary bundle (or an exception/circuit-open state)
    switches to the local hybrid backend.
    """

    backend_name = "es_with_hybrid_fallback"

    def __init__(
        self,
        primary: KnowledgeRetrievalService,
        fallback: KnowledgeRetrievalService,
        *,
        settings: Any = None,
        failure_registry: ToolFailureRegistry | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        effective_settings = settings or getattr(primary, "settings", None)
        self.failure_registry = failure_registry or ToolFailureRegistry(
            # The retrieval adapter is shared across runs, so an identical
            # query failure must not become a permanent process-wide block.
            # Per-run duplicate blocking remains in ToolRuntime; this layer
            # uses the service circuit and its cooldown/half-open probe.
            repeat_threshold=2**31 - 1,
            circuit_threshold=max(
                1,
                int(
                    getattr(
                        effective_settings,
                        "tool_circuit_threshold",
                        5,
                    )
                    or 5
                ),
            ),
            cooldown_seconds=max(
                1,
                int(
                    getattr(
                        effective_settings,
                        "tool_circuit_cooldown_seconds",
                        60,
                    )
                    or 60
                ),
            ),
        )
        base_url = str(
            getattr(effective_settings, "es_base_url", "") or ""
        ).rstrip("/")
        index = str(
            getattr(effective_settings, "es_index", "") or ""
        ).strip("/")
        self.target = "%s/%s" % (base_url, index)
        self.target = self.target.strip("/") or "elasticsearch"

    @staticmethod
    def _request_identity(
        request: KnowledgeRetrievalRequest,
    ) -> dict[str, Any]:
        knowledge_request = request.knowledge_request
        return {
            "query": collapse_whitespace(request.query).casefold(),
            "merchantId": request.merchant_id,
            "topics": sorted(
                str(item.value if hasattr(item, "value") else item)
                for item in request.topic_categories
            ),
            "requestKey": (
                str(knowledge_request.request_key or "")
                if knowledge_request is not None
                else ""
            ),
            "round": request.round,
            "requiredCapabilities": sorted(
                request.required_capabilities
            ),
        }

    def _fallback_bundle(
        self,
        request: KnowledgeRetrievalRequest,
        primary: KnowledgeBundle,
    ) -> KnowledgeBundle:
        try:
            fallback = self.fallback.retrieve(request)
        except Exception as exc:
            fallback = failed_knowledge_bundle(
                request,
                getattr(self.fallback, "backend_name", "hybrid"),
                "KNOWLEDGE_FALLBACK_FAILED",
                "%s:%s" % (type(exc).__name__, str(exc)[:400]),
                stage="fallback",
            )
        return merge_knowledge_fallback(primary, fallback)

    def retrieve(
        self,
        request: KnowledgeRetrievalRequest,
    ) -> KnowledgeBundle:
        args = self._request_identity(request)
        blocked = self.failure_registry.should_block(
            "retrieve_knowledge",
            args,
            service_name="elasticsearch",
            target=self.target,
        )
        if blocked is not None:
            primary = failed_knowledge_bundle(
                request,
                getattr(self.primary, "backend_name", "es"),
                "ES_RETRIEVAL_CIRCUIT_OPEN",
                str(blocked.reason or "Elasticsearch circuit is open"),
                stage="circuit",
            )
            return self._fallback_bundle(request, primary)

        try:
            primary = normalize_knowledge_bundle_status(
                self.primary.retrieve(request)
            )
        except Exception as exc:
            primary = failed_knowledge_bundle(
                request,
                getattr(self.primary, "backend_name", "es"),
                "ES_RETRIEVAL_FAILED",
                "%s:%s" % (type(exc).__name__, str(exc)[:400]),
            )

        if primary.retrieval_status != "failed":
            self.failure_registry.record_success(
                "retrieve_knowledge",
                args,
                service_name="elasticsearch",
                target=self.target,
            )
            return primary

        issue = next(iter(primary.retrieval_issues), None)
        self.failure_registry.record_failure(
            "retrieve_knowledge",
            args,
            str(getattr(issue, "code", "") or "ES_RETRIEVAL_FAILED"),
            str(
                getattr(issue, "message", "")
                or "Elasticsearch retrieval failed"
            ),
            service_name="elasticsearch",
            target=self.target,
        )
        return self._fallback_bundle(request, primary)

    def cache_trace(self) -> dict[str, Any]:
        primary_trace = (
            self.primary.cache_trace()
            if callable(getattr(self.primary, "cache_trace", None))
            else {}
        )
        fallback_trace = (
            self.fallback.cache_trace()
            if callable(getattr(self.fallback, "cache_trace", None))
            else {}
        )
        return {
            "primary": primary_trace,
            "fallback": fallback_trace,
            "failureRegistry": self.failure_registry.trace(),
        }


def merge_recall_items(primary: list[RecallItem], secondary: list[RecallItem]) -> list[RecallItem]:
    by_id: dict[str, RecallItem] = {}
    for item in list(primary or []) + list(secondary or []):
        key = item.doc_id or str((item.metadata or {}).get("semanticRefId") or "")
        if not key:
            continue
        current = by_id.get(key)
        if current is None:
            by_id[key] = item
            continue
        preferred = item if recall_item_sort_key(item) > recall_item_sort_key(current) else current
        other = current if preferred is item else item
        merged = merge_recall_item_metadata(preferred, other)
        has_final_score = any((candidate.metadata or {}).get("finalScore") is not None for candidate in [current, item])
        merged_score = (
            float(preferred.fusion_score or 0.0)
            if has_final_score
            else max(float(current.fusion_score or 0.0), float(item.fusion_score or 0.0))
        )
        by_id[key] = merged.model_copy(
            update={
                "fusion_score": merged_score,
            }
        )
    return sorted(by_id.values(), key=recall_item_sort_key, reverse=True)


def source_type_top_k_policy(
    include_rules: bool = False,
    query_text: str = "",
    topics: list[str] | None = None,
    metric_candidates: list[dict[str, Any]] | None = None,
    retrieval_profile: dict[str, Any] | None = None,
) -> dict[str, int]:
    profile = retrieval_profile or {}
    configured_caps = profile.get("sourceTypeCaps") or {}
    if isinstance(configured_caps, dict) and configured_caps:
        policy = {
            "SEMANTIC_METRIC": int(configured_caps.get("SEMANTIC_METRIC") or 12),
            "SEMANTIC_RELATIONSHIP": int(configured_caps.get("SEMANTIC_RELATIONSHIP") or 8),
            "SEMANTIC_TABLE_ASSET": int(configured_caps.get("SEMANTIC_TABLE_ASSET") or 6),
            "SEMANTIC_COLUMN": int(configured_caps.get("SEMANTIC_COLUMN") or 10),
            "SEMANTIC_TERM": int(configured_caps.get("SEMANTIC_TERM") or 6),
            "SEMANTIC_BUSINESS_RULE": int(configured_caps.get("SEMANTIC_BUSINESS_RULE") or (6 if include_rules else 3)),
            "GOVERNED_RULE": int(configured_caps.get("GOVERNED_RULE") or (6 if include_rules else 3)),
        }
    else:
        profile_kind = str(profile.get("profileKind") or "balanced")
        if profile_kind == "focused":
            policy = {
                "SEMANTIC_METRIC": 10,
                "SEMANTIC_RELATIONSHIP": 5,
                "SEMANTIC_TABLE_ASSET": 4,
                "SEMANTIC_COLUMN": 8,
                "SEMANTIC_TERM": 4,
                "SEMANTIC_BUSINESS_RULE": 4 if include_rules else 2,
                "GOVERNED_RULE": 4 if include_rules else 2,
            }
        elif profile_kind == "broad":
            policy = {
                "SEMANTIC_METRIC": 14,
                "SEMANTIC_RELATIONSHIP": 10,
                "SEMANTIC_TABLE_ASSET": 8,
                "SEMANTIC_COLUMN": 12,
                "SEMANTIC_TERM": 8,
                "SEMANTIC_BUSINESS_RULE": 8 if include_rules else 4,
                "GOVERNED_RULE": 8 if include_rules else 4,
            }
        else:
            policy = {
                "SEMANTIC_METRIC": 12,
                "SEMANTIC_RELATIONSHIP": 8,
                "SEMANTIC_TABLE_ASSET": 6,
                "SEMANTIC_COLUMN": 10,
                "SEMANTIC_TERM": 6,
                "SEMANTIC_BUSINESS_RULE": 6 if include_rules else 3,
                "GOVERNED_RULE": 6 if include_rules else 3,
            }
    relationship_heavy = str(profile.get("queryType") or "") in {
        "mixed_rule_data",
        "multi_hop_analysis",
    }
    metric_heavy = bool(metric_candidates)
    if relationship_heavy:
        policy["SEMANTIC_RELATIONSHIP"] = min(policy["SEMANTIC_RELATIONSHIP"] + 2, 12)
    if metric_heavy:
        policy["SEMANTIC_METRIC"] = min(policy["SEMANTIC_METRIC"] + 1, 16)
    if topics and len(topics) >= 3:
        policy["SEMANTIC_TABLE_ASSET"] = min(policy["SEMANTIC_TABLE_ASSET"] + 1, 10)
    return policy


def limit_recall_items_by_source_type(
    items: list[RecallItem], policy: dict[str, int], limit: int = 24
) -> list[RecallItem]:
    if not items:
        return []
    counts: dict[str, int] = {}
    selected: list[RecallItem] = []
    for item in sorted(items, key=recall_item_sort_key, reverse=True):
        source_type = str(item.source_type or "UNKNOWN").upper()
        cap = int(policy.get(source_type, max(1, limit)))
        if counts.get(source_type, 0) < cap:
            selected.append(item)
            counts[source_type] = counts.get(source_type, 0) + 1
    return selected[: max(1, int(limit or len(selected)))]


def recall_channels_for_items(items: list[RecallItem]) -> list[str]:
    channels: list[str] = []
    for item in items or []:
        metadata = item.metadata or {}
        raw_channels = metadata.get("recallChannels") or [metadata.get("recallChannel")]
        for raw in raw_channels or []:
            channel = str(raw or "").strip()
            if channel and channel not in channels:
                channels.append(channel)
    return channels


def retrieval_lane_trace(
    retrieval_profile: dict[str, Any],
    vector_enabled: bool,
    include_rules: bool,
    has_metric_candidates: bool,
    broad_enabled: bool,
    query_plan_size: int = 1,
    hierarchical_enabled: bool = False,
) -> list[dict[str, Any]]:
    lanes: list[dict[str, Any]] = []
    lanes.append(
        {
            "lane": "post_recall_metric_validation",
            "enabled": has_metric_candidates,
            "candidateCount": 6 if has_metric_candidates else 0,
        }
    )
    lanes.append({"lane": "bm25_lane", "enabled": True, "topK": int(retrieval_profile.get("textTopK") or 0)})
    lanes.append(
        {
            "lane": "vector_lane",
            "enabled": vector_enabled,
            "topK": int(retrieval_profile.get("vectorTopK") or 0) if vector_enabled else 0,
        }
    )
    broad_flag = bool(retrieval_profile.get("broadSearchEnabled", True)) and broad_enabled
    lanes.append(
        {
            "lane": "broad_bm25_lane",
            "enabled": broad_flag,
            "topK": int(retrieval_profile.get("broadTextTopK") or 0) if broad_flag else 0,
        }
    )
    lanes.append(
        {
            "lane": "broad_vector_lane",
            "enabled": broad_flag and vector_enabled,
            "topK": int(retrieval_profile.get("broadVectorTopK") or 0) if broad_flag and vector_enabled else 0,
        }
    )
    lanes.append(
        {
            "lane": "multi_query_lane",
            "enabled": query_plan_size > 1,
            "queryCount": max(1, int(query_plan_size or 1)),
        }
    )
    lanes.append(
        {
            "lane": "directory_recursive_lane",
            "enabled": bool(hierarchical_enabled and query_plan_size > 1),
            "maxDepth": 2,
        }
    )
    if include_rules:
        lanes.append(
            {
                "lane": "governed_rule_lane",
                "enabled": True,
                "topK": int((retrieval_profile.get("sourceTypeCaps") or {}).get("GOVERNED_RULE") or 0),
            }
        )
    return lanes


def rrf_fuse_recall_items(
    ranked_groups: list[tuple[str, list[RecallItem]]],
    rrf_k: int = 60,
    score_scale: float = 1000.0,
    limit: int = 24,
) -> list[RecallItem]:
    """Fuse ranked recall lists with reciprocal rank fusion.

    BM25 scores and vector similarities are not comparable. RRF only uses the
    rank position inside each channel, then normalizes the result to 0..1 so
    downstream ranking keeps the same score semantics when a channel degrades.
    """
    k = max(1, int(rrf_k or 60))
    scale = float(score_scale or 1.0)
    by_id: dict[str, RecallItem] = {}
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    channel_scores: dict[str, dict[str, float]] = {}
    for channel, items in ranked_groups:
        channel_name = str(channel or "unknown")
        seen_in_channel: set[str] = set()
        for rank, item in enumerate(items or [], start=1):
            key = item.doc_id or str((item.metadata or {}).get("semanticRefId") or "")
            if not key or key in seen_in_channel:
                continue
            seen_in_channel.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(key, {})[channel_name] = rank
            channel_scores.setdefault(key, {})[channel_name] = float(item.fusion_score or 0.0)
            if key not in by_id:
                by_id[key] = item
            else:
                by_id[key] = merge_recall_item_metadata(by_id[key], item)
    active_lane_count = max(1, sum(1 for _, items in ranked_groups if items))
    theoretical_max = active_lane_count / float(k + 1)
    fused: list[RecallItem] = []
    for key, item in by_id.items():
        metadata = dict(item.metadata or {})
        raw_score = scores.get(key, 0.0)
        normalized_score = max(0.0, min(1.0, raw_score / theoretical_max)) if theoretical_max else 0.0
        if metadata.get("rrfRanks"):
            metadata["upstreamRrfRanks"] = metadata.get("rrfRanks")
        if metadata.get("channelScores"):
            metadata["upstreamChannelScores"] = metadata.get("channelScores")
        if metadata.get("rrfNormalizedScore") is not None:
            metadata["upstreamRrfNormalizedScore"] = metadata.get("rrfNormalizedScore")
        metadata["recallFusion"] = "rrf"
        metadata["scoreVersion"] = "recall_v2"
        metadata["rrfScore"] = raw_score
        metadata["rrfNormalizedScore"] = normalized_score
        metadata["rrfDisplayScore"] = raw_score * scale
        metadata["retrievalScore"] = normalized_score
        metadata["rrfK"] = k
        metadata["rrfActiveLaneCount"] = active_lane_count
        metadata["rrfRanks"] = ranks.get(key, {})
        metadata["channelScores"] = channel_scores.get(key, {})
        metadata["recallChannels"] = sorted((ranks.get(key) or {}).keys())
        fused.append(item.model_copy(update={"fusion_score": round(normalized_score, 6), "metadata": metadata}))
    fused = sorted(fused, key=recall_item_sort_key, reverse=True)
    return fused[: max(1, int(limit or len(fused)))] if limit else fused


def merge_recall_item_metadata(primary: RecallItem, secondary: RecallItem) -> RecallItem:
    metadata = dict(primary.metadata or {})
    other = dict(secondary.metadata or {})
    for key, value in other.items():
        if key not in metadata or is_empty_metadata_value(metadata.get(key)):
            metadata[key] = value
    queries: list[str] = []
    for source in [metadata, other]:
        for raw in list(source.get("recallQueries") or []) + [source.get("recallQuery")]:
            query = str(raw or "").strip()
            if query and query not in queries:
                queries.append(query)
    if queries:
        metadata["recallQueries"] = queries
        metadata["recallQuery"] = queries[0]
    if secondary.content and len(secondary.content) > len(primary.content or ""):
        return secondary.model_copy(update={"metadata": metadata})
    return primary.model_copy(update={"metadata": metadata})


def is_empty_metadata_value(value: object) -> bool:
    return value is None or value == "" or value == []


def unique_source_refs(items: list[RecallItem]) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for item in items:
        ref = item.doc_id or str((item.metadata or {}).get("semanticRefId") or "")
        if not ref or ref in seen:
            continue
        seen.add(ref)
        refs.append(ref)
    return refs


def rewrite_retrieval_query(request: KnowledgeRetrievalRequest) -> str:
    """Turn a context-dependent follow-up into a standalone retrieval query.

    This is intentionally deterministic: it only inherits the previous user
    question when the current turn contains an explicit follow-up signal.
    """
    current = collapse_whitespace(request.query)
    previous = collapse_whitespace(request.previous_user_question)
    if not current or not previous or current == previous:
        return current
    language = load_language_policy().routing
    follow_up = current.startswith(language.follow_up_prefixes) or any(
        phrase in current for phrase in language.follow_up_phrases
    )
    if not follow_up:
        return current
    return "%s；追问补充：%s" % (previous[:600], current[:300])


def filter_recall_items_by_governance(
    items: list[RecallItem],
    request: KnowledgeRetrievalRequest,
) -> tuple[list[RecallItem], dict[str, int]]:
    kept: list[RecallItem] = []
    filtered: dict[str, int] = {}
    for item in items or []:
        reason = recall_governance_block_reason(item, request)
        if reason:
            filtered[reason] = filtered.get(reason, 0) + 1
            continue
        kept.append(item)
    return kept, filtered


def recall_governance_decisions(
    items: list[RecallItem],
    request: KnowledgeRetrievalRequest,
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for item in items or []:
        reason = recall_governance_block_reason(item, request)
        if not reason:
            continue
        metadata = dict(item.metadata or {})
        decisions.append(
            {
                "refId": item.doc_id or str(metadata.get("semanticRefId") or ""),
                "sourceType": str(item.source_type or "UNKNOWN").upper(),
                "topic": str(item.topic or metadata.get("topic") or ""),
                "table": str(item.table or metadata.get("tableName") or ""),
                "reasonCode": "GOVERNANCE_%s" % reason.upper(),
            }
        )
    return decisions


def recall_governance_metadata(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    mappings = {
        "status": payload.get("status") or payload.get("lifecycleStatus"),
        "version": payload.get("version") or payload.get("semanticVersion"),
        "activeVersion": payload.get("activeVersion") or payload.get("currentVersion"),
        "merchantId": payload.get("merchantId"),
        "merchantIds": payload.get("merchantIds") or payload.get("allowedMerchantIds"),
        "allowedRoles": payload.get("allowedRoles"),
        "requiredPermissions": payload.get("requiredPermissions"),
        "visibilityPolicy": payload.get("visibilityPolicy"),
        "expiresAt": payload.get("expiresAt") or payload.get("expiryAt"),
        "confidence": payload.get("confidence") or payload.get("knowledgeConfidence"),
    }
    return {
        key: value
        for key, value in mappings.items()
        if value is not None and value != "" and value != () and value != [] and value != {}
    }


def recall_governance_block_reason(item: RecallItem, request: KnowledgeRetrievalRequest) -> str:
    metadata = dict(item.metadata or {})
    status = (
        str(metadata.get("lifecycleStatus") or metadata.get("publishStatus") or metadata.get("status") or "")
        .strip()
        .lower()
    )
    blocked_statuses = {
        "pending",
        "pending_review",
        "draft",
        "rejected",
        "disabled",
        "inactive",
        "expired",
        "rolled_back",
        "deleted",
        "archived",
        "blocked",
    }
    if status in blocked_statuses:
        return "status"
    asset_status = str(metadata.get("assetStatus") or "").strip().upper()
    semantic_kind = str(metadata.get("semanticKind") or "").strip().upper()
    source_type = str(item.source_type or "").strip().upper()
    requires_published_asset = bool(
        source_type in {"SEMANTIC_TABLE_ASSET", "SEMANTIC_METRIC"}
        or semantic_kind in {"TABLE_ASSET", "TABLE_DETAIL", "METRIC"}
    )
    if requires_published_asset and asset_status not in {"ACTIVE", "PUBLISHED"}:
        return "semantic_activation"
    if asset_status and asset_status.lower() in blocked_statuses:
        return "status"

    expires_at = metadata.get("expiresAt") or metadata.get("expiryAt")
    if expires_at and timestamp_is_past(expires_at):
        return "expired"
    if metadata.get("assetExpiresAt") and timestamp_is_past(metadata.get("assetExpiresAt")):
        return "expired"

    active_version = str(metadata.get("activeVersion") or metadata.get("currentVersion") or "").strip()
    item_version = str(metadata.get("semanticVersion") or metadata.get("version") or "").strip()
    if active_version and item_version and active_version != item_version:
        return "version"

    merchant_id = str(request.merchant_id or "").strip()
    scoped_merchants = metadata.get("merchantIds") or metadata.get("allowedMerchantIds") or []
    if isinstance(scoped_merchants, str):
        scoped_merchants = [scoped_merchants]
    item_merchant = str(metadata.get("merchantId") or "").strip()
    if item_merchant and item_merchant not in {"*", "global", merchant_id}:
        return "merchant"
    asset_merchant = str(metadata.get("assetMerchantId") or "").strip()
    if asset_merchant and asset_merchant not in {"*", "global", merchant_id}:
        return "merchant"
    if (
        scoped_merchants
        and merchant_id not in {str(value) for value in scoped_merchants}
        and "*" not in scoped_merchants
    ):
        return "merchant"

    visibility = metadata.get("visibilityPolicy") if isinstance(metadata.get("visibilityPolicy"), dict) else {}
    allowed_roles = metadata.get("allowedRoles") or visibility.get("allowedRoles") or []
    if isinstance(allowed_roles, str):
        allowed_roles = [allowed_roles]
    authorization = load_authorization_policy()
    role = str(request.access_role or authorization.default_access_role).strip()
    global_access = "*" in authorization.permissions_for_access_role(role)
    normalized_roles = {str(value).strip().lower() for value in allowed_roles if str(value).strip()}
    if normalized_roles and role.lower() not in normalized_roles and not global_access:
        return "role"
    asset_roles = metadata.get("assetAllowedRoles") or []
    if isinstance(asset_roles, str):
        asset_roles = [asset_roles]
    normalized_asset_roles = {str(value).strip().lower() for value in asset_roles if str(value).strip()}
    if normalized_asset_roles and role.lower() not in normalized_asset_roles and not global_access:
        return "role"
    visibility_level = str(visibility.get("level") or metadata.get("visibility") or "").strip().lower()
    if visibility_level == "restricted" and not normalized_roles and not global_access:
        return "role"
    asset_visibility = (
        metadata.get("assetVisibilityPolicy") if isinstance(metadata.get("assetVisibilityPolicy"), dict) else {}
    )
    if (
        str(asset_visibility.get("level") or "").strip().lower() == "restricted"
        and not normalized_asset_roles
        and not global_access
    ):
        return "role"

    required_permissions = metadata.get("requiredPermissions") or []
    if isinstance(required_permissions, str):
        required_permissions = [required_permissions]
    granted = {str(value).strip() for value in request.permissions if str(value).strip()}
    if required_permissions and not set(map(str, required_permissions)).issubset(granted):
        return "permission"
    asset_permissions = metadata.get("assetRequiredPermissions") or []
    if isinstance(asset_permissions, str):
        asset_permissions = [asset_permissions]
    if asset_permissions and not set(map(str, asset_permissions)).issubset(granted):
        return "permission"
    return ""


def timestamp_is_past(value: object) -> bool:
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed <= datetime.now(timezone.utc)


def business_rerank_recall_items(
    items: list[RecallItem],
    query_text: str,
    request: KnowledgeRetrievalRequest,
) -> list[RecallItem]:
    """Apply deterministic ranking policy without blending business scores.

    Exact, unambiguous post-recall metric validation may establish a
    protection tier. Within the same tier the normalized retrieval score
    remains authoritative; Topic/name matches are only tie-break diagnostics.
    """

    query = str(query_text or "").lower()
    reranked: list[RecallItem] = []
    fallback_ranks = {
        id(item): rank
        for rank, item in enumerate(
            sorted(items or [], key=lambda value: float(value.fusion_score or 0.0), reverse=True), start=1
        )
    }
    for item in items or []:
        metadata = dict(item.metadata or {})
        source_type = str(item.source_type or "").upper()
        reasons: list[str] = []
        tie_break_tier = 0
        aliases = metadata.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        labels = [
            metadata.get("businessName"),
            metadata.get("metricKey"),
            *aliases,
        ]
        if any(str(label).strip().lower() in query for label in labels if len(str(label).strip()) >= 2):
            reasons.append("exact_business_label")
            tie_break_tier += 2
        if item.topic and str(item.topic).lower() in query:
            reasons.append("topic_match")
            tie_break_tier += 1
        retrieval_score = recall_item_retrieval_score(item, fallback_rank=fallback_ranks.get(id(item), 1))
        protection_tier, protection_reasons = metric_protection_tier(metadata, source_type)
        for stale_key in (
            "businessScore",
            "retrievalWeightedScore",
            "businessWeightedScore",
            "finalScore",
            "businessRerankBoost",
        ):
            metadata.pop(stale_key, None)
        metadata["scoreVersion"] = "recall_v3"
        metadata["retrievalScore"] = round(retrieval_score, 6)
        metadata["protectionTier"] = protection_tier
        metadata["protectionReasons"] = protection_reasons
        metadata["businessTieBreakTier"] = tie_break_tier
        metadata["rankingPolicyReasons"] = reasons
        metadata.pop("businessRerankReasons", None)
        reranked.append(
            item.model_copy(
                update={
                    "fusion_score": round(retrieval_score, 6),
                    "metadata": metadata,
                }
            )
        )
    return sorted(reranked, key=recall_item_sort_key, reverse=True)


def recall_item_retrieval_score(item: RecallItem, fallback_rank: int = 1) -> float:
    metadata = dict(item.metadata or {})
    for key in ["retrievalScore", "rrfNormalizedScore", "metricResolverScore"]:
        value = metadata.get(key)
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
    raw_score = float(item.fusion_score or 0.0)
    if 0.0 <= raw_score <= 1.0:
        return raw_score
    rank = max(1, int(fallback_rank or 1))
    return round(61.0 / float(60 + rank), 6)


def metric_protection_tier(
    metadata: dict[str, Any],
    source_type: str,
) -> tuple[int, list[str]]:
    if source_type != "SEMANTIC_METRIC":
        return 0, []
    resolution_type = str(metadata.get("metricResolutionType") or "")
    confidence = max(0.0, min(1.0, float(metadata.get("metricResolutionConfidence") or 0.0)))
    ambiguous = bool(metadata.get("metricResolutionAmbiguous") or False)
    exact = resolution_type.startswith("exact_")
    if exact and confidence >= 0.95 and not ambiguous:
        return 2, ["exact_metric", "high_confidence", "unambiguous"]
    if confidence >= 0.8 and not ambiguous:
        reasons = ["metric_candidate", "high_confidence"]
        if exact:
            reasons.append("exact_metric")
        return 1, reasons
    return 0, []


def recall_item_sort_key(item: RecallItem) -> tuple[int, float, float]:
    metadata = dict(item.metadata or {})
    if str(metadata.get("scoreVersion") or "") == "recall_v3":
        return (
            int(metadata.get("protectionTier") or 0),
            float(metadata.get("retrievalScore") or item.fusion_score or 0.0),
            float(metadata.get("businessTieBreakTier") or 0.0),
        )
    return (
        int(metadata.get("protectionTier") or 0),
        float(metadata.get("finalScore") if metadata.get("finalScore") is not None else item.fusion_score or 0.0),
        float(metadata.get("retrievalScore") or 0.0),
    )


def retrieval_query_text(request: KnowledgeRetrievalRequest, rewritten_query: str = "") -> str:
    parts = [rewritten_query or request.query]
    parts.extend(request.keywords or [])
    knowledge_request = request.knowledge_request
    if knowledge_request:
        parts.extend(
            [
                knowledge_request.query,
                knowledge_request.source_phrase,
                knowledge_request.reason,
                " ".join(knowledge_request.expected_refs or []),
            ]
        )
    seen: set[str] = set()
    values: list[str] = []
    for part in parts:
        value = str(part or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return " ".join(values)


def exact_metric_label_in_query(metric: dict[str, object], query_text: str) -> str:
    query = normalize_recall_label(query_text)
    if not query:
        return ""
    labels = [
        str(metric.get("businessName") or ""),
        str(metric.get("metricKey") or ""),
        *[str(alias) for alias in metric.get("aliases") or []],
    ]
    for label in labels:
        normalized = normalize_recall_label(label)
        if not is_protective_metric_label(normalized):
            continue
        if normalized and normalized in query:
            return label
    return ""


def normalize_recall_label(value: str) -> str:
    return "".join(str(value or "").lower().split())


def is_protective_metric_label(label: str) -> bool:
    if not label:
        return False
    if "_" in label:
        return len(label) >= 4
    return len(label) >= 3


def normalize_question_category(category: object) -> QuestionCategory | None:
    if isinstance(category, QuestionCategory):
        return category
    raw = str(category or "").strip()
    if not raw:
        return None
    try:
        return QuestionCategory(raw)
    except Exception:
        pass
    for item in QuestionCategory:
        if category_display(item) == raw:
            return item
    return None


def topic_categories_support_knowledge_capability(
    topic_assets: TopicAssetService,
    categories: list[QuestionCategory],
    capability: str,
) -> bool:
    """Resolve retrieval lanes from published topic roles, never topic IDs."""

    expected = safe_ascii_component(
        capability,
        extras=("_",),
        uppercase=True,
        strip="_",
    )
    if not expected:
        return False
    for topic in topic_assets.topic_names_for_categories(categories):
        contract = topic_assets.load_topic_contract(topic)
        metadata = contract.get("metadata") if isinstance(contract.get("metadata"), dict) else {}
        declared_values: list[Any] = []
        for source in (contract, metadata):
            for key in (
                "capabilities",
                "knowledgeCapabilities",
                "knowledgeCapability",
                "knowledgeRoles",
                "knowledgeRole",
                "retrievalCapabilities",
                "routingRole",
                "topicRole",
            ):
                value = source.get(key)
                declared_values.extend(value if isinstance(value, list) else [value])
        for value in declared_values:
            normalized = safe_ascii_component(
                value,
                extras=("_",),
                uppercase=True,
                strip="_",
            )
            tokens = [token for token in normalized.split("_") if token]
            if normalized == expected or expected in tokens:
                return True
    return False


def route_is_rule_sensitive(request: KnowledgeRetrievalRequest) -> bool:
    slots = request.route_slots or {}
    risk_level = str(slots.get("riskLevel") or slots.get("risk_level") or "").strip()
    return risk_level == "rule_sensitive"


def es_hit_to_recall_item(hit: dict[str, object], query_text: str, channel: str = "bm25") -> RecallItem:
    source = hit.get("_source") if isinstance(hit, dict) else {}
    source = source if isinstance(source, dict) else {}
    metadata = dict(source.get("metadata") or {})
    semantic_ref_id = str(
        source.get("semantic_ref_id") or metadata.get("semanticRefId") or source.get("doc_id") or hit.get("_id") or ""
    )
    semantic_path = str(source.get("semantic_path") or metadata.get("semanticPath") or "")
    merchant_uri = str(source.get("merchant_uri") or metadata.get("merchantUri") or "")
    context_layer = str(source.get("context_layer") or metadata.get("contextLayer") or "")
    retrieval_level = str(source.get("retrieval_level") or metadata.get("retrievalLevel") or "")
    directory_id = str(source.get("directory_id") or metadata.get("directoryId") or "")
    parent_directory_id = str(source.get("parent_directory_id") or metadata.get("parentDirectoryId") or "")
    metadata["semanticRefId"] = semantic_ref_id
    if semantic_path:
        metadata["semanticPath"] = semantic_path
    if merchant_uri:
        metadata["merchantUri"] = merchant_uri
    if context_layer:
        metadata["contextLayer"] = context_layer
    if retrieval_level:
        metadata["retrievalLevel"] = retrieval_level
    if directory_id:
        metadata["directoryId"] = directory_id
    if parent_directory_id:
        metadata["parentDirectoryId"] = parent_directory_id
    metadata["recallQuery"] = query_text
    metadata["recallQueries"] = [query_text] if query_text else []
    metadata["esScore"] = float(hit.get("_score") or 0.0)
    metadata["recallChannel"] = channel
    if channel == "bm25":
        metadata["bm25Score"] = float(hit.get("_score") or 0.0)
    elif channel == "vector":
        metadata["vectorScore"] = float(hit.get("_score") or 0.0)
    return attach_goal_recall_capabilities(
        RecallItem(
            doc_id=str(source.get("doc_id") or semantic_ref_id or hit.get("_id") or ""),
            title=str(source.get("title") or ""),
            content=str(source.get("content") or ""),
            source_type=str(source.get("source_type") or ""),
            topic=str(source.get("topic") or metadata.get("topic") or ""),
            table=str(source.get("table") or metadata.get("tableName") or ""),
            answer_mode=str(source.get("answer_mode") or ""),
            fusion_score=float(hit.get("_score") or 0.0),
            metadata=metadata,
        )
    )


def recall_queries_from_items(items: list[RecallItem]) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    for item in items:
        metadata = item.metadata or {}
        for raw in list(metadata.get("recallQueries") or []) + [metadata.get("recallQuery")]:
            query = str(raw or "").strip()
            if not query or query in seen:
                continue
            seen.add(query)
            queries.append(query)
    return queries


def semantic_hash_for_items(items: list[RecallItem]) -> str:
    records = [
        {
            "docId": item.doc_id,
            "sourceType": item.source_type,
            "semanticRefId": str((item.metadata or {}).get("semanticRefId") or ""),
            "merchantUri": str((item.metadata or {}).get("merchantUri") or ""),
            "sourcePath": str((item.metadata or {}).get("sourcePath") or ""),
        }
        for item in items
    ]
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16] if records else ""
