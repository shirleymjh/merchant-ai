from merchant_ai.config import get_settings
from merchant_ai.graph.workflow import (
    append_knowledge_retrieval_outcomes_to_run_result,
    create_workflow,
)
from merchant_ai.models import (
    AgentRunResult,
    ChatContext,
    KnowledgeBundle,
    KnowledgeRetrievalRequest,
    RecallBundle,
    RecallItem,
    RetrievalIssue,
)
from merchant_ai.services.assets import TopicAssetService
from merchant_ai.services.retrieval import (
    EsKnowledgeRetrievalService,
    merge_knowledge_fallback,
)


def retrieval_settings(tmp_path):
    return get_settings().model_copy(
        update={
            "cache_enabled": False,
            "harness_workspace_path": str(tmp_path),
            "agent_checkpointer_backend": "memory",
            "embedding_api_key": "",
            "llm_api_key": "",
        }
    )


def test_es_operational_failure_is_distinct_from_a_successful_empty_search(monkeypatch, tmp_path):
    settings = retrieval_settings(tmp_path)
    request = KnowledgeRetrievalRequest(query="unmapped operational probe")

    failed_service = EsKnowledgeRetrievalService(settings, TopicAssetService(settings))

    def fail_search(_query, _topics, include_rules=False):
        del include_rules
        raise RuntimeError("injected transport outage")

    monkeypatch.setattr(failed_service, "_search", fail_search)
    failed = failed_service.retrieve(request)

    empty_service = EsKnowledgeRetrievalService(settings, TopicAssetService(settings))
    monkeypatch.setattr(empty_service, "_search", lambda _query, _topics, include_rules=False: [])
    empty = empty_service.retrieve(request)

    assert failed.retrieval_status == "failed"
    assert failed.recall_bundle.items == []
    assert failed.retrieval_issues[0].code == "ES_RETRIEVAL_FAILED"
    assert failed.recall_rounds[0].blocked_reason.startswith("ES_RETRIEVAL_FAILED:")
    assert failed.recall_rounds[0].retrieval_status == "failed"

    assert empty.retrieval_status == "empty"
    assert empty.recall_bundle.items == []
    assert empty.retrieval_issues == []
    assert empty.recall_rounds[0].blocked_reason == ""
    assert empty.recall_rounds[0].retrieval_status == "empty"


def test_partial_es_lane_failure_is_structured_instead_of_silently_swallowed(monkeypatch, tmp_path):
    settings = retrieval_settings(tmp_path).model_copy(
        update={
            "es_vector_enabled": True,
            "es_vector_field": "embedding",
            "embedding_model": "embedding-model",
            "embedding_api_key": "test-key",
        }
    )
    service = EsKnowledgeRetrievalService(settings, TopicAssetService(settings))
    monkeypatch.setattr(
        service,
        "_text_search",
        lambda _query, _topics, include_rules=False: [
            RecallItem(
                doc_id="semantic:generic:asset",
                title="asset",
                content="asset evidence",
                source_type="SEMANTIC_TABLE_ASSET",
                fusion_score=1.0,
            )
        ],
    )

    def fail_embedding(_query):
        raise RuntimeError("injected embedding outage")

    monkeypatch.setattr(service, "_embed_text", fail_embedding)
    bundle = service.retrieve(KnowledgeRetrievalRequest(query="generic semantic query"))

    assert bundle.retrieval_status == "degraded"
    assert [item.doc_id for item in bundle.recall_bundle.items] == ["semantic:generic:asset"]
    assert bundle.retrieval_issues[0].code == "ES_RETRIEVAL_LANE_FAILED"
    assert bundle.retrieval_issues[0].lane == "vector"
    assert bundle.retrieval_issues[0].severity == "warning"


def test_successful_fallback_resolves_primary_failure_but_preserves_disclosure():
    issue = RetrievalIssue(
        code="PRIMARY_RETRIEVAL_FAILED",
        message="primary unavailable",
        backend="primary",
        lane="primary",
        severity="blocking",
    )
    primary = KnowledgeBundle(
        backend="primary",
        retrieval_status="failed",
        retrieval_issues=[issue],
    )
    fallback_item = RecallItem(doc_id="semantic:generic:fallback", source_type="SEMANTIC_TABLE_ASSET")
    fallback = KnowledgeBundle(
        backend="secondary",
        recall_bundle=RecallBundle(items=[fallback_item]),
        source_refs=[fallback_item.doc_id],
        retrieval_status="success",
    )

    merged = merge_knowledge_fallback(primary, fallback)

    assert merged.retrieval_status == "degraded"
    assert merged.source_refs == [fallback_item.doc_id]
    assert merged.retrieval_issues[0].resolved is True
    assert merged.retrieval_issues[0].fallback_used is True
    assert merged.retrieval_issues[0].severity == "warning"


def test_retrieval_failure_reaches_planner_evidence_and_response(tmp_path):
    settings = retrieval_settings(tmp_path)
    workflow = create_workflow(settings)
    state = workflow._initial_state(
        "generic business question",
        settings.merchant_id,
        ChatContext(),
        None,
        "thread_retrieval_failure",
        "run_retrieval_failure",
        [],
    )
    issue = RetrievalIssue(
        code="KNOWLEDGE_RETRIEVER_FAILED",
        message="injected semantic retrieval outage",
        backend="semantic",
        lane="primary",
        stage="retrieve",
        severity="blocking",
    )
    state["knowledge_bundle"] = KnowledgeBundle(
        backend="semantic",
        retrieval_status="failed",
        retrieval_issues=[issue],
    )
    state["knowledge_retrieval_status"] = "failed"
    state["knowledge_retrieval_issues"] = [issue.model_dump(by_alias=True)]
    state["knowledge_retrieval_outcomes"] = [
        {
            "stage": "topic_workspace",
            "requestKey": "",
            "query": state["question"],
            "backend": "semantic",
            "status": "failed",
            "itemCount": 0,
            "issues": [issue.model_dump(by_alias=True)],
        }
    ]

    state = workflow.compact_assets(state)
    health = state["planning_asset_pack"].metric_compaction["retrievalHealth"]
    planner_gaps = state["planning_asset_pack"].metric_compaction["knowledgeRequestGaps"]
    assert health["status"] == "failed"
    assert planner_gaps[0]["code"] == "KNOWLEDGE_RETRIEVER_FAILED"
    assert planner_gaps[0]["type"] == "retrieval_failure"

    run_result = AgentRunResult()
    append_knowledge_retrieval_outcomes_to_run_result(state, run_result)
    assert run_result.evidence_gaps[0].code == "KNOWLEDGE_RETRIEVER_FAILED"
    assert run_result.evidence_gaps[0].severity == "blocking"
    assert run_result.degraded_reasons[0]["stage"] == "retrieval"

    state["agent_run_result"] = run_result
    state["answer"] = ""
    response = workflow.to_response(state)
    retrieval_debug = response.debug_trace["harness"]["knowledgeRetrieval"]
    assert retrieval_debug["status"] == "failed"
    assert retrieval_debug["issues"][0]["code"] == "KNOWLEDGE_RETRIEVER_FAILED"
    assert response.debug_trace["evidenceGaps"][0]["code"] == "KNOWLEDGE_RETRIEVER_FAILED"
