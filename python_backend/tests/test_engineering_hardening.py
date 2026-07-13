from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import sys
from types import ModuleType, SimpleNamespace

from fastapi.testclient import TestClient

import app.main as app_main
from app.main import create_app
from merchant_ai.config import Settings
from merchant_ai.graph.state import emit, register_event_listener, unregister_event_listener
from merchant_ai.models import AgentTaskResult, NodeExecutionContext, NodePlanContract, NodePlanCritiqueResult
from merchant_ai.models import AgentRunResult, MerchantInfo, QueryBundle, QueryPlan, QuestionIntent, VerifiedEvidence
from merchant_ai.services.answer import answer_data_package, verified_answer_context
from merchant_ai.services.attachments import AttachmentStore
from merchant_ai.services.assets import SemanticAssetGovernanceService, TopicAssetService
from merchant_ai.services.context_assembly import build_llm_context_blocks, context_cache_layout, context_quarantine_policy
from merchant_ai.services.memory import MemoryIngestionService, MemoryStore, MemoryWriteGate
from merchant_ai.services.query_contracts import contract_gaps_from_task_results, tenant_scope_binding_error
from merchant_ai.services.repositories import AnswerRepository, write_json
from merchant_ai.services.runtime_state import FileRuntimeStateStore, NodeTaskState, RedisRuntimeStateStore, create_runtime_state_store
from merchant_ai.services.security import Permission, authorize_merchant_access, merchant_principal, ops_principal
from merchant_ai.services.tool_runtime import ToolRuntimePolicyRegistry, ToolRuntimeService
from merchant_ai.services.tools import node_runtime_tool_schemas, tool_registry_from_descriptions, validate_tool_result_contract


def test_settings_exposes_grouped_config_views(tmp_path):
    settings = Settings(
        ops_token="secret",
        merchant_id="100",
        allowed_merchant_ids="100,200",
        cors_allow_origins="https://merchant.example, http://localhost:5173",
        cors_allow_credentials=True,
        harness_workspace_path=str(tmp_path),
    )

    assert settings.security.ops_token == "secret"
    assert settings.merchant_allowed("100") is True
    assert settings.merchant_allowed("200") is True
    assert settings.merchant_allowed("300") is False
    assert settings.security.cors_allow_origins == ["https://merchant.example", "http://localhost:5173"]
    assert settings.runtime.workspace_path == tmp_path
    assert settings.grouped_summary()["security"]["opsTokenConfigured"] is True


def test_es_vector_recall_is_enabled_by_default():
    settings = Settings()

    assert settings.es_vector_enabled is True
    assert settings.es_vector_field == "content_vector"
    assert settings.embedding_model == "text-embedding-3-small"


def test_ops_token_protects_runtime_endpoints():
    client = TestClient(create_app(Settings(ops_token="secret")))

    assert client.get("/api/runtime/alerts").status_code == 401
    assert client.get("/api/runtime/alerts", headers={"X-Ops-Token": "bad"}).status_code == 401
    assert client.get("/api/runtime/alerts", headers={"Authorization": "Bearer secret"}).status_code == 200

    metrics = client.get("/api/runtime/metrics", headers={"X-Ops-Token": "secret"}).json()
    assert metrics["config"]["security"]["opsTokenConfigured"] is True
    assert "answerRepository" in metrics["degraded"]


def test_missing_ops_token_fails_closed_for_non_loopback_client():
    client = TestClient(create_app(Settings(ops_token="")))

    assert client.get("/api/runtime/alerts").status_code == 503


def test_attachment_store_extracts_excel_and_enforces_merchant_scope(tmp_path):
    from openpyxl import Workbook

    settings = Settings(harness_workspace_path=str(tmp_path))
    store = AttachmentStore(settings)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "经营日报"
    sheet.append(["日期", "GMV"])
    sheet.append(["2026-07-11", 1234])
    payload = BytesIO()
    workbook.save(payload)

    metadata = store.save("日报.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", payload.getvalue(), "100")
    reference = SimpleNamespace(id=metadata["attachmentId"])

    assert metadata["parser"] == "excel"
    assert "经营日报" in metadata["textPreview"]
    assert "1234" in store.context_for([reference], "100")
    assert store.context_for([reference], "200") == ""


def test_merchant_allowlist_blocks_cross_tenant_requests():
    client = TestClient(create_app(Settings(merchant_id="100", allowed_merchant_ids="100", ops_token="secret")))

    assert client.post("/api/chat", json={"message": "你好", "merchantId": "100"}).status_code == 200
    blocked = client.post("/api/chat", json={"message": "你好", "merchantId": "101"})
    assert blocked.status_code == 403

    memory_blocked = client.get("/api/memory/101", headers={"X-Ops-Token": "secret"})
    assert memory_blocked.status_code == 403


def test_merchant_can_confirm_chat_knowledge_suggestion_through_api(tmp_path):
    client = TestClient(
        create_app(
            Settings(
                merchant_id="100",
                allowed_merchant_ids="100",
                memory_backend="file",
                harness_workspace_path=str(tmp_path),
            )
        )
    )
    memory = app_main.workflow.memory_store.load("100")
    memory["knowledgeSuggestions"] = [
        {
            "suggestionId": "ks_store_rule",
            "status": "candidate",
            "scopeType": "merchant",
            "metricName": "退款预警",
            "payload": {"memoryType": "correction", "correctionText": "本店退款率超过8%时提醒"},
        }
    ]
    app_main.workflow.memory_store.save("100", memory)

    response = client.post(
        "/api/merchant/knowledge-suggestions/ks_store_rule/action",
        json={"action": "confirm_use", "merchantId": "100", "actor": "merchant_user"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "MERCHANT_ACTIVE"
    assert payload["noticeType"] == "merchant_rule_confirmed"
    assert payload["requestedAction"] == "confirm_use"
    assert payload["ruleText"] == "本店退款率超过8%时提醒"
    assert "suggestion" not in payload
    assert "scopeType" not in payload


def test_event_listener_registry_is_safe_for_parallel_runs():
    seen = []

    def run_once(index: int) -> None:
        run_id = "run_%s" % index

        def listener(event_type, node, payload):
            seen.append((run_id, event_type, node, payload["runId"]))

        register_event_listener(run_id, listener)
        try:
            emit({"run_id": run_id, "thread_id": "thread_%s" % index}, "run.test", "TEST", {})
        finally:
            unregister_event_listener(run_id)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(run_once, range(32)))

    assert len(seen) == 32
    assert all(item[0] == item[3] for item in seen)


def test_answer_repository_records_degraded_reason_on_fallback():
    repo = AnswerRepository.__new__(AnswerRepository)
    repo.available = True
    repo.last_degraded_reason = {}

    def broken_query(*_args, **_kwargs):
        raise RuntimeError("storage unavailable")

    repo.db = SimpleNamespace(query=broken_query)

    assert repo.exists("answer_1") is False
    assert repo.trace()["available"] is True
    assert repo.trace()["lastDegradedReason"]["component"] == "answer_repository"
    assert repo.trace()["lastDegradedReason"]["operation"] == "exists"


def test_contract_gaps_keep_repairable_issue_evidence():
    result = AgentTaskResult(
        task_id="gmv",
        node_plan_contract=NodePlanContract(
            task_id="gmv",
            preferred_table="ads_merchant_profile",
            allowed_columns=["seller_id", "pt"],
        ),
        node_plan_critique=NodePlanCritiqueResult(
            task_id="gmv",
            valid=False,
            code="MISSING_METRIC_COLUMN",
            message="metricColumn is not available in node schema",
            graph_repairable=True,
            issues=[{"code": "MISSING_METRIC_COLUMN", "evidence": "order_gmv_amt_1d"}],
        ),
    )

    gaps = contract_gaps_from_task_results([result])

    assert gaps[0].code == "MISSING_METRIC_COLUMN"
    assert gaps[0].evidence == "order_gmv_amt_1d"
    assert gaps[0].source == "node_contract_critic"


def test_tenant_scope_requires_backend_bound_merchant_parameter():
    contract = NodePlanContract(
        task_id="gmv",
        preferred_table="ads_merchant_profile",
        allowed_columns=["seller_id", "pt", "order_gmv_amt_1d"],
        merchant_filter_column="seller_id",
    )
    context = NodeExecutionContext(merchant_id="100")

    literal_error = tenant_scope_binding_error(
        "SELECT order_gmv_amt_1d FROM ads_merchant_profile WHERE seller_id = '200'",
        [],
        contract,
        context,
    )
    bound_error = tenant_scope_binding_error(
        "SELECT order_gmv_amt_1d FROM ads_merchant_profile WHERE seller_id = %s",
        ["100"],
        contract,
        context,
    )

    assert literal_error
    assert bound_error == ""


def test_principal_permission_enforces_merchant_scope(tmp_path):
    settings = Settings(merchant_id="100", allowed_merchant_ids="100,200", harness_workspace_path=str(tmp_path))

    assert authorize_merchant_access(settings, merchant_principal("100"), "100", Permission.CHAT_RUN) == "100"
    assert authorize_merchant_access(settings, ops_principal(), "200", Permission.OPS_READ) == "200"

    try:
        authorize_merchant_access(settings, merchant_principal("100"), "200", Permission.CHAT_RUN)
    except Exception as exc:
        assert getattr(exc, "status_code", 0) == 403
    else:
        raise AssertionError("cross-merchant principal access should be blocked")


def test_file_runtime_state_store_externalizes_node_task_state(tmp_path):
    store = FileRuntimeStateStore(Settings(harness_workspace_path=str(tmp_path)))
    state = store.upsert_node_task(
        NodeTaskState(
            run_id="run_1",
            task_id="node_1",
            status="running",
            idempotency_key="node:run_1:node_1:table",
            payload={"table": "ads_merchant_profile"},
        )
    )

    loaded = store.get_node_task("run_1", "node_1")

    assert loaded is not None
    assert loaded.idempotency_key == state.idempotency_key
    assert store.list_node_tasks("run_1")[0].status == "running"
    store.cancel_run("run_1", "user requested")
    assert store.run_canceled("run_1") is True


def test_file_runtime_state_store_supports_node_task_queue_lease_and_complete(tmp_path):
    store = FileRuntimeStateStore(Settings(harness_workspace_path=str(tmp_path)))

    queued = store.enqueue_node_task(
        NodeTaskState(
            run_id="run_1",
            task_id="node_1",
            idempotency_key="node:run_1:node_1:ads_merchant_profile",
        )
    )
    claimed = store.claim_node_task("run_1", "node_1", "worker_1", lease_seconds=30)
    completed = store.complete_node_task("run_1", "node_1", "completed", {"rows": 3})

    assert queued.status == "queued"
    assert claimed is not None
    assert claimed.status == "running"
    assert claimed.attempts == 1
    assert completed.status == "completed"
    assert completed.payload["rows"] == 3


def test_runtime_state_store_factory_defaults_to_file(tmp_path):
    store = create_runtime_state_store(Settings(harness_workspace_path=str(tmp_path)))

    assert isinstance(store, FileRuntimeStateStore)


def test_redis_runtime_state_store_uses_real_backend_protocol(monkeypatch):
    class FakeRedisClient:
        def __init__(self):
            self.hashes = {}
            self.sets = {}

        def ping(self):
            return True

        def hset(self, key, mapping):
            self.hashes[key] = dict(mapping)

        def hgetall(self, key):
            return dict(self.hashes.get(key) or {})

        def sadd(self, key, value):
            self.sets.setdefault(key, set()).add(value)

        def smembers(self, key):
            return set(self.sets.get(key) or set())

        def exists(self, key):
            return key in self.hashes

        def srem(self, key, value):
            self.sets.setdefault(key, set()).discard(value)

    fake_client = FakeRedisClient()
    fake_redis_module = ModuleType("redis")
    fake_redis_module.Redis = SimpleNamespace(from_url=lambda *_args, **_kwargs: fake_client)
    monkeypatch.setitem(sys.modules, "redis", fake_redis_module)

    store = RedisRuntimeStateStore(Settings(redis_enabled=True, runtime_state_backend="redis"))
    store.enqueue_node_task(NodeTaskState(run_id="run_1", task_id="node_1"))
    claimed = store.claim_node_task("run_1", "node_1", "worker")
    completed = store.complete_node_task("run_1", "node_1", "completed", {"rows": 2})

    assert claimed is not None
    assert claimed.status == "running"
    assert completed.status == "completed"
    assert store.list_node_tasks("run_1")[0].payload["rows"] == 2


def test_semantic_governance_creates_snapshot_impact_and_rolls_back(tmp_path):
    settings = Settings(harness_workspace_path=str(tmp_path))
    topic_assets = TopicAssetService(settings)
    table_dir = topic_assets.table_asset_dir("电商交易", "ads_merchant_profile")
    table_dir.mkdir(parents=True, exist_ok=True)
    write_json(table_dir / "asset.json", {"topic": "电商交易", "tableName": "ads_merchant_profile", "metrics": [{"metricKey": "gmv", "sourceColumns": ["pay_amt"]}]})
    write_json(table_dir / "schema.json", [{"name": "seller_id", "type": "varchar"}, {"name": "pay_amt", "type": "decimal"}])
    write_json(table_dir / "semantic_version.json", {"semanticVersion": "old_version", "schemaVersion": "old_schema", "sourceHash": "old_hash"})
    pending_dir = settings.resolved_topic_path / "电商交易" / "pending" / "ads_merchant_profile"
    pending_dir.mkdir(parents=True, exist_ok=True)
    write_json(pending_dir / "asset.json", {"topic": "电商交易", "tableName": "ads_merchant_profile", "metrics": [{"metricKey": "gmv", "sourceColumns": ["pay_amt_new"]}]})
    write_json(pending_dir / "schema.json", [{"name": "seller_id", "type": "varchar"}, {"name": "pay_amt_new", "type": "decimal"}])

    repo = SimpleNamespace(show_full_columns=lambda _table: [{"Field": "seller_id", "Type": "varchar"}, {"Field": "pay_amt_new", "Type": "decimal"}])
    governance = SemanticAssetGovernanceService(settings, repo, topic_assets)

    preflight = governance.preflight_publish("电商交易", "ads_merchant_profile")
    impact = governance.impact_analysis("电商交易", "ads_merchant_profile")
    rollback = governance.rollback("电商交易", "ads_merchant_profile", version="old_version", reviewer="ops", reason="test")

    assert preflight["rollbackSnapshot"]["semanticVersion"] == "old_version"
    assert impact["status"] == "IMPACT_ANALYZED"
    assert rollback["status"] == "ROLLED_BACK"
    assert "asset.json" in rollback["restoredFiles"]


def test_answer_prompt_uses_verified_answer_context_contract():
    run_result = AgentRunResult(
        merged_query_bundle=QueryBundle(
            tables=["ads_merchant_profile"],
            rows=[{"seller_id": "100", "order_gmv_amt_1d": 188}],
        ),
        verified_evidence=VerifiedEvidence(passed=True),
        degraded_reasons=[{"code": "DRAFT_STRUCTURED_SQL_FALLBACK", "reason": "LLM unavailable"}],
    )
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="看GMV",
                plan_task_id="gmv",
                preferred_table="ads_merchant_profile",
                metric_name="order_gmv_amt_1d",
            )
        ]
    )

    context = verified_answer_context("看GMV", plan, run_result, merchant=MerchantInfo(merchant_id="100"))
    payload = answer_data_package("看GMV", plan, run_result, merchant=MerchantInfo(merchant_id="100"))

    assert context.verified_passed is True
    assert payload["verifiedPassed"] is True
    assert payload["degradedReasons"][0]["code"] == "DRAFT_STRUCTURED_SQL_FALLBACK"
    assert "taskResults" not in payload


def test_context_manifest_blocks_cache_layout_and_quarantine_policy():
    package = SimpleNamespace(
        agent="planner",
        stage="plan",
        package_id="ctx_1",
        context_hash="hash_1",
        allowed_tables=["ads_merchant_profile"],
        allowed_metrics=["gmv"],
        artifact_refs=[],
    )
    state = {
        "runtime_injection": {"stage": "plan", "merchant": {"merchantId": "100"}},
        "memory_injection": {"selectedMemoryIds": ["mem_1"], "items": [{"content": "prefer GMV"}]},
    }

    blocks = build_llm_context_blocks(state, package, {"trimmedSections": ["memoryInjection"]})
    layout = context_cache_layout(blocks)
    quarantine = context_quarantine_policy(blocks)

    assert [item["name"] for item in blocks][:2] == ["system_prompt", "semantic_catalog"]
    assert "system_prompt" in layout["stablePrefix"]
    assert "memory" in quarantine["dataOnlyBlocks"]
    assert any(item["name"] == "memory" and item["truncatedReason"] == "memoryInjection" for item in blocks)


def test_memory_write_gate_rejects_too_short_events_and_reviews_sensitive_events():
    gate = MemoryWriteGate()

    rejected = gate.evaluate({"eventId": "e1", "memoryType": "query_event", "question": "hi", "confidence": 0.9}, "100")
    review = gate.evaluate(
        {
            "eventId": "e2",
            "memoryType": "query_event",
            "question": "以后默认看GMV，联系人 a@example.com",
            "answerPreview": "",
            "confidence": 0.8,
            "scope": {"merchantId": "100"},
        },
        "100",
    )

    assert rejected["allowed"] is False
    assert "too_short_to_remember" in rejected["reasons"]
    assert review["allowed"] is True
    assert review["action"] == "review"
    assert "possible_sensitive_identifier" in review["reasons"]


def test_memory_ingestion_service_persists_write_policy_trace():
    class DummyStore(MemoryStore):
        def __init__(self):
            self.payload = {"merchantId": "100", "events": [], "preferences": [], "facts": []}

        def load(self, merchant_id):
            return dict(self.payload)

        def save(self, merchant_id, payload):
            self.payload = payload
            return payload

        def select_for_question(self, state, budget_tokens=0, budget_chars=0):
            return {}

        def update_from_state(self, state):
            return {}

    service = MemoryIngestionService(Settings(merchant_id="100"))
    result = service.update_store(DummyStore(), {"question": "hi", "requested_merchant_id": "100"})

    assert result["memoryIngestionTrace"]["written"] is False
    assert result["memoryIngestionTrace"]["writePolicy"]["allowed"] is False


def test_tool_registry_capability_and_result_contract_validation():
    descriptions = {"execute_sql": "execute SQL in Doris", "semantic_read": "read semantic asset"}
    registry = tool_registry_from_descriptions(descriptions)
    schemas = node_runtime_tool_schemas(descriptions, ["execute_sql"])

    invalid = validate_tool_result_contract("execute_sql", {"error": "missing rows"}, registry)
    valid = validate_tool_result_contract("execute_sql", {"rows": []}, registry)

    assert schemas[0]["capability"]["permission"] == "agent.sql.execute"
    assert invalid["valid"] is False
    assert invalid["missingKeys"] == ["rows"]
    assert valid["valid"] is True
    assert valid["resultHash"]


def test_tool_runtime_attaches_contract_and_result_hash(tmp_path):
    runtime = ToolRuntimeService(Settings(harness_workspace_path=str(tmp_path), cache_enabled=False), policy_registry=ToolRuntimePolicyRegistry(Settings()))

    result = runtime.execute("execute_sql", {"sql": "select 1"}, lambda _args: {"rows": [{"x": 1}]}, call_id="call_1")

    assert result.status == "success"
    assert result.contract["valid"] is True
    assert result.result_hash == result.contract["resultHash"]


def test_tool_runtime_fail_closes_high_risk_tool_contract_violation(tmp_path):
    runtime = ToolRuntimeService(Settings(harness_workspace_path=str(tmp_path), cache_enabled=False), policy_registry=ToolRuntimePolicyRegistry(Settings()))

    result = runtime.execute("execute_sql", {"sql": "select 1"}, lambda _args: {"ok": True}, call_id="call_1")

    assert result.status == "failed"
    assert result.error_type == "TOOL_CONTRACT_VIOLATION"
    assert result.recommended_action == "repair_tool_handler_or_degrade"
