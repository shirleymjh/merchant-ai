from __future__ import annotations

from threading import RLock

from merchant_ai.models import RecallItem
from merchant_ai.services.grounded_rule_artifact import (
    build_verified_rule_artifact,
    render_verified_rule_answer,
    verified_rule_candidate_refs,
)
from merchant_ai.services.grounded_runtime_kernel import (
    GroundedRuntimeKernel,
    GroundedRuntimeSession,
)


def governed_rule() -> RecallItem:
    return RecallItem(
        doc_id="semantic:rules:platform_rules:chunk:0001",
        title="平台规则 / 商品与上架",
        content="商品审核被拒后，应优先查看拒绝原因、类目、品牌资质和图文描述。",
        source_type="GOVERNED_RULE",
        metadata={
            "status": "PUBLISHED",
            "semanticPath": "rules/platform_rules.md",
            "headingPath": ["商品与上架"],
            "chunkIndex": 1,
            "visibilityPolicy": {"level": "public"},
        },
    )


def test_builds_rule_artifact_only_from_current_verified_recall() -> None:
    rule = governed_rule()

    artifact = build_verified_rule_artifact(
        question="商品审核被拒怎么办？",
        goal_contract_fingerprint="f" * 64,
        goal_ids=["g_rule"],
        requested_ref_ids=[rule.doc_id],
        core_semantic_evidence=[],
        recall_items=[rule],
    )

    assert artifact.verification_passed is True
    assert artifact.goal_ids == ["g_rule"]
    assert artifact.evidence_refs[0].ref_id == rule.doc_id
    assert "品牌资质" in artifact.rule_context


def test_accepts_completely_read_topic_rule_leaf() -> None:
    ref_id = "semantic:商品管理:dwm_goods_detail_df:rule:商品审核明细口径"
    artifact = build_verified_rule_artifact(
        question="商品审核拒绝怎么识别？",
        goal_contract_fingerprint="f" * 64,
        goal_ids=["g_rule"],
        requested_ref_ids=[ref_id],
        core_semantic_evidence=[
            {
                "refId": ref_id,
                "kind": "BUSINESS_RULE",
                "topic": "商品管理",
                "table": "dwm_goods_detail_df",
                "path": "topics/商品管理/tables/dwm_goods_detail_df/rules/商品审核明细口径.json",
                "contentComplete": True,
                "contentHash": "a" * 64,
                "contentSnippet": (
                    '{"definition":{"title":"商品审核明细口径",'
                    '"content":"审核拒绝按 is_audit_pass = 0 识别。"}}'
                ),
            }
        ],
        recall_items=[],
    )

    assert artifact.evidence_refs[0].source_type == "TOPIC_RULE_LEAF"
    assert "is_audit_pass" in artifact.rule_context


def test_rejects_core_authored_or_unpublished_rule_text() -> None:
    unpublished = governed_rule().model_copy(
        update={"metadata": {"status": "PENDING_REVIEW"}}
    )

    try:
        build_verified_rule_artifact(
            question="商品审核被拒怎么办？",
            goal_contract_fingerprint="f" * 64,
            goal_ids=["g_rule"],
            requested_ref_ids=[unpublished.doc_id],
            core_semantic_evidence=[],
            recall_items=[unpublished],
        )
    except ValueError as exc:
        assert "not verified" in str(exc)
    else:
        raise AssertionError("unpublished rule evidence must fail closed")


def test_rule_artifact_id_is_stable_for_same_verified_evidence() -> None:
    rule = governed_rule()
    kwargs = {
        "question": "商品审核被拒怎么办？",
        "goal_contract_fingerprint": "f" * 64,
        "goal_ids": ["g_rule"],
        "requested_ref_ids": [rule.doc_id],
        "core_semantic_evidence": [],
        "recall_items": [rule],
    }

    first = build_verified_rule_artifact(**kwargs)
    second = build_verified_rule_artifact(**kwargs)

    assert first.artifact_id == second.artifact_id


def test_candidate_refs_and_rendered_answer_never_use_core_authored_text() -> None:
    rule = governed_rule()
    candidates = verified_rule_candidate_refs(
        core_semantic_evidence=[],
        recall_items=[rule],
    )
    artifact = build_verified_rule_artifact(
        question="商品审核被拒怎么办？",
        goal_contract_fingerprint="f" * 64,
        goal_ids=["g_rule"],
        requested_ref_ids=[candidates[0]["refId"]],
        core_semantic_evidence=[],
        recall_items=[rule],
    )

    answer = render_verified_rule_answer([artifact])

    assert "优先查看拒绝原因" in answer
    assert "重新提交" not in answer


def test_kernel_publishes_rule_artifact_and_composes_terminal_answer() -> None:
    rule = governed_rule()
    artifact = build_verified_rule_artifact(
        question="商品审核被拒怎么办？",
        goal_contract_fingerprint="f" * 64,
        goal_ids=["g_rule"],
        requested_ref_ids=[rule.doc_id],
        core_semantic_evidence=[],
        recall_items=[rule],
    )
    kernel = object.__new__(GroundedRuntimeKernel)
    kernel._lock = RLock()
    session = GroundedRuntimeSession(
        session_id="rule-session",
        question="商品审核被拒怎么办？",
        merchant_id="100",
    )

    published = kernel.publish_rule_artifact(session, artifact)
    answer = kernel.compose_rule_answer(
        session,
        artifact_ids=[published.artifact_id],
    )

    assert session.phase == "ANSWERED"
    assert session.answer_rule_artifact_ids == [artifact.artifact_id]
    assert "优先查看拒绝原因" in answer
