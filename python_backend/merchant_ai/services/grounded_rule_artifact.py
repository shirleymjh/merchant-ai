from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from pydantic import Field

from merchant_ai.models import APIModel, RecallItem


class GroundedRuleEvidenceRef(APIModel):
    ref_id: str
    source_type: str
    title: str = ""
    topic: str = ""
    table: str = ""
    path: str = ""
    content_hash: str
    content: str


class GroundedVerifiedRuleArtifact(APIModel):
    artifact_id: str
    question: str
    goal_contract_fingerprint: str
    goal_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[GroundedRuleEvidenceRef] = Field(default_factory=list)
    rule_context: str = ""
    verification_passed: bool = False
    created_at: str = ""


def verified_rule_candidate_refs(
    *,
    core_semantic_evidence: Iterable[dict[str, Any]],
    recall_items: Iterable[RecallItem | dict[str, Any]],
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in core_semantic_evidence:
        evidence = _exact_rule_evidence(raw)
        if evidence is None or evidence.ref_id in seen:
            continue
        seen.add(evidence.ref_id)
        candidates.append(
            {
                "refId": evidence.ref_id,
                "sourceType": evidence.source_type,
                "title": evidence.title,
            }
        )
    for raw in recall_items:
        item = raw if isinstance(raw, RecallItem) else RecallItem.model_validate(raw)
        evidence = _recalled_rule_evidence(item)
        if evidence is None or evidence.ref_id in seen:
            continue
        seen.add(evidence.ref_id)
        candidates.append(
            {
                "refId": evidence.ref_id,
                "sourceType": evidence.source_type,
                "title": evidence.title,
            }
        )
    return candidates


def render_verified_rule_answer(
    artifacts: Sequence[GroundedVerifiedRuleArtifact],
) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for artifact in artifacts:
        if not artifact.verification_passed:
            continue
        for evidence in artifact.evidence_refs:
            include = True
            for raw_line in evidence.content.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("标题路径："):
                    continue
                if "常见测试问法" in line:
                    include = False
                    continue
                if not include:
                    continue
                normalized = line.lstrip("-* ").strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                lines.append(normalized)
                if len(lines) >= 8:
                    break
            if len(lines) >= 8:
                break
        if len(lines) >= 8:
            break
    if not lines:
        return "当前没有可验证的已发布规则依据，暂时不能给出确定处理办法。"
    if len(lines) == 1:
        return "根据已发布规则：%s" % lines[0]
    return "根据已发布规则：\n\n%s" % "\n".join(
        "- %s" % line for line in lines
    )


def build_verified_rule_artifact(
    *,
    question: str,
    goal_contract_fingerprint: str,
    goal_ids: Sequence[str],
    requested_ref_ids: Sequence[str],
    core_semantic_evidence: Iterable[dict[str, Any]],
    recall_items: Iterable[RecallItem | dict[str, Any]],
) -> GroundedVerifiedRuleArtifact:
    """Build a rule artifact only from Kernel-observed reads or recall.

    The Core selects identifiers, never rule text.  Exact Topic rule leaves
    must have been completely read by the root Core.  Governed rule snippets
    must come from the current session recall and carry published status.
    """

    requested = _dedupe_strings(requested_ref_ids)
    if not requested:
        raise ValueError("rule artifact requires at least one selected evidence ref")

    available: dict[str, GroundedRuleEvidenceRef] = {}
    for raw in core_semantic_evidence:
        evidence = _exact_rule_evidence(raw)
        if evidence is not None:
            available[evidence.ref_id] = evidence
    for raw in recall_items:
        item = raw if isinstance(raw, RecallItem) else RecallItem.model_validate(raw)
        evidence = _recalled_rule_evidence(item)
        if evidence is not None:
            available[evidence.ref_id] = evidence

    missing = [ref_id for ref_id in requested if ref_id not in available]
    if missing:
        raise ValueError(
            "selected rule refs are not verified in the current session: %s"
            % ",".join(missing[:8])
        )
    selected = [available[ref_id] for ref_id in requested]
    context = "\n\n".join(
        "[%s] %s\n%s"
        % (
            evidence.ref_id,
            evidence.title or evidence.path or evidence.source_type,
            evidence.content,
        )
        for evidence in selected
    )[:16_000]
    if not context.strip():
        raise ValueError("verified rule evidence has no usable content")

    normalized_goal_ids = _dedupe_strings(goal_ids)
    fingerprint_payload = {
        "question": str(question or "").strip(),
        "goalContractFingerprint": str(goal_contract_fingerprint or "").strip(),
        "goalIds": normalized_goal_ids,
        "evidence": [
            {
                "refId": evidence.ref_id,
                "contentHash": evidence.content_hash,
            }
            for evidence in selected
        ],
    }
    artifact_id = "rule_artifact_%s" % hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    return GroundedVerifiedRuleArtifact(
        artifact_id=artifact_id,
        question=str(question or "").strip(),
        goal_contract_fingerprint=str(goal_contract_fingerprint or "").strip(),
        goal_ids=normalized_goal_ids,
        evidence_refs=selected,
        rule_context=context,
        verification_passed=True,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _exact_rule_evidence(raw: Any) -> GroundedRuleEvidenceRef | None:
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("kind") or "").strip().upper()
    if kind not in {"BUSINESS_RULE", "RULE"} or not bool(
        raw.get("contentComplete")
    ):
        return None
    ref_id = str(raw.get("refId") or "").strip()
    content = str(raw.get("contentSnippet") or "").strip()
    if not ref_id or not content:
        return None
    rule_text, title = _rule_text_from_exact_content(content)
    if not rule_text:
        return None
    content_hash = str(raw.get("contentHash") or "").strip() or hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()
    return GroundedRuleEvidenceRef(
        ref_id=ref_id,
        source_type="TOPIC_RULE_LEAF",
        title=title,
        topic=str(raw.get("topic") or ""),
        table=str(raw.get("table") or ""),
        path=str(raw.get("path") or ""),
        content_hash=content_hash,
        content=rule_text,
    )


def _recalled_rule_evidence(item: RecallItem) -> GroundedRuleEvidenceRef | None:
    if str(item.source_type or "").strip().upper() != "GOVERNED_RULE":
        return None
    metadata = dict(item.metadata or {})
    status = str(metadata.get("status") or "PUBLISHED").strip().upper()
    if status != "PUBLISHED":
        return None
    visibility = metadata.get("visibilityPolicy") or {}
    if isinstance(visibility, dict) and str(
        visibility.get("level") or "public"
    ).strip().lower() not in {"public", "merchant"}:
        return None
    content = str(item.content or "").strip()
    if not content:
        return None
    ref_id = _governed_rule_ref_id(item)
    return GroundedRuleEvidenceRef(
        ref_id=ref_id,
        source_type="GOVERNED_RULE",
        title=str(item.title or metadata.get("headingText") or ""),
        topic=str(item.topic or ""),
        table=str(item.table or ""),
        path=str(metadata.get("semanticPath") or metadata.get("sourcePath") or ""),
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        content=content,
    )


def _governed_rule_ref_id(item: RecallItem) -> str:
    metadata = dict(item.metadata or {})
    semantic_ref = str(metadata.get("semanticRefId") or "").strip()
    if semantic_ref.startswith("semantic:rules:"):
        return semantic_ref
    doc_id = str(item.doc_id or "").strip()
    if doc_id.startswith("semantic:rules:"):
        return doc_id
    path = str(metadata.get("semanticPath") or metadata.get("sourcePath") or "rule")
    heading = str(metadata.get("headingPath") or metadata.get("headingText") or "")
    chunk_index = int(metadata.get("chunkIndex") or 0)
    digest = hashlib.sha256(
        ("%s:%s:%d" % (path, heading, chunk_index)).encode("utf-8")
    ).hexdigest()[:16]
    return "semantic:rules:published:%s" % digest


def _rule_text_from_exact_content(content: str) -> tuple[str, str]:
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return content.strip(), ""
    if not isinstance(payload, dict):
        return content.strip(), ""
    definition = payload.get("definition")
    source = definition if isinstance(definition, dict) else payload
    title = str(source.get("title") or payload.get("key") or "").strip()
    text = str(
        source.get("content")
        or source.get("rule")
        or source.get("description")
        or ""
    ).strip()
    return text, title


def _dedupe_strings(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result
