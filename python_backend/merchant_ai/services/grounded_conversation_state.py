from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

from merchant_ai.config import Settings
from merchant_ai.services.time_semantics import (
    has_explicit_time_expression,
    resolve_time_range,
)


GROUNDED_CONVERSATION_STATE_VERSION = "grounded_conversation_state.v1"


class GroundedConversationStateError(RuntimeError):
    """Base error for durable Grounded conversation state."""


class GroundedConversationStateCorruptError(GroundedConversationStateError):
    """The persisted state cannot be trusted or restored."""


class GroundedConversationStateConflictError(GroundedConversationStateError):
    """The state changed after the caller loaded it."""


@dataclass(frozen=True)
class GroundedConversationState:
    thread_id: str
    snapshot: dict[str, Any]
    revision: int
    updated_at: str
    version: str = GROUNDED_CONVERSATION_STATE_VERSION


StateUpdater = Callable[[Optional[dict[str, Any]]], Mapping[str, Any]]


_REFERENCE_PATTERN = re.compile(
    r"(?:这里面|这里边|其中|上述|前面(?:的|提到的)?|刚才(?:的|查到的|那些)?|"
    r"上一轮|上一次|上面(?:的)?|该(?:批|组|范围|结果|数据|订单)|"
    r"这(?:些|几)(?:个|笔|条|张|单|订单|商品|退款|记录|结果|数据)?|"
    r"那(?:些|几)(?:个|笔|条|张|单|订单|商品|退款|记录|结果|数据)?|"
    r"在此基础上|沿用(?:这个|上述|前面)?范围|"
    r"\b(?:these|those|them|the above|previous results?|same (?:set|scope|period)|"
    r"within (?:that|those|the results?)|among (?:these|those|them))\b)",
    re.IGNORECASE,
)
_IMPLICIT_CONTINUATION_PATTERN = re.compile(
    r"^(?:再|然后|接着|继续|还有|顺便|另外).{0,80}$|.{1,80}(?:呢|又如何|怎么样)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GroundedConversationResolution:
    """Server-side interpretation of one turn before Topic routing.

    The effective question may inherit only constraints retained from a
    verified server-owned scope (or, during rollout, an explicit time phrase
    from server-reconstructed user history). Assistant prose and client tool
    messages are never treated as scope authority.
    """

    original_question: str
    effective_question: str
    status: str = "STANDALONE"
    reference_detected: bool = False
    reference_phrases: tuple[str, ...] = ()
    antecedent_question: str = ""
    inherited_time_expression: str = ""
    inherited_filters: tuple[str, ...] = ()
    source_revision: int = 0
    source_artifact_ids: tuple[str, ...] = ()
    source: str = ""
    clarification_question: str = ""
    clarification_options: tuple[str, ...] = ()
    reference_contract: "GroundedReferenceContractResolution" = field(
        default_factory=lambda: GroundedReferenceContractResolution()
    )

    @property
    def needs_clarification(self) -> bool:
        return bool(self.clarification_question)

    def trace(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "referenceDetected": self.reference_detected,
            "referencePhrases": list(self.reference_phrases),
            "antecedentQuestion": self.antecedent_question,
            "inheritedTimeExpression": self.inherited_time_expression,
            "inheritedFilters": list(self.inherited_filters),
            "sourceRevision": self.source_revision,
            "sourceArtifactIds": list(self.source_artifact_ids),
            "source": self.source,
            "originalQuestion": self.original_question,
            "effectiveQuestion": self.effective_question,
            "needsClarification": self.needs_clarification,
            "referenceContract": self.reference_contract.trace(),
        }


@dataclass(frozen=True)
class GroundedReferenceContractResolution:
    """Typed meaning of a cross-turn BI reference.

    A conversational antecedent is not always a row population.  It can be a
    predicate scope, an exact entity set, a result artifact, a metric value or
    a comparison baseline.  Keeping the distinction here prevents downstream
    query planning from treating every phrase such as ``这里面`` as an ``IN``
    list or as a repeated time filter.
    """

    status: str = "NONE"
    referent_type: str = "NONE"
    downstream_operation: str = "UNSPECIFIED"
    source_artifact_id: str = ""
    source_contract_fingerprint: str = ""
    source_sql_fingerprint: str = ""
    source_query_shape: str = ""
    source_topics: tuple[str, ...] = ()
    source_tables: tuple[str, ...] = ()
    source_goal_ids: tuple[str, ...] = ()
    source_entity_identities: tuple[str, ...] = ()
    source_data_grains: tuple[str, ...] = ()
    coverage_status: str = "UNKNOWN"
    snapshot_semantics: str = "ABSOLUTE_PREDICATE_SNAPSHOT"
    population_required: bool = False
    complete_membership_required: bool = False
    membership_handle_type: str = ""
    membership_handle_id: str = ""
    membership_values_hash: str = ""
    reason: str = ""

    @property
    def bound(self) -> bool:
        return self.status == "BOUND" and bool(self.referent_type)

    def trace(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "referentType": self.referent_type,
            "downstreamOperation": self.downstream_operation,
            "sourceArtifactId": self.source_artifact_id,
            "sourceContractFingerprint": self.source_contract_fingerprint,
            "sourceSqlFingerprint": self.source_sql_fingerprint,
            "sourceQueryShape": self.source_query_shape,
            "sourceTopics": list(self.source_topics),
            "sourceTables": list(self.source_tables),
            "sourceGoalIds": list(self.source_goal_ids),
            "sourceEntityIdentities": list(self.source_entity_identities),
            "sourceDataGrains": list(self.source_data_grains),
            "coverageStatus": self.coverage_status,
            "snapshotSemantics": self.snapshot_semantics,
            "populationRequired": self.population_required,
            "completeMembershipRequired": self.complete_membership_required,
            "membershipHandleType": self.membership_handle_type,
            "membershipHandleId": self.membership_handle_id,
            "membershipValuesHash": self.membership_values_hash,
            "reason": self.reason,
        }


def grounded_conversation_principal_fingerprint(
    merchant_id: str,
    user_scope: Mapping[str, Any] | None,
) -> str:
    """Fingerprint the authorization scope that is allowed to reuse a state."""

    scope = dict(user_scope or {})
    payload = {
        "merchantId": str(merchant_id or scope.get("merchantId") or "").strip(),
        "userId": str(scope.get("userId") or scope.get("user_id") or "").strip(),
        "role": str(scope.get("role") or "").strip(),
        "region": str(scope.get("region") or "").strip(),
        "storeIds": sorted(
            {
                str(item).strip()
                for item in (scope.get("storeIds") or scope.get("store_ids") or [])
                if str(item).strip()
            }
        ),
        "permissions": sorted(
            {
                str(item).strip()
                for item in (scope.get("permissions") or [])
                if str(item).strip()
            }
        ),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resolve_grounded_conversation_turn(
    question: str,
    *,
    persisted_snapshot: Mapping[str, Any] | None = None,
    persisted_revision: int = 0,
    message_history: Sequence[Any] | None = None,
    request_context: Any = None,
) -> GroundedConversationResolution:
    """Resolve clarification replies and cross-turn references before routing.

    A reference such as ``这里面`` inherits the complete prior query scope,
    never the visible preview rows. If the previous turn contained multiple
    incompatible result scopes, the resolver asks the user instead of choosing
    one silently.
    """

    original = str(question or "").strip()
    if not original:
        raise ValueError("conversation question is required")
    snapshot = dict(persisted_snapshot or {})
    pending = _pending_clarification(snapshot, request_context)
    if pending:
        pending_question = str(pending.get("pendingQuestion") or "").strip()
        if pending_question:
            effective = "%s\n\n用户对上一轮澄清的回答：%s" % (pending_question, original)
            return GroundedConversationResolution(
                original_question=original,
                effective_question=effective,
                status="CLARIFICATION_RESUMED",
                antecedent_question=pending_question,
                source_revision=max(0, int(persisted_revision or 0)),
                source="SERVER_PENDING_CLARIFICATION",
            )

    explicit_matches = tuple(dict.fromkeys(match.group(0) for match in _REFERENCE_PATTERN.finditer(original)))
    implicit = bool(_IMPLICIT_CONTINUATION_PATTERN.search(original))
    reference_detected = bool(explicit_matches or implicit)
    if not reference_detected:
        return GroundedConversationResolution(
            original_question=original,
            effective_question=original,
        )

    active_scope = dict(snapshot.get("activeScope") or {})
    antecedent = str(
        (snapshot.get("lastTurn") or {}).get("originalQuestion")
        or _latest_prior_user_question(message_history or (), original)
        or ""
    ).strip()
    result_sets = _unique_result_sets(
        item
        for item in active_scope.get("resultSets") or []
        if isinstance(item, Mapping)
    )
    selected_result_set = _select_reference_result_set(original, result_sets)
    active_artifact_ids = _dedupe_strings(active_scope.get("artifactIds") or [])
    if (
        selected_result_set is not None
        and not str(selected_result_set.get("queryArtifactId") or "").strip()
        and len(active_artifact_ids) == 1
    ):
        selected_result_set = {
            **selected_result_set,
            "queryArtifactId": active_artifact_ids[0],
        }
    if bool(active_scope.get("ambiguous")) or (
        result_sets and selected_result_set is None
    ):
        labels = _result_scope_labels(result_sets)
        return GroundedConversationResolution(
            original_question=original,
            effective_question=original,
            status="AMBIGUOUS_REFERENCE",
            reference_detected=True,
            reference_phrases=explicit_matches,
            antecedent_question=antecedent,
            source_revision=max(0, int(persisted_revision or 0)),
            source="VERIFIED_CONVERSATION_SCOPE",
            clarification_question=(
                "你说的“%s”可能对应上一轮的多个结果范围，请确认要基于哪一个继续。"
                % (explicit_matches[0] if explicit_matches else "上一轮结果")
            ),
            clarification_options=tuple(labels[:4]),
            reference_contract=GroundedReferenceContractResolution(
                status="AMBIGUOUS",
                referent_type="AMBIGUOUS",
                downstream_operation=_classify_downstream_operation(original),
                reason="MULTIPLE_VERIFIED_RESULT_ARTIFACTS",
            ),
        )

    reference_contract = _resolve_reference_contract(
        original,
        selected_result_set,
    )
    if reference_contract.status == "UNSUPPORTED":
        label = _result_scope_labels([selected_result_set] if selected_result_set else [])
        return GroundedConversationResolution(
            original_question=original,
            effective_question=original,
            status="AMBIGUOUS_REFERENCE",
            reference_detected=True,
            reference_phrases=explicit_matches,
            antecedent_question=antecedent,
            source_revision=max(0, int(persisted_revision or 0)),
            source="VERIFIED_CONVERSATION_SCOPE",
            clarification_question=(
                "上一轮结果不能安全地作为这次分析的完整对象（%s）。"
                "请确认要沿用上一轮筛选范围，还是只分析已展示的结果。"
                % (reference_contract.reason or "结果范围不完整")
            ),
            clarification_options=tuple(
                [
                    "沿用上一轮完整筛选范围",
                    "只分析上一轮已展示结果",
                    *label[:1],
                ]
            ),
            reference_contract=reference_contract,
        )

    time_expressions = _dedupe_strings(active_scope.get("timeExpressions") or [])
    filters = _dedupe_strings(active_scope.get("filterSummaries") or [])
    artifact_ids = _dedupe_strings(
        [
            str((selected_result_set or {}).get("queryArtifactId") or "")
        ]
        if selected_result_set is not None
        else active_artifact_ids
    )
    source = "VERIFIED_CONVERSATION_SCOPE" if active_scope else ""
    if not time_expressions and not filters and not reference_contract.bound:
        history_time = _explicit_time_from_question(antecedent)
        if history_time:
            time_expressions = [history_time]
            source = "SERVER_USER_HISTORY"

    if len(time_expressions) > 1 and not has_explicit_time_expression(original):
        return GroundedConversationResolution(
            original_question=original,
            effective_question=original,
            status="AMBIGUOUS_REFERENCE",
            reference_detected=True,
            reference_phrases=explicit_matches,
            antecedent_question=antecedent,
            source_revision=max(0, int(persisted_revision or 0)),
            source=source,
            clarification_question="上一轮包含多个时间范围，请确认这次要沿用哪一个。",
            clarification_options=tuple(time_expressions[:4]),
            reference_contract=reference_contract,
        )
    if not time_expressions and not filters and not reference_contract.bound:
        return GroundedConversationResolution(
            original_question=original,
            effective_question=original,
            status="AMBIGUOUS_REFERENCE",
            reference_detected=True,
            reference_phrases=explicit_matches,
            antecedent_question=antecedent,
            source_revision=max(0, int(persisted_revision or 0)),
            source=source or "NO_VERIFIED_SCOPE",
            clarification_question=(
                "我无法安全确定“%s”指的是哪个数据范围。请补充时间范围或筛选条件。"
                % (explicit_matches[0] if explicit_matches else "这个追问")
            ),
            reference_contract=reference_contract,
        )

    inherited_time = "" if has_explicit_time_expression(original) else (time_expressions[0] if time_expressions else "")
    constraints: list[str] = []
    if inherited_time:
        constraints.append("时间范围=%s" % inherited_time)
    constraints.extend(filters[:12])
    scope_text = "；".join(constraints)
    effective = original
    if scope_text:
        effective = (
            "%s\n\n【服务端已解析的上一轮数据范围】%s。"
            "必须复用上一轮完整查询范围，不得把预览行或截断行误当作完整集合。"
        ) % (original, scope_text)
    return GroundedConversationResolution(
        original_question=original,
        effective_question=effective,
        status="RESOLVED_REFERENCE",
        reference_detected=True,
        reference_phrases=explicit_matches,
        antecedent_question=antecedent,
        inherited_time_expression=inherited_time,
        inherited_filters=tuple(filters[:12]),
        source_revision=max(0, int(persisted_revision or 0)),
        source_artifact_ids=tuple(artifact_ids),
        source=source,
        reference_contract=reference_contract,
    )


def _pending_clarification(snapshot: Mapping[str, Any], request_context: Any) -> dict[str, Any]:
    pending = snapshot.get("pendingClarification")
    if isinstance(pending, Mapping) and str(pending.get("pendingQuestion") or "").strip():
        return dict(pending)
    if request_context is None:
        return {}
    question = str(getattr(request_context, "pending_question", "") or "").strip()
    stage = str(getattr(request_context, "pending_clarification_stage", "") or "").strip()
    if not question or not stage:
        return {}
    return {
        "pendingQuestion": question,
        "stage": stage,
        "type": str(getattr(request_context, "pending_clarification_type", "") or ""),
        "options": list(getattr(request_context, "pending_clarification_options", None) or []),
    }


def _latest_prior_user_question(messages: Sequence[Any], current_question: str) -> str:
    current = str(current_question or "").strip()
    for item in reversed(list(messages or [])):
        if isinstance(item, Mapping):
            role = str(item.get("role") or "").lower()
            text = str(item.get("text") or item.get("content") or "").strip()
        else:
            role = str(getattr(item, "role", "") or "").lower()
            text = str(getattr(item, "text", "") or "").strip()
        if role != "user" or not text or text == current:
            continue
        return text
    return ""


def _explicit_time_from_question(question: str) -> str:
    text = str(question or "").strip()
    if not text or not has_explicit_time_expression(text):
        return ""
    resolved = resolve_time_range(text)
    return str(resolved.label or "").strip()


def _incompatible_result_scopes(result_sets: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether a bare reference has more than one artifact antecedent.

    Equal time/topic/filter predicates do *not* make two result artifacts the
    same referent.  A DETAIL order set and a RANKED refund set can share all of
    those attributes while having different entity, grain and row membership.
    """

    return len(_unique_result_sets(result_sets)) > 1


def _unique_result_sets(
    result_sets: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in result_sets:
        item = dict(raw)
        identity = str(item.get("queryArtifactId") or "").strip()
        if not identity:
            identity = str(item.get("contractFingerprint") or "").strip()
        if not identity:
            identity = hashlib.sha256(
                json.dumps(
                    item,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(item)
    return unique


def _select_reference_result_set(
    question: str,
    result_sets: Sequence[Mapping[str, Any]],
) -> Optional[dict[str, Any]]:
    if not result_sets:
        return None
    if len(result_sets) == 1:
        return dict(result_sets[0])
    text = str(question or "").strip().casefold()
    ordinal_patterns = [
        (0, r"(?:第\s*(?:一|1)\s*(?:个|张|组|份)|\bfirst\b)"),
        (1, r"(?:第\s*(?:二|2|两)\s*(?:个|张|组|份)|\bsecond\b)"),
        (2, r"(?:第\s*(?:三|3)\s*(?:个|张|组|份)|\bthird\b)"),
    ]
    for index, pattern in ordinal_patterns:
        if index < len(result_sets) and re.search(pattern, text, re.IGNORECASE):
            return dict(result_sets[index])

    matched: list[dict[str, Any]] = []
    for raw in result_sets:
        item = dict(raw)
        candidates = _dedupe_strings(
            [
                item.get("label"),
                item.get("queryShape"),
                *(item.get("topics") or []),
                *(item.get("tables") or []),
                *(item.get("entityIdentities") or []),
            ]
        )
        # Require a meaningful business token.  Matching generic shape words
        # such as DETAIL or RANKED would make a bare reference accidentally
        # select an artifact.
        if any(
            len(candidate) >= 2
            and candidate.casefold() not in {"detail", "ranked", "grouped", "scalar", "trend"}
            and candidate.casefold() in text
            for candidate in candidates
        ):
            matched.append(item)
    return matched[0] if len(matched) == 1 else None


def _classify_downstream_operation(question: str) -> str:
    text = str(question or "").strip()
    patterns = [
        ("RANK", r"(?:最多|最少|最高|最低|排名|排行|前\s*[一二三四五六七八九十\d]+|后\s*[一二三四五六七八九十\d]+|\btop\s*\d*|\bbottom\s*\d*)"),
        ("DETAIL_DRILLDOWN", r"(?:明细|详情|逐笔|逐单|列出|展开|下钻|drill\s*down|details?)"),
        ("TREND", r"(?:趋势|走势|按[日周月季年]|变化曲线|trend|over\s+time)"),
        ("COMPARE", r"(?:同比|环比|对比|比较|差异|相比|versus|\bvs\.?\b|compare)"),
        ("EXPLAIN", r"(?:为什么|为何|原因|根因|归因|异常|怎么回事|why|root\s*cause)"),
        ("AGGREGATE", r"(?:多少|合计|总计|总额|数量|平均|占比|汇总|aggregate|total|average|share)"),
    ]
    for operation, pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return operation
    return "FOLLOW_UP"


def _resolve_reference_contract(
    question: str,
    result_set: Optional[Mapping[str, Any]],
) -> GroundedReferenceContractResolution:
    operation = _classify_downstream_operation(question)
    if result_set is None:
        return GroundedReferenceContractResolution(
            status="UNVERIFIED_HISTORY",
            referent_type="PREDICATE_SCOPE",
            downstream_operation=operation,
            population_required=_operation_requires_population(operation),
            reason="NO_VERIFIED_RESULT_ARTIFACT",
        )

    item = dict(result_set)
    shape = str(item.get("queryShape") or "").strip().upper()
    coverage = str(
        item.get("coverageStatus")
        or item.get("resultCoverage")
        or item.get("coverage")
        or ("TOP_N" if shape == "RANKED" else "UNKNOWN")
    ).strip().upper()
    question_text = str(question or "")
    population_required = _operation_requires_population(operation)
    explicit_scope_reference = bool(
        re.search(
            r"(?:范围|筛选条件|时间段|期间|这批数据|same\s+(?:scope|range)|within\s+that\s+range)",
            question_text,
            re.IGNORECASE,
        )
    )
    scalar_reference = bool(
        re.search(r"(?:这个数|该指标|这个指标|这个值|that\s+(?:number|metric|value))", question_text, re.IGNORECASE)
    )
    if scalar_reference or shape == "SCALAR":
        referent_type = "METRIC_VALUE"
    elif explicit_scope_reference or shape == "DETAIL":
        # DETAIL commonly renders only a preview.  Its verified query Contract
        # still defines the complete predicate scope, so reuse that scope and
        # never turn visible rows into the population by accident.
        referent_type = "PREDICATE_SCOPE"
    elif shape == "ENTITY_LOOKUP":
        referent_type = "ENTITY_SET"
    elif shape in {"RANKED", "GROUPED", "TREND"}:
        referent_type = "RESULT_ARTIFACT"
    else:
        referent_type = "RESULT_ARTIFACT"

    complete_membership_required = bool(
        population_required and referent_type in {"ENTITY_SET", "RESULT_ARTIFACT"}
    )
    complete_coverages = {"ALL_ROWS", "TOP_N", "COMPLETE", "EXACT_ENTITY_SET"}
    membership_handle_type = str(
        item.get("membershipHandleType")
        or ("VERIFIED_ENTITY_SET" if item.get("entitySetArtifactId") else "")
        or ("RESULT_RELATION" if item.get("resultRelationHandle") else "")
    ).strip().upper()
    membership_handle_id = str(
        item.get("membershipHandleId")
        or item.get("entitySetArtifactId")
        or item.get("resultRelationHandle")
        or ""
    ).strip()
    membership_values_hash = str(item.get("membershipValuesHash") or "").strip()
    status = "BOUND"
    reason = "VERIFIED_ARTIFACT_REFERENCE"
    if referent_type == "METRIC_VALUE" and population_required:
        status = "UNSUPPORTED"
        reason = "SCALAR_DOES_NOT_DEFINE_A_ROW_POPULATION"
    elif complete_membership_required and coverage not in complete_coverages:
        status = "UNSUPPORTED"
        reason = "REFERENCED_RESULT_MEMBERSHIP_NOT_COMPLETE"
    elif complete_membership_required and not membership_handle_id:
        status = "UNSUPPORTED"
        reason = "REFERENCE_MEMBERSHIP_HANDLE_MISSING"

    entity_identities = item.get("entityIdentities") or item.get(
        "outputEntityIdentities"
    ) or []
    if isinstance(entity_identities, Mapping):
        entity_identities = list(entity_identities.values())
    return GroundedReferenceContractResolution(
        status=status,
        referent_type=referent_type,
        downstream_operation=operation,
        source_artifact_id=str(item.get("queryArtifactId") or "").strip(),
        source_contract_fingerprint=str(
            item.get("contractFingerprint") or ""
        ).strip(),
        source_sql_fingerprint=str(item.get("sqlFingerprint") or "").strip(),
        source_query_shape=shape,
        source_topics=tuple(_dedupe_strings(item.get("topics") or [])),
        source_tables=tuple(_dedupe_strings(item.get("tables") or [])),
        source_goal_ids=tuple(_dedupe_strings(item.get("goalIds") or [])),
        source_entity_identities=tuple(_dedupe_strings(entity_identities)),
        source_data_grains=tuple(_dedupe_strings(item.get("dataGrains") or [])),
        coverage_status=coverage,
        snapshot_semantics=str(
            item.get("snapshotSemantics") or "ABSOLUTE_PREDICATE_SNAPSHOT"
        ),
        population_required=population_required,
        complete_membership_required=complete_membership_required,
        membership_handle_type=membership_handle_type,
        membership_handle_id=membership_handle_id,
        membership_values_hash=membership_values_hash,
        reason=reason,
    )


def _operation_requires_population(operation: str) -> bool:
    return str(operation or "").upper() in {
        "RANK",
        "DETAIL_DRILLDOWN",
        "TREND",
        "COMPARE",
        "EXPLAIN",
        "AGGREGATE",
        "FOLLOW_UP",
    }


def _result_scope_labels(result_sets: Sequence[Mapping[str, Any]]) -> list[str]:
    labels: list[str] = []
    for index, item in enumerate(result_sets, start=1):
        label = str(item.get("label") or item.get("queryShape") or "").strip()
        times = _dedupe_strings(item.get("timeExpressions") or [])
        topics = _dedupe_strings(item.get("topics") or [])
        pieces = [label, "/".join(topics), "/".join(times)]
        rendered = " · ".join(piece for piece in pieces if piece)
        labels.append(rendered or "上一轮结果 %d" % index)
    return list(dict.fromkeys(labels))


def _dedupe_strings(values: Iterable[Any]) -> list[str]:
    return list(
        dict.fromkeys(
            str(item).strip()
            for item in values
            if str(item).strip()
        )
    )


class GroundedConversationStateStore:
    """Durable, thread-scoped business state for the Grounded runtime.

    Each thread is stored in one versioned JSON envelope under the harness
    workspace. Writes use a same-directory temporary file and ``os.replace``
    so readers observe either the old or the new complete snapshot.

    ``locked(thread_id)`` can wrap an entire load -> execute -> save business
    transaction. Store methods are re-entrant inside that context for the same
    thread. The lock combines an in-process ``RLock`` with ``flock`` so separate
    store instances and worker processes sharing the workspace are serialized.
    """

    _registry_lock = threading.Lock()
    _thread_mutexes: dict[str, threading.RLock] = {}
    _held_file_locks = threading.local()

    def __init__(self, settings: Settings, root: Path | None = None):
        self.root = Path(root) if root is not None else settings.resolved_workspace_path / "grounded_conversation_state"
        self.threads_dir = self.root / "threads"
        self.locks_dir = self.root / "locks"
        self.threads_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir.mkdir(parents=True, exist_ok=True)

    def load(self, thread_id: str) -> Optional[GroundedConversationState]:
        normalized_thread_id = self._normalize_thread_id(thread_id)
        with self._thread_file_lock(normalized_thread_id, exclusive=False):
            return self._load_unlocked(normalized_thread_id)

    def load_snapshot(self, thread_id: str) -> Optional[dict[str, Any]]:
        state = self.load(thread_id)
        return state.snapshot if state is not None else None

    def save(
        self,
        thread_id: str,
        snapshot: Mapping[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> GroundedConversationState:
        normalized_thread_id = self._normalize_thread_id(thread_id)
        normalized_snapshot = self._normalize_snapshot(snapshot)
        with self._thread_file_lock(normalized_thread_id, exclusive=True):
            current = self._load_unlocked(normalized_thread_id)
            self._check_expected_revision(current, expected_revision)
            return self._save_unlocked(normalized_thread_id, normalized_snapshot, current)

    def save_snapshot(
        self,
        thread_id: str,
        snapshot: Mapping[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> GroundedConversationState:
        return self.save(thread_id, snapshot, expected_revision=expected_revision)

    def update(
        self,
        thread_id: str,
        updater: StateUpdater,
        *,
        expected_revision: int | None = None,
    ) -> GroundedConversationState:
        """Atomically load, transform, and persist one thread snapshot."""

        normalized_thread_id = self._normalize_thread_id(thread_id)
        with self._thread_file_lock(normalized_thread_id, exclusive=True):
            current = self._load_unlocked(normalized_thread_id)
            self._check_expected_revision(current, expected_revision)
            current_snapshot = self._json_copy(current.snapshot) if current is not None else None
            replacement = updater(current_snapshot)
            normalized_snapshot = self._normalize_snapshot(replacement)
            return self._save_unlocked(normalized_thread_id, normalized_snapshot, current)

    def clear(self, thread_id: str, *, expected_revision: int | None = None) -> bool:
        normalized_thread_id = self._normalize_thread_id(thread_id)
        with self._thread_file_lock(normalized_thread_id, exclusive=True):
            current = self._load_unlocked(normalized_thread_id)
            self._check_expected_revision(current, expected_revision)
            path = self.path_for(normalized_thread_id)
            try:
                path.unlink()
            except FileNotFoundError:
                return False
            self._fsync_directory(path.parent)
            return True

    def exists(self, thread_id: str) -> bool:
        return self.load(thread_id) is not None

    @contextmanager
    def locked(self, thread_id: str) -> Iterator[None]:
        """Hold the exclusive thread lock across a complete business transaction."""

        normalized_thread_id = self._normalize_thread_id(thread_id)
        with self._thread_file_lock(normalized_thread_id, exclusive=True):
            yield

    def path_for(self, thread_id: str) -> Path:
        normalized_thread_id = self._normalize_thread_id(thread_id)
        return self.threads_dir / (self._thread_file_stem(normalized_thread_id) + ".json")

    def lock_path_for(self, thread_id: str) -> Path:
        normalized_thread_id = self._normalize_thread_id(thread_id)
        return self.locks_dir / (self._thread_file_stem(normalized_thread_id) + ".lock")

    def _load_unlocked(self, thread_id: str) -> Optional[GroundedConversationState]:
        path = self.path_for(thread_id)
        try:
            encoded = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise GroundedConversationStateError("failed to read Grounded conversation state") from exc

        try:
            payload = json.loads(encoded)
        except (TypeError, json.JSONDecodeError) as exc:
            raise GroundedConversationStateCorruptError("Grounded conversation state is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise GroundedConversationStateCorruptError("Grounded conversation state envelope must be an object")

        version = str(payload.get("version") or "")
        persisted_thread_id = str(payload.get("threadId") or "")
        snapshot = payload.get("snapshot")
        revision = payload.get("revision")
        updated_at = str(payload.get("updatedAt") or "")
        if version != GROUNDED_CONVERSATION_STATE_VERSION:
            raise GroundedConversationStateCorruptError("unsupported Grounded conversation state version: %s" % version)
        if persisted_thread_id != thread_id:
            raise GroundedConversationStateCorruptError("Grounded conversation state thread scope mismatch")
        if not isinstance(snapshot, dict):
            raise GroundedConversationStateCorruptError("Grounded conversation snapshot must be an object")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            raise GroundedConversationStateCorruptError("Grounded conversation state revision is invalid")
        if not updated_at:
            raise GroundedConversationStateCorruptError("Grounded conversation state updatedAt is missing")
        return GroundedConversationState(
            thread_id=thread_id,
            snapshot=snapshot,
            revision=revision,
            updated_at=updated_at,
            version=version,
        )

    def _save_unlocked(
        self,
        thread_id: str,
        snapshot: dict[str, Any],
        current: GroundedConversationState | None,
    ) -> GroundedConversationState:
        state = GroundedConversationState(
            thread_id=thread_id,
            snapshot=snapshot,
            revision=(current.revision if current is not None else 0) + 1,
            updated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        payload = {
            "version": state.version,
            "threadId": state.thread_id,
            "revision": state.revision,
            "updatedAt": state.updated_at,
            "snapshot": state.snapshot,
        }
        self._atomic_write_json(self.path_for(thread_id), payload)
        return state

    @contextmanager
    def _thread_file_lock(self, thread_id: str, *, exclusive: bool) -> Iterator[None]:
        lock_path = self.lock_path_for(thread_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_key = str(lock_path.resolve())
        mutex = self._mutex_for(lock_key)
        with mutex:
            held = self._held_locks()
            if lock_key in held:
                if exclusive and not held[lock_key]:
                    raise GroundedConversationStateError("cannot upgrade a shared Grounded conversation state lock")
                yield
                return

            with lock_path.open("a+b") as lock_handle:
                operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
                fcntl.flock(lock_handle.fileno(), operation)
                held[lock_key] = exclusive
                try:
                    yield
                finally:
                    held.pop(lock_key, None)
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    @classmethod
    def _mutex_for(cls, lock_key: str) -> threading.RLock:
        with cls._registry_lock:
            mutex = cls._thread_mutexes.get(lock_key)
            if mutex is None:
                mutex = threading.RLock()
                cls._thread_mutexes[lock_key] = mutex
            return mutex

    @classmethod
    def _held_locks(cls) -> dict[str, bool]:
        held = getattr(cls._held_file_locks, "locks", None)
        if held is None:
            held = {}
            cls._held_file_locks.locks = held
        return held

    @staticmethod
    def _check_expected_revision(
        current: GroundedConversationState | None,
        expected_revision: int | None,
    ) -> None:
        if expected_revision is None:
            return
        actual_revision = current.revision if current is not None else 0
        if actual_revision != expected_revision:
            raise GroundedConversationStateConflictError(
                "Grounded conversation state revision conflict: expected %s, found %s"
                % (expected_revision, actual_revision)
            )

    @staticmethod
    def _normalize_thread_id(thread_id: str) -> str:
        value = str(thread_id or "").strip()
        if not value:
            raise ValueError("thread_id is required")
        if len(value) > 512 or any(ord(character) < 32 for character in value):
            raise ValueError("thread_id is invalid")
        return value

    @staticmethod
    def _normalize_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(snapshot, Mapping):
            raise TypeError("Grounded conversation snapshot must be a JSON mapping")
        try:
            encoded = json.dumps(dict(snapshot), ensure_ascii=False, separators=(",", ":"))
            decoded = json.loads(encoded)
        except (TypeError, ValueError) as exc:
            raise TypeError("Grounded conversation snapshot must contain only JSON values") from exc
        if not isinstance(decoded, dict):
            raise TypeError("Grounded conversation snapshot must be a JSON mapping")
        return decoded

    @staticmethod
    def _json_copy(snapshot: dict[str, Any]) -> dict[str, Any]:
        return json.loads(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")))

    @staticmethod
    def _thread_file_stem(thread_id: str) -> str:
        safe_prefix = re.sub(r"[^a-zA-Z0-9_.-]+", "_", thread_id).strip("._-")[:80] or "thread"
        digest = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()[:16]
        return "%s--%s" % (safe_prefix, digest)

    @staticmethod
    def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            ".%s.%s.%s.%s.tmp" % (path.name, os.getpid(), threading.get_ident(), uuid.uuid4().hex)
        )
        encoded = json.dumps(dict(payload), ensure_ascii=False, indent=2, separators=(",", ": ")) + "\n"
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            GroundedConversationStateStore._fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            # Some mounted filesystems do not support directory fsync. The
            # same-directory replace remains atomic even in that environment.
            pass
        finally:
            os.close(descriptor)
