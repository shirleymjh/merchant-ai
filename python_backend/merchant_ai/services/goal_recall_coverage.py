from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from pydantic import Field

from merchant_ai.models import APIModel, RecallItem
from merchant_ai.services.grounded_goal_contract import (
    OriginalQuestionGoalContract,
)


_DEFAULT_PROTOCOL_PATH = Path(__file__).resolve().parents[2] / "resources" / "runtime" / "goal_recall_capabilities.json"


class GoalRecallCapabilityProtocol(APIModel):
    protocol_version: str = "goal_recall_capabilities.v1"
    goal_requirements: dict[str, list[str]] = Field(default_factory=dict)
    source_capabilities: dict[str, list[str]] = Field(default_factory=dict)

    def requirements_for(self, goal_kind: str) -> list[str]:
        return _normalized_capabilities(self.goal_requirements.get(str(goal_kind or "").strip().upper(), []))

    def capabilities_for_source(self, source_type: str) -> list[str]:
        return _normalized_capabilities(
            self.source_capabilities.get(
                str(source_type or "").strip().upper(),
                [],
            )
        )

    def source_types_for(self, capabilities: Iterable[str]) -> list[str]:
        required = set(_normalized_capabilities(capabilities))
        return sorted(
            source_type
            for source_type, provided in self.source_capabilities.items()
            if required.intersection(_normalized_capabilities(provided))
        )


class GoalRecallCoverageItem(APIModel):
    goal_id: str
    goal_kind: str
    status: str
    required_capabilities: list[str] = Field(default_factory=list)
    candidate_refs: list[str] = Field(default_factory=list)
    exact_candidate_refs: list[str] = Field(default_factory=list)


class GoalSupplementalRecallRequest(APIModel):
    request_id: str
    target_goal_ids: list[str] = Field(default_factory=list)
    query: str
    query_terms: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    request_fingerprint: str


class GoalRecallCoverageReceipt(APIModel):
    receipt_version: str = "goal_recall_coverage.v1"
    receipt_id: str
    status: str
    protocol_version: str
    index_version: str = ""
    recall_fingerprint: str
    scope_fingerprint: str = ""
    items: list[GoalRecallCoverageItem] = Field(default_factory=list)
    covered_goal_ids: list[str] = Field(default_factory=list)
    ambiguous_goal_ids: list[str] = Field(default_factory=list)
    missing_goal_ids: list[str] = Field(default_factory=list)
    supplemental_requests: list[GoalSupplementalRecallRequest] = Field(default_factory=list)


@lru_cache(maxsize=4)
def load_goal_recall_capability_protocol(
    path: str = "",
) -> GoalRecallCapabilityProtocol:
    protocol_path = Path(path) if path else _DEFAULT_PROTOCOL_PATH
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    return GoalRecallCapabilityProtocol.model_validate(payload)


def recall_item_capabilities(
    item: RecallItem,
    *,
    protocol: GoalRecallCapabilityProtocol | None = None,
) -> list[str]:
    active_protocol = protocol or load_goal_recall_capability_protocol()
    metadata = dict(item.metadata or {})
    published = metadata.get("goalRecallCapabilities") or metadata.get("goal_recall_capabilities")
    if isinstance(published, str):
        published = [published]
    if isinstance(published, list) and published:
        return _normalized_capabilities(published)
    return active_protocol.capabilities_for_source(item.source_type)


def attach_goal_recall_capabilities(
    item: RecallItem,
    *,
    protocol: GoalRecallCapabilityProtocol | None = None,
) -> RecallItem:
    active_protocol = protocol or load_goal_recall_capability_protocol()
    metadata = dict(item.metadata or {})
    capabilities = recall_item_capabilities(item, protocol=active_protocol)
    if capabilities:
        metadata["goalRecallCapabilities"] = capabilities
    metadata["goalRecallCapabilityProtocolVersion"] = active_protocol.protocol_version
    return item.model_copy(update={"metadata": metadata})


def filter_and_tag_goal_recall_items(
    items: Iterable[RecallItem],
    *,
    target_goal_ids: Iterable[str],
    required_capabilities: Iterable[str],
    coverage_receipt_id: str = "",
    protocol: GoalRecallCapabilityProtocol | None = None,
) -> list[RecallItem]:
    active_protocol = protocol or load_goal_recall_capability_protocol()
    targets = _normalized_strings(target_goal_ids)
    required = set(_normalized_capabilities(required_capabilities))
    tagged: list[RecallItem] = []
    for raw_item in items:
        item = attach_goal_recall_capabilities(
            raw_item,
            protocol=active_protocol,
        )
        provided = set(recall_item_capabilities(item, protocol=active_protocol))
        if required and not required.intersection(provided):
            continue
        metadata = dict(item.metadata or {})
        metadata["targetGoalIds"] = list(
            dict.fromkeys(
                [
                    *_normalized_strings(metadata.get("targetGoalIds") or []),
                    *targets,
                ]
            )
        )
        if coverage_receipt_id:
            metadata["goalRecallCoverageReceiptId"] = coverage_receipt_id
        tagged.append(item.model_copy(update={"metadata": metadata}))
    return tagged


class GoalRecallCoverageService:
    """Evaluate candidate coverage without selecting semantic bindings."""

    def __init__(
        self,
        protocol: GoalRecallCapabilityProtocol | None = None,
    ):
        self.protocol = protocol or load_goal_recall_capability_protocol()

    def evaluate(
        self,
        contract: OriginalQuestionGoalContract,
        recall_items: Iterable[RecallItem],
        *,
        index_version: str = "",
        scope_fingerprint: str = "",
    ) -> GoalRecallCoverageReceipt:
        items = [attach_goal_recall_capabilities(item, protocol=self.protocol) for item in recall_items]
        requirement_counts: dict[tuple[str, ...], int] = {}
        for goal in contract.goals:
            if not bool(getattr(goal, "required", True)):
                continue
            signature = tuple(self.protocol.requirements_for(str(getattr(goal, "kind", "") or "").strip().upper()))
            if signature:
                requirement_counts[signature] = requirement_counts.get(signature, 0) + 1
        coverage_items: list[GoalRecallCoverageItem] = []
        for goal in contract.goals:
            if not bool(getattr(goal, "required", True)):
                continue
            goal_id = str(getattr(goal, "goal_id", "") or "").strip()
            goal_kind = str(getattr(goal, "kind", "") or "").strip().upper()
            required = self.protocol.requirements_for(goal_kind)
            if not required:
                coverage_items.append(
                    GoalRecallCoverageItem(
                        goal_id=goal_id,
                        goal_kind=goal_kind,
                        status="NOT_APPLICABLE",
                    )
                )
                continue
            compatible = [
                item
                for item in items
                if set(required).intersection(recall_item_capabilities(item, protocol=self.protocol))
            ]
            candidates = [item for item in compatible if _item_matches_goal(item, goal)]
            if not candidates and requirement_counts.get(tuple(required)) == 1:
                exact_compatible = [item for item in compatible if _item_is_unambiguous_exact_candidate(item)]
                if len(exact_compatible) == 1:
                    candidates = exact_compatible
            candidate_refs = _candidate_refs(candidates)
            exact_refs = _candidate_refs(item for item in candidates if _item_exactly_matches_goal(item, goal))
            status = "MISSING" if not candidate_refs else "AMBIGUOUS" if len(exact_refs) > 1 else "COVERED"
            coverage_items.append(
                GoalRecallCoverageItem(
                    goal_id=goal_id,
                    goal_kind=goal_kind,
                    status=status,
                    required_capabilities=required,
                    candidate_refs=candidate_refs,
                    exact_candidate_refs=exact_refs,
                )
            )

        covered = [item.goal_id for item in coverage_items if item.status == "COVERED"]
        ambiguous = [item.goal_id for item in coverage_items if item.status == "AMBIGUOUS"]
        missing = [item.goal_id for item in coverage_items if item.status == "MISSING"]
        recall_fingerprint = _fingerprint(
            [
                {
                    "ref": ref,
                    "capabilities": recall_item_capabilities(
                        item,
                        protocol=self.protocol,
                    ),
                    "targetGoalIds": _normalized_strings((item.metadata or {}).get("targetGoalIds") or []),
                }
                for item in items
                if (ref := _candidate_ref(item))
            ]
        )
        receipt_seed = {
            "contract": contract.model_dump(by_alias=True, mode="json"),
            "indexVersion": index_version,
            "recallFingerprint": recall_fingerprint,
            "scopeFingerprint": scope_fingerprint,
            "protocolVersion": self.protocol.protocol_version,
        }
        receipt_id = "goal_recall_%s" % _fingerprint(receipt_seed)[:20]
        receipt = GoalRecallCoverageReceipt(
            receipt_id=receipt_id,
            status="PARTIAL" if missing else "COMPLETE",
            protocol_version=self.protocol.protocol_version,
            index_version=index_version,
            recall_fingerprint=recall_fingerprint,
            scope_fingerprint=scope_fingerprint,
            items=coverage_items,
            covered_goal_ids=covered,
            ambiguous_goal_ids=ambiguous,
            missing_goal_ids=missing,
        )
        return receipt.model_copy(
            update={
                "supplemental_requests": self.plan_supplements(
                    contract,
                    receipt,
                )
            }
        )

    def plan_supplements(
        self,
        contract: OriginalQuestionGoalContract,
        receipt: GoalRecallCoverageReceipt,
    ) -> list[GoalSupplementalRecallRequest]:
        goal_map = contract.goal_map()
        requests: list[GoalSupplementalRecallRequest] = []
        item_by_goal = {item.goal_id: item for item in receipt.items}
        for goal_id in receipt.missing_goal_ids:
            goal = goal_map.get(goal_id)
            coverage = item_by_goal.get(goal_id)
            if goal is None or coverage is None:
                continue
            terms = _normalized_strings(
                [
                    str(getattr(goal, "label", "") or ""),
                    *(getattr(goal, "source_spans", ()) or ()),
                ]
            )
            if not terms:
                continue
            query = " ".join(terms)
            fingerprint = _fingerprint(
                {
                    "receiptId": receipt.receipt_id,
                    "goalId": goal_id,
                    "query": query,
                    "requiredCapabilities": coverage.required_capabilities,
                    "indexVersion": receipt.index_version,
                    "scopeFingerprint": receipt.scope_fingerprint,
                }
            )
            requests.append(
                GoalSupplementalRecallRequest(
                    request_id="goal_supplement_%s" % fingerprint[:16],
                    target_goal_ids=[goal_id],
                    query=query,
                    query_terms=terms,
                    required_capabilities=coverage.required_capabilities,
                    request_fingerprint=fingerprint,
                )
            )
        return requests


def _item_matches_goal(item: RecallItem, goal: Any) -> bool:
    metadata = dict(item.metadata or {})
    goal_id = str(getattr(goal, "goal_id", "") or "").strip()
    if goal_id in _normalized_strings(metadata.get("targetGoalIds") or []):
        return True
    phrases = _goal_phrase_keys(goal)
    labels = _item_label_keys(item)
    return bool(
        phrases.intersection(labels)
        or any(len(phrase) >= 2 and (phrase in label or label in phrase) for phrase in phrases for label in labels)
    )


def _item_exactly_matches_goal(item: RecallItem, goal: Any) -> bool:
    metadata = dict(item.metadata or {})
    if bool(metadata.get("metricResolutionAmbiguous")):
        return False
    return bool(
        _goal_phrase_keys(goal).intersection(_item_label_keys(item)) or int(metadata.get("protectionTier") or 0) >= 2
    )


def _item_is_unambiguous_exact_candidate(item: RecallItem) -> bool:
    metadata = dict(item.metadata or {})
    resolution_type = str(metadata.get("metricResolutionType") or "").strip().lower()
    try:
        confidence = float(metadata.get("metricResolutionConfidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return bool(
        not metadata.get("metricResolutionAmbiguous")
        and (
            int(metadata.get("protectionTier") or 0) >= 2
            or (resolution_type.startswith("exact_") and confidence >= 0.95)
        )
    )


def _goal_phrase_keys(goal: Any) -> set[str]:
    return {
        normalized
        for value in [
            getattr(goal, "label", ""),
            *(getattr(goal, "source_spans", ()) or ()),
        ]
        if (normalized := _match_key(value))
    }


def _item_label_keys(item: RecallItem) -> set[str]:
    metadata = dict(item.metadata or {})
    aliases = metadata.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    values = [
        item.title,
        metadata.get("matchedMetricLabel"),
        metadata.get("matchedExactMetricLabel"),
        metadata.get("businessName"),
        metadata.get("displayName"),
        metadata.get("naturalName"),
        metadata.get("metricKey"),
        metadata.get("canonicalMetricKey"),
        metadata.get("columnName"),
        metadata.get("term"),
        metadata.get("ruleTitle"),
        metadata.get("relationshipId"),
        metadata.get("tableName"),
        *aliases,
    ]
    return {normalized for value in values if (normalized := _match_key(value))}


def _candidate_refs(items: Iterable[RecallItem]) -> list[str]:
    return list(dict.fromkeys(ref for item in items if (ref := _candidate_ref(item))))


def _candidate_ref(item: RecallItem) -> str:
    return str((item.metadata or {}).get("semanticRefId") or item.doc_id or "").strip()


def _match_key(value: Any) -> str:
    return "".join(character.casefold() for character in str(value or "") if character.isalnum())


def _normalized_capabilities(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value or "").strip().upper() for value in values if str(value or "").strip()))


def _normalized_strings(values: Iterable[Any]) -> list[str]:
    if isinstance(values, str):
        values = [values]
    return list(dict.fromkeys(str(value or "").strip() for value in values if str(value or "").strip()))


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "GoalRecallCapabilityProtocol",
    "GoalRecallCoverageItem",
    "GoalRecallCoverageReceipt",
    "GoalRecallCoverageService",
    "GoalSupplementalRecallRequest",
    "attach_goal_recall_capabilities",
    "filter_and_tag_goal_recall_items",
    "load_goal_recall_capability_protocol",
    "recall_item_capabilities",
]
