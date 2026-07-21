from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

from pydantic import Field

from merchant_ai.models import APIModel


GroundedContractRepairType = Literal["NONE", "EVIDENCE", "BINDING", "TOPOLOGY"]


class GroundedContractRepairDirective(APIModel):
    """Kernel-owned instructions for repairing one rejected query Contract.

    The model may choose between the listed repair options, but it does not
    infer the failure class, invent semantic paths, or decide which tools are
    legal.  Those decisions are derived from typed validation gaps here.
    """

    directive_id: str = ""
    repair_type: GroundedContractRepairType = "NONE"
    status: str = "READY"
    next_action: str = ""
    source_gap_codes: list[str] = Field(default_factory=list)
    read_next: list[dict[str, Any]] = Field(default_factory=list)
    repair_options: list[dict[str, Any]] = Field(default_factory=list)
    allowed_actions: list[str] = Field(default_factory=list)
    base_attempt_id: str = ""
    base_contract_fingerprint: str = ""
    contract_version: int = 0
    parent_attempt_id: str = ""
    parent_contract_fingerprint: str = ""


_READ_CAPABILITY_KEYS = (
    "readNext",
    "read_next",
    "candidateReads",
    "candidate_reads",
)
_REF_CAPABILITY_KEYS = (
    "candidateRefIds",
    "candidateTimeFieldRefs",
    "requiredRefIds",
    "requiredFieldRefs",
    "requiredMetricRefs",
    "semanticRefIds",
    "readRefIds",
)
_PATH_CAPABILITY_KEYS = (
    "candidatePaths",
    "requiredPaths",
    "readPaths",
)


def _model_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump(by_alias=True, mode="json"))
    return {}


def _items(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _normalized_path(path: str, prefix: str) -> str:
    normalized = str(path or "").strip()
    if not normalized:
        return ""
    if not prefix:
        return normalized
    normalized_prefix = "/%s" % str(prefix or "").strip("/")
    if normalized == normalized_prefix or normalized.startswith(
        normalized_prefix + "/"
    ):
        return normalized
    return "%s/%s" % (normalized_prefix, normalized.lstrip("/"))


def collect_contract_repair_reads(
    gaps: Iterable[Any],
    *,
    paths_by_ref: Mapping[str, str] | None = None,
    resolve_ref_path: Callable[[str], str] | None = None,
    path_prefix: str = "",
) -> list[dict[str, Any]]:
    """Collect exact semantic leaves from every governed gap capability.

    Rejected refs are not automatically reread: for binding/topology failures
    they identify bindings to replace.  They become read targets only when the
    gap explicitly says that the ref was missing or not read.
    """

    known_paths = {
        str(ref_id or "").strip(): str(path or "").strip()
        for ref_id, path in dict(paths_by_ref or {}).items()
        if str(ref_id or "").strip()
    }
    candidates: list[dict[str, Any]] = []

    def resolve_path(ref_id: str, supplied_path: str = "") -> str:
        path = str(supplied_path or "").strip() or known_paths.get(ref_id, "")
        if not path and ref_id and resolve_ref_path is not None:
            try:
                path = str(resolve_ref_path(ref_id) or "").strip()
            except Exception:
                path = ""
        return _normalized_path(path, path_prefix)

    def append_candidate(value: Any, *, gap_code: str) -> None:
        if isinstance(value, Mapping):
            ref_id = str(
                value.get("refId")
                or value.get("semanticRefId")
                or value.get("ref_id")
                or ""
            ).strip()
            path = str(
                value.get("path")
                or value.get("semanticPath")
                or value.get("filePath")
                or ""
            ).strip()
            if ref_id or path:
                candidates.append(
                    {
                        "refId": ref_id,
                        "path": resolve_path(ref_id, path),
                        "sourceGapCode": gap_code,
                    }
                )
            return
        normalized = str(value or "").strip()
        if not normalized:
            return
        is_path = "/" in normalized and not normalized.startswith("semantic:")
        ref_id = "" if is_path else normalized
        candidates.append(
            {
                "refId": ref_id,
                "path": resolve_path(ref_id, normalized if is_path else ""),
                "sourceGapCode": gap_code,
            }
        )

    for raw_gap in gaps:
        gap = _model_payload(raw_gap)
        code = str(gap.get("code") or "").strip()
        resolution = str(
            gap.get("resolution") or gap.get("nextAction") or ""
        ).upper()
        capability = dict(gap.get("requiredCapability") or {})
        for key in _READ_CAPABILITY_KEYS:
            for item in _items(capability.get(key)):
                append_candidate(item, gap_code=code)
        for key in _REF_CAPABILITY_KEYS:
            for item in _items(capability.get(key)):
                append_candidate(item, gap_code=code)
        for key in _PATH_CAPABILITY_KEYS:
            for item in _items(capability.get(key)):
                append_candidate(item, gap_code=code)

        normalized_code = code.upper()
        rejected_refs_are_missing = any(
            marker in normalized_code
            for marker in (
                "NOT_READ",
                "REF_MISSING",
                "EVIDENCE_MISSING",
                "EVIDENCE_REQUIRED",
            )
        ) or resolution.startswith("READ")
        if rejected_refs_are_missing:
            for ref_id in list(gap.get("rejectedRefIds") or []):
                append_candidate(ref_id, gap_code=code)

    deduped: list[dict[str, Any]] = []
    index_by_identity: dict[tuple[str, str], int] = {}
    for item in candidates:
        ref_id = str(item.get("refId") or "")
        path = str(item.get("path") or "")
        if not ref_id and not path:
            continue
        identity = ("ref", ref_id) if ref_id else ("path", path)
        existing_index = index_by_identity.get(identity)
        if existing_index is not None:
            if path and not str(deduped[existing_index].get("path") or ""):
                deduped[existing_index] = item
            continue
        index_by_identity[identity] = len(deduped)
        deduped.append(item)
    return deduped


def _gap_has_topology_repair(gap: Mapping[str, Any]) -> bool:
    evidence_kind = str(gap.get("evidenceKind") or "").upper()
    code = str(gap.get("code") or "").upper()
    resolution = str(
        gap.get("resolution") or gap.get("nextAction") or ""
    ).upper()
    capability = dict(gap.get("requiredCapability") or {})
    allowed_repairs = {
        str(item or "").upper()
        for item in _items(capability.get("allowedRepairs"))
    }
    return bool(
        evidence_kind == "QUERY_TOPOLOGY"
        or "TOPOLOGY" in code
        or "EXECUTION_GRAPH" in resolution
        or "SPLIT" in resolution
        or bool(capability.get("splitExecutionRequired"))
        or str(capability.get("topology") or "").strip()
        or any("SPLIT" in item for item in allowed_repairs)
    )


def _gap_has_evidence_repair(gap: Mapping[str, Any]) -> bool:
    capability = dict(gap.get("requiredCapability") or {})
    if any(capability.get(key) not in (None, "", [], {}) for key in (
        *_READ_CAPABILITY_KEYS,
        *_REF_CAPABILITY_KEYS,
        *_PATH_CAPABILITY_KEYS,
    )):
        return True
    code = str(gap.get("code") or "").upper()
    return any(
        marker in code
        for marker in (
            "NOT_READ",
            "REF_MISSING",
            "EVIDENCE_MISSING",
            "EVIDENCE_REQUIRED",
        )
    )


def classify_contract_repair(
    gaps: Sequence[Any],
    *,
    contract_status: str = "",
) -> GroundedContractRepairType:
    payloads = [_model_payload(gap) for gap in gaps]
    payloads = [gap for gap in payloads if bool(gap.get("blocking", True))]
    if not payloads:
        return "NONE"
    explicit = [
        str(gap.get("repairType") or "").upper()
        for gap in payloads
        if str(gap.get("repairType") or "").upper()
        in {"EVIDENCE", "BINDING", "TOPOLOGY"}
    ]
    if "TOPOLOGY" in explicit or any(
        _gap_has_topology_repair(gap) for gap in payloads
    ):
        return "TOPOLOGY"
    if "BINDING" in explicit:
        return "BINDING"
    if "EVIDENCE" in explicit:
        return "EVIDENCE"
    if any(_gap_has_evidence_repair(gap) for gap in payloads):
        return "EVIDENCE"

    for gap in payloads:
        resolution = str(
            gap.get("resolution") or gap.get("nextAction") or ""
        ).upper()
        code = str(gap.get("code") or "").upper()
        if any(
            marker in resolution
            for marker in (
                "REBIND",
                "REVISE_BINDING",
                "RESELECT",
                "REMOVE_",
                "RESUBMIT",
            )
        ) or any(
            marker in code
            for marker in (
                "BINDING_AMBIGUOUS",
                "EXTRA_OUTPUT",
                "GRAIN_CONFLICT",
                "TABLE_INSUFFICIENT",
            )
        ):
            return "BINDING"
    if str(contract_status or "").upper() == "REVISE_BINDINGS":
        return "BINDING"
    return "EVIDENCE"


def _repair_status(repair_type: GroundedContractRepairType) -> str:
    return "READY" if repair_type == "NONE" else "REPAIR_REQUIRED"


def _repair_next_action(repair_type: GroundedContractRepairType) -> str:
    return (
        ""
        if repair_type == "NONE"
        else "CHOOSE_SAFE_REPAIR_AND_SUBMIT_NEW_VERSION"
    )


def _repair_options(
    gaps: Sequence[dict[str, Any]],
    repair_type: GroundedContractRepairType,
    read_next: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if repair_type == "NONE":
        return []
    options: list[dict[str, Any]] = []
    if repair_type == "TOPOLOGY":
        table_groups = [
            dict(group)
            for gap in gaps
            for group in (
                dict(gap.get("requiredCapability") or {}).get("tableGroups")
                or []
            )
            if isinstance(group, Mapping)
        ]
        options.extend(
            [
                {
                    "type": "REBIND_TO_COMPATIBLE_SINGLE_CONTRACT",
                    "nextAction": "PROPOSE_GROUNDED_CONTRACT",
                    "requirements": {
                        "sameMetricOwnerTable": True,
                        "compatibleTimeSemantics": True,
                        "compatibleGroupingAndPopulation": True,
                    },
                },
                {
                    "type": "SPLIT_OR_REVISE_EXECUTION_GRAPH",
                    "nextAction": "PROPOSE_OR_REVISE_GROUNDED_EXECUTION_GRAPH",
                    "dependencyMode": "PARALLEL_WHEN_INDEPENDENT",
                    "tableGroups": table_groups,
                },
            ]
        )
    for gap in gaps:
        code = str(gap.get("code") or "")
        resolution = str(
            gap.get("resolution") or gap.get("nextAction") or ""
        )
        capability = dict(gap.get("requiredCapability") or {})
        if not resolution and not capability:
            continue
        options.append(
            {
                "gapCode": code,
                "type": resolution or (
                    "READ_REQUIRED_SEMANTIC_ASSETS"
                    if repair_type == "EVIDENCE"
                    else "REVISE_BINDINGS"
                ),
                "resolution": resolution,
                "searchScope": str(gap.get("searchScope") or ""),
                "requiredCapability": capability,
                "rejectedRefIds": list(gap.get("rejectedRefIds") or []),
            }
        )
    if repair_type == "EVIDENCE" and read_next and not options:
        options.append(
            {
                "type": "READ_REQUIRED_SEMANTIC_ASSETS",
                "nextAction": "READ_THEN_RESUBMIT_CONTRACT",
            }
        )
    return options


def build_contract_repair_directive(
    gaps: Sequence[Any],
    *,
    contract_status: str = "",
    paths_by_ref: Mapping[str, str] | None = None,
    resolve_ref_path: Callable[[str], str] | None = None,
    path_prefix: str = "",
    base_attempt_id: str = "",
    base_contract_fingerprint: str = "",
    contract_version: int = 0,
    parent_attempt_id: str = "",
    parent_contract_fingerprint: str = "",
) -> GroundedContractRepairDirective:
    payloads = [
        _model_payload(gap)
        for gap in gaps
        if bool(_model_payload(gap).get("blocking", True))
    ]
    repair_type = classify_contract_repair(
        payloads,
        contract_status=contract_status,
    )
    read_next = collect_contract_repair_reads(
        payloads,
        paths_by_ref=paths_by_ref,
        resolve_ref_path=resolve_ref_path,
        path_prefix=path_prefix,
    )
    allowed_actions: list[str] = []
    if repair_type != "NONE":
        # These are affordances, not a fixed transition.  Core may read an
        # alternative asset, remove an optional bad binding, or resubmit from
        # already-read evidence.  Harness only adds graph mutation when the
        # typed gaps actually expose a topology repair.
        allowed_actions = [
            "READ_SEMANTIC_ASSET",
            "RESUBMIT_GROUNDED_CONTRACT",
        ]
    if repair_type == "TOPOLOGY":
        allowed_actions.extend(
            [
            "PROPOSE_GROUNDED_EXECUTION_GRAPH",
            "REVISE_GROUNDED_EXECUTION_GRAPH",
            ]
        )
    if any(
        "AMBIGUOUS" in str(gap.get("code") or "").upper()
        for gap in payloads
    ):
        allowed_actions.append("ASK_HUMAN_FOR_BUSINESS_AMBIGUITY")
    source_gap_codes = list(
        dict.fromkeys(
            str(gap.get("code") or "")
            for gap in payloads
            if str(gap.get("code") or "")
        )
    )
    fingerprint_payload = {
        "repairType": repair_type,
        "sourceGapCodes": source_gap_codes,
        "readNext": read_next,
        "baseAttemptId": base_attempt_id,
        "baseContractFingerprint": base_contract_fingerprint,
        "contractVersion": int(contract_version or 0),
    }
    directive_id = (
        "repair_%s"
        % hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:16]
        if repair_type != "NONE"
        else ""
    )
    return GroundedContractRepairDirective(
        directive_id=directive_id,
        repair_type=repair_type,
        status=_repair_status(repair_type),
        next_action=_repair_next_action(repair_type),
        source_gap_codes=source_gap_codes,
        read_next=read_next,
        repair_options=_repair_options(payloads, repair_type, read_next),
        allowed_actions=allowed_actions,
        base_attempt_id=base_attempt_id,
        base_contract_fingerprint=base_contract_fingerprint,
        contract_version=int(contract_version or 0),
        parent_attempt_id=parent_attempt_id,
        parent_contract_fingerprint=parent_contract_fingerprint,
    )
