from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Sequence

from pydantic import Field, computed_field

from merchant_ai.models import (
    APIModel,
    AnswerMode,
    GraphValidationGap,
    GraphValidationResult,
    IntentType,
    PlanningAssetEntry,
    PlanningAssetPack,
    QueryPlan,
    QuestionIntent,
    RecallItem,
    RelationshipEntry,
    ResolvedTimeRange,
    SnapshotAlignmentContract,
    TaskRole,
)
from merchant_ai.graph.query_graph_contract import query_graph_fingerprint
from merchant_ai.services.semantic_metrics import seal_semantic_metric_resolution
from merchant_ai.services.time_semantics import resolve_time_range


class GroundedContractGap(APIModel):
    code: str
    message: str
    blocking: bool = True
    evidence_kind: str = ""
    topic: str = ""
    table: str = ""
    phrase: str = ""
    resolution: str = ""
    search_scope: str = ""
    required_capability: dict[str, Any] = Field(default_factory=dict)
    rejected_ref_ids: list[str] = Field(default_factory=list)


class GroundedRejectedBinding(APIModel):
    fingerprint: str
    code: str = "TABLE_INSUFFICIENT"
    topic: str = ""
    table: str = ""
    ref_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    required_capability: dict[str, Any] = Field(default_factory=dict)


class GroundedEvidenceRef(APIModel):
    ref_id: str
    content_hash: str
    kind: str
    topic: str
    table: str = ""
    path: str = ""


class GroundedTableBinding(APIModel):
    topic: str
    table: str
    title: str = ""
    data_grain: str = ""
    time_column: str = ""
    merchant_filter_column: str = ""
    detail_ref_id: str = ""


class GroundedMetricBinding(APIModel):
    requested_phrase: str
    semantic_ref_id: str
    topic: str
    table: str
    metric_key: str
    business_name: str = ""
    formula: str = ""
    source_columns: list[str] = Field(default_factory=list)
    aggregation_policy: str = ""
    metric_grain: str = ""
    applicable_time_grain: str = ""
    time_column: str = ""
    unit: str = ""
    aliases: list[str] = Field(default_factory=list)
    anchor_policy: str = ""
    time_semantics: dict[str, Any] = Field(default_factory=dict)
    binding_type: str = "published_metric"
    field_aggregation: str = ""
    source_field_ref_id: str = ""
    calculation_capabilities: dict[str, Any] = Field(default_factory=dict)


class GroundedDimensionBinding(APIModel):
    requested_phrase: str
    semantic_ref_id: str
    topic: str
    table: str
    column: str
    business_name: str = ""
    role: str = ""
    aliases: list[str] = Field(default_factory=list)
    usage: str = "group_by"


class GroundedRelationshipBinding(APIModel):
    semantic_ref_id: str
    topic: str
    name: str
    left_table: str
    right_table: str
    join_type: str = ""
    keys: list[list[str]] = Field(default_factory=list)
    grain: str = ""
    cautions: list[str] = Field(default_factory=list)


class GroundedRankingBinding(APIModel):
    enabled: bool = False
    direction: str = ""
    limit: int = 0
    metric_ref_id: str = ""
    dimension_ref_id: str = ""


class GroundedRankingHint(APIModel):
    order: str = ""
    limit: int = 0
    metric_ref: str = ""


class GroundedFieldAggregationHint(APIModel):
    field_ref: str = ""
    aggregation: str = ""
    requested_phrase: str = ""


class GroundedBindingHints(APIModel):
    table_refs: list[str] = Field(default_factory=list)
    metric_refs: list[str] = Field(default_factory=list)
    field_aggregations: list[GroundedFieldAggregationHint] = Field(default_factory=list)
    dimension_refs: list[str] = Field(default_factory=list)
    group_by_ref: str = ""
    label_refs: dict[str, str] = Field(default_factory=dict)
    relationship_refs: list[str] = Field(default_factory=list)
    ranking: GroundedRankingHint = Field(default_factory=GroundedRankingHint)
    analysis_mode: str = ""
    time_expression: str = ""


class GroundedQueryContract(APIModel):
    contract_version: str = "grounded_query_contract.v1"
    status: str = "UNRESOLVED"
    question: str
    topics: list[str] = Field(default_factory=list)
    analysis_mode: str = ""
    binding_hints: GroundedBindingHints = Field(default_factory=GroundedBindingHints)
    query_shape: str = ""
    execution_shape: str = ""
    primary_table: str = ""
    tables: list[GroundedTableBinding] = Field(default_factory=list)
    metrics: list[GroundedMetricBinding] = Field(default_factory=list)
    dimensions: list[GroundedDimensionBinding] = Field(default_factory=list)
    relationships: list[GroundedRelationshipBinding] = Field(default_factory=list)
    time_range: ResolvedTimeRange = Field(default_factory=ResolvedTimeRange)
    ranking: GroundedRankingBinding = Field(default_factory=GroundedRankingBinding)
    evidence_refs: list[str] = Field(default_factory=list)
    evidence: list[GroundedEvidenceRef] = Field(default_factory=list)
    unresolved_gaps: list[GroundedContractGap] = Field(default_factory=list)
    rejected_bindings: list[GroundedRejectedBinding] = Field(default_factory=list)
    provenance: str = "successful_core_read_file_calls"

    @computed_field
    @property
    def ready(self) -> bool:
        return self.status == "READY" and not any(gap.blocking for gap in self.unresolved_gaps)


class GroundedQueryContractValidationResult(APIModel):
    valid: bool = False
    gaps: list[GroundedContractGap] = Field(default_factory=list)


@dataclass(frozen=True)
class _EvidenceDocument:
    ref: GroundedEvidenceRef
    payload: Any


@dataclass(frozen=True)
class GroundedExecutionPreparation:
    """Execution hand-off validated only against the Grounded Contract."""

    plan: QueryPlan
    validation: GraphValidationResult
    source_plan_fingerprint: str
    execution_plan_fingerprint: str
    question_fingerprint: str
    asset_pack_fingerprint: str
    changed: bool = False
    optimization_notes: tuple[str, ...] = ()
    validator_name: str = "GroundedContractProjectionValidator"
    freshness_reports: tuple[Any, ...] = ()
    runtime_fallback_task_ids: tuple[str, ...] = ()
    runtime_source_plan_fingerprint: str = ""
    snapshot_alignment: SnapshotAlignmentContract = field(
        default_factory=SnapshotAlignmentContract
    )

    @property
    def executable(self) -> bool:
        return bool(self.validation.valid)

    def require_executable(self) -> QueryPlan:
        if self.validation.valid:
            return self.plan
        codes = [gap.code for gap in self.validation.gaps if gap.code]
        raise ValueError(
            "grounded execution preparation is invalid; gaps=%s"
            % (",".join(codes[:8]) or "GROUNDED_PROJECTION_INVALID")
        )


class GroundedQueryContractBuilder:
    """Build a planning hand-off using only successful Core semantic reads.

    Recall snippets, table manifests and model-authored refs are intentionally
    insufficient. Every executable binding must resolve to a hashed semantic
    document from the Core's ``read_file`` ledger.
    """

    def __init__(self, validator: "GroundedQueryContractValidator | None" = None):
        self.validator = validator or GroundedQueryContractValidator()

    def build(
        self,
        question: str,
        topics: Iterable[str],
        core_semantic_evidence: Iterable[dict[str, Any]],
        binding_hints: dict[str, Any] | GroundedBindingHints | None = None,
        timezone_name: str = "Asia/Shanghai",
        now: datetime | None = None,
        default_days: int = 7,
    ) -> GroundedQueryContract:
        normalized_topics = _dedupe(str(topic or "").strip() for topic in topics)
        hints = _normalize_binding_hints(binding_hints)
        requested_refs = {
            *hints.table_refs,
            *hints.metric_refs,
            *(item.field_ref for item in hints.field_aggregations),
            *hints.dimension_refs,
            *hints.relationship_refs,
            *hints.label_refs.keys(),
            *([hints.group_by_ref] if hints.group_by_ref else []),
        }
        for ref_id in list(requested_refs):
            if ":column:" in ref_id:
                requested_refs.add(ref_id.replace(":column:", ":field:", 1))
            if ":field:" in ref_id:
                requested_refs.add(ref_id.replace(":field:", ":column:", 1))
        documents, discovery_gaps = self._trusted_documents(
            core_semantic_evidence,
            normalized_topics,
            requested_refs,
        )
        document_refs = {document.ref.ref_id for document in documents}
        hints = _canonicalize_binding_hints(hints, document_refs)
        document_kinds = {document.ref.ref_id: document.ref.kind for document in documents}
        discovery_gaps.extend(_missing_binding_ref_gaps(hints, document_kinds))
        table_details = self._table_details(documents)
        metrics = self._metric_bindings(
            documents,
            hints.metric_refs,
            hints.label_refs,
            question=str(question or ""),
        )
        field_metrics, field_metric_gaps = self._field_aggregation_bindings(
            documents,
            hints.field_aggregations,
            hints.label_refs,
            table_details,
        )
        metrics.extend(field_metrics)
        discovery_gaps.extend(field_metric_gaps)
        dimensions = self._dimension_bindings(
            documents,
            _dedupe([*hints.dimension_refs, hints.group_by_ref]),
            hints.label_refs,
            hints.group_by_ref,
        )
        ranking = self._ranking_binding(hints, metrics, dimensions)

        selected_tables = _dedupe(
            [binding.table for binding in metrics]
            + [binding.table for binding in dimensions]
            + [
                detail.table
                for detail in table_details.values()
                if detail.detail_ref_id in set(hints.table_refs)
            ]
        )
        tables: list[GroundedTableBinding] = []
        for table in selected_tables:
            detail = table_details.get(table)
            if detail is not None:
                tables.append(detail)
                continue
            topic = next(
                (
                    binding.topic
                    for binding in [*metrics, *dimensions]
                    if binding.table == table
                ),
                "",
            )
            tables.append(GroundedTableBinding(topic=topic, table=table))
            discovery_gaps.append(
                GroundedContractGap(
                    code="TABLE_DETAIL_EVIDENCE_REQUIRED",
                    message="Selected table %s has no trusted TABLE_DETAIL read" % table,
                    evidence_kind="TABLE_DETAIL",
                    topic=topic,
                    table=table,
                )
            )

        relationships = self._relationship_bindings(
            documents,
            selected_tables,
            hints.relationship_refs,
        )
        time_expression = hints.time_expression or question
        time_range = resolve_time_range(
            time_expression,
            timezone_name=timezone_name,
            now=now,
            default_days=default_days,
        )
        anchor_policies = _dedupe(metric.anchor_policy for metric in metrics if metric.anchor_policy)
        if len(anchor_policies) == 1:
            time_range.anchor_policy = anchor_policies[0]

        supporting_refs = _dedupe(
            [table.detail_ref_id for table in tables if table.detail_ref_id]
            + [metric.semantic_ref_id for metric in metrics]
            + [dimension.semantic_ref_id for dimension in dimensions]
            + [relationship.semantic_ref_id for relationship in relationships]
        )
        evidence_by_ref = {document.ref.ref_id: document.ref for document in documents}
        evidence = [evidence_by_ref[ref_id] for ref_id in supporting_refs if ref_id in evidence_by_ref]
        contract = GroundedQueryContract(
            question=str(question or "").strip(),
            topics=normalized_topics,
            analysis_mode=hints.analysis_mode,
            binding_hints=hints,
            query_shape=_canonical_query_shape(
                hints,
                metrics,
                dimensions,
                selected_tables,
                ranking,
            ),
            execution_shape=_execution_shape(metrics, dimensions, selected_tables, ranking),
            primary_table=metrics[0].table if metrics else (selected_tables[0] if selected_tables else ""),
            tables=tables,
            metrics=metrics,
            dimensions=dimensions,
            relationships=relationships,
            time_range=time_range,
            ranking=ranking,
            evidence_refs=supporting_refs,
            evidence=evidence,
            unresolved_gaps=_dedupe_gaps(discovery_gaps),
        )
        validation = self.validator.validate(contract)
        gaps = _dedupe_gaps([*contract.unresolved_gaps, *validation.gaps])
        rejected_bindings = _rejected_bindings_for_contract(contract, gaps)
        return contract.model_copy(
            update={
                "status": _grounded_contract_status(gaps),
                "unresolved_gaps": gaps,
                "rejected_bindings": rejected_bindings,
            }
        )

    def _trusted_documents(
        self,
        raw_evidence: Iterable[dict[str, Any]],
        topics: Sequence[str],
        required_refs: set[str] | None = None,
    ) -> tuple[list[_EvidenceDocument], list[GroundedContractGap]]:
        documents: list[_EvidenceDocument] = []
        gaps: list[GroundedContractGap] = []
        allowed_topics = set(topics)
        seen: set[tuple[str, str]] = set()
        for raw in raw_evidence:
            if not isinstance(raw, dict):
                gaps.append(_gap("INVALID_CORE_EVIDENCE", "Core evidence entry is not an object"))
                continue
            ref_id = str(raw.get("refId") or raw.get("ref_id") or "").strip()
            if required_refs and ref_id not in required_refs:
                continue
            topic = str(raw.get("topic") or "").strip()
            table = str(raw.get("table") or "").strip()
            kind = str(raw.get("kind") or "").strip().upper()
            content = str(raw.get("contentSnippet") or raw.get("content_snippet") or "")
            content_hash = str(raw.get("contentHash") or raw.get("content_hash") or "").strip()
            if not ref_id.startswith("semantic:") or not content or not content_hash:
                gaps.append(
                    _gap(
                        "UNTRUSTED_CORE_EVIDENCE",
                        "Semantic evidence requires refId, contentHash and contentSnippet",
                        kind,
                        topic,
                        table,
                    )
                )
                continue
            if not allowed_topics or topic not in allowed_topics:
                gaps.append(
                    _gap(
                        "EVIDENCE_TOPIC_OUT_OF_SCOPE",
                        "Evidence %s is outside the active Topic workspace" % ref_id,
                        kind,
                        topic,
                        table,
                    )
                )
                continue
            actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if actual_hash != content_hash:
                gaps.append(
                    _gap(
                        "EVIDENCE_HASH_MISMATCH",
                        "Evidence content hash does not match %s" % ref_id,
                        kind,
                        topic,
                        table,
                    )
                )
                continue
            identity = (ref_id, content_hash)
            if identity in seen:
                continue
            try:
                payload = json.loads(content)
            except (TypeError, ValueError):
                gaps.append(
                    _gap(
                        "EVIDENCE_CONTENT_INVALID",
                        "Evidence %s is not a complete JSON semantic document" % ref_id,
                        kind,
                        topic,
                        table,
                    )
                )
                continue
            seen.add(identity)
            documents.append(
                _EvidenceDocument(
                    ref=GroundedEvidenceRef(
                        ref_id=ref_id,
                        content_hash=content_hash,
                        kind=kind,
                        topic=topic,
                        table=table,
                        path=str(raw.get("path") or ""),
                    ),
                    payload=payload,
                )
            )
        return documents, gaps

    @staticmethod
    def _table_details(documents: Sequence[_EvidenceDocument]) -> dict[str, GroundedTableBinding]:
        details: dict[str, GroundedTableBinding] = {}
        for document in documents:
            if document.ref.kind != "TABLE_DETAIL" or not isinstance(document.payload, dict):
                continue
            payload = document.payload
            table = str(payload.get("tableName") or document.ref.table or "").strip()
            if not table:
                continue
            details[table] = GroundedTableBinding(
                topic=str(payload.get("topic") or document.ref.topic or ""),
                table=table,
                title=str(payload.get("title") or ""),
                data_grain=str(payload.get("dataGrain") or payload.get("grain") or ""),
                time_column=str(payload.get("timeColumn") or ""),
                merchant_filter_column=str(
                    payload.get("merchantFilterColumn")
                    or payload.get("scopeFilterColumn")
                    or ""
                ),
                detail_ref_id=document.ref.ref_id,
            )
        return details

    @staticmethod
    def _metric_bindings(
        documents: Sequence[_EvidenceDocument],
        selected_refs: Sequence[str],
        label_refs: dict[str, str],
        question: str = "",
    ) -> list[GroundedMetricBinding]:
        selected_set = set(selected_refs)
        bindings: list[GroundedMetricBinding] = []
        metric_documents = [document for document in documents if document.ref.kind == "METRIC"]
        for document in metric_documents:
            if document.ref.ref_id not in selected_set:
                continue
            payload = document.payload if isinstance(document.payload, dict) else {}
            metric = payload.get("metric") if isinstance(payload.get("metric"), dict) else payload
            metric_key = str(metric.get("metricKey") or "").strip()
            table = str(payload.get("tableName") or document.ref.table or "").strip()
            if not metric_key or not table:
                continue
            aliases = _dedupe(
                [
                    str(metric.get("businessName") or ""),
                    str(metric.get("displayName") or ""),
                    str(metric.get("naturalName") or ""),
                    *[str(alias) for alias in metric.get("aliases") or []],
                ]
            )
            phrase = str(
                label_refs.get(document.ref.ref_id)
                or _metric_phrase_in_question(question, aliases)
                or metric.get("businessName")
                or metric_key
            )
            binding = GroundedMetricBinding(
                requested_phrase=phrase,
                semantic_ref_id=document.ref.ref_id,
                topic=str(payload.get("topic") or document.ref.topic or ""),
                table=table,
                metric_key=metric_key,
                business_name=str(metric.get("businessName") or metric_key),
                formula=str(metric.get("formula") or metric.get("metricFormula") or ""),
                source_columns=[str(column) for column in metric.get("sourceColumns") or []],
                aggregation_policy=str(metric.get("aggregationPolicy") or ""),
                metric_grain=str(metric.get("metricGrain") or ""),
                applicable_time_grain=str(metric.get("applicableTimeGrain") or ""),
                time_column=str(metric.get("timeColumn") or ""),
                unit=str(metric.get("unit") or ""),
                aliases=aliases,
                anchor_policy=str((metric.get("timeSemantics") or {}).get("asOfPolicy") or "")
                if isinstance(metric.get("timeSemantics"), dict)
                else "",
                time_semantics=(
                    dict(metric.get("timeSemantics") or {})
                    if isinstance(metric.get("timeSemantics"), dict)
                    else {}
                ),
                calculation_capabilities=semantic_evidence_calculation_capabilities(
                    "METRIC",
                    payload,
                ),
            )
            bindings.append(binding)
        order = {ref_id: index for index, ref_id in enumerate(selected_refs)}
        bindings.sort(key=lambda item: (order.get(item.semantic_ref_id, len(order)), item.table, item.metric_key))
        return bindings

    @staticmethod
    def _field_aggregation_bindings(
        documents: Sequence[_EvidenceDocument],
        selected: Sequence[GroundedFieldAggregationHint],
        label_refs: dict[str, str],
        table_details: dict[str, GroundedTableBinding],
    ) -> tuple[list[GroundedMetricBinding], list[GroundedContractGap]]:
        """Compile an allowlisted aggregate from an exact governed COLUMN read.

        This is deliberately not a published business metric.  The binding is
        run-scoped and formula-free at the tool boundary: Core selects a field
        ref plus an allowlisted operator, and the contract deterministically
        derives the executable formula and metricSpec identity.
        """

        documents_by_ref = {
            document.ref.ref_id: document
            for document in documents
            if document.ref.kind == "COLUMN"
        }
        bindings: list[GroundedMetricBinding] = []
        gaps: list[GroundedContractGap] = []
        seen: set[tuple[str, str]] = set()
        for hint in selected:
            ref_id = str(hint.field_ref or "").strip()
            aggregation = _normalize_field_aggregation(hint.aggregation)
            if not aggregation:
                gaps.append(
                    _gap(
                        "FIELD_AGGREGATION_UNSUPPORTED",
                        "Field aggregation %s is not allowlisted; supported values are COUNT and COUNT_DISTINCT"
                        % str(hint.aggregation or ""),
                        "COLUMN",
                        phrase=str(hint.requested_phrase or ""),
                    )
                )
                continue
            identity = (ref_id, aggregation)
            if identity in seen:
                continue
            seen.add(identity)
            document = documents_by_ref.get(ref_id)
            if document is None:
                continue
            payload = document.payload if isinstance(document.payload, dict) else {}
            definition = payload.get("definition") if isinstance(payload.get("definition"), dict) else {}
            field_capabilities = semantic_evidence_calculation_capabilities("COLUMN", payload)
            supported_aggregations = {
                str(item or "").strip().upper()
                for item in field_capabilities.get("allowedAggregations") or []
                if str(item or "").strip()
            }
            if supported_aggregations and aggregation not in supported_aggregations:
                gaps.append(
                    _gap(
                        "FIELD_AGGREGATION_NOT_DECLARED",
                        "Field %s does not declare %s as an allowed derivation"
                        % (ref_id, aggregation),
                        "COLUMN",
                        document.ref.topic,
                        document.ref.table,
                        str(hint.requested_phrase or ""),
                    )
                )
                continue
            column = str(definition.get("columnName") or definition.get("Field") or "").strip()
            table = str(payload.get("tableName") or document.ref.table or "").strip()
            if not column or not table:
                gaps.append(
                    _gap(
                        "FIELD_AGGREGATION_DEFINITION_INVALID",
                        "Field aggregation ref %s has no governed table/column definition" % ref_id,
                        "COLUMN",
                        document.ref.topic,
                        document.ref.table,
                        str(hint.requested_phrase or ""),
                    )
                )
                continue
            detail = table_details.get(table)
            business_name = str(definition.get("businessName") or column).strip()
            phrase = str(
                hint.requested_phrase
                or label_refs.get(ref_id)
                or _field_aggregation_business_name(business_name, aggregation)
            ).strip()
            bindings.append(
                GroundedMetricBinding(
                    requested_phrase=phrase,
                    semantic_ref_id=ref_id,
                    topic=str(payload.get("topic") or document.ref.topic or ""),
                    table=table,
                    metric_key=_field_aggregation_metric_key(column, aggregation),
                    business_name=phrase or _field_aggregation_business_name(business_name, aggregation),
                    formula=_field_aggregation_formula(column, aggregation),
                    source_columns=[column],
                    aggregation_policy="period_recompute",
                    metric_grain=detail.data_grain if detail else "",
                    applicable_time_grain="period",
                    time_column=detail.time_column if detail else "",
                    aliases=_dedupe(
                        [
                            phrase,
                            _field_aggregation_business_name(business_name, aggregation),
                            business_name,
                            column,
                        ]
                    ),
                    binding_type="field_aggregation",
                    field_aggregation=aggregation,
                    source_field_ref_id=ref_id,
                    calculation_capabilities={
                        **field_capabilities,
                        "declaredAggregation": aggregation,
                        "timeRollupPolicy": str(
                            field_capabilities.get("derivedMeasureTimeRollupPolicy")
                            or "RECOMPUTE_FROM_DETAIL"
                        ),
                    },
                )
            )
        return bindings, gaps

    @staticmethod
    def _dimension_bindings(
        documents: Sequence[_EvidenceDocument],
        selected_refs: Sequence[str],
        label_refs: dict[str, str],
        group_by_ref: str,
    ) -> list[GroundedDimensionBinding]:
        selected_set = set(selected_refs)
        bindings: list[GroundedDimensionBinding] = []
        for document in documents:
            if document.ref.ref_id not in selected_set:
                continue
            definitions: list[dict[str, Any]] = []
            payload = document.payload
            if document.ref.kind == "COLUMN" and isinstance(payload, dict):
                definition = payload.get("definition")
                if isinstance(definition, dict):
                    definitions = [definition]
            else:
                continue
            for definition in definitions:
                column = str(definition.get("columnName") or definition.get("Field") or "").strip()
                role = str(definition.get("role") or definition.get("semanticRole") or "").upper()
                if not column or role in {"MEASURE", "METRIC"}:
                    continue
                table = str(payload.get("tableName") or document.ref.table or "").strip()
                aliases = _dedupe(
                    [
                        str(definition.get("businessName") or ""),
                        str(definition.get("description") or ""),
                        column,
                        *[str(alias) for alias in definition.get("aliases") or []],
                    ]
                )
                phrase = str(label_refs.get(document.ref.ref_id) or definition.get("businessName") or column)
                bindings.append(
                    GroundedDimensionBinding(
                        requested_phrase=phrase,
                        semantic_ref_id=document.ref.ref_id,
                        topic=str(payload.get("topic") or document.ref.topic or ""),
                        table=table,
                        column=column,
                        business_name=str(definition.get("businessName") or column),
                        role=role,
                        aliases=aliases,
                        usage="group_by" if document.ref.ref_id == group_by_ref else "label",
                    )
                )
        order = {ref_id: index for index, ref_id in enumerate(selected_refs)}
        bindings.sort(key=lambda item: (0 if item.semantic_ref_id == group_by_ref else 1, order.get(item.semantic_ref_id, len(order)), item.column))
        return bindings

    @staticmethod
    def _relationship_bindings(
        documents: Sequence[_EvidenceDocument],
        selected_tables: Sequence[str],
        selected_refs: Sequence[str],
    ) -> list[GroundedRelationshipBinding]:
        wanted = set(selected_tables)
        selected_set = set(selected_refs)
        relationships: list[GroundedRelationshipBinding] = []
        if len(wanted) < 2:
            return relationships
        for document in documents:
            if document.ref.kind != "RELATIONSHIPS" or document.ref.ref_id not in selected_set:
                continue
            payload = document.payload
            raw_relationships = payload if isinstance(payload, list) else payload.get("relationships") if isinstance(payload, dict) else []
            for raw in raw_relationships or []:
                if not isinstance(raw, dict):
                    continue
                left = str(raw.get("leftTable") or "")
                right = str(raw.get("rightTable") or "")
                if left not in wanted or right not in wanted:
                    continue
                relationships.append(
                    GroundedRelationshipBinding(
                        semantic_ref_id=document.ref.ref_id,
                        topic=document.ref.topic,
                        name=str(raw.get("name") or "%s_%s" % (left, right)),
                        left_table=left,
                        right_table=right,
                        join_type=str(raw.get("joinType") or ""),
                        keys=[
                            [str(value) for value in pair]
                            for pair in raw.get("keys") or []
                            if isinstance(pair, (list, tuple)) and len(pair) == 2
                        ],
                        grain=str(raw.get("grain") or ""),
                        cautions=[str(item) for item in raw.get("cautions") or []],
                    )
                )
        return relationships

    @staticmethod
    def _ranking_binding(
        hints: GroundedBindingHints,
        metrics: Sequence[GroundedMetricBinding],
        dimensions: Sequence[GroundedDimensionBinding],
    ) -> GroundedRankingBinding:
        direction = str(hints.ranking.order or "").strip().upper()
        ranking_mode = hints.analysis_mode.lower() in {"topn", "ranking", "ranked_group"}
        # A limit is an execution bound, not a ranking declaration.  Ranking is
        # only meaningful when Core explicitly binds a grouping dimension and
        # either declares an order or selects a ranking analysis mode.
        enabled = bool(hints.group_by_ref and (direction or ranking_mode))
        if not enabled:
            return GroundedRankingBinding()
        if direction not in {"ASC", "DESC"}:
            direction = ""
        metric_ref = str(hints.ranking.metric_ref or "")
        metric = next((item for item in metrics if item.semantic_ref_id == metric_ref), None)
        if metric is None and not metric_ref and len(metrics) == 1:
            metric = metrics[0]
        dimension = next(
            (item for item in dimensions if item.semantic_ref_id == hints.group_by_ref),
            None,
        )
        return GroundedRankingBinding(
            enabled=True,
            direction=direction,
            limit=int(hints.ranking.limit or 0),
            metric_ref_id=metric.semantic_ref_id if metric else "",
            dimension_ref_id=dimension.semantic_ref_id if dimension else "",
        )


class GroundedSemanticFitValidator:
    """Enforce usage constraints declared by progressively-read semantic assets.

    The validator does not interpret business phrases and does not choose a
    table. Core owns that reasoning. This gate only compares the proposed
    Contract (time window, formula, grain and analysis shape) with the selected
    asset's governed calculation semantics.
    """

    def validate(self, contract: GroundedQueryContract) -> list[GroundedContractGap]:
        gaps: list[GroundedContractGap] = []
        table_by_name = {table.table: table for table in contract.tables}
        for metric in contract.metrics:
            violations = _metric_usage_policy_violations(contract, metric)
            if not violations:
                continue
            table = table_by_name.get(metric.table)
            policy = dict(metric.calculation_capabilities or {})
            alternative = dict(
                policy.get("alternativeCapability")
                or policy.get("alternative_capability")
                or {}
            )
            required_capability = alternative or {
                "requiredUsagePolicy": str(policy.get("timeRollupPolicy") or "compatible_binding"),
                "nativeTimeGrain": str(policy.get("nativeTimeGrain") or ""),
                "nativeWindowDays": int(policy.get("nativeWindowDays") or 0),
            }
            message = str(policy.get("violationMessage") or "").strip() or (
                "Binding %s on table %s violates governed semantic usage policy: %s"
                % (metric.metric_key, metric.table, ", ".join(violations))
            )
            gaps.append(
                _gap(
                    "TABLE_INSUFFICIENT",
                    message,
                    "METRIC",
                    metric.topic,
                    metric.table,
                    metric.requested_phrase,
                    resolution=str(policy.get("resolution") or "RESELECT_TABLE"),
                    search_scope="READ_BINDINGS_THEN_TABLE_MANIFEST_THEN_TOPIC_INDEX",
                    required_capability=required_capability,
                    rejected_ref_ids=_dedupe(
                        [
                            metric.semantic_ref_id,
                            table.detail_ref_id if table is not None else "",
                        ]
                    ),
                )
            )
        return gaps


class GroundedQueryContractValidator:
    def validate(self, contract: GroundedQueryContract) -> GroundedQueryContractValidationResult:
        gaps: list[GroundedContractGap] = []
        evidence_refs = set(contract.evidence_refs)
        if not contract.question.strip():
            gaps.append(_gap("QUESTION_REQUIRED", "Original question is required"))
        if not contract.topics:
            gaps.append(_gap("TOPIC_REQUIRED", "At least one active Topic is required"))
        if not contract.metrics:
            gaps.append(
                _gap(
                    "METRIC_EVIDENCE_REQUIRED",
                    "No requested metric was bound from trusted Core reads",
                    "METRIC",
                )
            )
        table_by_name = {table.table: table for table in contract.tables}
        for metric in contract.metrics:
            if metric.semantic_ref_id not in evidence_refs:
                gaps.append(_binding_ref_gap("METRIC_EVIDENCE_REF_MISSING", metric.semantic_ref_id, metric.topic, metric.table))
            if not metric.formula:
                gaps.append(_gap("METRIC_FORMULA_REQUIRED", "Metric %s has no governed formula" % metric.metric_key, "METRIC", metric.topic, metric.table, metric.requested_phrase))
            if not metric.aggregation_policy:
                gaps.append(_gap("METRIC_AGGREGATION_POLICY_REQUIRED", "Metric %s has no aggregation policy" % metric.metric_key, "METRIC", metric.topic, metric.table, metric.requested_phrase))
            table = table_by_name.get(metric.table)
            if table is None or not table.detail_ref_id:
                gaps.append(_gap("TABLE_DETAIL_EVIDENCE_REQUIRED", "Metric owner table %s lacks TABLE_DETAIL evidence" % metric.table, "TABLE_DETAIL", metric.topic, metric.table))
            effective_time_column = metric.time_column or (table.time_column if table else "")
            if not effective_time_column:
                gaps.append(_gap("TIME_COLUMN_EVIDENCE_REQUIRED", "Metric %s has no grounded time column" % metric.metric_key, "METRIC", metric.topic, metric.table))
        for table in contract.tables:
            if table.detail_ref_id and table.detail_ref_id not in evidence_refs:
                gaps.append(_binding_ref_gap("TABLE_EVIDENCE_REF_MISSING", table.detail_ref_id, table.topic, table.table))
            if table.detail_ref_id and not table.merchant_filter_column:
                gaps.append(_gap("MERCHANT_SCOPE_COLUMN_REQUIRED", "Table %s has no governed merchant scope column" % table.table, "TABLE_DETAIL", table.topic, table.table))
        for dimension in contract.dimensions:
            if dimension.semantic_ref_id not in evidence_refs:
                gaps.append(_binding_ref_gap("DIMENSION_EVIDENCE_REF_MISSING", dimension.semantic_ref_id, dimension.topic, dimension.table))
            table = table_by_name.get(dimension.table)
            if (
                dimension.usage == "group_by"
                and table is not None
                and table.merchant_filter_column
                and dimension.column == table.merchant_filter_column
            ):
                gaps.append(
                    _gap(
                        "MERCHANT_SCOPE_DIMENSION_FORBIDDEN",
                        "Merchant scope column %s is an access filter and cannot be grouped or returned"
                        % dimension.column,
                        "COLUMN",
                        dimension.topic,
                        dimension.table,
                        dimension.requested_phrase,
                    )
                )
        grouped_dimensions = [
            dimension for dimension in contract.dimensions if dimension.usage == "group_by"
        ]
        if contract.query_shape == "SCALAR":
            if grouped_dimensions:
                gaps.append(
                    _gap(
                        "SCALAR_GROUP_BY_FORBIDDEN",
                        "SCALAR shape cannot contain a groupBy dimension",
                        "COLUMN",
                    )
                )
            if contract.ranking.enabled:
                gaps.append(
                    _gap(
                        "SCALAR_RANKING_FORBIDDEN",
                        "SCALAR shape cannot contain ranking semantics",
                    )
                )
        elif contract.query_shape in {"GROUPED", "TREND", "RANKED"}:
            if len(grouped_dimensions) != 1:
                gaps.append(
                    _gap(
                        "%s_DIMENSION_REQUIRED" % contract.query_shape,
                        "%s shape requires exactly one explicitly read groupBy dimension"
                        % contract.query_shape,
                        "COLUMN",
                    )
                )
        if contract.query_shape == "TREND" and len(grouped_dimensions) == 1:
            dimension = grouped_dimensions[0]
            table = table_by_name.get(dimension.table)
            time_roles = {"TIME", "DATE", "DATETIME", "TIMESTAMP", "TIME_DIMENSION"}
            if not table or not (
                dimension.column == table.time_column or dimension.role in time_roles
            ):
                gaps.append(
                    _gap(
                        "TREND_TIME_DIMENSION_REQUIRED",
                        "TREND shape groupBy must be the governed time dimension",
                        "COLUMN",
                        dimension.topic,
                        dimension.table,
                        dimension.requested_phrase,
                    )
                )
        if contract.query_shape == "RANKED" and not contract.ranking.enabled:
            gaps.append(
                _gap(
                    "RANKING_BINDING_REQUIRED",
                    "RANKED shape requires explicit ranking bindings",
                )
            )
        if contract.ranking.enabled:
            if not contract.metrics or not contract.ranking.metric_ref_id:
                gaps.append(_gap("RANKING_METRIC_REQUIRED", "Ranking requires one grounded metric", "METRIC"))
            if not contract.dimensions or not contract.ranking.dimension_ref_id:
                gaps.append(_gap("RANKING_DIMENSION_REQUIRED", "Ranking requires one grounded grouping dimension", "COLUMN"))
            if contract.ranking.direction not in {"ASC", "DESC"}:
                gaps.append(_gap("RANKING_DIRECTION_REQUIRED", "Ranking requires an explicit ASC or DESC direction"))
            if contract.ranking.limit <= 0:
                gaps.append(_gap("RANKING_LIMIT_REQUIRED", "Ranking requires an explicit positive limit"))
        selected_tables = {table.table for table in contract.tables if table.table}
        if len(selected_tables) > 1:
            if not _tables_connected(selected_tables, contract.relationships):
                gaps.append(
                    _gap(
                        "RELATIONSHIP_EVIDENCE_REQUIRED",
                        "Selected tables are not connected by trusted relationship evidence",
                        "RELATIONSHIPS",
                    )
                )
            for relationship in contract.relationships:
                if relationship.semantic_ref_id not in evidence_refs:
                    gaps.append(_binding_ref_gap("RELATIONSHIP_EVIDENCE_REF_MISSING", relationship.semantic_ref_id, relationship.topic, ""))
                if not relationship.keys:
                    gaps.append(_gap("RELATIONSHIP_KEYS_REQUIRED", "Relationship %s has no governed keys" % relationship.name, "RELATIONSHIPS", relationship.topic))
        if str(contract.time_range.source or "") == "default_days":
            gaps.append(
                _gap(
                    "TIME_RANGE_REQUIRED",
                    "The user did not specify a time window; ask how far back to query before execution",
                )
            )
        elif contract.time_range.days <= 0:
            gaps.append(_gap("TIME_RANGE_REQUIRED", "A positive time window is required"))
        gaps.extend(GroundedSemanticFitValidator().validate(contract))
        return GroundedQueryContractValidationResult(valid=not gaps, gaps=_dedupe_gaps(gaps))


def build_grounded_query_contract_from_refs(
    question: str,
    topics: Iterable[str],
    read_ref_ids: Iterable[str],
    semantic_catalog: Any,
    binding_hints: dict[str, Any] | GroundedBindingHints | None = None,
    timezone_name: str = "Asia/Shanghai",
    now: datetime | None = None,
    default_days: int = 7,
) -> GroundedQueryContract:
    """Resolve already-selected Core refs and build a hash-sealed contract."""

    ledger: list[dict[str, Any]] = []
    for ref_id in _dedupe(str(item or "").strip() for item in read_ref_ids):
        try:
            result = semantic_catalog.read(ref_id=ref_id, max_chars=2_000_000, offset=0)
        except Exception:
            result = {}
        if not isinstance(result, dict) or not result.get("success") or result.get("truncated"):
            ledger.append({"refId": ref_id})
            continue
        content = str(result.get("content") or "")
        ledger.append(
            {
                "refId": str(result.get("refId") or ref_id),
                "path": str(result.get("path") or ""),
                "kind": str(result.get("kind") or ""),
                "topic": str(result.get("topic") or ""),
                "table": str(result.get("table") or ""),
                "contentSnippet": content,
                "contentHash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "offset": 0,
            }
        )
    return GroundedQueryContractBuilder().build(
        question,
        topics,
        ledger,
        binding_hints=binding_hints,
        timezone_name=timezone_name,
        now=now,
        default_days=default_days,
    )


def materialize_grounded_asset_pack(
    contract: GroundedQueryContract | dict[str, Any],
    topic_assets: Any,
) -> PlanningAssetPack:
    """Project only contract-bound semantics and execution columns into a pack."""

    grounded = contract if isinstance(contract, GroundedQueryContract) else GroundedQueryContract.model_validate(contract)
    pack = PlanningAssetPack()
    table_bindings = {table.table: table for table in grounded.tables}
    required_columns: dict[str, list[str]] = {table: [] for table in table_bindings}
    column_evidence: dict[tuple[str, str], str] = {}
    for table in grounded.tables:
        required_columns[table.table] = _dedupe(
            [table.merchant_filter_column, table.time_column]
        )
        for column in required_columns[table.table]:
            column_evidence[(table.table, column)] = table.detail_ref_id
    for metric in grounded.metrics:
        required_columns.setdefault(metric.table, [])
        required_columns[metric.table] = _dedupe(
            [*required_columns[metric.table], *metric.source_columns, metric.time_column]
        )
        for column in metric.source_columns:
            column_evidence[(metric.table, column)] = metric.semantic_ref_id
    for dimension in grounded.dimensions:
        required_columns.setdefault(dimension.table, [])
        required_columns[dimension.table] = _dedupe(
            [*required_columns[dimension.table], dimension.column]
        )
        column_evidence[(dimension.table, dimension.column)] = dimension.semantic_ref_id
    relationship_key_columns: set[tuple[str, str]] = set()
    for relationship in grounded.relationships:
        for left, right in relationship.keys:
            required_columns.setdefault(relationship.left_table, [])
            required_columns.setdefault(relationship.right_table, [])
            required_columns[relationship.left_table] = _dedupe([*required_columns[relationship.left_table], left])
            required_columns[relationship.right_table] = _dedupe([*required_columns[relationship.right_table], right])
            column_evidence[(relationship.left_table, left)] = relationship.semantic_ref_id
            column_evidence[(relationship.right_table, right)] = relationship.semantic_ref_id
            relationship_key_columns.update(
                {(relationship.left_table, left), (relationship.right_table, right)}
            )

    dimension_semantics = {
        (dimension.table, dimension.column): {
            "columnName": dimension.column,
            "businessName": dimension.business_name or dimension.column,
            "aliases": list(dimension.aliases),
            "role": dimension.role,
            "usage": dimension.usage,
            "groundedEvidenceRef": dimension.semantic_ref_id,
        }
        for dimension in grounded.dimensions
    }

    for table_name, table in table_bindings.items():
        asset = _safe_mapping_call(topic_assets, "load_table_asset", table.topic, table_name)
        schema_rows = _safe_list_call(topic_assets, "load_table_schema", table.topic, table_name)
        schema_by_name = {
            str(row.get("columnName") or row.get("Field") or ""): row
            for row in schema_rows
            if isinstance(row, dict) and str(row.get("columnName") or row.get("Field") or "")
        }
        columns = required_columns.get(table_name, [])
        raw_usage_profile = (
            asset.get("tableUsageProfile")
            if isinstance(asset.get("tableUsageProfile"), dict)
            else {}
        )
        execution_usage_policy = {
            key: deepcopy(raw_usage_profile.get(key))
            for key in ("contractStatus", "queryableByAgent")
            if raw_usage_profile.get(key) is not None
        }
        table_metadata = {
            "tableName": table_name,
            "tableComment": table.title or table_name,
            "dataGrain": table.data_grain,
            "timeColumn": table.time_column,
            "merchantFilterColumn": table.merchant_filter_column,
            "scopeFilterColumn": table.merchant_filter_column,
            "regionFilterColumn": str(asset.get("regionFilterColumn") or ""),
            "storeFilterColumn": str(asset.get("storeFilterColumn") or ""),
            "freshnessType": str(asset.get("freshnessType") or ""),
            "rowAccessPolicy": deepcopy(asset.get("rowAccessPolicy")),
            "resultAccessPolicies": deepcopy(asset.get("resultAccessPolicies")),
            "tableUsageProfile": execution_usage_policy,
            "status": asset.get("status"),
            "version": asset.get("version") or asset.get("semanticVersion"),
            "visibilityPolicy": deepcopy(asset.get("visibilityPolicy")),
            "allowedRoles": deepcopy(asset.get("allowedRoles")),
            "requiredPermissions": deepcopy(asset.get("requiredPermissions")),
            "groundedEvidenceRef": table.detail_ref_id,
        }
        pack.tables.append(
            PlanningAssetEntry(
                key=table_name,
                table=table_name,
                topic=table.topic,
                title=table_metadata["tableComment"],
                columns=columns,
                aliases=_dedupe([table_name, table_metadata["tableComment"]]),
                description=json.dumps(table_metadata, ensure_ascii=False),
                source_ref_id=table.detail_ref_id,
                metadata=table_metadata,
            )
        )
        pack.schema_source[table_name] = "grounded_query_contract"
        for column in columns:
            schema = dict(schema_by_name.get(column) or {"columnName": column})
            semantic = dict(dimension_semantics.get((table_name, column)) or {})
            title = str(semantic.get("businessName") or schema.get("comment") or schema.get("Comment") or column)
            source_ref_id = column_evidence.get((table_name, column), table.detail_ref_id)
            field = PlanningAssetEntry(
                key=column,
                table=table_name,
                topic=table.topic,
                title=title,
                aliases=_dedupe([column, title, *[str(item) for item in semantic.get("aliases") or []]]),
                description=json.dumps({"schema": schema, "semantic": semantic}, ensure_ascii=False),
                source_ref_id=source_ref_id,
                metadata={"schema": schema, "semantic": semantic},
            )
            pack.fields.append(field)
            role = str(semantic.get("semanticRole") or semantic.get("role") or "").upper()
            if (
                column == table.merchant_filter_column
                or (table_name, column) in relationship_key_columns
                or role in {"KEY", "ENTITY_KEY", "JOIN_KEY", "PRIMARY_KEY"}
            ):
                pack.entity_keys.append(field.model_copy(deep=True))

    for metric in grounded.metrics:
        metadata = {
            "metricKey": metric.metric_key,
            "businessName": metric.business_name,
            "formula": metric.formula,
            "sourceColumns": list(metric.source_columns),
            "aggregationPolicy": metric.aggregation_policy,
            "metricGrain": metric.metric_grain,
            "applicableTimeGrain": metric.applicable_time_grain,
            "timeColumn": metric.time_column,
            "unit": metric.unit,
            "aliases": list(metric.aliases),
            "timeSemantics": (
                dict(metric.time_semantics)
                if metric.time_semantics
                else {"asOfPolicy": metric.anchor_policy}
            ),
            "groundedEvidenceRef": metric.semantic_ref_id,
            "bindingType": metric.binding_type,
            "fieldAggregation": metric.field_aggregation,
            "sourceFieldRefId": metric.source_field_ref_id,
            "calculationSemantics": dict(metric.calculation_capabilities),
        }
        planning_asset_ref = _metric_planning_asset_ref(metric)
        pack.metrics.append(
            PlanningAssetEntry(
                key=metric.metric_key,
                table=metric.table,
                topic=metric.topic,
                title=metric.business_name or metric.metric_key,
                columns=list(metric.source_columns),
                aliases=list(metric.aliases),
                description=json.dumps(metadata, ensure_ascii=False),
                source_ref_id=planning_asset_ref,
                metadata=metadata,
            )
        )

    for relationship in grounded.relationships:
        pack.relationships.append(
            RelationshipEntry(
                relationship_id=relationship.name,
                left_table=relationship.left_table,
                right_table=relationship.right_table,
                join_keys=[
                    {"leftColumn": left, "rightColumn": right}
                    for left, right in relationship.keys
                ],
                grain=relationship.grain,
                cautions=list(relationship.cautions),
                source_ref_id=relationship.semantic_ref_id,
                description=json.dumps(relationship.model_dump(by_alias=True), ensure_ascii=False),
            )
        )

    pack.table_manifest = {
        "mode": "grounded_query_contract",
        "questionIndependent": False,
        "topics": list(grounded.topics),
        "tables": [
            {
                "topic": table.topic,
                "table": table.table,
                "title": table.title or table.table,
                "detailRefId": table.detail_ref_id,
            }
            for table in grounded.tables
        ],
        "tableCount": len(grounded.tables),
        "policy": "Only tables explicitly bound by GroundedQueryContract are executable",
    }
    pack.metric_compaction = {
        "groundedQueryContract": {
            "status": grounded.status,
            "executionShape": grounded.execution_shape,
            "evidenceRefs": list(grounded.evidence_refs),
        },
        "loadedSourceRefs": list(grounded.evidence_refs),
        "recalledMetricEvidence": [
            {
                "metricKey": metric.metric_key,
                "ownerTable": metric.table,
                "businessName": metric.business_name,
                "aliases": list(metric.aliases),
                "formula": metric.formula,
                "sourceColumns": list(metric.source_columns),
                "aggregationPolicy": metric.aggregation_policy,
                "metricGrain": metric.metric_grain,
                "applicableTimeGrain": metric.applicable_time_grain,
                "timeColumn": metric.time_column,
                "timeSemantics": (
                    dict(metric.time_semantics)
                    if metric.time_semantics
                    else {"asOfPolicy": metric.anchor_policy}
                ),
                "semanticRefId": metric.semantic_ref_id,
                "bindingType": metric.binding_type,
                "fieldAggregation": metric.field_aggregation,
                "sourceFieldRefId": metric.source_field_ref_id,
                "calculationSemantics": dict(metric.calculation_capabilities),
                "recallQuery": metric.requested_phrase,
                "recallQueries": [metric.requested_phrase],
                "matchedMetricLabel": metric.requested_phrase,
                "metricResolutionType": (
                    "grounded_field_aggregation"
                    if metric.binding_type == "field_aggregation"
                    else "core_explicit_binding"
                ),
                "metricResolutionConfidence": 1.0,
                "metricResolutionAmbiguous": False,
            }
            for metric in grounded.metrics
        ],
    }
    pack.source_refs = {
        evidence.ref_id: RecallItem(
            doc_id=evidence.ref_id,
            title=evidence.ref_id,
            source_type=evidence.kind,
            topic=evidence.topic,
            table=evidence.table,
            metadata={
                "semanticRefId": evidence.ref_id,
                "semanticKind": evidence.kind,
                "semanticPath": evidence.path,
                "contentHash": evidence.content_hash,
                "trustedSource": grounded.provenance,
            },
        )
        for evidence in grounded.evidence
    }
    return pack


def compile_grounded_query(
    contract: GroundedQueryContract | dict[str, Any],
    pack: PlanningAssetPack | dict[str, Any],
) -> Any:
    """Compile a grounded Contract directly into an executable QueryPlan.

    This compiler is intentionally independent from the legacy
    ``QuestionUnderstanding`` compiler.  The Contract has already bound the
    executable semantic objects, so this stage may project those bindings but
    must not infer a metric, dimension, ranking shape, or owner table again.
    """

    grounded = contract if isinstance(contract, GroundedQueryContract) else GroundedQueryContract.model_validate(contract)
    asset_pack = pack if isinstance(pack, PlanningAssetPack) else PlanningAssetPack.model_validate(pack)
    if not grounded.ready:
        raise ValueError(
            "grounded query contract is unresolved: %s"
            % ",".join(gap.code for gap in grounded.unresolved_gaps if gap.blocking)
        )
    if not grounded.metrics:
        raise ValueError("grounded query contract has no metric bindings")
    metric_tables = {metric.table for metric in grounded.metrics if metric.table}
    if grounded.query_shape == "MULTI_TABLE" or len(metric_tables) != 1:
        raise ValueError(
            "grounded direct compiler does not infer cross-table metric execution; "
            "submit an explicit executable graph contract"
        )
    execution_table = next(iter(metric_tables))
    table_binding = next(
        (table for table in grounded.tables if table.table == execution_table),
        None,
    )
    if table_binding is None:
        raise ValueError("grounded metric owner table has no table binding")

    explicit_group_dimensions = [
        dimension for dimension in grounded.dimensions if dimension.usage == "group_by"
    ]
    if len(explicit_group_dimensions) > 1:
        raise ValueError("grounded direct compiler supports exactly one explicit groupBy binding")
    group_dimension = explicit_group_dimensions[0] if explicit_group_dimensions else None
    if group_dimension is not None and group_dimension.table != execution_table:
        raise ValueError(
            "grounded direct compiler does not infer cross-table grouping; "
            "submit an explicit executable graph contract"
        )
    if (
        group_dimension is not None
        and table_binding.merchant_filter_column
        and group_dimension.column == table_binding.merchant_filter_column
    ):
        raise ValueError("merchantFilterColumn is an access filter and cannot be a groupBy dimension")

    if grounded.query_shape == "RANKED":
        if group_dimension is None:
            raise ValueError("grounded ranking requires an explicit groupBy dimension binding")
        if grounded.ranking.dimension_ref_id != group_dimension.semantic_ref_id:
            raise ValueError("grounded ranking dimension does not match the explicit groupBy binding")
        ranking_metric = next(
            (
                metric
                for metric in grounded.metrics
                if metric.semantic_ref_id == grounded.ranking.metric_ref_id
            ),
            None,
        )
        if ranking_metric is None:
            raise ValueError("grounded ranking metric is not an explicitly bound metric")
        if grounded.ranking.direction not in {"", "DESC"}:
            raise ValueError("grounded direct compiler does not yet support ascending ranking")

    if grounded.query_shape == "RANKED":
        answer_mode = AnswerMode.TOPN
    elif grounded.query_shape in {"GROUPED", "TREND"}:
        answer_mode = AnswerMode.GROUP_AGG
    elif grounded.query_shape == "SCALAR":
        answer_mode = AnswerMode.METRIC
    else:
        raise ValueError("unsupported grounded query shape: %s" % grounded.query_shape)
    group_by = group_dimension.column if group_dimension is not None else ""

    def metric_resolution(metric: GroundedMetricBinding) -> dict[str, Any]:
        return seal_semantic_metric_resolution(
            {
                "requestedMetricRef": metric.semantic_ref_id,
                "metricKey": metric.metric_key,
                "displayName": metric.requested_phrase or metric.business_name or metric.metric_key,
                "sourcePhrase": metric.requested_phrase or metric.business_name or metric.metric_key,
                "businessName": metric.business_name or metric.metric_key,
                "ownerTable": metric.table,
                "semanticRefId": metric.semantic_ref_id,
                "formula": metric.formula,
                "sourceColumns": list(metric.source_columns),
                "confidence": 1.0,
                "resolutionSource": "grounded_query_contract",
                "aggregationPolicy": metric.aggregation_policy,
                "metricGrain": metric.metric_grain,
                "applicableTimeGrain": metric.applicable_time_grain,
                "timeColumn": metric.time_column or table_binding.time_column,
                "timeSemantics": (
                    dict(metric.time_semantics)
                    if metric.time_semantics
                    else {"asOfPolicy": metric.anchor_policy}
                ),
                "unit": metric.unit,
                "bindingType": metric.binding_type,
                "fieldAggregation": metric.field_aggregation,
                "sourceFieldRefId": metric.source_field_ref_id,
                "contractProvenance": {
                    "kind": "grounded_semantic_read",
                    "refId": metric.semantic_ref_id,
                },
            },
            force=True,
        )

    def metric_spec(metric: GroundedMetricBinding, task_id: str) -> dict[str, Any]:
        return {
            "metricName": metric.metric_key,
            "displayName": metric.requested_phrase or metric.business_name or metric.metric_key,
            "sourcePhrase": metric.requested_phrase or metric.business_name or metric.metric_key,
            "businessName": metric.business_name or metric.metric_key,
            "metricColumn": metric.source_columns[0] if metric.source_columns else "",
            "metricFormula": metric.formula,
            "sourceColumns": list(metric.source_columns),
            "sourceTaskId": task_id,
            "semanticRefId": metric.semantic_ref_id,
            "ownerTable": metric.table,
            "aggregationPolicy": metric.aggregation_policy,
            "metricGrain": metric.metric_grain,
            "applicableTimeGrain": metric.applicable_time_grain,
            "timeColumn": metric.time_column or table_binding.time_column,
            "timeSemantics": (
                dict(metric.time_semantics)
                if metric.time_semantics
                else {"asOfPolicy": metric.anchor_policy}
            ),
            "unit": metric.unit,
            "bindingType": metric.binding_type,
            "fieldAggregation": metric.field_aggregation,
            "sourceFieldRefId": metric.source_field_ref_id,
        }

    ordered_metrics = list(grounded.metrics)
    if grounded.query_shape == "RANKED":
        ordered_metrics.sort(
            key=lambda metric: 0
            if metric.semantic_ref_id == grounded.ranking.metric_ref_id
            else 1
        )
    anchor_metric = ordered_metrics[0]
    task_id = "grounded_%s_%s" % (
        grounded.query_shape.lower(),
        _safe_task_token(anchor_metric.metric_key),
    )
    source_columns = _dedupe(
        column for metric in ordered_metrics for column in metric.source_columns
    )
    intents = [
        QuestionIntent(
            question=grounded.question,
            intent_type=IntentType.VALID,
            answer_mode=answer_mode,
            plan_task_id=task_id,
            task_role=TaskRole.ANCHOR,
            preferred_table=anchor_metric.table,
            metric_column=(
                anchor_metric.source_columns[0]
                if anchor_metric.source_columns
                else ""
            ),
            metric_name=anchor_metric.metric_key,
            metric_formula=anchor_metric.formula,
            metric_specs=[metric_spec(metric, task_id) for metric in ordered_metrics],
            group_by_column=group_by,
            group_by_name=group_dimension.business_name if group_dimension else "",
            days=grounded.time_range.days,
            limit=(
                grounded.ranking.limit
                if grounded.query_shape == "RANKED"
                else 20 if group_dimension is not None else 1
            ),
            required_evidence=_dedupe([*source_columns, group_by]),
            # Physical output columns are produced only by explicit grouping.
            # Tenant/access columns remain internal filters.
            output_keys=[group_by] if group_by else [],
            knowledge_ref_ids=_dedupe(
                [
                    table_binding.detail_ref_id,
                    *(metric.semantic_ref_id for metric in ordered_metrics),
                    group_dimension.semantic_ref_id if group_dimension else "",
                ]
            ),
            analysis_source="grounded_query_contract",
            analysis_note="metricRefs=%s"
            % ",".join(metric.semantic_ref_id for metric in ordered_metrics),
            sql_strategy="structured_first",
            metric_resolution=metric_resolution(anchor_metric),
            time_range=grounded.time_range.model_copy(deep=True),
        )
    ]

    plan = QueryPlan(
        intents=intents,
        agent_trace=[
            "planner=grounded_query_contract_direct_compiler",
            "planner_llm_calls=0",
        ],
        question_understanding={
            "source": "grounded_query_contract",
            "contractVersion": grounded.contract_version,
            "queryShape": grounded.query_shape,
            "executionShape": grounded.execution_shape,
            "semanticSelectionRefs": list(grounded.evidence_refs),
        },
        compiler_trace=[
            "GROUNDED_DIRECT_COMPILE:%s" % grounded.execution_shape,
            "GROUNDED_GROUP_BY:%s" % (group_by or "none"),
        ],
        planner_loaded_refs=list(grounded.evidence_refs),
    )
    validation = _validate_grounded_plan_projection(grounded, plan, asset_pack)
    plan_fingerprint = query_graph_fingerprint(plan)
    return GroundedExecutionPreparation(
        plan=plan,
        validation=validation,
        source_plan_fingerprint=plan_fingerprint,
        execution_plan_fingerprint=plan_fingerprint,
        question_fingerprint=_stable_hash(grounded.question),
        asset_pack_fingerprint=_stable_hash(
            asset_pack.model_dump(by_alias=True, mode="json")
        ),
    )


def _safe_task_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "")).strip("_")
    return token[:64] or "metric"


def _validate_grounded_plan_projection(
    contract: GroundedQueryContract,
    plan: QueryPlan,
    asset_pack: PlanningAssetPack,
) -> GraphValidationResult:
    """Validate exact Contract-to-plan projection without question heuristics."""

    gaps: list[GraphValidationGap] = []
    if not contract.ready:
        gaps.append(
            GraphValidationGap(
                code="GROUNDED_CONTRACT_NOT_READY",
                reason="Only a READY GroundedQueryContract may be compiled",
            )
        )
    if len(plan.intents) != 1:
        gaps.append(
            GraphValidationGap(
                code="GROUNDED_NODE_COUNT_MISMATCH",
                reason="The current grounded compiler requires exactly one explicit execution node",
            )
        )
        return GraphValidationResult(valid=False, gaps=gaps, repairable=False)

    intent = plan.intents[0]
    shape_modes = {
        "SCALAR": AnswerMode.METRIC,
        "GROUPED": AnswerMode.GROUP_AGG,
        "TREND": AnswerMode.GROUP_AGG,
        "RANKED": AnswerMode.TOPN,
    }
    expected_mode = shape_modes.get(contract.query_shape)
    if expected_mode is None or intent.answer_mode != expected_mode:
        gaps.append(
            GraphValidationGap(
                code="GROUNDED_SHAPE_PROJECTION_MISMATCH",
                evidence=contract.query_shape,
                task_id=intent.plan_task_id,
                reason="answerMode is not the canonical Contract queryShape projection",
            )
        )

    metric_tables = {metric.table for metric in contract.metrics}
    expected_table = next(iter(metric_tables)) if len(metric_tables) == 1 else ""
    if not expected_table or intent.preferred_table != expected_table:
        gaps.append(
            GraphValidationGap(
                code="GROUNDED_TABLE_PROJECTION_MISMATCH",
                evidence=intent.preferred_table,
                task_id=intent.plan_task_id,
                reason="preferredTable is not the explicit Contract metric owner",
            )
        )

    group_dimensions = [
        dimension for dimension in contract.dimensions if dimension.usage == "group_by"
    ]
    expected_group = group_dimensions[0].column if len(group_dimensions) == 1 else ""
    if intent.group_by_column != expected_group:
        gaps.append(
            GraphValidationGap(
                code="GROUNDED_GROUP_PROJECTION_MISMATCH",
                evidence=intent.group_by_column,
                task_id=intent.plan_task_id,
                reason="groupByColumn differs from the explicit Contract dimension",
            )
        )

    table_binding = next(
        (table for table in contract.tables if table.table == intent.preferred_table),
        None,
    )
    merchant_column = table_binding.merchant_filter_column if table_binding else ""
    if merchant_column and (
        merchant_column == intent.group_by_column
        or merchant_column in intent.output_keys
        or merchant_column in intent.required_evidence
    ):
        gaps.append(
            GraphValidationGap(
                code="GROUNDED_SCOPE_COLUMN_EXPOSED",
                evidence=merchant_column,
                task_id=intent.plan_task_id,
                reason="merchantFilterColumn may only be consumed by execution scope filtering",
            )
        )

    spec_by_ref = {
        str(spec.get("semanticRefId") or ""): spec
        for spec in intent.metric_specs
        if isinstance(spec, dict) and str(spec.get("semanticRefId") or "")
    }
    expected_refs = {metric.semantic_ref_id for metric in contract.metrics}
    if set(spec_by_ref) != expected_refs:
        gaps.append(
            GraphValidationGap(
                code="GROUNDED_METRIC_SET_MISMATCH",
                evidence=",".join(sorted(set(spec_by_ref) ^ expected_refs)),
                task_id=intent.plan_task_id,
                reason="metricSpecs do not exactly cover the Contract metric bindings",
            )
        )
    for metric in contract.metrics:
        spec = spec_by_ref.get(metric.semantic_ref_id) or {}
        if (
            str(spec.get("metricName") or "") != metric.metric_key
            or str(spec.get("ownerTable") or "") != metric.table
            or str(spec.get("metricFormula") or "") != metric.formula
            or list(spec.get("sourceColumns") or []) != list(metric.source_columns)
        ):
            gaps.append(
                GraphValidationGap(
                    code="GROUNDED_METRIC_PROJECTION_MISMATCH",
                    evidence=metric.semantic_ref_id,
                    task_id=intent.plan_task_id,
                    reason="metricSpec drifted from its exact Contract binding",
                )
            )

    known_tables = set(asset_pack.known_tables())
    contract_tables = {table.table for table in contract.tables if table.table}
    if known_tables != contract_tables:
        gaps.append(
            GraphValidationGap(
                code="GROUNDED_PACK_TABLE_SCOPE_MISMATCH",
                evidence=",".join(sorted(known_tables ^ contract_tables)),
                reason="PlanningAssetPack must contain exactly the Contract-bound tables",
            )
        )
    known_columns = set(asset_pack.known_columns(intent.preferred_table))
    required_columns = {
        *[column for metric in contract.metrics for column in metric.source_columns],
        *([expected_group] if expected_group else []),
        *([table_binding.time_column] if table_binding and table_binding.time_column else []),
        *(
            [table_binding.merchant_filter_column]
            if table_binding and table_binding.merchant_filter_column
            else []
        ),
    }
    missing_columns = sorted(required_columns - known_columns)
    if missing_columns:
        gaps.append(
            GraphValidationGap(
                code="GROUNDED_PACK_COLUMN_MISSING",
                evidence=",".join(missing_columns),
                task_id=intent.plan_task_id,
                reason="Contract execution columns are missing from the materialized pack",
            )
        )
    return GraphValidationResult(valid=not gaps, gaps=gaps, repairable=False)


def _stable_hash(value: Any) -> str:
    payload = (
        value
        if isinstance(value, str)
        else json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )
    return hashlib.sha256(str(payload).encode("utf-8")).hexdigest()


def _execution_shape(
    metrics: Sequence[GroundedMetricBinding],
    dimensions: Sequence[GroundedDimensionBinding],
    tables: Sequence[str],
    ranking: GroundedRankingBinding,
) -> str:
    if ranking.enabled:
        return "ranked_group"
    if len(set(tables)) > 1:
        return "multi_table"
    if len(metrics) > 1:
        return "same_table_multi_metric"
    if metrics and any(dimension.usage == "group_by" for dimension in dimensions):
        return "grouped_metric"
    if metrics:
        return "single_metric"
    return "unresolved"


def _canonical_query_shape(
    hints: GroundedBindingHints,
    metrics: Sequence[GroundedMetricBinding],
    dimensions: Sequence[GroundedDimensionBinding],
    tables: Sequence[str],
    ranking: GroundedRankingBinding,
) -> str:
    """Canonicalize candidate bindings into one authoritative query shape.

    ``analysis_mode`` is only a shape request.  Exact semantic bindings remain
    mandatory and the validator rejects an incomplete requested shape.  The
    compiler consumes this normalized value and never reconstructs shape from
    question text or incidental execution controls such as ``limit``.
    """

    mode = str(hints.analysis_mode or "").strip().lower()
    if mode in {"topn", "ranking", "ranked_group"} or ranking.enabled:
        return "RANKED"
    if mode in {"trend", "time_series", "timeseries"}:
        return "TREND"
    if len(set(tables)) > 1:
        return "MULTI_TABLE"
    if any(dimension.usage == "group_by" for dimension in dimensions):
        return "GROUPED"
    if metrics:
        return "SCALAR"
    return "UNRESOLVED"


def _normalize_binding_hints(
    value: dict[str, Any] | GroundedBindingHints | None,
) -> GroundedBindingHints:
    if isinstance(value, GroundedBindingHints):
        return value.model_copy(deep=True)
    payload = dict(value or {})

    def ref_from(item: Any) -> str:
        if isinstance(item, str):
            return item.strip()
        if not isinstance(item, dict):
            return ""
        return str(
            item.get("refId")
            or item.get("ref_id")
            or item.get("metricRefId")
            or item.get("metric_ref_id")
            or item.get("semanticRefId")
            or item.get("semantic_ref_id")
            or item.get("detailRefId")
            or item.get("detail_ref_id")
            or ""
        ).strip()

    def refs_from(value: Any) -> list[str]:
        items = value if isinstance(value, list) else [value]
        return _dedupe(ref_from(item) for item in items if item is not None)

    if not (payload.get("tableRefs") or payload.get("table_refs")):
        table_refs: list[str] = []
        for key in (
            "tableRef",
            "table_ref",
            "selectedTableDetailRefId",
            "selected_table_detail_ref_id",
            "tableDetailRef",
            "table_detail_ref",
        ):
            table_refs.extend(refs_from(payload.get(key)))
        for key in ("table", "selectedTable", "selected_table"):
            table_refs.extend(refs_from(payload.get(key)))
        if table_refs:
            payload["tableRefs"] = _dedupe(table_refs)

    metric_items: list[Any] = []
    for key in (
        "metricBindings",
        "metric_bindings",
        "requestedMetrics",
        "requested_metrics",
        "metrics",
    ):
        raw = payload.get(key)
        if isinstance(raw, dict):
            metric_items.extend(raw.values())
        elif isinstance(raw, list):
            metric_items.extend(raw)
    if not (payload.get("metricRefs") or payload.get("metric_refs")):
        metric_refs: list[str] = []
        for key in ("metricRef", "metric_ref"):
            metric_refs.extend(refs_from(payload.get(key)))
        metric_refs.extend(refs_from(metric_items))
        if metric_refs:
            payload["metricRefs"] = _dedupe(metric_refs)

    dimension_items: list[Any] = []
    for key in ("dimensionBindings", "dimension_bindings", "dimensions"):
        raw = payload.get(key)
        if isinstance(raw, dict):
            dimension_items.extend(raw.values())
        elif isinstance(raw, list):
            dimension_items.extend(raw)
    if not (payload.get("dimensionRefs") or payload.get("dimension_refs")):
        dimension_refs = refs_from(dimension_items)
        if dimension_refs:
            payload["dimensionRefs"] = dimension_refs

    if not (payload.get("groupByRef") or payload.get("group_by_ref")):
        for key in ("groupBy", "group_by", "groupDimension", "group_dimension"):
            ref_id = ref_from(payload.get(key))
            if ref_id:
                payload["groupByRef"] = ref_id
                break

    if not (payload.get("timeExpression") or payload.get("time_expression")):
        time_expression = ""
        for key in ("timeWindow", "time_window", "timeRange", "time_range", "time"):
            raw = payload.get(key)
            if isinstance(raw, str):
                time_expression = raw.strip()
            elif isinstance(raw, dict):
                time_expression = str(
                    raw.get("userPhrase")
                    or raw.get("user_phrase")
                    or raw.get("phrase")
                    or raw.get("window")
                    or raw.get("label")
                    or ""
                ).strip()
            if time_expression:
                break
        if time_expression:
            payload["timeExpression"] = time_expression

    if not (payload.get("analysisMode") or payload.get("analysis_mode")):
        intent = str(payload.get("intent") or payload.get("shape") or "").strip().upper()
        if any(token in intent for token in ("RANK", "TOPN")):
            payload["analysisMode"] = "topn"
        elif any(token in intent for token in ("TREND", "TIME_SERIES")):
            payload["analysisMode"] = "trend"
        elif any(token in intent for token in ("GROUP", "DIMENSION")):
            payload["analysisMode"] = "grouped_metric"
        elif any(token in intent for token in ("METRIC", "SCALAR", "SUMMARY")):
            payload["analysisMode"] = "metric_total"

    raw_labels = payload.get("labelRefs", payload.get("label_refs", {}))
    if isinstance(raw_labels, list):
        labels: dict[str, str] = {}
        for item in raw_labels:
            if not isinstance(item, dict):
                continue
            ref_id = str(item.get("refId") or item.get("ref_id") or "")
            label = str(item.get("label") or item.get("phrase") or "")
            if ref_id and label:
                labels[ref_id] = label
        payload["labelRefs"] = labels
    elif not isinstance(raw_labels, dict):
        payload["labelRefs"] = {}
    if not payload.get("labelRefs") and not payload.get("label_refs"):
        labels: dict[str, str] = {}
        for item in [*metric_items, *dimension_items]:
            if not isinstance(item, dict):
                continue
            ref_id = ref_from(item)
            label = str(
                item.get("phrase")
                or item.get("requestedPhrase")
                or item.get("requested_phrase")
                or item.get("alias")
                or item.get("businessName")
                or item.get("business_name")
                or ""
            ).strip()
            if ref_id and label:
                labels[ref_id] = label
        if labels:
            payload["labelRefs"] = labels
    return GroundedBindingHints.model_validate(payload)


def _canonicalize_binding_hints(
    hints: GroundedBindingHints,
    available_refs: set[str],
) -> GroundedBindingHints:
    """Resolve legacy/guessed column ref spelling only against trusted reads.

    The Core may not manufacture executable refs.  This compatibility step is
    deliberately evidence-bound: ``:column:`` and ``:field:`` are considered
    aliases only when the alternate canonical ref already exists in the
    successful read ledger.
    """

    def canonical(ref_id: str) -> str:
        value = str(ref_id or "").strip()
        if not value or value in available_refs:
            return value
        alternates = []
        if ":column:" in value:
            alternates.append(value.replace(":column:", ":field:", 1))
        if ":field:" in value:
            alternates.append(value.replace(":field:", ":column:", 1))
        return next((item for item in alternates if item in available_refs), value)

    return hints.model_copy(
        update={
            "table_refs": _dedupe(canonical(item) for item in hints.table_refs),
            "metric_refs": _dedupe(canonical(item) for item in hints.metric_refs),
            "field_aggregations": [
                item.model_copy(update={"field_ref": canonical(item.field_ref)})
                for item in hints.field_aggregations
            ],
            "dimension_refs": _dedupe(canonical(item) for item in hints.dimension_refs),
            "group_by_ref": canonical(hints.group_by_ref),
            "relationship_refs": _dedupe(canonical(item) for item in hints.relationship_refs),
            "label_refs": {
                canonical(ref_id): label
                for ref_id, label in hints.label_refs.items()
            },
        }
    )


def _missing_binding_ref_gaps(
    hints: GroundedBindingHints,
    available_ref_kinds: dict[str, str],
) -> list[GroundedContractGap]:
    groups = [
        ("TABLE_BINDING_REF_NOT_READ", "TABLE_DETAIL", hints.table_refs),
        ("METRIC_BINDING_REF_NOT_READ", "METRIC", hints.metric_refs),
        (
            "FIELD_AGGREGATION_REF_NOT_READ",
            "COLUMN",
            [item.field_ref for item in hints.field_aggregations],
        ),
        ("DIMENSION_BINDING_REF_NOT_READ", "COLUMN", hints.dimension_refs),
        ("RELATIONSHIP_BINDING_REF_NOT_READ", "RELATIONSHIPS", hints.relationship_refs),
    ]
    gaps: list[GroundedContractGap] = []
    for code, kind, refs in groups:
        for ref_id in refs:
            observed_kind = str(available_ref_kinds.get(ref_id) or "").upper()
            if observed_kind == kind:
                continue
            if observed_kind:
                gaps.append(
                    _gap(
                        code.replace("_NOT_READ", "_WRONG_KIND"),
                        "Core selected ref %s as %s but the trusted read kind is %s"
                        % (ref_id, kind, observed_kind),
                        kind,
                    )
                )
                continue
            gaps.append(
                _gap(
                    code,
                    "Core selected ref %s but no successful trusted read exists" % ref_id,
                    kind,
                )
            )
    if hints.group_by_ref:
        observed_kind = str(available_ref_kinds.get(hints.group_by_ref) or "").upper()
        if observed_kind != "COLUMN":
            code = "GROUP_BY_BINDING_REF_WRONG_KIND" if observed_kind else "GROUP_BY_BINDING_REF_NOT_READ"
            message = (
                "Core selected groupByRef %s but the trusted read kind is %s"
                % (hints.group_by_ref, observed_kind)
                if observed_kind
                else "Core selected groupByRef %s but no successful trusted read exists" % hints.group_by_ref
            )
            gaps.append(_gap(code, message, "COLUMN"))
    return gaps


def _normalize_field_aggregation(value: str) -> str:
    normalized = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "COUNT": "COUNT",
        "COUNT_DISTINCT": "COUNT_DISTINCT",
        "DISTINCT_COUNT": "COUNT_DISTINCT",
        "COUNTDISTINCT": "COUNT_DISTINCT",
    }
    return aliases.get(normalized, "")


def _field_aggregation_formula(column: str, aggregation: str) -> str:
    if aggregation == "COUNT_DISTINCT":
        return "COUNT(DISTINCT `%s`)" % column
    if aggregation == "COUNT":
        return "COUNT(`%s`)" % column
    return ""


def _field_aggregation_metric_key(column: str, aggregation: str) -> str:
    prefix = "count_distinct" if aggregation == "COUNT_DISTINCT" else "count"
    return "%s_%s" % (prefix, column)


def _field_aggregation_business_name(business_name: str, aggregation: str) -> str:
    suffix = "去重数" if aggregation == "COUNT_DISTINCT" else "非空数"
    return "%s%s" % (business_name or "字段", suffix)


def _metric_planning_asset_ref(metric: GroundedMetricBinding) -> str:
    if metric.binding_type != "field_aggregation":
        return metric.semantic_ref_id
    return "grounded-field-aggregation:%s:%s" % (
        metric.semantic_ref_id,
        metric.field_aggregation.lower(),
    )


_REVISE_BINDING_GAP_CODES = {
    "TABLE_INSUFFICIENT",
    "REQUIRED_CAPABILITY_NOT_BOUND",
    "REJECTED_BINDING_REUSED",
}


def _grounded_contract_status(gaps: Sequence[GroundedContractGap]) -> str:
    blocking = [gap for gap in gaps if gap.blocking]
    if not blocking:
        return "READY"
    if any(gap.code in _REVISE_BINDING_GAP_CODES for gap in blocking):
        return "REVISE_BINDINGS"
    return "UNRESOLVED"


def _metric_usage_policy_violations(
    contract: GroundedQueryContract,
    metric: GroundedMetricBinding,
) -> list[str]:
    policy = dict(metric.calculation_capabilities or {})
    if not policy:
        return []
    violations: list[str] = []
    requested_days = max(0, int(contract.time_range.days or 0))
    native_window_days = max(0, int(policy.get("nativeWindowDays") or 0))
    window_policy = str(policy.get("windowPolicy") or "").strip().upper()
    time_rollup_policy = str(policy.get("timeRollupPolicy") or "").strip().upper()
    preserves_native_time_grain = _contract_preserves_native_time_grain(contract, policy)
    crosses_native_window = (
        native_window_days > 0
        and requested_days > native_window_days
        and not preserves_native_time_grain
    )

    if (
        window_policy == "EXACT_ONLY"
        and native_window_days > 0
        and requested_days > 0
        and requested_days != native_window_days
    ):
        violations.append("WINDOW_MISMATCH")
    if (
        time_rollup_policy == "NOT_COMPOSABLE"
        and crosses_native_window
    ):
        violations.append("TIME_ROLLUP_NOT_COMPOSABLE")

    actual_aggregation = _formula_top_level_aggregation(metric.formula)
    forbidden = {
        str(item or "").strip().upper()
        for item in policy.get("forbiddenAggregations") or []
        if str(item or "").strip()
    }
    allowed = {
        str(item or "").strip().upper()
        for item in policy.get("allowedAggregations") or []
        if str(item or "").strip()
    }
    enforce_aggregation_at_native_grain = bool(policy.get("enforceAggregationAtNativeGrain"))
    if (
        actual_aggregation
        and actual_aggregation in forbidden
        and (crosses_native_window or enforce_aggregation_at_native_grain)
    ):
        violations.append("FORBIDDEN_AGGREGATION:%s" % actual_aggregation)
    if (
        allowed
        and actual_aggregation
        and actual_aggregation not in allowed
        and (crosses_native_window or enforce_aggregation_at_native_grain)
    ):
        violations.append("AGGREGATION_NOT_ALLOWED:%s" % actual_aggregation)

    required_components = {
        str(item or "").strip()
        for item in policy.get("requiredComponents") or []
        if str(item or "").strip()
    }
    source_columns = set(metric.source_columns)
    missing_components = sorted(required_components - source_columns)
    if missing_components:
        violations.append("REQUIRED_COMPONENTS_MISSING:%s" % ",".join(missing_components))
    required_weight = str(policy.get("requiredWeightRef") or "").strip()
    if required_weight and required_weight not in source_columns:
        violations.append("WEIGHT_REF_MISSING:%s" % required_weight)

    allowed_grains = {
        str(item or "").strip()
        for item in policy.get("allowedTableGrains") or []
        if str(item or "").strip()
    }
    if allowed_grains and metric.metric_grain and metric.metric_grain not in allowed_grains:
        violations.append("TABLE_GRAIN_NOT_ALLOWED:%s" % metric.metric_grain)
    return _dedupe(violations)


def _contract_preserves_native_time_grain(
    contract: GroundedQueryContract,
    policy: dict[str, Any],
) -> bool:
    native_time_grain = str(policy.get("nativeTimeGrain") or "").strip().upper()
    if not native_time_grain:
        return False
    analysis_mode = str(contract.analysis_mode or "").strip().upper()
    declared_modes = {
        str(item or "").strip().upper()
        for item in policy.get("nativeGrainAnalysisModes") or []
        if str(item or "").strip()
    }
    if declared_modes and analysis_mode in declared_modes:
        return True
    return any(
        dimension.role.upper() == "TIME"
        and str(dimension.usage or "").lower() == "group_by"
        for dimension in contract.dimensions
    )


def _formula_top_level_aggregation(formula: str) -> str:
    compact = re.sub(r"\s+", "", str(formula or "")).upper()
    if compact.startswith("COUNT(DISTINCT"):
        return "COUNT_DISTINCT"
    for operation in ("SUM", "AVG", "COUNT", "MIN", "MAX", "MAX_BY", "MIN_BY", "NDV"):
        if compact.startswith(operation + "("):
            return operation
    if "/" in compact or compact.startswith("SAFE_DIVIDE("):
        return "RATIO"
    return "EXPRESSION" if compact else ""


def semantic_evidence_calculation_capabilities(
    kind: str,
    payload: Any,
) -> dict[str, Any]:
    """Read calculation/usage semantics without interpreting business names."""

    if not isinstance(payload, dict):
        return {}
    normalized_kind = str(kind or "").upper()
    if normalized_kind == "METRIC":
        definition = payload.get("metric") if isinstance(payload.get("metric"), dict) else payload
        explicit = (
            definition.get("calculationSemantics")
            or definition.get("usagePolicy")
            or definition.get("calculationCapabilities")
            or {}
        )
        result = dict(explicit) if isinstance(explicit, dict) else {}
        result.setdefault("declaredAggregation", _formula_top_level_aggregation(str(definition.get("formula") or definition.get("metricFormula") or "")))
        return result
    if normalized_kind == "COLUMN":
        definition = payload.get("definition") if isinstance(payload.get("definition"), dict) else payload
        explicit = (
            definition.get("calculationSemantics")
            or definition.get("usagePolicy")
            or definition.get("calculationCapabilities")
            or {}
        )
        return dict(explicit) if isinstance(explicit, dict) else {}
    return {}


def grounded_rejection_fingerprint(
    question: str,
    table: str,
    ref_ids: Sequence[str],
    required_capability: dict[str, Any],
) -> str:
    payload = {
        "question": str(question or "").strip(),
        "table": str(table or "").strip(),
        "refIds": sorted(str(item) for item in ref_ids if str(item)),
        "requiredCapability": required_capability,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _rejected_bindings_for_contract(
    contract: GroundedQueryContract,
    gaps: Sequence[GroundedContractGap],
) -> list[GroundedRejectedBinding]:
    rejected: list[GroundedRejectedBinding] = []
    seen: set[str] = set()
    for gap in gaps:
        if gap.code not in {"TABLE_INSUFFICIENT", "REQUIRED_CAPABILITY_NOT_BOUND"}:
            continue
        target_tables = [gap.table] if gap.table else [table.table for table in contract.tables if table.table]
        for table_name in _dedupe(target_tables):
            table = next((item for item in contract.tables if item.table == table_name), None)
            ref_ids = _dedupe(
                [
                    *gap.rejected_ref_ids,
                    *(
                        [table.detail_ref_id]
                        if table is not None and table.detail_ref_id
                        else []
                    ),
                    *[
                        metric.semantic_ref_id
                        for metric in contract.metrics
                        if metric.table == table_name
                    ],
                ]
            )
            fingerprint = grounded_rejection_fingerprint(
                contract.question,
                table_name,
                ref_ids,
                gap.required_capability,
            )
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            rejected.append(
                GroundedRejectedBinding(
                    fingerprint=fingerprint,
                    code=gap.code,
                    topic=(table.topic if table is not None else gap.topic),
                    table=table_name,
                    ref_ids=ref_ids,
                    reason=gap.message,
                    required_capability=dict(gap.required_capability),
                )
            )
    return rejected


def merge_grounded_rejected_bindings(
    prior: Iterable[GroundedRejectedBinding | dict[str, Any]],
    current: Iterable[GroundedRejectedBinding | dict[str, Any]],
) -> list[GroundedRejectedBinding]:
    merged: list[GroundedRejectedBinding] = []
    seen: set[str] = set()
    for raw in [*list(prior), *list(current)]:
        try:
            item = raw if isinstance(raw, GroundedRejectedBinding) else GroundedRejectedBinding.model_validate(raw)
        except Exception:
            continue
        if item.fingerprint in seen:
            continue
        seen.add(item.fingerprint)
        merged.append(item)
    return merged[-32:]


def _safe_mapping_call(target: Any, method: str, *args: Any) -> dict[str, Any]:
    callback = getattr(target, method, None)
    if not callable(callback):
        return {}
    try:
        value = callback(*args)
    except Exception:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _safe_list_call(target: Any, method: str, *args: Any) -> list[dict[str, Any]]:
    callback = getattr(target, method, None)
    if not callable(callback):
        return []
    try:
        value = callback(*args)
    except Exception:
        return []
    return [dict(item) for item in value or [] if isinstance(item, dict)] if isinstance(value, list) else []


def _tables_connected(tables: set[str], relationships: Sequence[GroundedRelationshipBinding]) -> bool:
    if len(tables) <= 1:
        return True
    graph: dict[str, set[str]] = {table: set() for table in tables}
    for relationship in relationships:
        if relationship.left_table in graph and relationship.right_table in graph:
            graph[relationship.left_table].add(relationship.right_table)
            graph[relationship.right_table].add(relationship.left_table)
    visited: set[str] = set()
    pending = [next(iter(tables))]
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        pending.extend(graph.get(current, set()) - visited)
    return visited == tables


def _metric_phrase_in_question(question: str, aliases: Sequence[str]) -> str:
    """Preserve the user's wording after semantic binding has already succeeded."""

    normalized_question = re.sub(r"[\s_\-—·]+", "", str(question or "")).lower()
    matches = [
        str(alias).strip()
        for alias in aliases
        if str(alias or "").strip()
        and re.sub(r"[\s_\-—·]+", "", str(alias)).lower() in normalized_question
    ]
    return max(matches, key=lambda item: len(re.sub(r"\s+", "", item)), default="")


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _dedupe_gaps(gaps: Iterable[GroundedContractGap]) -> list[GroundedContractGap]:
    result: list[GroundedContractGap] = []
    seen: set[tuple[str, str, str, str]] = set()
    for gap in gaps:
        identity = (gap.code, gap.topic, gap.table, gap.phrase)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(gap)
    return result


def _gap(
    code: str,
    message: str,
    evidence_kind: str = "",
    topic: str = "",
    table: str = "",
    phrase: str = "",
    resolution: str = "",
    search_scope: str = "",
    required_capability: dict[str, Any] | None = None,
    rejected_ref_ids: Sequence[str] | None = None,
) -> GroundedContractGap:
    return GroundedContractGap(
        code=code,
        message=message,
        evidence_kind=evidence_kind,
        topic=topic,
        table=table,
        phrase=phrase,
        resolution=resolution,
        search_scope=search_scope,
        required_capability=dict(required_capability or {}),
        rejected_ref_ids=_dedupe(str(item) for item in rejected_ref_ids or []),
    )


def _binding_ref_gap(code: str, ref_id: str, topic: str, table: str) -> GroundedContractGap:
    return _gap(code, "Binding ref %s is not present in the trusted evidence set" % ref_id, topic=topic, table=table)
