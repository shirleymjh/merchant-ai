from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Literal, Mapping, Sequence

from pydantic import Field, computed_field, model_validator

from merchant_ai.models import APIModel


GroundedArtifactType = Literal[
    "SCALAR",
    "ENTITY_SET",
    "RELATION",
    "BASELINE",
    "RULE",
    "ANALYSIS",
]

_CALCULATION_NODE_TYPES = {
    "PHYSICAL_METRIC",
    "METRIC_REF",
    "DIMENSION_REF",
    "TIME_REF",
    "TRUSTED_ARTIFACT_REF",
    "AGGREGATE",
    "ALIGN",
    "JOIN",
    "FORMULA",
    "COMPOSITE_METRIC",
    "WINDOW",
    "FILTER",
    "PROJECT",
    "CONSTANT",
}


class GroundedSemanticType(APIModel):
    """Business type carried by every semantic calculation node.

    ``row_grain`` describes what one output row represents.  ``time_grain``
    and ``window_policy`` make temporal compatibility explicit instead of
    leaving it to SQL text or table names.  The model is intentionally
    dialect-independent and can therefore be shared by SQL, batch and Skill
    execution adapters.
    """

    row_grain: list[str] = Field(default_factory=list)
    entity_keys: list[str] = Field(default_factory=list)
    time_grain: str = ""
    time_column: str = ""
    window_policy: str = ""
    value_type: str = ""
    unit: str = ""


class GroundedCalculationNode(APIModel):
    node_id: str
    node_type: str
    semantic_ref_id: str = ""
    metric_key: str = ""
    binding_key: str = ""
    table: str = ""
    expression: str = ""
    source_columns: list[str] = Field(default_factory=list)
    input_node_ids: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    semantic_type: GroundedSemanticType = Field(
        default_factory=GroundedSemanticType
    )
    internal: bool = True

    @model_validator(mode="after")
    def normalize_identity(self) -> "GroundedCalculationNode":
        self.node_id = str(self.node_id or "").strip()
        self.node_type = str(self.node_type or "").strip().upper()
        if not self.node_id:
            raise ValueError("calculation node id must not be empty")
        if not self.node_type:
            raise ValueError("calculation node type must not be empty")
        if self.node_type not in _CALCULATION_NODE_TYPES:
            raise ValueError(
                "unsupported calculation node type: %s" % self.node_type
            )
        self.input_node_ids = _dedupe(self.input_node_ids)
        self.source_columns = _dedupe(self.source_columns)
        return self


class GroundedCalculationGraph(APIModel):
    """Canonical semantic calculation IR for one requested output metric."""

    graph_version: str = "grounded_calculation_graph.v1"
    output_node_ids: list[str] = Field(default_factory=list)
    nodes: list[GroundedCalculationNode] = Field(default_factory=list)
    expression: str = ""
    component_metric_refs: list[str] = Field(default_factory=list)
    component_metric_keys: list[str] = Field(default_factory=list)
    alignment: dict[str, Any] = Field(default_factory=dict)
    provenance: str = "published_semantic_metric"

    @computed_field
    @property
    def composite(self) -> bool:
        return bool(
            self.component_metric_refs
            or self.component_metric_keys
            or any(
                node.node_type
                in {
                    "COMPOSITE_METRIC",
                    "FORMULA",
                    "WINDOW",
                    "ALIGN",
                    "JOIN",
                }
                and node.input_node_ids
                for node in self.nodes
            )
        )

    @model_validator(mode="after")
    def validate_graph(self) -> "GroundedCalculationGraph":
        self.output_node_ids = _dedupe(self.output_node_ids)
        self.component_metric_refs = _dedupe(self.component_metric_refs)
        self.component_metric_keys = _dedupe(self.component_metric_keys)
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("calculation graph node ids must be unique")
        known = set(node_ids)
        missing_outputs = sorted(set(self.output_node_ids) - known)
        if missing_outputs:
            raise ValueError(
                "calculation graph output nodes are missing: %s"
                % ",".join(missing_outputs)
            )
        missing_inputs = sorted(
            {
                input_id
                for node in self.nodes
                for input_id in node.input_node_ids
                if input_id not in known
            }
        )
        if missing_inputs:
            raise ValueError(
                "calculation graph input nodes are missing: %s"
                % ",".join(missing_inputs)
            )
        if _graph_has_cycle(self.nodes):
            raise ValueError("calculation graph must be acyclic")
        return self


class GroundedOutputProjection(APIModel):
    semantic_ref_id: str
    output_alias: str
    binding_kind: str = "METRIC"
    calculation_node_id: str = ""


class GroundedTrustedArtifactDescriptor(APIModel):
    """One protocol for verified scalar, entity, relation and baseline data."""

    descriptor_version: str = "grounded_trusted_artifact.v1"
    artifact_id: str
    artifact_type: GroundedArtifactType
    output_schema: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias="schema",
        serialization_alias="schema",
    )
    semantic_lineage: dict[str, list[str]] = Field(default_factory=dict)
    covered_goal_ids: list[str] = Field(default_factory=list)
    source_artifact_ids: list[str] = Field(default_factory=list)
    merchant_scope_fingerprint: str = ""
    time_scope: dict[str, Any] = Field(default_factory=dict)
    grain: GroundedSemanticType = Field(default_factory=GroundedSemanticType)
    coverage_status: str = "UNKNOWN"
    snapshot_id: str = ""
    immutable: bool = True
    values_hash: str = ""
    row_count: int = 0


def calculation_graph_from_metric_asset(
    metric: Mapping[str, Any],
    *,
    semantic_ref_id: str,
    topic: str,
    table: str,
) -> GroundedCalculationGraph:
    """Compile legacy and new metric asset fields into one stable IR.

    ``requiresMetrics`` and ``metricDependencies`` remain accepted migration
    inputs.  New assets should prefer ``componentMetricRefs`` and
    ``calculationExpression`` so dependency identity is unambiguous.
    """

    published_graph = metric.get("calculationGraph")
    if isinstance(published_graph, Mapping):
        return GroundedCalculationGraph.model_validate(published_graph)

    metric_key = str(metric.get("metricKey") or "").strip()
    expression = str(
        metric.get("calculationExpression")
        or metric.get("compositeExpression")
        or metric.get("formula")
        or metric.get("metricFormula")
        or ""
    ).strip()
    component_refs: list[str] = []
    component_keys: list[str] = []
    raw_components = [
        *list(metric.get("componentMetricRefs") or []),
        *list(metric.get("requiresMetrics") or []),
        *list(metric.get("metricDependencies") or []),
    ]
    component_specs: list[dict[str, str]] = []
    for raw in raw_components:
        if isinstance(raw, Mapping):
            ref_id = str(
                raw.get("semanticRefId")
                or raw.get("metricRef")
                or raw.get("refId")
                or ""
            ).strip()
            key = str(
                raw.get("metricKey")
                or raw.get("key")
                or raw.get("alias")
                or ""
            ).strip()
        else:
            value = str(raw or "").strip()
            ref_id = value if value.startswith("semantic:") else ""
            key = "" if ref_id else value
        if ref_id:
            component_refs.append(ref_id)
        if key:
            component_keys.append(key)
        if ref_id or key:
            component_specs.append({"refId": ref_id, "metricKey": key})

    component_refs = _dedupe(component_refs)
    component_keys = _dedupe(component_keys)
    component_nodes: list[GroundedCalculationNode] = []
    component_node_ids: list[str] = []
    seen_components: set[tuple[str, str]] = set()
    for index, spec in enumerate(component_specs, 1):
        identity = (spec["refId"], spec["metricKey"])
        if identity in seen_components:
            continue
        seen_components.add(identity)
        node_id = "component_%d" % index
        component_node_ids.append(node_id)
        component_nodes.append(
            GroundedCalculationNode(
                node_id=node_id,
                node_type="METRIC_REF",
                semantic_ref_id=spec["refId"],
                metric_key=spec["metricKey"],
                internal=True,
            )
        )

    alignment = (
        dict(metric.get("alignment") or {})
        if isinstance(metric.get("alignment"), Mapping)
        else {}
    )
    type_spec = GroundedSemanticType(
        row_grain=_string_list(
            alignment.get("entityGrain")
        ),
        entity_keys=_string_list(
            alignment.get("entityKeys")
        ),
        time_grain=str(
            alignment.get("timeGrain")
            or metric.get("applicableTimeGrain")
            or ""
        ),
        time_column=str(metric.get("timeColumn") or ""),
        window_policy=str(
            alignment.get("timePolicy")
            or ((metric.get("timeSemantics") or {}).get("selectionPolicy") or "")
            if isinstance(metric.get("timeSemantics"), Mapping)
            else alignment.get("timePolicy") or ""
        ),
        value_type=str(metric.get("metricType") or ""),
        unit=str(metric.get("unit") or ""),
    )
    output_node = GroundedCalculationNode(
        node_id="output",
        node_type=("COMPOSITE_METRIC" if component_node_ids else "PHYSICAL_METRIC"),
        semantic_ref_id=semantic_ref_id,
        metric_key=metric_key,
        table=table,
        expression=expression,
        source_columns=_string_list(metric.get("sourceColumns") or []),
        input_node_ids=component_node_ids,
        parameters={
            "topic": topic,
            "aggregationPolicy": str(metric.get("aggregationPolicy") or ""),
        },
        semantic_type=type_spec,
        internal=False,
    )
    return GroundedCalculationGraph(
        output_node_ids=["output"],
        nodes=[*component_nodes, output_node],
        expression=expression,
        component_metric_refs=component_refs,
        component_metric_keys=component_keys,
        alignment=alignment,
        provenance=(
            "compiled_legacy_metric_dependencies"
            if raw_components and not metric.get("componentMetricRefs")
            else "published_semantic_metric"
        ),
    )


def bind_calculation_graph_components(
    graph: GroundedCalculationGraph,
    components: Sequence[Mapping[str, str]],
) -> GroundedCalculationGraph:
    """Return a graph whose component nodes carry exact governed identity."""

    remaining = [dict(item) for item in components]
    nodes: list[GroundedCalculationNode] = []
    exact_refs: list[str] = []
    exact_keys: list[str] = []
    for node in graph.nodes:
        if node.node_type != "METRIC_REF":
            nodes.append(node.model_copy(deep=True))
            continue
        match_index = next(
            (
                index
                for index, item in enumerate(remaining)
                if (
                    node.semantic_ref_id
                    and node.semantic_ref_id
                    == str(item.get("semanticRefId") or "")
                )
                or (
                    node.metric_key
                    and node.metric_key == str(item.get("metricKey") or "")
                )
            ),
            -1,
        )
        if match_index < 0:
            nodes.append(node.model_copy(deep=True))
            continue
        item = remaining.pop(match_index)
        ref_id = str(item.get("semanticRefId") or node.semantic_ref_id or "")
        key = str(item.get("metricKey") or node.metric_key or "")
        exact_refs.append(ref_id)
        exact_keys.append(key)
        nodes.append(
            node.model_copy(
                update={
                    "semantic_ref_id": ref_id,
                    "metric_key": key,
                    "table": str(item.get("table") or ""),
                    "semantic_type": GroundedSemanticType.model_validate(
                        item.get("semanticType") or {}
                    ),
                },
                deep=True,
            )
        )
    return graph.model_copy(
        update={
            "nodes": nodes,
            "component_metric_refs": _dedupe(exact_refs),
            "component_metric_keys": _dedupe(exact_keys),
        },
        deep=True,
    )


def compose_query_calculation_graph(
    requested_metrics: Sequence[Any],
    internal_metrics: Sequence[Any],
    upstream_artifacts: Sequence[Any] = (),
) -> tuple[GroundedCalculationGraph, dict[str, str]]:
    """Compose metric-local graphs into one query-level semantic DAG."""

    metric_bindings = [*internal_metrics, *requested_metrics]
    output_node_by_ref: dict[str, str] = {}
    prefix_by_ref: dict[str, str] = {}
    for metric in metric_bindings:
        ref_id = str(getattr(metric, "semantic_ref_id", "") or "")
        prefix = "metric_%s" % hashlib.sha256(
            ref_id.encode("utf-8")
        ).hexdigest()[:16]
        prefix_by_ref[ref_id] = prefix
        graph = getattr(metric, "calculation_graph", None)
        output_ids = list(getattr(graph, "output_node_ids", None) or [])
        if output_ids:
            output_node_by_ref[ref_id] = "%s__%s" % (
                prefix,
                output_ids[0],
            )

    nodes: list[GroundedCalculationNode] = []
    for metric in metric_bindings:
        ref_id = str(getattr(metric, "semantic_ref_id", "") or "")
        graph = getattr(metric, "calculation_graph", None)
        prefix = prefix_by_ref.get(ref_id, "")
        for node in list(getattr(graph, "nodes", None) or []):
            input_ids = [
                "%s__%s" % (prefix, input_id)
                for input_id in node.input_node_ids
            ]
            if node.node_type == "METRIC_REF" and node.semantic_ref_id:
                component_output = output_node_by_ref.get(
                    node.semantic_ref_id,
                    "",
                )
                if component_output and component_output not in input_ids:
                    input_ids.append(component_output)
            nodes.append(
                node.model_copy(
                    update={
                        "node_id": "%s__%s" % (prefix, node.node_id),
                        "input_node_ids": input_ids,
                    },
                    deep=True,
                )
            )

    for binding in upstream_artifacts:
        descriptor = getattr(binding, "descriptor", None)
        artifact_id = str(getattr(binding, "artifact_id", "") or "")
        target_ref = str(
            getattr(binding, "target_binding_ref", "") or ""
        )
        node_id = "artifact_%s" % hashlib.sha256(
            artifact_id.encode("utf-8")
        ).hexdigest()[:16]
        nodes.append(
            GroundedCalculationNode(
                node_id=node_id,
                node_type="TRUSTED_ARTIFACT_REF",
                semantic_ref_id=target_ref,
                parameters={
                    "artifactId": artifact_id,
                    "artifactKind": str(
                        getattr(binding, "artifact_kind", "") or ""
                    ),
                    "descriptor": (
                        descriptor.model_dump(
                            by_alias=True,
                            mode="json",
                        )
                        if hasattr(descriptor, "model_dump")
                        else {}
                    ),
                },
                semantic_type=(
                    descriptor.grain.model_copy(deep=True)
                    if descriptor is not None
                    else GroundedSemanticType()
                ),
                internal=True,
            )
        )
        for index, node in enumerate(nodes):
            if (
                node.node_id == node_id
                or node.semantic_ref_id != target_ref
                or node.node_type == "TRUSTED_ARTIFACT_REF"
            ):
                continue
            if node_id not in node.input_node_ids:
                nodes[index] = node.model_copy(
                    update={
                        "input_node_ids": [
                            *node.input_node_ids,
                            node_id,
                        ]
                    },
                    deep=True,
                )

    requested_refs = [
        str(getattr(metric, "semantic_ref_id", "") or "")
        for metric in requested_metrics
    ]
    output_ids = [
        output_node_by_ref[ref_id]
        for ref_id in requested_refs
        if ref_id in output_node_by_ref
    ]
    return (
        GroundedCalculationGraph(
            graph_version="grounded_query_calculation_graph.v1",
            output_node_ids=output_ids,
            nodes=nodes,
            provenance="grounded_query_contract",
        ),
        output_node_by_ref,
    )


def calculation_graph_fingerprint(graph: GroundedCalculationGraph) -> str:
    encoded = json.dumps(
        graph.model_dump(by_alias=True, mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def trusted_query_artifact_descriptor(
    artifact: Any,
    *,
    merchant_scope_fingerprint: str = "",
    covered_goal_ids: Sequence[str] = (),
) -> GroundedTrustedArtifactDescriptor:
    """Adapt the existing verified-query object to the unified protocol."""

    contract = getattr(artifact, "contract", None)
    run_result = getattr(artifact, "run_result", None)
    bundle = getattr(run_result, "merged_query_bundle", None)
    rows = list(getattr(bundle, "rows", None) or [])
    output_columns = _dedupe(getattr(artifact, "output_columns", None) or [])
    semantic_refs = dict(
        getattr(artifact, "output_semantic_refs", None) or {}
    )
    lineage = {
        str(key): _dedupe(value or [])
        for key, value in dict(
            getattr(artifact, "output_lineage", None) or {}
        ).items()
    }
    query_shape = str(getattr(contract, "query_shape", "") or "").upper()
    artifact_type: GroundedArtifactType = (
        "SCALAR" if query_shape == "SCALAR" and len(rows) <= 1 else "RELATION"
    )
    grain = GroundedSemanticType()
    for metric in list(getattr(contract, "metrics", None) or []):
        graph = getattr(metric, "calculation_graph", None)
        output_ids = list(getattr(graph, "output_node_ids", None) or [])
        nodes = list(getattr(graph, "nodes", None) or [])
        output_node = next(
            (
                node
                for node in nodes
                if str(getattr(node, "node_id", "") or "") in output_ids
            ),
            None,
        )
        if output_node is not None:
            grain = getattr(output_node, "semantic_type", grain).model_copy(
                deep=True
            )
            break
    time_range = getattr(contract, "time_range", None)
    time_scope = (
        time_range.model_dump(by_alias=True, mode="json")
        if hasattr(time_range, "model_dump")
        else {}
    )
    snapshot = getattr(bundle, "data_snapshot", None)
    snapshot_payload = (
        snapshot.model_dump(by_alias=True, mode="json")
        if hasattr(snapshot, "model_dump")
        else {}
    )
    values_hash = str(
        getattr(artifact, "run_result_fingerprint", "") or ""
    ) or _stable_hash(rows)
    return GroundedTrustedArtifactDescriptor(
        artifact_id=str(getattr(artifact, "artifact_id", "") or ""),
        artifact_type=artifact_type,
        schema={
            "columns": [
                {
                    "name": column,
                    "semanticRefId": str(semantic_refs.get(column) or ""),
                }
                for column in output_columns
            ]
        },
        semantic_lineage=lineage,
        covered_goal_ids=_dedupe(covered_goal_ids),
        merchant_scope_fingerprint=str(merchant_scope_fingerprint or ""),
        time_scope=time_scope,
        grain=grain,
        coverage_status=str(
            getattr(bundle, "result_coverage", "") or "UNKNOWN"
        ),
        snapshot_id=_stable_hash(snapshot_payload) if snapshot_payload else "",
        immutable=True,
        values_hash=values_hash,
        row_count=len(rows),
    )


def trusted_entity_set_descriptor(
    artifact: Any,
    *,
    merchant_scope_fingerprint: str = "",
    covered_goal_ids: Sequence[str] = (),
) -> GroundedTrustedArtifactDescriptor:
    """Adapt a verified entity set without exposing its values to the model."""

    semantic_ref = str(
        getattr(artifact, "source_semantic_ref_id", "") or ""
    )
    source_column = str(getattr(artifact, "source_column", "") or "")
    return GroundedTrustedArtifactDescriptor(
        artifact_id=str(getattr(artifact, "artifact_id", "") or ""),
        artifact_type="ENTITY_SET",
        schema={
            "columns": [
                {
                    "name": source_column,
                    "semanticRefId": semantic_ref,
                }
            ]
        },
        semantic_lineage={source_column: [semantic_ref] if semantic_ref else []},
        covered_goal_ids=_dedupe(covered_goal_ids),
        source_artifact_ids=_dedupe(
            [str(getattr(artifact, "source_query_artifact_id", "") or "")]
        ),
        merchant_scope_fingerprint=str(merchant_scope_fingerprint or ""),
        grain=GroundedSemanticType(
            row_grain=_dedupe(
                [str(getattr(artifact, "source_entity_identity", "") or "")]
            ),
            entity_keys=[source_column] if source_column else [],
        ),
        coverage_status=(
            "TRUNCATED" if bool(getattr(artifact, "truncated", False)) else "ALL_ROWS"
        ),
        immutable=True,
        values_hash=str(getattr(artifact, "values_hash", "") or ""),
        row_count=int(getattr(artifact, "value_count", 0) or 0),
    )


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _graph_has_cycle(nodes: Sequence[GroundedCalculationNode]) -> bool:
    graph = {node.node_id: list(node.input_node_ids) for node in nodes}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> bool:
        if node_id in visiting:
            return True
        if node_id in visited:
            return False
        visiting.add(node_id)
        for dependency in graph.get(node_id, []):
            if visit(dependency):
                return True
        visiting.remove(node_id)
        visited.add(node_id)
        return False

    return any(visit(node_id) for node_id in graph)


def _string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return _dedupe(str(item or "").strip() for item in value)
    return [str(value)]


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value or "").strip() for value in values if str(value or "").strip()))
