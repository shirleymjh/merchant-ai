from __future__ import annotations

import json

from merchant_ai.models import PlanningAssetEntry, PlanningAssetPack, RelationshipEntry
from merchant_ai.services.semantic_joins import plan_governed_joins


def _table(
    name: str,
    *,
    id_type: str = "bigint",
    parent_type: str = "bigint",
    tenant_scoped: bool | None = None,
) -> PlanningAssetEntry:
    columns = ["id", "parent_id"]
    schema_columns = [
        {"columnName": "id", "dataType": id_type, "semanticRole": "PRIMARY_KEY"},
        {"columnName": "parent_id", "dataType": parent_type, "semanticRole": "JOIN_KEY"},
    ]
    metadata: dict = {"schemaColumns": schema_columns}
    if tenant_scoped is True:
        columns.append("scope_key")
        schema_columns.append(
            {
                "columnName": "scope_key",
                "dataType": "string",
                "semanticRole": "TENANT_SCOPE_KEY",
            }
        )
        metadata["rowAccessPolicy"] = {
            "scopeType": "TENANT",
            "filterColumn": "scope_key",
            "required": True,
        }
    elif tenant_scoped is False:
        metadata["tenantScoped"] = False
    return PlanningAssetEntry(key=name, table=name, columns=columns, metadata=metadata)


def _relationship(
    ref_id: str,
    left: str,
    right: str,
    *,
    left_column: str = "parent_id",
    right_column: str = "id",
    join_keys: list[dict[str, str]] | None = None,
    metadata: dict | None = None,
) -> RelationshipEntry:
    return RelationshipEntry(
        relationship_id=ref_id.rsplit(":", 1)[-1],
        source_ref_id=ref_id,
        left_table=left,
        right_table=right,
        join_keys=join_keys or [{"leftColumn": left_column, "rightColumn": right_column}],
        description=json.dumps(metadata or {}),
    )


def test_equal_minimum_relationship_paths_fail_closed_as_ambiguous() -> None:
    pack = PlanningAssetPack(
        tables=[_table(name) for name in ["base", "path_a", "path_b", "target"]],
        relationships=[
            _relationship("semantic:rel:base_a", "base", "path_a"),
            _relationship("semantic:rel:a_target", "path_a", "target"),
            _relationship("semantic:rel:base_b", "base", "path_b"),
            _relationship("semantic:rel:b_target", "path_b", "target"),
        ],
    )

    result = plan_governed_joins(pack, "base", ["target"])

    assert result.plan is None
    assert [gap.code for gap in result.gaps] == ["JOIN_PATH_AMBIGUOUS"]
    assert "semantic:rel:base_a" in result.gaps[0].evidence
    assert "semantic:rel:base_b" in result.gaps[0].evidence


def test_relationship_without_join_keys_is_not_executable() -> None:
    relationship = _relationship("semantic:rel:missing_keys", "base", "target")
    relationship.join_keys = []
    pack = PlanningAssetPack(tables=[_table("base"), _table("target")], relationships=[relationship])

    result = plan_governed_joins(pack, "base", ["target"])

    assert result.plan is None
    assert [gap.code for gap in result.gaps] == ["JOIN_KEYS_MISSING", "JOIN_PATH_NOT_FOUND"]
    assert result.gaps[0].relationship_ref_id == "semantic:rel:missing_keys"


def test_filter_scope_uses_exists_without_requiring_fanout_metadata() -> None:
    relationship = _relationship("semantic:rel:base_children", "base", "children")
    pack = PlanningAssetPack(tables=[_table("base"), _table("children")], relationships=[relationship])

    result = plan_governed_joins(
        pack,
        "base",
        ["children"],
        relationship_ref_ids=["semantic:rel:base_children"],
        usage="filter_scope",
    )

    assert result.valid
    assert result.plan is not None
    assert result.plan.execution_strategy == "semi_join"
    assert len(result.plan.steps) == 1
    assert result.plan.steps[0].strategy == "exists"
    assert result.plan.steps[0].fanout_safe is True
    assert result.plan.steps[0].keys[0].from_column == "parent_id"
    assert result.plan.steps[0].keys[0].to_column == "id"


def test_aggregate_relationship_join_rejects_unproven_fanout_safety() -> None:
    relationship = _relationship("semantic:rel:base_children", "base", "children")
    pack = PlanningAssetPack(tables=[_table("base"), _table("children")], relationships=[relationship])

    result = plan_governed_joins(pack, "base", ["children"], usage="aggregate")

    assert result.plan is None
    assert [gap.code for gap in result.gaps] == ["JOIN_FANOUT_UNPROVEN"]
    assert "no recognized cardinality" in result.gaps[0].evidence


def test_aggregate_many_to_one_is_safe_only_in_declared_direction() -> None:
    relationship = _relationship(
        "semantic:rel:events_parent",
        "events",
        "parent",
        metadata={"metadata": {"cardinality": "many_to_one", "joinType": "left"}},
    )
    pack = PlanningAssetPack(tables=[_table("events"), _table("parent")], relationships=[relationship])

    forward = plan_governed_joins(pack, "events", ["parent"], usage="aggregate")
    reverse = plan_governed_joins(pack, "parent", ["events"], usage="aggregate")

    assert forward.valid
    assert forward.plan is not None
    assert forward.plan.steps[0].cardinality == "many_to_one"
    assert forward.plan.steps[0].fanout_safe is True
    assert reverse.plan is None
    assert [gap.code for gap in reverse.gaps] == ["JOIN_FANOUT_UNPROVEN"]
    assert "direction=right_to_left" in reverse.gaps[0].evidence


def test_conflicting_fanout_declarations_cannot_be_rescued_by_cardinality() -> None:
    relationship = _relationship(
        "semantic:rel:conflicting_safety",
        "events",
        "parent",
        metadata={
            "cardinality": "many_to_one",
            "fanoutSafe": True,
            "leftToRightFanoutSafe": False,
        },
    )
    pack = PlanningAssetPack(tables=[_table("events"), _table("parent")], relationships=[relationship])

    result = plan_governed_joins(pack, "events", ["parent"], usage="aggregate")

    assert result.plan is None
    assert [gap.code for gap in result.gaps] == ["JOIN_FANOUT_UNPROVEN"]
    assert "conflicting" in result.gaps[0].evidence


def test_explicit_relationship_ref_requires_exact_governed_identity() -> None:
    relationship = _relationship("semantic:rel:governed", "base", "target")
    pack = PlanningAssetPack(tables=[_table("base"), _table("target")], relationships=[relationship])

    by_short_id = plan_governed_joins(
        pack,
        "base",
        ["target"],
        relationship_ref_ids=[relationship.relationship_id],
    )
    wrong_case = plan_governed_joins(
        pack,
        "base",
        ["target"],
        relationship_ref_ids=["SEMANTIC:REL:GOVERNED"],
    )

    assert [gap.code for gap in by_short_id.gaps] == ["JOIN_RELATIONSHIP_REF_UNKNOWN"]
    assert [gap.code for gap in wrong_case.gaps] == ["JOIN_RELATIONSHIP_REF_UNKNOWN"]


def test_join_key_must_belong_to_its_declared_table() -> None:
    relationship = _relationship(
        "semantic:rel:bad_field",
        "base",
        "target",
        left_column="not_on_base",
    )
    pack = PlanningAssetPack(tables=[_table("base"), _table("target")], relationships=[relationship])

    result = plan_governed_joins(pack, "base", ["target"])

    assert result.plan is None
    assert result.gaps[0].code == "JOIN_KEY_FIELD_NOT_IN_TABLE"
    assert result.gaps[0].tables == ("base",)


def test_explicit_relationship_scope_cannot_include_an_extra_branch() -> None:
    pack = PlanningAssetPack(
        tables=[_table(name) for name in ["base", "target", "unused"]],
        relationships=[
            _relationship("semantic:rel:target", "base", "target"),
            _relationship("semantic:rel:unused", "base", "unused"),
        ],
    )

    result = plan_governed_joins(
        pack,
        "base",
        ["target"],
        relationship_ref_ids=["semantic:rel:target", "semantic:rel:unused"],
    )

    assert result.plan is None
    assert [gap.code for gap in result.gaps] == ["JOIN_RELATIONSHIP_SCOPE_NON_MINIMAL"]
    assert result.gaps[0].evidence == "semantic:rel:unused"


def test_join_key_type_must_be_governed_on_both_sides() -> None:
    relationship = _relationship("semantic:rel:unknown_types", "base", "target")
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="base", columns=["id", "parent_id"]),
            PlanningAssetEntry(table="target", columns=["id", "parent_id"]),
        ],
        relationships=[relationship],
    )

    result = plan_governed_joins(
        pack,
        "base",
        ["target"],
        relationship_ref_ids=["semantic:rel:unknown_types"],
        usage="filter_scope",
    )

    assert result.plan is None
    assert [gap.code for gap in result.gaps] == ["JOIN_KEY_TYPE_UNKNOWN"]
    assert "base.parent_id=unknown" in result.gaps[0].evidence
    assert "target.id=unknown" in result.gaps[0].evidence


def test_join_key_incompatible_type_families_fail_closed() -> None:
    relationship = _relationship("semantic:rel:mismatched_types", "base", "target")
    pack = PlanningAssetPack(
        tables=[_table("base"), _table("target", id_type="string")],
        relationships=[relationship],
    )

    result = plan_governed_joins(
        pack,
        "base",
        ["target"],
        relationship_ref_ids=["semantic:rel:mismatched_types"],
        usage="filter_scope",
    )

    assert result.plan is None
    assert [gap.code for gap in result.gaps] == ["JOIN_KEY_TYPE_MISMATCH"]
    assert "base.parent_id=numeric" in result.gaps[0].evidence
    assert "target.id=string" in result.gaps[0].evidence


def test_compatible_numeric_join_key_types_are_executable() -> None:
    relationship = _relationship("semantic:rel:numeric_types", "base", "target")
    pack = PlanningAssetPack(
        tables=[
            _table("base", parent_type="int"),
            _table("target", id_type="decimal(20,0)"),
        ],
        relationships=[relationship],
    )

    result = plan_governed_joins(pack, "base", ["target"], usage="filter_scope")

    assert result.valid
    assert result.plan is not None
    assert result.plan.steps[0].strategy == "exists"


def test_join_key_types_can_come_from_recalled_field_entries() -> None:
    relationship = _relationship("semantic:rel:field_types", "base", "target")
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="base", columns=["id", "parent_id"]),
            PlanningAssetEntry(table="target", columns=["id", "parent_id"]),
        ],
        fields=[
            PlanningAssetEntry(
                key="parent_id",
                table="base",
                metadata={"schema": {"dataType": "varchar(128)"}},
            ),
            PlanningAssetEntry(
                key="id",
                table="target",
                metadata={"schema": {"dataType": "string"}},
            ),
        ],
        relationships=[relationship],
    )

    result = plan_governed_joins(pack, "base", ["target"], usage="filter_scope")

    assert result.valid


def test_tenant_scoped_relationship_rejects_entity_key_only_join() -> None:
    relationship = _relationship("semantic:rel:tenant_unsafe", "base", "target")
    pack = PlanningAssetPack(
        tables=[_table("base", tenant_scoped=True), _table("target", tenant_scoped=True)],
        relationships=[relationship],
    )

    result = plan_governed_joins(
        pack,
        "base",
        ["target"],
        relationship_ref_ids=["semantic:rel:tenant_unsafe"],
        usage="filter_scope",
    )

    assert result.plan is None
    assert [gap.code for gap in result.gaps] == ["JOIN_TENANT_ISOLATION_UNPROVEN"]
    assert "scopeKeyPairs=0,totalKeyPairs=1" in result.gaps[0].evidence


def test_tenant_scoped_relationship_accepts_governed_composite_key() -> None:
    relationship = _relationship(
        "semantic:rel:tenant_composite",
        "base",
        "target",
        join_keys=[
            {"leftColumn": "scope_key", "rightColumn": "scope_key"},
            {"leftColumn": "parent_id", "rightColumn": "id"},
        ],
    )
    pack = PlanningAssetPack(
        tables=[_table("base", tenant_scoped=True), _table("target", tenant_scoped=True)],
        relationships=[relationship],
    )

    result = plan_governed_joins(pack, "base", ["target"], usage="filter_scope")

    assert result.valid
    assert result.plan is not None
    assert len(result.plan.steps[0].keys) == 2


def test_explicit_unsafe_tenant_declaration_overrides_inferred_composite_key() -> None:
    relationship = _relationship(
        "semantic:rel:tenant_denied",
        "base",
        "target",
        join_keys=[
            {"leftColumn": "scope_key", "rightColumn": "scope_key"},
            {"leftColumn": "parent_id", "rightColumn": "id"},
        ],
        metadata={"tenantSafeCompositeKey": False},
    )
    pack = PlanningAssetPack(
        tables=[_table("base", tenant_scoped=True), _table("target", tenant_scoped=True)],
        relationships=[relationship],
    )

    result = plan_governed_joins(
        pack,
        "base",
        ["target"],
        relationship_ref_ids=["semantic:rel:tenant_denied"],
        usage="filter_scope",
    )

    assert result.plan is None
    assert [gap.code for gap in result.gaps] == ["JOIN_TENANT_ISOLATION_UNPROVEN"]
    assert "declaration=false" in result.gaps[0].evidence


def test_explicit_tenant_safe_composite_declaration_is_honored() -> None:
    relationship = _relationship(
        "semantic:rel:declared_tenant_composite",
        "base",
        "target",
        join_keys=[
            {"leftColumn": "id", "rightColumn": "parent_id"},
            {"leftColumn": "parent_id", "rightColumn": "id"},
        ],
        metadata={"tenantSafety": {"safe": True, "compositeKey": True}},
    )
    pack = PlanningAssetPack(
        tables=[_table("base", tenant_scoped=True), _table("target", tenant_scoped=True)],
        relationships=[relationship],
    )

    result = plan_governed_joins(pack, "base", ["target"], usage="filter_scope")

    assert result.valid


def test_tenant_scoped_table_can_join_explicit_global_table() -> None:
    relationship = _relationship("semantic:rel:global_dimension", "base", "dimension")
    pack = PlanningAssetPack(
        tables=[_table("base", tenant_scoped=True), _table("dimension", tenant_scoped=False)],
        relationships=[relationship],
    )

    result = plan_governed_joins(pack, "base", ["dimension"], usage="filter_scope")

    assert result.valid


def test_detail_join_rejects_unproven_result_grain() -> None:
    relationship = _relationship(
        "semantic:rel:detail_many",
        "base",
        "children",
        metadata={"cardinality": "one_to_many"},
    )
    pack = PlanningAssetPack(tables=[_table("base"), _table("children")], relationships=[relationship])

    result = plan_governed_joins(pack, "base", ["children"], usage="detail")

    assert result.plan is None
    assert [gap.code for gap in result.gaps] == ["DETAIL_JOIN_GRAIN_UNPROVEN"]
    assert "cardinality=one_to_many" in result.gaps[0].evidence


def test_detail_join_accepts_one_to_one_cardinality() -> None:
    relationship = _relationship(
        "semantic:rel:detail_one",
        "base",
        "profile",
        metadata={"cardinality": "one_to_one"},
    )
    pack = PlanningAssetPack(tables=[_table("base"), _table("profile")], relationships=[relationship])

    result = plan_governed_joins(pack, "base", ["profile"], usage="detail")

    assert result.valid
    assert result.plan is not None
    assert result.plan.steps[0].fanout_safe is True


def test_detail_join_accepts_explicit_directional_result_grain_proof() -> None:
    relationship = _relationship(
        "semantic:rel:detail_grain",
        "base",
        "profile",
        metadata={
            "cardinality": "many_to_one",
            "rowIdentityPreserved": {"leftToRight": True, "rightToLeft": False},
        },
    )
    pack = PlanningAssetPack(tables=[_table("base"), _table("profile")], relationships=[relationship])

    forward = plan_governed_joins(pack, "base", ["profile"], usage="detail")
    reverse = plan_governed_joins(pack, "profile", ["base"], usage="detail")

    assert forward.valid
    assert reverse.plan is None
    assert [gap.code for gap in reverse.gaps] == ["DETAIL_JOIN_GRAIN_UNPROVEN"]


def test_aggregate_inner_join_requires_referential_completeness() -> None:
    relationship = _relationship(
        "semantic:rel:aggregate_inner",
        "events",
        "parent",
        metadata={"cardinality": "many_to_one", "joinType": "inner"},
    )
    pack = PlanningAssetPack(tables=[_table("events"), _table("parent")], relationships=[relationship])

    result = plan_governed_joins(pack, "events", ["parent"], usage="aggregate")

    assert result.plan is None
    assert [gap.code for gap in result.gaps] == ["JOIN_BASE_ROW_PRESERVATION_UNPROVEN"]
    assert "referential completeness is not proven" in result.gaps[0].evidence


def test_aggregate_inner_join_accepts_directional_referential_completeness() -> None:
    relationship = _relationship(
        "semantic:rel:aggregate_complete",
        "events",
        "parent",
        metadata={
            "cardinality": "many_to_one",
            "joinType": "inner",
            "referentialCompleteness": {"leftToRight": True, "rightToLeft": False},
        },
    )
    pack = PlanningAssetPack(tables=[_table("events"), _table("parent")], relationships=[relationship])

    result = plan_governed_joins(pack, "events", ["parent"], usage="aggregate")

    assert result.valid
    assert result.plan is not None
    assert result.plan.steps[0].join_type == "inner"
    assert result.plan.steps[0].base_rows_preserved is True


def test_aggregate_outer_join_must_preserve_the_current_base_direction() -> None:
    relationship = _relationship(
        "semantic:rel:aggregate_wrong_outer_direction",
        "events",
        "parent",
        metadata={"cardinality": "many_to_one", "joinType": "right"},
    )
    pack = PlanningAssetPack(tables=[_table("events"), _table("parent")], relationships=[relationship])

    result = plan_governed_joins(pack, "events", ["parent"], usage="aggregate")

    assert result.plan is None
    assert [gap.code for gap in result.gaps] == ["JOIN_BASE_ROW_PRESERVATION_UNPROVEN"]
    assert "joinType=right_outer,direction=left_to_right" in result.gaps[0].evidence
