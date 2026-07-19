import ast
from pathlib import Path

from merchant_ai.models import PlanningAssetEntry
from merchant_ai.services.doris_physical_plan_governance import (
    DorisPhysicalPlanGovernor,
    PartitionPruningRequirement,
    PhysicalPlanGovernancePolicy,
    assess_doris_physical_plan,
)


def _asset(
    table: str,
    *,
    partitions=(),
    partition_count: int = 0,
    buckets=(),
    bucket_count: int = 0,
    colocate_group: str = "",
    indexes=(),
    status: str = "PUBLISHED",
):
    return {
        "topic": "opaque_topic",
        "tableName": table,
        "status": status,
        "groundedEvidenceRef": "asset:%s" % table,
        "physicalMetadata": {
            "partitionColumns": list(partitions),
            "partitionCount": partition_count,
            "bucketColumns": list(buckets),
            "bucketCount": bucket_count,
            "colocateGroup": colocate_group,
            "invertedIndexes": list(indexes),
            "ddlHash": "ddl:%s" % table,
        },
    }


def _codes(assessment):
    return {item.code for item in assessment.gaps}


def _obligation(assessment, kind):
    return next(item for item in assessment.obligations if item.kind == kind)


def test_contract_declared_partition_pruning_is_verified_from_explain_ratio() -> None:
    assessment = assess_doris_physical_plan(
        sql=(
            "SELECT f.opaque_value FROM opaque_fact f "
            "WHERE f.opaque_day >= '2026-07-01'"
        ),
        formal_assets=[
            _asset(
                "opaque_fact",
                partitions=["opaque_day"],
                partition_count=30,
            )
        ],
        explain_payload=[
            {
                "Explain String(Nereids Planner)": (
                    "TABLE: opaque_fact\npartitions=3/30"
                )
            }
        ],
        partition_requirements=[
            PartitionPruningRequirement(
                table="opaque_fact",
                partition_column="opaque_day",
                semantic_evidence_ref="semantic-time:opaque_day",
            )
        ],
    )

    assert assessment.status == "VERIFIED"
    assert assessment.executable is True
    obligation = _obligation(assessment, "PARTITION_PRUNING")
    assert obligation.status == "VERIFIED"
    assert obligation.capability["businessTimeInferredFromPartition"] is False
    plan_evidence = next(item for item in obligation.evidence if item.source == "EXPLAIN_VERBOSE")
    assert plan_evidence.attributes["selected"] == 3
    assert plan_evidence.attributes["total"] == 30


def test_partition_layout_alone_never_creates_business_time_obligation() -> None:
    assessment = assess_doris_physical_plan(
        sql="SELECT f.opaque_value FROM opaque_fact f",
        formal_assets=[
            _asset(
                "opaque_fact",
                partitions=["opaque_storage_key"],
                partition_count=30,
            )
        ],
        explain_payload=[{"plan": "TABLE: opaque_fact\npartitions=30/30"}],
    )

    assert assessment.status == "NOT_REQUIRED"
    assert assessment.obligations == []
    assert assessment.gaps == []


def test_partition_scan_without_reduction_returns_structured_blocking_gap() -> None:
    sql = "SELECT * FROM opaque_fact WHERE opaque_day BETWEEN '2026-07-01' AND '2026-07-07'"
    assessment = DorisPhysicalPlanGovernor().assess(
        sql=sql,
        formal_assets=[
            _asset(
                "opaque_fact",
                partitions=["opaque_day"],
                partition_count=30,
            )
        ],
        explain_payload=[{"plan": "TABLE: opaque_fact\npartitions=30/30"}],
        partition_requirements=[
            PartitionPruningRequirement(
                table="opaque_fact",
                partition_column="opaque_day",
                semantic_evidence_ref="semantic-time:opaque_day",
            )
        ],
    )

    assert assessment.status == "GAPPED"
    assert assessment.executable is False
    assert "PARTITION_SCAN_NOT_REDUCED" in _codes(assessment)
    assert assessment.sql_was_rewritten is False


def test_partition_requirement_must_bind_semantic_and_physical_evidence() -> None:
    assessment = assess_doris_physical_plan(
        sql="SELECT * FROM opaque_fact WHERE business_clock >= '2026-07-01'",
        formal_assets=[
            _asset(
                "opaque_fact",
                partitions=["physical_day"],
                partition_count=20,
            )
        ],
        explain_payload=[{"plan": "TABLE: opaque_fact\npartitions=1/20"}],
        partition_requirements=[
            PartitionPruningRequirement(
                table="opaque_fact",
                partition_column="business_clock",
                semantic_evidence_ref="",
            )
        ],
    )

    assert assessment.status == "GAPPED"
    assert {
        "PARTITION_REQUIREMENT_SEMANTIC_EVIDENCE_MISSING",
        "PARTITION_COLUMN_NOT_DECLARED",
    } <= _codes(assessment)


def test_reused_aliases_in_separate_ctes_do_not_cross_contaminate_filter_lineage() -> None:
    assessment = assess_doris_physical_plan(
        sql=(
            "WITH a AS (SELECT x.p FROM opaque_left x WHERE x.p = 'v'), "
            "b AS (SELECT x.p FROM opaque_right x) "
            "SELECT a.p FROM a JOIN b ON a.p = b.p"
        ),
        formal_assets=[
            _asset("opaque_left", partitions=["p"], partition_count=10),
            _asset("opaque_right", partitions=["p"], partition_count=10),
        ],
        explain_payload=[{"plan": "TABLE: opaque_right\npartitions=1/10"}],
        partition_requirements=[
            PartitionPruningRequirement(
                table="opaque_right",
                partition_column="p",
                semantic_evidence_ref="semantic:opaque_right.p",
            )
        ],
    )

    assert assessment.status == "GAPPED"
    assert assessment.executable is False
    assert "PARTITION_FILTER_ABSENT" in _codes(assessment)
    assert _obligation(assessment, "PARTITION_PRUNING").status == "GAP"


def test_ambiguous_unqualified_physical_filter_returns_typed_lineage_gap() -> None:
    assessment = assess_doris_physical_plan(
        sql=(
            "SELECT l.p FROM opaque_left l JOIN opaque_right r "
            "ON l.k = r.k WHERE p = 'v'"
        ),
        formal_assets=[
            _asset("opaque_left", partitions=["p"], buckets=["k"]),
            _asset("opaque_right", partitions=["p"], buckets=["k"]),
        ],
        explain_payload=[],
    )

    assert assessment.status == "GAPPED"
    assert assessment.executable is False
    assert "PHYSICAL_FILTER_LINEAGE_UNRESOLVED" in _codes(assessment)


def test_full_bucket_equality_filter_requires_tablet_pruning_proof() -> None:
    asset = _asset(
        "opaque_fact",
        buckets=["scope_key", "entity_key"],
        bucket_count=16,
    )
    verified = assess_doris_physical_plan(
        sql=(
            "SELECT * FROM opaque_fact f "
            "WHERE f.scope_key = 's' AND f.entity_key IN ('a', 'b')"
        ),
        formal_assets=[asset],
        explain_payload=[{"plan": "TABLE: opaque_fact\ntabletRatio=2/16"}],
    )
    partial = assess_doris_physical_plan(
        sql="SELECT * FROM opaque_fact f WHERE f.scope_key = 's'",
        formal_assets=[asset],
        explain_payload=[{"plan": "TABLE: opaque_fact\ntabletRatio=16/16"}],
    )

    assert _obligation(verified, "BUCKET_FILTER_PRUNING").status == "VERIFIED"
    assert "BUCKET_FILTER_PARTIAL" in _codes(partial)
    assert partial.executable is True


def test_bucket_filter_policy_can_promote_missing_proof_to_blocking() -> None:
    inputs = {
        "sql": "SELECT * FROM opaque_fact WHERE scope_key = 's'",
        "formal_assets": [
            _asset(
                "opaque_fact",
                buckets=["scope_key"],
                bucket_count=16,
            )
        ],
        "explain_payload": [{"plan": "TABLE: opaque_fact\ntablets=16/16"}],
    }
    advisory = assess_doris_physical_plan(**inputs)
    assessment = assess_doris_physical_plan(
        **inputs,
        policy=PhysicalPlanGovernancePolicy(bucket_filter_blocking=True),
    )

    assert "TABLET_SCAN_NOT_REDUCED" in _codes(assessment)
    assert assessment.executable is False
    assert advisory.executable is True
    assert assessment.policy_fingerprint != advisory.policy_fingerprint
    assert assessment.assessment_id != advisory.assessment_id


def test_aligned_colocate_join_requires_positive_explain_evidence() -> None:
    assets = [
        _asset(
            "opaque_left",
            buckets=["scope_key", "entity_key"],
            bucket_count=16,
            colocate_group="opaque_group",
        ),
        _asset(
            "opaque_right",
            buckets=["scope_key", "foreign_entity_key"],
            bucket_count=16,
            colocate_group="opaque_group",
        ),
    ]
    sql = (
        "SELECT l.entity_key FROM opaque_left l JOIN opaque_right r "
        "ON l.scope_key = r.scope_key "
        "AND l.entity_key = r.foreign_entity_key"
    )
    verified = assess_doris_physical_plan(
        sql=sql,
        formal_assets=assets,
        explain_payload=[
            {"plan": "HASH JOIN opaque_left opaque_right\ncolocate: true"}
        ],
    )
    missing = assess_doris_physical_plan(
        sql=sql,
        formal_assets=assets,
        explain_payload=[
            {"plan": "HASH JOIN opaque_left opaque_right\ncolocate: false"}
        ],
    )

    assert _obligation(verified, "BUCKET_JOIN_ALIGNMENT").status == "VERIFIED"
    assert _obligation(verified, "COLOCATE_JOIN").status == "VERIFIED"
    assert "COLOCATE_PLAN_EVIDENCE_MISSING" in _codes(missing)


def test_colocate_evidence_is_bound_to_its_exact_join_table_pair() -> None:
    assets = [
        _asset("opaque_a1", buckets=["k"], bucket_count=8, colocate_group="group_a"),
        _asset("opaque_a2", buckets=["k"], bucket_count=8, colocate_group="group_a"),
        _asset("opaque_b1", buckets=["k"], bucket_count=8, colocate_group="group_b"),
        _asset("opaque_b2", buckets=["k"], bucket_count=8, colocate_group="group_b"),
    ]
    assessment = assess_doris_physical_plan(
        sql=(
            "SELECT a1.k FROM opaque_a1 a1 JOIN opaque_a2 a2 ON a1.k = a2.k "
            "JOIN opaque_b1 b1 ON a1.k = b1.k "
            "JOIN opaque_b2 b2 ON b1.k = b2.k"
        ),
        formal_assets=assets,
        explain_payload=[
            {
                "plan": (
                    "JOIN opaque_a1 opaque_a2\ncolocate: true\n"
                    "JOIN opaque_b1 opaque_b2\ncolocate: false"
                )
            }
        ],
    )

    colocate = {
        frozenset(item.tables): item
        for item in assessment.obligations
        if item.kind == "COLOCATE_JOIN"
    }
    assert colocate[frozenset({"opaque_a1", "opaque_a2"})].status == "VERIFIED"
    assert colocate[frozenset({"opaque_b1", "opaque_b2"})].status == "GAP"
    assert "COLOCATE_PLAN_EVIDENCE_MISSING" in _codes(assessment)


def test_join_that_omits_distribution_key_returns_alignment_gap() -> None:
    assessment = assess_doris_physical_plan(
        sql=(
            "SELECT l.entity_key FROM opaque_left l JOIN opaque_right r "
            "ON l.entity_key = r.foreign_entity_key"
        ),
        formal_assets=[
            _asset(
                "opaque_left",
                buckets=["scope_key", "entity_key"],
                bucket_count=16,
            ),
            _asset(
                "opaque_right",
                buckets=["scope_key", "foreign_entity_key"],
                bucket_count=16,
            ),
        ],
        explain_payload=[{"plan": "HASH JOIN"}],
    )

    assert "BUCKET_JOIN_NOT_ALIGNED" in _codes(assessment)
    obligation = _obligation(assessment, "BUCKET_JOIN_ALIGNMENT")
    assert obligation.capability["bucketAligned"] is False


def test_unresolved_join_lineage_cannot_silently_skip_bucket_governance() -> None:
    assessment = assess_doris_physical_plan(
        sql=(
            "WITH left_scope AS (SELECT entity_key FROM opaque_left), "
            "right_scope AS (SELECT entity_key FROM opaque_right) "
            "SELECT l.entity_key FROM left_scope l JOIN right_scope r "
            "ON l.entity_key > r.entity_key"
        ),
        formal_assets=[
            _asset("opaque_left", buckets=["entity_key"], bucket_count=16),
            _asset("opaque_right", buckets=["entity_key"], bucket_count=16),
        ],
        explain_payload=[{"plan": "NESTED LOOP JOIN"}],
    )

    assert "BUCKET_JOIN_LINEAGE_UNRESOLVED" in _codes(assessment)
    assert _obligation(assessment, "BUCKET_JOIN_ALIGNMENT").status == "GAP"


def test_filtered_declared_inverted_index_requires_plan_hit() -> None:
    asset = _asset(
        "opaque_fact",
        indexes=[
            {
                "name": "idx_opaque_text",
                "column": "opaque_text",
                "type": "INVERTED",
            }
        ],
    )
    sql = "SELECT * FROM opaque_fact f WHERE match_any(f.opaque_text, 'needle')"
    verified = assess_doris_physical_plan(
        sql=sql,
        formal_assets=[asset],
        explain_payload=[
            {
                "plan": (
                    "TABLE: opaque_fact\n"
                    "INVERTED INDEX idx_opaque_text column opaque_text"
                )
            }
        ],
    )
    missing = assess_doris_physical_plan(
        sql=sql,
        formal_assets=[asset],
        explain_payload=[{"plan": "TABLE: opaque_fact\nPREDICATES: opaque_text"}],
    )

    assert _obligation(verified, "INVERTED_INDEX_USAGE").status == "VERIFIED"
    assert "INVERTED_INDEX_PLAN_EVIDENCE_MISSING" in _codes(missing)


def test_index_declaration_does_not_create_obligation_without_sql_filter() -> None:
    assessment = assess_doris_physical_plan(
        sql="SELECT opaque_text FROM opaque_fact",
        formal_assets=[
            _asset(
                "opaque_fact",
                indexes=[
                    {
                        "name": "idx_opaque_text",
                        "column": "opaque_text",
                        "type": "INVERTED",
                    }
                ],
            )
        ],
        explain_payload=[{"plan": "TABLE: opaque_fact"}],
    )

    assert assessment.status == "NOT_REQUIRED"


def test_unpublished_or_missing_physical_asset_fails_closed() -> None:
    unpublished = assess_doris_physical_plan(
        sql="SELECT * FROM opaque_fact",
        formal_assets=[_asset("opaque_fact", status="DRAFT")],
        explain_payload=[],
    )
    missing = assess_doris_physical_plan(
        sql="SELECT * FROM opaque_fact",
        formal_assets=[],
        explain_payload=[],
    )

    assert unpublished.status == "INVALID"
    assert unpublished.executable is False
    assert "PHYSICAL_ASSET_NOT_PUBLISHED" in _codes(unpublished)
    assert "FORMAL_PHYSICAL_ASSET_MISSING" in _codes(missing)


def test_planning_asset_pack_table_entry_is_accepted_without_business_adapter() -> None:
    entry = PlanningAssetEntry(
        key="opaque_fact",
        table="opaque_fact",
        topic="opaque_topic",
        metadata={
            "status": "PUBLISHED",
            "groundedEvidenceRef": "asset:opaque_fact",
            "physicalMetadata": {
                "bucketColumns": ["scope_key"],
                "bucketCount": 8,
                "ddlHash": "opaque-ddl",
            },
        },
    )

    assessment = assess_doris_physical_plan(
        sql="SELECT * FROM opaque_fact WHERE scope_key = 's'",
        formal_assets=[entry],
        explain_payload=[{"plan": "TABLE: opaque_fact\ntablets=1/8"}],
    )

    assert assessment.status == "VERIFIED"
    assert assessment.formal_assets[0].asset_ref == "asset:opaque_fact"


def test_multiple_statements_are_rejected_without_sql_rewrite() -> None:
    assessment = assess_doris_physical_plan(
        sql="SELECT * FROM opaque_fact; SELECT * FROM opaque_fact",
        formal_assets=[_asset("opaque_fact")],
        explain_payload=[],
    )

    assert assessment.status == "INVALID"
    assert "PHYSICAL_PLAN_SQL_INVALID" in _codes(assessment)
    assert assessment.sql_was_rewritten is False


def test_governor_source_has_no_regular_expression_dependency() -> None:
    module_path = (
        Path(__file__).parents[2]
        / "merchant_ai"
        / "services"
        / "doris_physical_plan_governance.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert "re" not in imported_modules
