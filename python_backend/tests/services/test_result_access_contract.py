from merchant_ai.config import get_settings
from merchant_ai.models import (
    AnswerMode,
    NodeExecutionContext,
    PlanningAssetEntry,
    PlanningAssetPack,
    QuestionIntent,
)
from merchant_ai.services.assets import validate_semantic_asset
from merchant_ai.services.planning import generic_output_keys
from merchant_ai.services.query import NodeWorkerExecutor, SqlValidationService


class UnconfiguredLlm:
    configured = False


class NeverQueriedRepository:
    pass


def governed_pack() -> PlanningAssetPack:
    table = "fact_daily"
    return PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table=table,
                columns=["tenant_key", "business_day", "measure_value", "private_note", "public_label"],
                metadata={
                    "timeColumn": "business_day",
                    "merchantFilterColumn": "tenant_key",
                    "rowAccessPolicy": {
                        "scopeType": "tenant",
                        "filterColumn": "tenant_key",
                        "operator": "eq",
                        "valueSource": "merchant_id",
                        "required": True,
                    },
                    "resultAccessPolicies": {
                        "METRIC": {
                            "visibilityPolicy": {"level": "public", "reason": "published aggregate"},
                            "maskingPolicy": {"strategy": "none"},
                        },
                        "TIME": {
                            "visibilityPolicy": {"level": "public", "reason": "published dimension"},
                            "maskingPolicy": {"strategy": "none"},
                        },
                    },
                },
            )
        ],
        fields=[
            PlanningAssetEntry(
                key="tenant_key",
                table=table,
                metadata={"semantic": {"role": "KEY"}},
            ),
            PlanningAssetEntry(
                key="business_day",
                table=table,
                metadata={"semantic": {"role": "TIME"}},
            ),
            PlanningAssetEntry(
                key="measure_value",
                table=table,
                metadata={"semantic": {"role": "METRIC"}},
            ),
            PlanningAssetEntry(
                key="private_note",
                table=table,
                metadata={"semantic": {"role": "ATTRIBUTE"}},
            ),
            PlanningAssetEntry(
                key="public_label",
                table=table,
                metadata={
                    "semantic": {
                        "role": "DIMENSION",
                        "defaultVisible": True,
                        "visibilityPolicy": {"level": "public"},
                        "maskingPolicy": {"strategy": "none"},
                    }
                },
            ),
        ],
        metrics=[
            PlanningAssetEntry(
                key="measure_value",
                table=table,
                columns=["measure_value"],
                metadata={
                    "metricKey": "measure_value",
                    "formula": "SUM(measure_value)",
                    "sourceColumns": ["measure_value"],
                },
            ),
            PlanningAssetEntry(
                key="derived_measure",
                table=table,
                columns=["measure_value", "private_note"],
                metadata={
                    "metricKey": "derived_measure",
                    "formula": "SUM(measure_value) / NULLIF(SUM(private_note), 0)",
                    "sourceColumns": ["measure_value", "private_note"],
                },
            ),
        ],
    )


def test_result_role_policy_authorizes_aggregate_without_exposing_raw_detail():
    worker = NodeWorkerExecutor(
        UnconfiguredLlm(),
        NeverQueriedRepository(),
        SqlValidationService(),
        get_settings(),
    )
    pack = governed_pack()
    aggregate = QuestionIntent(
        question="generic aggregate",
        answer_mode=AnswerMode.GROUP_AGG,
        plan_task_id="aggregate",
        preferred_table="fact_daily",
        group_by_column="business_day",
        metric_name="measure_value",
        metric_column="measure_value",
        metric_formula="SUM(measure_value)",
        output_keys=["business_day"],
    )

    aggregate_contract = worker._node_plan_contract(
        aggregate,
        pack,
        NodeExecutionContext(merchant_id="tenant-1"),
    )

    assert "business_day" in aggregate_contract.visible_columns
    assert "measure_value" in aggregate_contract.visible_columns
    assert "private_note" not in aggregate_contract.visible_columns
    assert "tenant_key" not in aggregate_contract.visible_columns
    assert worker.node_contract_validator.review(aggregate_contract).valid

    detail = aggregate.model_copy(
        update={
            "answer_mode": AnswerMode.DETAIL,
            "group_by_column": "",
            "metric_name": "",
            "metric_column": "",
            "metric_formula": "",
            "required_evidence": [],
            "output_keys": ["measure_value"],
        }
    )
    detail_contract = worker._node_plan_contract(
        detail,
        pack,
        NodeExecutionContext(merchant_id="tenant-1"),
    )
    assert "measure_value" not in detail_contract.visible_columns
    assert not worker.node_contract_validator.review(detail_contract).valid


def test_generic_output_keys_require_explicit_display_and_access_contracts():
    pack = governed_pack()
    columns = set(pack.known_columns("fact_daily"))

    output = generic_output_keys(
        QuestionIntent(preferred_table="fact_daily", filter_column="tenant_key"),
        columns,
        pack,
    )

    assert output == ["tenant_key", "public_label"]
    assert "business_day" not in output
    assert "private_note" not in output


def test_published_metric_sources_are_internal_computation_not_display_evidence():
    worker = NodeWorkerExecutor(
        UnconfiguredLlm(),
        NeverQueriedRepository(),
        SqlValidationService(),
        get_settings(),
    )
    pack = governed_pack()
    intent = QuestionIntent(
        question="generic derived aggregate",
        answer_mode=AnswerMode.GROUP_AGG,
        plan_task_id="derived",
        preferred_table="fact_daily",
        group_by_column="business_day",
        metric_name="derived_measure",
        metric_formula="SUM(measure_value) / NULLIF(SUM(private_note), 0)",
        metric_specs=[
            {
                "metricName": "derived_measure",
                "metricFormula": "SUM(measure_value) / NULLIF(SUM(private_note), 0)",
                "sourceColumns": ["measure_value", "private_note"],
            }
        ],
        required_evidence=["business_day", "measure_value", "private_note"],
        output_keys=["business_day"],
    )

    contract = worker._node_plan_contract(
        intent,
        pack,
        NodeExecutionContext(merchant_id="tenant-1"),
    )
    critique = worker.node_contract_validator.review(contract)

    assert "measure_value" in contract.internal_only_columns
    assert "private_note" in contract.internal_only_columns
    assert "measure_value" not in contract.visible_columns
    assert "private_note" not in contract.visible_columns
    assert critique.valid


def test_queryable_asset_publish_fails_closed_without_result_access_policy():
    asset = {
        "tableName": "fact_daily",
        "timeColumn": "business_day",
        "tableUsageProfile": {"contractStatus": "APPROVED", "queryableByAgent": True},
        "schemaColumns": [
            {"columnName": "business_day"},
            {"columnName": "measure_value"},
        ],
        "semanticColumns": [],
        "metrics": [
            {
                "metricKey": "measure_value",
                "formula": "SUM(measure_value)",
                "sourceColumns": ["measure_value"],
            }
        ],
    }

    validation = validate_semantic_asset(asset, [])

    codes = {item["code"] for item in validation["errors"]}
    assert "METRIC_RESULT_ACCESS_POLICY_UNDECLARED" in codes
    assert "TIME_RESULT_ACCESS_POLICY_UNDECLARED" in codes
