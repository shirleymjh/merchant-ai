from merchant_ai.services.grounded_contract_repair import (
    build_contract_repair_directive,
)
from merchant_ai.services.grounded_query_contract import GroundedContractGap


def test_candidate_time_ref_becomes_exact_read_next_path() -> None:
    time_ref = "semantic:商品管理:dwm_goods_detail_df:field:pt"
    directive = build_contract_repair_directive(
        [
            GroundedContractGap(
                code="TIME_FIELD_BINDING_REQUIRED",
                message="read and bind the governed time field",
                evidence_kind="COLUMN",
                required_capability={
                    "candidateTimeFieldRefs": [time_ref]
                },
            )
        ],
        contract_status="UNRESOLVED",
        resolve_ref_path=lambda ref_id: (
            "topics/商品管理/tables/dwm_goods_detail_df/columns/pt.json"
            if ref_id == time_ref
            else ""
        ),
        path_prefix="knowledge",
        base_attempt_id="attempt-v1",
        contract_version=1,
    )

    assert directive.status == "REPAIR_REQUIRED"
    assert directive.repair_type == "EVIDENCE"
    assert directive.read_next == [
        {
            "refId": time_ref,
            "path": (
                "/knowledge/topics/商品管理/tables/"
                "dwm_goods_detail_df/columns/pt.json"
            ),
            "sourceGapCode": "TIME_FIELD_BINDING_REQUIRED",
        }
    ]
    assert set(directive.allowed_actions) >= {
        "READ_SEMANTIC_ASSET",
        "RESUBMIT_GROUNDED_CONTRACT",
    }


def test_rejected_topology_refs_are_not_misreported_as_reads() -> None:
    metric_ref = "semantic:交易:orders:metric:order_cnt"
    directive = build_contract_repair_directive(
        [
            GroundedContractGap(
                code="INDEPENDENT_QUERY_SPLIT_REQUIRED",
                message="split independently aggregatable metrics",
                evidence_kind="QUERY_TOPOLOGY",
                resolution=(
                    "REBIND_TO_COMPATIBLE_SINGLE_TABLE_OR_PROPOSE_EXECUTION_GRAPH"
                ),
                required_capability={
                    "topology": "PARALLEL_INDEPENDENT_QUERIES"
                },
                rejected_ref_ids=[metric_ref],
            )
        ],
        contract_status="REVISE_BINDINGS",
    )

    assert directive.repair_type == "TOPOLOGY"
    assert directive.read_next == []
    assert set(directive.allowed_actions) >= {
        "READ_SEMANTIC_ASSET",
        "RESUBMIT_GROUNDED_CONTRACT",
        "PROPOSE_GROUNDED_EXECUTION_GRAPH",
        "REVISE_GROUNDED_EXECUTION_GRAPH",
    }
