import json

from merchant_ai.models import AgentTaskResult, EntitySet, QueryBundle
from merchant_ai.services.query import merge_query_bundles, merge_task_result_bundles


def write_rows(tmp_path, name, rows):
    path = tmp_path / (name + "_rows.json")
    path.write_text(json.dumps(rows), encoding="utf-8")
    return str(path)


def test_merge_reads_full_result_artifact_instead_of_preview(tmp_path):
    rows = [{"dimension_key": value} for value in ["a", "b", "c"]]
    bundle = QueryBundle(
        rows=rows[:1],
        original_row_count=3,
        offloaded_files=[write_rows(tmp_path, "source", rows)],
    )

    merged = merge_query_bundles([bundle], ["source_task"])

    assert merged.lineage_complete is True
    assert merged.rows == rows
    assert merged.source_row_counts == {"source_task": 3}
    assert merged.source_artifact_refs["source_task"] == bundle.offloaded_files


def test_merge_fails_closed_when_full_result_artifact_is_missing():
    bundle = QueryBundle(
        rows=[{"dimension_key": "preview_only"}],
        original_row_count=3,
        offloaded_files=["/missing/source_rows.json"],
    )

    merged = merge_query_bundles([bundle], ["source_task"])

    assert merged.failed is True
    assert merged.lineage_complete is False
    assert merged.rows == []
    assert merged.runtime_events[0]["code"] == "MERGE_INPUT_INCOMPLETE"


def test_entity_merge_uses_complete_rows_from_each_source(tmp_path):
    left_rows = [
        {"dimension_key": "a", "measure_a": 1},
        {"dimension_key": "b", "measure_a": 2},
    ]
    right_rows = [
        {"dimension_key": "a", "measure_b": 3},
        {"dimension_key": "b", "measure_b": 4},
    ]
    results = [
        AgentTaskResult(
            task_id="left",
            success=True,
            query_bundle=QueryBundle(
                rows=left_rows[:1],
                original_row_count=2,
                offloaded_files=[write_rows(tmp_path, "left", left_rows)],
            ),
            entity_set=EntitySet(join_key="dimension_key", column_values={"dimension_key": ["a", "b"]}),
        ),
        AgentTaskResult(
            task_id="right",
            success=True,
            query_bundle=QueryBundle(
                rows=right_rows[:1],
                original_row_count=2,
                offloaded_files=[write_rows(tmp_path, "right", right_rows)],
            ),
            entity_set=EntitySet(join_key="dimension_key", column_values={"dimension_key": ["a", "b"]}),
        ),
    ]

    merged = merge_task_result_bundles(results)

    assert merged.lineage_complete is True
    assert merged.original_row_count == 2
    assert merged.rows == [
        {"dimension_key": "a", "__fieldLineage": {"measure_a": ["left"], "measure_b": ["right"]}, "__fieldConflicts": {}, "measure_a": 1, "measure_b": 3},
        {"dimension_key": "b", "__fieldLineage": {"measure_a": ["left"], "measure_b": ["right"]}, "__fieldConflicts": {}, "measure_a": 2, "measure_b": 4},
    ]
