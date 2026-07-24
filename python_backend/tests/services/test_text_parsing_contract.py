from __future__ import annotations

import ast
from pathlib import Path

from merchant_ai.services.language_policy import load_language_policy
from merchant_ai.services.text_parsing import (
    collapse_whitespace,
    exact_path_segments,
    is_ascii_hex,
    literal_spans,
    parse_prefixed_reference,
    safe_ascii_component,
)


def test_character_parsing_primitives_are_deterministic_and_fail_closed() -> None:
    assert collapse_whitespace("  A\n\tB  ") == "A B"
    assert is_ascii_hex("a0F9", minimum=4)
    assert not is_ascii_hex("a0-z", minimum=4)
    assert safe_ascii_component(" Alias / V1 ", extras=("_", "-"), lowercase=True) == "alias_v1"
    assert parse_prefixed_reference(
        "semantic:topic:table:metric:key",
        prefix="semantic:",
        separator=":",
        part_count=4,
    ) == ("topic", "table", "metric", "key")
    assert parse_prefixed_reference(
        "semantic:topic::metric:key",
        prefix="semantic:",
        separator=":",
        part_count=4,
    ) is None
    assert exact_path_segments("topics/topic/tables/table", prefix="topics") == (
        "topics",
        "topic",
        "tables",
        "table",
    )
    assert exact_path_segments("topics/../table", prefix="topics") is None


def test_literal_spans_support_ascii_boundaries_without_partial_identifiers() -> None:
    assert literal_spans(
        "net revenue and revenue",
        ["revenue"],
        ascii_word_boundary=True,
    ) == [(4, 11, "revenue"), (16, 23, "revenue")]
    assert literal_spans(
        "prerevenue revenue_id revenue",
        ["revenue"],
        ascii_word_boundary=True,
    ) == [(22, 29, "revenue")]


def test_packaged_language_policy_is_structured_and_versioned() -> None:
    policy = load_language_policy()
    assert policy.policy_id
    assert policy.revision
    assert policy.temporal.units
    assert policy.routing.ranking_operators
    assert policy.routing.ranking_default_order in {"asc", "desc"}
    assert policy.routing.ranking_link_particles
    assert policy.routing.scope_lock_markers
    assert policy.routing.correction_markers
    assert policy.routing.time_clarification_options
    assert policy.answer.definition_markers
    assert set(policy.answer.trend_direction_markers) == {"up", "down", "flat"}


def test_assigned_production_files_have_no_pattern_engine_import_or_call() -> None:
    root = Path(__file__).resolve().parents[2] / "merchant_ai"
    paths = [
        root / "config.py",
        *[
            root / "services" / name
            for name in (
                "time_semantics.py",
                "routing.py",
                "assets.py",
                "retrieval.py",
                "semantic_request.py",
                "semantic_joins.py",
                "recall_index.py",
                "semantic_asset_migrations.py",
                "knowledge_requests.py",
                "clarification.py",
                "attachments.py",
                "answer.py",
                "answer_claims.py",
                "answer_formatting.py",
                "evidence.py",
                "formulas.py",
                "grounded_analysis_artifact.py",
                "query.py",
                "quick_metrics.py",
                "text_parsing.py",
                "language_policy.py",
            )
        ],
    ]
    violations: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(alias.name == "re" for alias in node.names):
                violations.append(str(path))
            if isinstance(node, ast.ImportFrom) and node.module == "re":
                violations.append(str(path))
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "re"
            ):
                violations.append(str(path))
    assert violations == []
