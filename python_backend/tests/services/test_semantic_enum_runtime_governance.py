import json
from pathlib import Path

from merchant_ai.config import get_settings
from merchant_ai.services.assets import SemanticCatalogService, TopicAssetService


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_runtime_asset_quarantines_unapproved_enums_and_keeps_approved_values(tmp_path: Path) -> None:
    settings = get_settings().model_copy(update={"topic_path": str(tmp_path / "topics")})
    write_json(
        tmp_path / "topics" / "domain" / "tables" / "events" / "asset.json",
        {
            "topic": "domain",
            "tableName": "events",
            "semanticColumns": [
                {
                    "columnName": "unreviewed_state",
                    "enumValues": ["unreviewed-secret-label"],
                    "enumMappings": {"x": "unreviewed-secret-meaning"},
                    "sampleValues": ["unreviewed-secret-sample"],
                    "evidence": "schema profile; samples=[unreviewed-secret-evidence]",
                },
                {
                    "columnName": "reviewed_state",
                    "enumValues": ["approved-label"],
                    "enumMappings": {"y": "approved-meaning"},
                    "enumMetadata": {"reviewStatus": "APPROVED"},
                },
            ],
        },
    )

    assets = TopicAssetService(settings)
    loaded = assets.load_table_asset("domain", "events")
    columns = {item["columnName"]: item for item in loaded["semanticColumns"]}

    assert columns["unreviewed_state"]["enumValues"] == []
    assert columns["unreviewed_state"]["enumMappings"] == {}
    assert "sampleValues" not in columns["unreviewed_state"]
    assert "unreviewed-secret-evidence" not in columns["unreviewed_state"]["evidence"]
    assert columns["unreviewed_state"]["enumMetadata"]["runtimeSuppressed"] is True
    assert columns["unreviewed_state"]["enumMetadata"]["runtimePolicy"] == "QUARANTINED_UNTIL_APPROVED"
    assert columns["reviewed_state"]["enumValues"] == ["approved-label"]
    assert columns["reviewed_state"]["enumMappings"] == {"y": "approved-meaning"}


def test_semantic_read_never_exposes_unapproved_enum_values(tmp_path: Path) -> None:
    settings = get_settings().model_copy(update={"topic_path": str(tmp_path / "topics")})
    write_json(
        tmp_path / "topics" / "domain" / "tables" / "events" / "asset.json",
        {
            "topic": "domain",
            "tableName": "events",
            "semanticColumns": [
                {
                    "columnName": "state",
                    "enumValues": ["unreviewed-secret-label"],
                    "enumMetadata": {"reviewStatus": "PENDING_REVIEW"},
                }
            ],
        },
    )

    catalog = SemanticCatalogService(TopicAssetService(settings))
    content = catalog.table_ref("domain", "events")["content"]

    assert "unreviewed-secret-label" not in content
    assert "QUARANTINED_UNTIL_APPROVED" in content


def test_table_ref_quarantines_a_raw_asset_argument() -> None:
    settings = get_settings()
    catalog = SemanticCatalogService(TopicAssetService(settings))
    content = catalog.table_ref(
        "domain",
        "events",
        asset={
            "topic": "domain",
            "tableName": "events",
            "semanticColumns": [
                {
                    "columnName": "state",
                    "enumMappings": {"x": "raw-bypass-secret-meaning"},
                }
            ],
        },
    )["content"]

    assert "raw-bypass-secret-meaning" not in content
    assert "QUARANTINED_UNTIL_APPROVED" in content
