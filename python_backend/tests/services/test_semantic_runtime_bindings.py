from __future__ import annotations

import json

from merchant_ai.config import get_settings
from merchant_ai.services.repositories import MerchantService
from merchant_ai.services.runtime_bindings import SemanticRuntimeBindingRegistry


def _write_asset(root, topic: str, table: str, role: str = "principal_profile") -> None:
    path = root / topic / "tables" / table / "asset.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "PUBLISHED",
                "tableName": table,
                "runtimeBindings": [
                    {
                        "role": role,
                        "lookupColumn": "principal_key",
                        "idColumn": "principal_key",
                        "displayColumns": ["display_value"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_runtime_binding_is_resolved_from_arbitrary_published_asset(tmp_path):
    _write_asset(tmp_path, "topic_alpha", "table_alpha")
    settings = get_settings().model_copy(update={"topic_path": str(tmp_path)})

    binding = SemanticRuntimeBindingRegistry(settings).resolve("principal_profile")

    assert binding["table"] == "table_alpha"
    assert binding["lookupColumn"] == "principal_key"


def test_runtime_binding_fails_closed_when_role_has_multiple_owners(tmp_path):
    _write_asset(tmp_path, "topic_alpha", "table_alpha")
    _write_asset(tmp_path, "topic_beta", "table_beta")
    settings = get_settings().model_copy(update={"topic_path": str(tmp_path)})

    assert SemanticRuntimeBindingRegistry(settings).resolve("principal_profile") == {}


def test_merchant_lookup_uses_semantic_binding_without_compiled_table_or_column_names():
    class Repository:
        def __init__(self):
            self.calls = []

        def query(self, sql, params):
            self.calls.append((sql, params))
            return [{"principal_key": "p-1", "display_value": "Profile Alpha"}]

    repository = Repository()
    service = MerchantService(
        get_settings(),
        repository,
        {
            "table": "table_alpha",
            "lookupColumn": "principal_key",
            "idColumn": "principal_key",
            "displayColumns": ["display_value"],
        },
    )

    merchant = service.current_merchant("p-1")

    assert repository.calls == [("SELECT * FROM `table_alpha` WHERE `principal_key` = %s LIMIT 1", ["p-1"])]
    assert merchant.merchant_id == "p-1"
    assert merchant.merchant_name == "Profile Alpha"
