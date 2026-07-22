import json
import os
from pathlib import Path

from merchant_ai.config import get_settings
from merchant_ai.models import RecallItem
from merchant_ai.services.recall_index import (
    EsRecallIndexAdapter,
    RecallIndexManager,
    changed_refs,
)


class VersionedProvider:
    def __init__(self, settings):
        self.settings = settings
        self.version = "v1"
        self.cleared = 0

    def clear_cache(self):
        self.cleared += 1

    def _load_documents(self):
        return [
            RecallItem(
                doc_id="metric:return_rate",
                title="return rate",
                content=self.version,
                source_type="SEMANTIC_METRIC",
                topic="经营画像",
                table="ads_merchant_profile",
                metadata={
                    "semanticRefId": "metric:return_rate",
                    "semanticPath": "topics/经营画像/tables/ads_merchant_profile/asset.json#metric:return_rate",
                },
            )
        ]


class FailingStageAdapter:
    def stage(self, docs, manifest):
        return {
            "success": False,
            "mode": "es",
            "transactionMode": "physical_index_alias_swap",
            "errorCode": "ES_STAGE_DOC_COUNT_MISMATCH",
        }


class RecordingTransactionalAdapter:
    def __init__(self, manifest_path: Path, commit_success: bool = True):
        self.manifest_path = manifest_path
        self.commit_success = commit_success
        self.events = []

    def stage(self, docs, manifest):
        self.events.append(("stage", self._active_version()))
        return {
            "success": True,
            "mode": "es",
            "transactionMode": "physical_index_alias_swap",
            "alias": "recall",
            "physicalIndex": "recall__candidate",
            "previousState": {"kind": "alias", "indices": ["recall__old"]},
            "committed": False,
        }

    def commit_stage(self, stage):
        self.events.append(("alias_swap", self._active_version()))
        if not self.commit_success:
            return {"success": False, "errorCode": "ES_ALIAS_SWAP_FAILED"}
        stage["committed"] = True
        return {"success": True}

    def rollback_stage(self, stage):
        self.events.append(("alias_rollback", self._active_version()))
        stage["committed"] = False
        return {"success": True, "rolledBack": True}

    def abort_stage(self, stage):
        self.events.append(("abort", self._active_version()))
        return {"success": True}

    def finalize_stage(self, stage):
        self.events.append(("finalize", self._active_version()))
        return {"success": True, "removed": ["recall__old"], "retained": []}

    def _active_version(self):
        if not self.manifest_path.exists():
            return ""
        return json.loads(self.manifest_path.read_text(encoding="utf-8")).get("indexVersion", "")


class RecordingIncrementalAdapter(RecordingTransactionalAdapter):
    def __init__(self, manifest_path: Path):
        super().__init__(manifest_path)
        self.incremental_call = {}

    def stage_incremental(
        self,
        docs,
        manifest,
        *,
        previous_manifest,
        updated_refs,
        replace_all=False,
    ):
        self.incremental_call = {
            "docIds": [doc.doc_id for doc in docs],
            "manifest": manifest,
            "previousManifest": previous_manifest,
            "updatedRefs": list(updated_refs),
            "replaceAll": replace_all,
        }
        return self.stage(docs, manifest)


def _settings(tmp_path, es_enabled=False):
    return get_settings().model_copy(
        update={
            "es_enabled": es_enabled,
            "es_index": "recall",
            "topic_path": str(tmp_path / "topics"),
            "harness_workspace_path": str(tmp_path / "workspace"),
        }
    )


def _active_v1(tmp_path):
    settings = _settings(tmp_path, es_enabled=False)
    provider = VersionedProvider(settings)
    manager = RecallIndexManager(settings, provider)
    first = manager.rebuild()
    return settings, provider, first


def test_es_stage_failure_preserves_old_active_manifest_and_writes_retry_record(tmp_path):
    settings, provider, first = _active_v1(tmp_path)
    manifest_path = settings.resolved_workspace_path / "recall_index_manifest.json"
    old_payload = manifest_path.read_text(encoding="utf-8")
    provider.version = "v2"
    settings.es_enabled = True
    manager = RecallIndexManager(settings, provider, es_adapter=FailingStageAdapter())

    result = manager.rebuild(topic="经营画像", table_name="ads_merchant_profile")

    assert result["success"] is False
    assert result["status"] == "PENDING_RETRY"
    assert result["activeManifestAdvanced"] is False
    assert result["indexVersion"] == first["indexVersion"]
    assert result["candidateIndexVersion"] != first["indexVersion"]
    assert manifest_path.read_text(encoding="utf-8") == old_payload
    retry_path = Path(result["retryRecord"])
    assert retry_path.exists()
    retry = json.loads(retry_path.read_text(encoding="utf-8"))
    assert retry["activeManifest"]["indexVersion"] == first["indexVersion"]
    assert retry["candidateManifest"]["indexVersion"] == result["candidateIndexVersion"]


def test_alias_swap_happens_before_active_manifest_advance(tmp_path):
    settings, provider, first = _active_v1(tmp_path)
    provider.version = "v2"
    settings.es_enabled = True
    manager = RecallIndexManager(settings, provider)
    adapter = RecordingTransactionalAdapter(manager.manifest_path)
    manager.es_adapter = adapter

    result = manager.rebuild(topic="经营画像", table_name="ads_merchant_profile")

    assert result["success"] is True
    assert result["indexVersion"] != first["indexVersion"]
    assert adapter.events[0] == ("stage", first["indexVersion"])
    assert adapter.events[1] == ("alias_swap", first["indexVersion"])
    assert adapter.events[2] == ("finalize", result["indexVersion"])
    active = json.loads(manager.manifest_path.read_text(encoding="utf-8"))
    assert active["indexVersion"] == result["indexVersion"]


def test_alias_swap_failure_does_not_advance_active_manifest(tmp_path):
    settings, provider, first = _active_v1(tmp_path)
    provider.version = "v2"
    settings.es_enabled = True
    manager = RecallIndexManager(settings, provider)
    adapter = RecordingTransactionalAdapter(manager.manifest_path, commit_success=False)
    manager.es_adapter = adapter

    result = manager.rebuild(topic="经营画像", table_name="ads_merchant_profile")

    assert result["success"] is False
    assert result["errorCode"] == "ES_ALIAS_SWAP_FAILED"
    assert result["indexVersion"] == first["indexVersion"]
    assert json.loads(manager.manifest_path.read_text(encoding="utf-8"))["indexVersion"] == first["indexVersion"]
    assert [event[0] for event in adapter.events] == ["stage", "alias_swap", "abort"]


def test_manifest_commit_failure_rolls_alias_back_and_keeps_old_manifest(tmp_path, monkeypatch):
    settings, provider, first = _active_v1(tmp_path)
    provider.version = "v2"
    settings.es_enabled = True
    manager = RecallIndexManager(settings, provider)
    adapter = RecordingTransactionalAdapter(manager.manifest_path)
    manager.es_adapter = adapter
    real_replace = os.replace

    def fail_active_manifest_replace(source, destination):
        if Path(destination) == manager.manifest_path:
            raise OSError("manifest fsync boundary failed")
        return real_replace(source, destination)

    monkeypatch.setattr("merchant_ai.services.recall_index.os.replace", fail_active_manifest_replace)

    result = manager.rebuild(topic="经营画像", table_name="ads_merchant_profile")

    assert result["success"] is False
    assert result["errorCode"] == "ACTIVE_MANIFEST_COMMIT_FAILED"
    assert result["indexVersion"] == first["indexVersion"]
    assert json.loads(manager.manifest_path.read_text(encoding="utf-8"))["indexVersion"] == first["indexVersion"]
    assert [event[0] for event in adapter.events] == ["stage", "alias_swap", "alias_rollback", "abort"]


def test_physical_stage_fails_closed_when_count_hash_validation_fails(tmp_path, monkeypatch):
    settings = _settings(tmp_path, es_enabled=True)
    provider = VersionedProvider(settings)
    docs = provider._load_documents()
    adapter = EsRecallIndexAdapter(settings)
    deleted = []
    monkeypatch.setattr(adapter, "_alias_state", lambda alias: {"kind": "absent", "indices": []})
    monkeypatch.setattr(adapter, "_create_index", lambda index_name, metadata: None)
    monkeypatch.setattr(adapter, "_bulk_upsert", lambda items, index_name="": {"success": True, "count": len(items)})
    monkeypatch.setattr(adapter, "_refresh_index", lambda index_name="": None)
    monkeypatch.setattr(
        adapter,
        "_validate_staged_index",
        lambda index_name, manifest: {"success": False, "errorCode": "ES_STAGE_DOC_COUNT_MISMATCH"},
    )
    monkeypatch.setattr(adapter, "_delete_index", lambda index_name="": deleted.append(index_name))
    manifest = {
        "indexVersion": "abc123",
        "semanticSourceHash": "abc123full",
        "docCount": 1,
    }

    result = adapter.stage(docs, manifest)

    assert result["success"] is False
    assert result["errorCode"] == "ES_STAGE_FAILED"
    assert result["physicalIndex"] in deleted


def test_incremental_stage_copies_active_and_applies_only_manifest_delta(
    tmp_path,
    monkeypatch,
):
    settings = _settings(tmp_path, es_enabled=True)
    adapter = EsRecallIndexAdapter(settings)
    changed = RecallItem(
        doc_id="metric:return_rate",
        title="return rate",
        content="v2",
        source_type="SEMANTIC_METRIC",
        topic="经营画像",
        table="ads_merchant_profile",
        metadata={
            "semanticRefId": "metric:return_rate",
            "semanticPath": (
                "topics/经营画像/tables/ads_merchant_profile/"
                "asset.json#metric:return_rate"
            ),
        },
    )
    stable = RecallItem(
        doc_id="metric:gmv",
        title="gmv",
        content="stable",
        source_type="SEMANTIC_METRIC",
        topic="电商交易",
        table="orders",
        metadata={
            "semanticRefId": "metric:gmv",
            "semanticPath": "topics/电商交易/tables/orders/asset.json#metric:gmv",
        },
    )
    updated_ref = (
        "经营画像/tables/ads_merchant_profile/"
        "asset.json#metric:return_rate"
    )
    removed_ref = "经营画像/tables/ads_merchant_profile/asset.json#metric:old"
    previous_manifest = {
        "indexVersion": "v1",
        "semanticSourceHash": "source-v1",
        "indexConfigHash": "same-config",
        "docCount": 3,
    }
    manifest = {
        "indexVersion": "v2",
        "semanticSourceHash": "source-v2",
        "indexConfigHash": "same-config",
        "docCount": 2,
    }
    events = []
    monkeypatch.setattr(
        adapter,
        "_alias_state",
        lambda alias: {"kind": "alias", "indices": ["recall__v_v1"]},
    )
    monkeypatch.setattr(
        adapter,
        "_create_index",
        lambda index_name, metadata: events.append(
            ("create", index_name, metadata)
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_copy_index",
        lambda source, target: (
            events.append(("copy", source, target))
            or {"success": True, "count": 3}
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_delete_refs",
        lambda refs, index_name="": (
            events.append(("delete", list(refs), index_name))
            or {"success": True, "count": 2}
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_bulk_upsert",
        lambda docs, index_name="": (
            events.append(("upsert", [doc.doc_id for doc in docs], index_name))
            or {"success": True, "count": len(docs)}
        ),
    )
    monkeypatch.setattr(adapter, "_refresh_index", lambda index_name="": None)
    monkeypatch.setattr(
        adapter,
        "_validate_staged_index",
        lambda index_name, candidate: {
            "success": True,
            "docCount": 2,
            "semanticSourceHash": candidate["semanticSourceHash"],
        },
    )

    result = adapter.stage_incremental(
        [changed, stable],
        manifest,
        previous_manifest=previous_manifest,
        updated_refs=[updated_ref, "deleted:%s" % removed_ref],
    )

    assert result["success"] is True
    assert result["candidateBuildMode"] == "incremental_copy_on_write"
    assert result["baseIndex"] == "recall__v_v1"
    assert result["copied"] == 3
    assert result["upserted"] == 1
    assert any(event[:2] == ("copy", "recall__v_v1") for event in events)
    assert next(event for event in events if event[0] == "upsert")[1] == [
        "metric:return_rate"
    ]
    assert next(event for event in events if event[0] == "delete")[1] == [
        updated_ref,
        "deleted:%s" % removed_ref,
    ]


def test_incremental_stage_falls_back_to_full_for_index_config_change(
    tmp_path,
    monkeypatch,
):
    settings = _settings(tmp_path, es_enabled=True)
    adapter = EsRecallIndexAdapter(settings)
    docs = VersionedProvider(settings)._load_documents()
    monkeypatch.setattr(
        adapter,
        "_alias_state",
        lambda alias: {"kind": "alias", "indices": ["recall__v_v1"]},
    )
    monkeypatch.setattr(adapter, "_create_index", lambda *args: None)
    monkeypatch.setattr(
        adapter,
        "_copy_index",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("config changes must not copy the old generation")
        ),
    )
    captured = {}

    def bulk(values, index_name=""):
        captured["docIds"] = [item.doc_id for item in values]
        return {"success": True, "count": len(values)}

    monkeypatch.setattr(adapter, "_bulk_upsert", bulk)
    monkeypatch.setattr(adapter, "_refresh_index", lambda index_name="": None)
    monkeypatch.setattr(
        adapter,
        "_validate_staged_index",
        lambda index_name, candidate: {
            "success": True,
            "docCount": 1,
            "semanticSourceHash": "source-v2",
        },
    )

    result = adapter.stage_incremental(
        docs,
        {
            "indexVersion": "v2",
            "semanticSourceHash": "source-v2",
            "indexConfigHash": "new-config",
            "docCount": 1,
        },
        previous_manifest={"indexConfigHash": "old-config"},
        updated_refs=["经营画像/tables/ads_merchant_profile/asset.json"],
    )

    assert result["success"] is True
    assert result["candidateBuildMode"] == "full_rebuild"
    assert result["fullRebuildReason"] == "INDEX_CONFIG_CHANGED_OR_UNKNOWN"
    assert captured["docIds"] == ["metric:return_rate"]


def test_incremental_stage_falls_back_when_active_index_mismatches_manifest(
    tmp_path,
    monkeypatch,
):
    settings = _settings(tmp_path, es_enabled=True)
    adapter = EsRecallIndexAdapter(settings)
    docs = VersionedProvider(settings)._load_documents()
    monkeypatch.setattr(
        adapter,
        "_alias_state",
        lambda alias: {"kind": "alias", "indices": ["recall__v_stale"]},
    )
    monkeypatch.setattr(adapter, "_create_index", lambda *args: None)
    monkeypatch.setattr(
        adapter,
        "_copy_index",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("an invalid base must not be copied")
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_bulk_upsert",
        lambda values, index_name="": {
            "success": True,
            "count": len(values),
        },
    )
    monkeypatch.setattr(adapter, "_refresh_index", lambda index_name="": None)
    validations = iter(
        [
            {"success": False, "errorCode": "ES_STAGE_SOURCE_HASH_MISMATCH"},
            {
                "success": True,
                "docCount": 1,
                "semanticSourceHash": "source-v2",
            },
        ]
    )
    monkeypatch.setattr(
        adapter,
        "_validate_staged_index",
        lambda index_name, candidate: next(validations),
    )

    result = adapter.stage_incremental(
        docs,
        {
            "indexVersion": "v2",
            "semanticSourceHash": "source-v2",
            "indexConfigHash": "same-config",
            "docCount": 1,
        },
        previous_manifest={
            "indexVersion": "v1",
            "semanticSourceHash": "source-v1",
            "indexConfigHash": "same-config",
            "docCount": 1,
        },
        updated_refs=["经营画像/tables/ads_merchant_profile/asset.json"],
    )

    assert result["success"] is True
    assert result["candidateBuildMode"] == "full_rebuild"
    assert result["fullRebuildReason"] == "ACTIVE_INDEX_MANIFEST_MISMATCH"


def test_manifest_diff_uses_doc_id_when_chunks_share_one_semantic_path():
    path = "rules/refund.md"
    previous = {
        "docs": [
            {"docId": "rule:chunk:0", "ref": path, "hash": "old-0"},
            {"docId": "rule:chunk:1", "ref": path, "hash": "old-1"},
        ]
    }
    current = [
        {"docId": "rule:chunk:0", "ref": path, "hash": "new-0"},
    ]

    assert changed_refs(current, previous, changed_only=True) == [
        path,
        "deleted:%s" % path,
    ]


def test_candidate_base_copy_uses_server_side_reindex(tmp_path, monkeypatch):
    settings = _settings(tmp_path, es_enabled=True)
    adapter = EsRecallIndexAdapter(settings)
    captured = {}

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"created": 7, "version_conflicts": 0, "failures": []}

    def post(url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs["json"]
        return Response()

    monkeypatch.setattr("merchant_ai.services.recall_index.requests.post", post)

    result = adapter._copy_index("recall__v_old", "recall__v_candidate")

    assert result == {"success": True, "count": 7, "error": ""}
    assert captured["url"].endswith(
        "/_reindex?refresh=false&wait_for_completion=true"
    )
    assert captured["json"] == {
        "source": {"index": "recall__v_old"},
        "dest": {"index": "recall__v_candidate", "op_type": "create"},
    }


def test_manager_passes_complete_manifest_delta_to_incremental_adapter(
    tmp_path,
):
    settings, provider, _ = _active_v1(tmp_path)
    provider.version = "v2"
    settings.es_enabled = True
    manager = RecallIndexManager(settings, provider)
    adapter = RecordingIncrementalAdapter(manager.manifest_path)
    manager.es_adapter = adapter

    result = manager.rebuild(
        changed_only=True,
        topic="经营画像",
        table_name="ads_merchant_profile",
    )

    assert result["success"] is True
    assert adapter.incremental_call["docIds"] == ["metric:return_rate"]
    assert adapter.incremental_call["updatedRefs"] == [
        "经营画像/tables/ads_merchant_profile/asset.json#metric:return_rate"
    ]
    assert adapter.incremental_call["replaceAll"] is False
    assert adapter.incremental_call["previousManifest"]["indexConfigHash"]


def test_alias_swap_removes_old_and_adds_candidate_in_one_request(tmp_path, monkeypatch):
    settings = _settings(tmp_path, es_enabled=True)
    adapter = EsRecallIndexAdapter(settings)
    captured = {}

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

    def post(url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs["json"]
        return Response()

    monkeypatch.setattr("merchant_ai.services.recall_index.requests.post", post)
    stage = {
        "alias": "recall",
        "physicalIndex": "recall__candidate",
        "previousState": {"kind": "alias", "indices": ["recall__old"]},
        "committed": False,
    }

    result = adapter.commit_stage(stage)

    assert result["success"] is True
    assert captured["url"].endswith("/_aliases")
    assert captured["json"]["actions"] == [
        {"remove": {"index": "recall__old", "alias": "recall"}},
        {"add": {"index": "recall__candidate", "alias": "recall", "is_write_index": True}},
    ]


def test_abort_keeps_legacy_backup_when_it_is_restored_as_active_alias(tmp_path, monkeypatch):
    settings = _settings(tmp_path, es_enabled=True)
    adapter = EsRecallIndexAdapter(settings)
    deleted = []
    monkeypatch.setattr(adapter, "_index_has_aliases", lambda index_name: index_name == "recall__legacy_backup")
    monkeypatch.setattr(adapter, "_delete_index", lambda index_name="": deleted.append(index_name))
    stage = {
        "alias": "recall",
        "physicalIndex": "recall__candidate",
        "legacyBackupIndex": "recall__legacy_backup",
        "previousState": {"kind": "concrete", "indices": ["recall"]},
        "committed": False,
    }

    result = adapter.abort_stage(stage)

    assert result["success"] is True
    assert result["retainedActive"] == ["recall__legacy_backup"]
    assert deleted == ["recall__candidate"]
