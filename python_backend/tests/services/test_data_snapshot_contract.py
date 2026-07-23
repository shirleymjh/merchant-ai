import pytest

from merchant_ai.config import get_settings
from merchant_ai.models import DataSnapshotContract, QueryBundle
from merchant_ai.services.data_snapshot_contract import validate_query_bundle_snapshots
from merchant_ai.services.repositories import DorisRepository


class EpochAwareDb:
    def __init__(self, epoch: str = "epoch-1"):
        self.epoch = epoch
        self.data_calls = 0

    def query(self, sql, params=None):
        if sql == "SELECT generation AS data_epoch FROM governed_epoch":
            return [{"data_epoch": self.epoch}]
        self.data_calls += 1
        return [{"value": self.data_calls}]


def _settings(tmp_path):
    return get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "cache_enabled": True,
            "cache_doris_select_ttl_seconds": 60,
            "doris_datasource_environment": "test",
            "doris_data_epoch_sql": "SELECT generation AS data_epoch FROM governed_epoch",
            "doris_data_epoch_column": "data_epoch",
            "doris_cache_generation": "cache-schema-v1",
        }
    )


def test_doris_cache_is_disabled_without_trusted_snapshot_identity(tmp_path):
    repository = DorisRepository(
        get_settings().model_copy(
            update={
                "harness_workspace_path": str(tmp_path),
                "cache_enabled": True,
                "cache_doris_select_ttl_seconds": 60,
            }
        )
    )
    database = EpochAwareDb()
    repository.db = database

    first = repository.query("SELECT value FROM facts")
    second = repository.query("SELECT value FROM facts")

    assert first != second
    assert database.data_calls == 2
    assert not repository.last_cache_hit
    assert repository.last_cache_key == ""


def test_doris_cache_key_is_bound_to_data_and_semantic_generations(tmp_path):
    repository = DorisRepository(_settings(tmp_path))
    database = EpochAwareDb()
    repository.db = database
    snapshot = repository.capture_data_snapshot("semantic-v1")

    first = repository.query(
        "SELECT value FROM facts",
        data_snapshot_contract=snapshot,
    )
    first_key = repository.last_cache_key
    second = repository.query(
        "SELECT   value FROM facts",
        data_snapshot_contract=snapshot,
    )

    assert snapshot.consistency_mode == "OBSERVED_EPOCH"
    assert second == first
    assert repository.last_cache_hit
    assert database.data_calls == 1

    changed_semantics = snapshot.model_copy(
        update={"semantic_activation_fingerprint": "semantic-v2"}
    )
    repository.query(
        "SELECT value FROM facts",
        data_snapshot_contract=changed_semantics,
    )
    assert not repository.last_cache_hit
    assert repository.last_cache_key != first_key

    database.epoch = "epoch-2"
    changed_epoch = repository.capture_data_snapshot("semantic-v1")
    repository.query(
        "SELECT value FROM facts",
        data_snapshot_contract=changed_epoch,
    )
    assert not repository.last_cache_hit
    assert repository.last_cache_key not in {"", first_key}

    epoch_key = repository.last_cache_key
    repository.settings.doris_cache_generation = "cache-schema-v2"
    changed_cache_generation = repository.capture_data_snapshot("semantic-v1")
    repository.query(
        "SELECT value FROM facts",
        data_snapshot_contract=changed_cache_generation,
    )
    assert not repository.last_cache_hit
    assert repository.last_cache_key not in {"", first_key, epoch_key}


def test_doris_cache_reuses_verified_semantic_request_across_equivalent_sql(tmp_path):
    repository = DorisRepository(_settings(tmp_path))
    database = EpochAwareDb()
    repository.db = database
    snapshot = repository.capture_data_snapshot("semantic-v1")
    semantic_fingerprint = "a" * 64
    scope_fingerprint = "b" * 64

    first = repository.query(
        "SELECT SUM(amount) AS total_amount FROM facts",
        data_snapshot_contract=snapshot,
        semantic_request_fingerprint=semantic_fingerprint,
        scope_fingerprint=scope_fingerprint,
    )
    second = repository.query(
        "SELECT SUM(amount) total_amount FROM facts",
        data_snapshot_contract=snapshot,
        semantic_request_fingerprint=semantic_fingerprint,
        scope_fingerprint=scope_fingerprint,
    )

    assert second == first
    assert database.data_calls == 1
    assert repository.last_cache_hit
    assert repository.cache_trace()["lastCacheIdentityKind"] == "SEMANTIC_REQUEST"

    repository.query(
        "SELECT SUM(amount) total_amount FROM facts",
        data_snapshot_contract=snapshot,
        semantic_request_fingerprint=semantic_fingerprint,
        scope_fingerprint="c" * 64,
    )

    assert database.data_calls == 2
    assert not repository.last_cache_hit


def test_multi_query_snapshot_validation_rejects_mixed_or_non_atomic_reads():
    observed = DataSnapshotContract(
        datasource_fingerprint="datasource",
        datasource_environment="prod",
        data_epoch="epoch-1",
        consistency_mode="OBSERVED_EPOCH",
        semantic_activation_fingerprint="semantic-v1",
        cache_generation="cache-v1",
    )
    same = [
        QueryBundle(rows=[{"value": 1}], data_snapshot=observed),
        QueryBundle(rows=[{"value": 2}], data_snapshot=observed),
    ]

    assert validate_query_bundle_snapshots(
        same,
        require_atomic_multi_query=False,
    ) == []
    assert validate_query_bundle_snapshots(
        same,
        require_atomic_multi_query=True,
    ) == ["ATOMIC_MULTI_QUERY_SNAPSHOT_UNSUPPORTED"]

    changed = observed.model_copy(update={"data_epoch": "epoch-2"})
    assert validate_query_bundle_snapshots(
        [same[0], QueryBundle(rows=[{"value": 2}], data_snapshot=changed)],
        require_atomic_multi_query=False,
    ) == ["DATA_SNAPSHOT_MISMATCH"]


def test_repository_rejects_snapshot_if_epoch_changes_before_query(tmp_path):
    repository = DorisRepository(_settings(tmp_path))
    database = EpochAwareDb()
    repository.db = database
    snapshot = repository.capture_data_snapshot("semantic-v1")
    database.epoch = "epoch-2"

    with pytest.raises(RuntimeError) as raised:
        repository.query(
            "SELECT value FROM facts",
            data_snapshot_contract=snapshot,
        )

    assert "DATA_SNAPSHOT_STALE" in str(raised.value)
    assert database.data_calls == 0


def test_atomic_as_of_snapshot_can_compose_multiple_queries():
    as_of = DataSnapshotContract(
        datasource_fingerprint="datasource",
        datasource_environment="prod",
        data_epoch="epoch-1",
        consistency_mode="AS_OF_READ",
        semantic_activation_fingerprint="semantic-v1",
        cache_generation="cache-v1",
    )
    bundles = [
        QueryBundle(rows=[{"value": 1}], data_snapshot=as_of),
        QueryBundle(rows=[{"value": 2}], data_snapshot=as_of),
    ]

    assert validate_query_bundle_snapshots(
        bundles,
        require_atomic_multi_query=True,
    ) == []
