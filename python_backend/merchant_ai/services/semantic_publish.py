from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from merchant_ai.config import Settings
from merchant_ai.models import RecallItem
from merchant_ai.services.assets import HybridRecallService, SemanticAssetGovernanceService, TopicAssetService
from merchant_ai.services.recall_index import RecallIndexManager


class SemanticPublishCoordinator:
    """Coordinate semantic filesystem activation with the versioned recall index."""

    def __init__(
        self,
        settings: Settings,
        topic_assets: TopicAssetService,
        semantic_governance: SemanticAssetGovernanceService,
        recall_index_manager: RecallIndexManager,
    ):
        self.settings = settings
        self.topic_assets = topic_assets
        self.semantic_governance = semantic_governance
        self.recall_index_manager = recall_index_manager

    @property
    def transaction_root(self) -> Path:
        return self.settings.resolved_workspace_path / "semantic_publish_transactions"

    def publish_approved(
        self,
        topic: str,
        table_name: str,
        reviewer: str,
        review_note: str,
        preflight: dict[str, Any],
    ) -> dict[str, Any]:
        transaction_id = uuid.uuid4().hex
        work_dir = self.transaction_root / "work" / transaction_id
        record_path = self.transaction_root / "transactions" / (transaction_id + ".json")
        state: dict[str, Any] = {
            "transactionId": transaction_id,
            "state": "PREPARING_CANDIDATE",
            "topic": topic,
            "tableName": table_name,
            "workDir": str(work_dir),
            "activeFilesystemAdvanced": False,
            "activeManifestAdvanced": False,
            "updatedAt": self._now(),
        }
        self._write_state(record_path, state)

        with self.recall_index_manager.publication_transaction():
            try:
                documents, candidate_publish = self._candidate_documents(
                    work_dir,
                    topic,
                    table_name,
                    reviewer,
                    review_note,
                )
            except Exception as exc:
                state.update({"state": "CANDIDATE_BUILD_FAILED", "error": str(exc)[:500], "updatedAt": self._now()})
                self._write_state(record_path, state)
                self._cleanup_work_dir(work_dir)
                return {
                    "success": False,
                    "status": "CANDIDATE_BUILD_FAILED",
                    "publishState": "PENDING",
                    "topic": topic,
                    "tableName": table_name,
                    "preflight": preflight,
                    "errorCode": "CANDIDATE_RECALL_BUILD_FAILED",
                    "error": str(exc)[:500],
                    "transactionId": transaction_id,
                    "transactionPath": str(record_path),
                }

            state.update(
                {
                    "state": "PREPARING_INDEX",
                    "candidatePublish": candidate_publish,
                    "candidateDocCount": len(documents),
                    "updatedAt": self._now(),
                }
            )
            self._write_state(record_path, state)
            preparation = self.recall_index_manager.prepare_rebuild(
                changed_only=True,
                topic=topic,
                table_name=table_name,
                documents=documents,
            )
            if not preparation.success:
                index_result = self.recall_index_manager.preparation_result(preparation)
                state.update(
                    {
                        "state": "PENDING_INDEX_RETRY",
                        "indexTransactionId": preparation.transaction_id,
                        "activeManifestAdvanced": False,
                        "errorCode": index_result.get("errorCode") or "INDEX_PREPARE_FAILED",
                        "updatedAt": self._now(),
                    }
                )
                self._write_state(record_path, state)
                self._cleanup_work_dir(work_dir)
                return self._pending_result(topic, table_name, preflight, transaction_id, record_path, index_result)

            target_dir = self.topic_assets.table_asset_dir(topic, table_name)
            governance_dir = self.settings.resolved_workspace_path / "semantic_governance" / topic / table_name
            target_existed = self._snapshot_directory(target_dir, work_dir / "backup" / "target")
            governance_existed = self._snapshot_directory(governance_dir, work_dir / "backup" / "governance")
            state.update(
                {
                    "state": "INDEX_STAGED",
                    "indexTransactionId": preparation.transaction_id,
                    "candidateIndexVersion": preparation.candidate_manifest.get("indexVersion", ""),
                    "candidateSemanticSourceHash": preparation.candidate_manifest.get("semanticSourceHash", ""),
                    "updatedAt": self._now(),
                }
            )
            self._write_state(record_path, state)

            publish_result: dict[str, Any] = {}
            governance_result: dict[str, Any] = {}
            try:
                publish_result = self.topic_assets.publish(topic, table_name, True, reviewer, review_note)
                if not publish_result.get("success") or publish_result.get("status") != "PUBLISHED":
                    raise RuntimeError("FILESYSTEM_PUBLISH_DID_NOT_REACH_PUBLISHED")
                state.update({"state": "FILESYSTEM_STAGED", "activeFilesystemAdvanced": True, "updatedAt": self._now()})
                self._write_state(record_path, state)
                governance_result = self.semantic_governance.after_publish(topic, table_name, reviewer, review_note)
                if not governance_result.get("success", False):
                    raise RuntimeError(str(governance_result.get("status") or "SEMANTIC_GOVERNANCE_COMMIT_FAILED"))
            except Exception as exc:
                abort_result = self.recall_index_manager.abort_prepared(preparation, reason="FILESYSTEM_PUBLISH_FAILED")
                compensation = self._restore_active_filesystem(
                    target_dir,
                    work_dir / "backup" / "target",
                    target_existed,
                    governance_dir,
                    work_dir / "backup" / "governance",
                    governance_existed,
                )
                state.update(
                    {
                        "state": "PENDING_INDEX_RETRY" if compensation.get("success") else "COMPENSATION_FAILED",
                        "activeFilesystemAdvanced": not compensation.get("success", False),
                        "activeManifestAdvanced": False,
                        "errorCode": "FILESYSTEM_PUBLISH_FAILED",
                        "error": str(exc)[:500],
                        "compensation": compensation,
                        "updatedAt": self._now(),
                    }
                )
                self._write_state(record_path, state)
                self._cleanup_work_dir(work_dir)
                return {
                    **publish_result,
                    "success": False,
                    "status": "PUBLISH_COMPENSATION_FAILED" if not compensation.get("success") else "PUBLISH_PENDING_INDEX_RETRY",
                    "publishState": "FAILED" if not compensation.get("success") else "PENDING",
                    "topic": topic,
                    "tableName": table_name,
                    "preflight": preflight,
                    "semanticGovernance": governance_result,
                    "recallIndex": abort_result,
                    "esUpsert": abort_result.get("es", {}),
                    "compensation": compensation,
                    "errorCode": "FILESYSTEM_PUBLISH_FAILED",
                    "error": str(exc)[:500],
                    "transactionId": transaction_id,
                    "transactionPath": str(record_path),
                }

            index_result = self.recall_index_manager.commit_prepared(preparation)
            if not index_result.get("success", False):
                compensation = self._restore_active_filesystem(
                    target_dir,
                    work_dir / "backup" / "target",
                    target_existed,
                    governance_dir,
                    work_dir / "backup" / "governance",
                    governance_existed,
                )
                state.update(
                    {
                        "state": "PENDING_INDEX_RETRY" if compensation.get("success") else "COMPENSATION_FAILED",
                        "activeFilesystemAdvanced": not compensation.get("success", False),
                        "activeManifestAdvanced": False,
                        "errorCode": index_result.get("errorCode") or "INDEX_COMMIT_FAILED",
                        "compensation": compensation,
                        "updatedAt": self._now(),
                    }
                )
                self._write_state(record_path, state)
                self._cleanup_work_dir(work_dir)
                return {
                    **publish_result,
                    "success": False,
                    "status": "PUBLISH_COMPENSATION_FAILED" if not compensation.get("success") else "PUBLISH_PENDING_INDEX_RETRY",
                    "publishState": "FAILED" if not compensation.get("success") else "PENDING",
                    "preflight": preflight,
                    "semanticGovernance": governance_result,
                    "recallIndex": index_result,
                    "esUpsert": index_result.get("es", {}),
                    "cacheInvalidated": bool(index_result.get("cacheInvalidated")),
                    "compensation": compensation,
                    "transactionId": transaction_id,
                    "transactionPath": str(record_path),
                }

            state.update(
                {
                    "state": "ACTIVE",
                    "activeFilesystemAdvanced": True,
                    "activeManifestAdvanced": True,
                    "activeIndexVersion": index_result.get("indexVersion", ""),
                    "activeSemanticSourceHash": index_result.get("semanticSourceHash", ""),
                    "updatedAt": self._now(),
                }
            )
            self._write_state(record_path, state)
            self._cleanup_work_dir(work_dir)
            return {
                **publish_result,
                "success": True,
                "status": "PUBLISHED",
                "publishState": "ACTIVE",
                "preflight": preflight,
                "semanticGovernance": governance_result,
                "recallIndex": index_result,
                "esUpsert": index_result.get("es", {}),
                "cacheInvalidated": bool(index_result.get("cacheInvalidated")),
                "transactionId": transaction_id,
                "transactionPath": str(record_path),
            }

    def _candidate_documents(
        self,
        work_dir: Path,
        topic: str,
        table_name: str,
        reviewer: str,
        review_note: str,
    ) -> tuple[list[RecallItem], dict[str, Any]]:
        staged_topics = work_dir / "staging" / "topics"
        live_topics = self.settings.resolved_topic_path
        staged_topics.parent.mkdir(parents=True, exist_ok=True)
        if live_topics.exists():
            shutil.copytree(live_topics, staged_topics)
        else:
            staged_topics.mkdir(parents=True, exist_ok=True)
        staged_settings = self.settings.model_copy(update={"topic_path": str(staged_topics)})
        staged_assets = TopicAssetService(staged_settings)
        publish_result = staged_assets.publish(topic, table_name, True, reviewer, review_note)
        if not publish_result.get("success") or publish_result.get("status") != "PUBLISHED":
            raise RuntimeError(str(publish_result.get("status") or "CANDIDATE_PUBLISH_FAILED"))
        provider = HybridRecallService(staged_settings, staged_assets)
        return provider._load_documents(), publish_result

    def _pending_result(
        self,
        topic: str,
        table_name: str,
        preflight: dict[str, Any],
        transaction_id: str,
        record_path: Path,
        index_result: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "success": False,
            "status": "PUBLISH_PENDING_INDEX_RETRY",
            "publishState": "PENDING",
            "topic": topic,
            "tableName": table_name,
            "preflight": preflight,
            "recallIndex": index_result,
            "esUpsert": index_result.get("es", {}),
            "cacheInvalidated": bool(index_result.get("cacheInvalidated")),
            "errorCode": index_result.get("errorCode") or "INDEX_PREPARE_FAILED",
            "transactionId": transaction_id,
            "transactionPath": str(record_path),
        }

    @staticmethod
    def _snapshot_directory(source: Path, destination: Path) -> bool:
        if not source.exists():
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        return True

    def _restore_active_filesystem(
        self,
        target_dir: Path,
        target_backup: Path,
        target_existed: bool,
        governance_dir: Path,
        governance_backup: Path,
        governance_existed: bool,
    ) -> dict[str, Any]:
        try:
            self._restore_directory(target_dir, target_backup, target_existed)
            self._restore_directory(governance_dir, governance_backup, governance_existed)
            self._clear_topic_asset_caches()
            self.recall_index_manager.clear_caches()
            return {"success": True, "status": "OLD_ACTIVE_RESTORED"}
        except Exception as exc:
            return {
                "success": False,
                "status": "OLD_ACTIVE_RESTORE_FAILED",
                "errorCode": "FILESYSTEM_COMPENSATION_FAILED",
                "error": str(exc)[:500],
            }

    def _clear_topic_asset_caches(self) -> None:
        self.topic_assets._table_asset_cache.clear()
        self.topic_assets._manifest_cache.clear()
        self.topic_assets._relationship_cache.clear()
        self.topic_assets._topic_names_cache = None
        self.topic_assets._semantic_source_hash_cache.clear()

    @staticmethod
    def _restore_directory(target: Path, backup: Path, existed: bool) -> None:
        if target.exists():
            shutil.rmtree(target)
        if existed:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(backup, target)

    @staticmethod
    def _cleanup_work_dir(work_dir: Path) -> None:
        try:
            shutil.rmtree(work_dir)
        except FileNotFoundError:
            return

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _write_state(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(".%s.%s.tmp" % (path.name, uuid.uuid4().hex))
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
