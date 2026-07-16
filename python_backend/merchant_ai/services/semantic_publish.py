from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from merchant_ai.config import Settings
from merchant_ai.models import RecallItem
from merchant_ai.services.assets import (
    HybridRecallService,
    SemanticAssetGovernanceService,
    TopicAssetService,
    semantic_candidate_source_hash,
)
from merchant_ai.services.recall_index import RecallIndexManager


class _PublishAbort(RuntimeError):
    def __init__(self, error_code: str, message: str = ""):
        super().__init__(message or error_code)
        self.error_code = error_code


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
        activation_verifier: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        with self.recall_index_manager.publication_transaction():
            submitted_validation = self._validate_preflight_binding(topic, table_name, preflight)
            if not submitted_validation.get("success"):
                return self._preflight_failure_result(topic, table_name, preflight, submitted_validation)
            try:
                authoritative_preflight = self.semantic_governance.preflight_publish(topic, table_name)
            except Exception as exc:
                return self._preflight_failure_result(
                    topic,
                    table_name,
                    preflight,
                    {
                        "success": False,
                        "status": "PREFLIGHT_VALIDATION_FAILED",
                        "errorCode": "PREFLIGHT_VALIDATION_FAILED",
                        "error": str(exc)[:500],
                    },
                )
            authoritative_validation = self._validate_preflight_binding(
                topic,
                table_name,
                authoritative_preflight,
            )
            if not authoritative_validation.get("success"):
                return self._preflight_failure_result(
                    topic,
                    table_name,
                    authoritative_preflight,
                    authoritative_validation,
                )
            if submitted_validation.get("sourceHash") != authoritative_validation.get("sourceHash"):
                return self._preflight_failure_result(
                    topic,
                    table_name,
                    authoritative_preflight,
                    {
                        "success": False,
                        "status": "PREFLIGHT_STALE",
                        "errorCode": "PREFLIGHT_STALE",
                        "expectedSourceHash": submitted_validation.get("sourceHash", ""),
                        "actualSourceHash": authoritative_validation.get("sourceHash", ""),
                    },
                )

            preflight = authoritative_preflight
            transaction_id = uuid.uuid4().hex
            work_dir = self.transaction_root / "work" / transaction_id
            record_path = self.transaction_root / "transactions" / (transaction_id + ".json")
            state: dict[str, Any] = {
                "transactionId": transaction_id,
                "state": "PREPARING_CANDIDATE",
                "topic": topic,
                "tableName": table_name,
                "preflightSourceHash": authoritative_validation.get("sourceHash", ""),
                "workDir": str(work_dir),
                "activeFilesystemAdvanced": False,
                "activeManifestAdvanced": False,
                "updatedAt": self._now(),
            }
            self._write_state(record_path, state)

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

            candidate_binding = self._validate_preflight_binding(topic, table_name, preflight)
            if not candidate_binding.get("success"):
                state.update(
                    {
                        "state": str(candidate_binding.get("status") or "PREFLIGHT_STALE"),
                        "errorCode": candidate_binding.get("errorCode") or "PREFLIGHT_STALE",
                        "updatedAt": self._now(),
                    }
                )
                self._write_state(record_path, state)
                self._cleanup_work_dir(work_dir)
                return self._preflight_failure_result(
                    topic,
                    table_name,
                    preflight,
                    candidate_binding,
                    transaction_id=transaction_id,
                    record_path=record_path,
                )

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

            staged_binding = self._validate_preflight_binding(topic, table_name, preflight)
            if not staged_binding.get("success"):
                abort_result = self.recall_index_manager.abort_prepared(
                    preparation,
                    reason=str(staged_binding.get("errorCode") or "PREFLIGHT_STALE"),
                )
                state.update(
                    {
                        "state": str(staged_binding.get("status") or "PREFLIGHT_STALE"),
                        "indexTransactionId": preparation.transaction_id,
                        "activeManifestAdvanced": False,
                        "errorCode": staged_binding.get("errorCode") or "PREFLIGHT_STALE",
                        "updatedAt": self._now(),
                    }
                )
                self._write_state(record_path, state)
                self._cleanup_work_dir(work_dir)
                return self._preflight_failure_result(
                    topic,
                    table_name,
                    preflight,
                    staged_binding,
                    transaction_id=transaction_id,
                    record_path=record_path,
                    recall_index=abort_result,
                )

            target_dir = self.topic_assets.table_asset_dir(topic, table_name)
            governance_dir = self.settings.resolved_workspace_path / "semantic_governance" / topic / table_name
            topic_manifest_path = self.topic_assets.root / topic / "manifest.json"
            target_existed = self._snapshot_directory(target_dir, work_dir / "backup" / "target")
            governance_existed = self._snapshot_directory(governance_dir, work_dir / "backup" / "governance")
            topic_manifest_existed = self._snapshot_file(
                topic_manifest_path,
                work_dir / "backup" / "topic_manifest.json",
            )
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
            activation_verification: dict[str, Any] = {}
            try:
                publish_result = self.topic_assets.publish(topic, table_name, True, reviewer, review_note)
                if not publish_result.get("success") or publish_result.get("status") != "PUBLISHED":
                    raise RuntimeError("FILESYSTEM_PUBLISH_DID_NOT_REACH_PUBLISHED")
                live_binding = self._validate_preflight_binding(topic, table_name, preflight)
                if not live_binding.get("success"):
                    raise _PublishAbort(
                        str(live_binding.get("errorCode") or "PREFLIGHT_STALE"),
                        str(live_binding.get("status") or "PREFLIGHT_STALE"),
                    )
                state.update({"state": "FILESYSTEM_STAGED", "activeFilesystemAdvanced": True, "updatedAt": self._now()})
                self._write_state(record_path, state)
                governance_result = self.semantic_governance.after_publish(topic, table_name, reviewer, review_note)
                governance_issue = self._post_publish_governance_issue(topic, table_name, governance_result)
                if governance_issue:
                    raise _PublishAbort(governance_issue, str(governance_result.get("status") or governance_issue))
                final_binding = self._validate_preflight_binding(topic, table_name, preflight)
                if not final_binding.get("success"):
                    raise _PublishAbort(
                        str(final_binding.get("errorCode") or "PREFLIGHT_STALE"),
                        str(final_binding.get("status") or "PREFLIGHT_STALE"),
                    )
                if activation_verifier is not None:
                    try:
                        activation_verification = activation_verifier(
                            {
                                "topic": topic,
                                "tableName": table_name,
                                "candidateDocuments": documents,
                                "previousRecallManifest": preparation.previous_manifest,
                                "candidateRecallManifest": preparation.candidate_manifest,
                                "publishResult": publish_result,
                                "governanceResult": governance_result,
                                "preflight": preflight,
                            }
                        )
                    except Exception as verifier_exc:
                        raise _PublishAbort("ACTIVATION_VERIFICATION_FAILED", str(verifier_exc)[:500]) from verifier_exc
                    if not isinstance(activation_verification, dict) or activation_verification.get("success") is not True:
                        verification_code = (
                            str(activation_verification.get("errorCode") or activation_verification.get("status") or "")
                            if isinstance(activation_verification, dict)
                            else ""
                        )
                        raise _PublishAbort(
                            verification_code or "ACTIVATION_VERIFICATION_FAILED",
                            verification_code or "ACTIVATION_VERIFICATION_FAILED",
                        )
                    state.update(
                        {
                            "state": "ACTIVATION_VERIFIED",
                            "activationVerification": activation_verification,
                            "updatedAt": self._now(),
                        }
                    )
                    self._write_state(record_path, state)
            except Exception as exc:
                error_code = exc.error_code if isinstance(exc, _PublishAbort) else "FILESYSTEM_PUBLISH_FAILED"
                abort_result = self.recall_index_manager.abort_prepared(preparation, reason=error_code)
                compensation = self._restore_active_filesystem(
                    target_dir,
                    work_dir / "backup" / "target",
                    target_existed,
                    governance_dir,
                    work_dir / "backup" / "governance",
                    governance_existed,
                    topic_manifest_path,
                    work_dir / "backup" / "topic_manifest.json",
                    topic_manifest_existed,
                )
                state.update(
                    {
                        "state": "PENDING_INDEX_RETRY" if compensation.get("success") else "COMPENSATION_FAILED",
                        "activeFilesystemAdvanced": not compensation.get("success", False),
                        "activeManifestAdvanced": False,
                        "errorCode": error_code,
                        "error": str(exc)[:500],
                        "compensation": compensation,
                        "updatedAt": self._now(),
                    }
                )
                self._write_state(record_path, state)
                self._cleanup_work_dir(work_dir)
                failure_status = (
                    error_code
                    if error_code
                    in {
                        "PREFLIGHT_STALE",
                        "POST_PUBLISH_RELEASE_GATE_FAILED",
                        "POST_PUBLISH_SCOPE_MISMATCH",
                        "ACTIVATION_VERIFICATION_FAILED",
                        "PUBLISHED_READBACK_FAILED",
                        "RECALL_READBACK_FAILED",
                        "RECALL_INDEX_UNCHANGED",
                    }
                    else "PUBLISH_PENDING_INDEX_RETRY"
                )
                return {
                    **publish_result,
                    "success": False,
                    "status": "PUBLISH_COMPENSATION_FAILED" if not compensation.get("success") else failure_status,
                    "publishState": "FAILED" if not compensation.get("success") else "PENDING",
                    "topic": topic,
                    "tableName": table_name,
                    "preflight": preflight,
                    "semanticGovernance": governance_result,
                    "activationVerification": activation_verification,
                    "recallIndex": abort_result,
                    "esUpsert": abort_result.get("es", {}),
                    "compensation": compensation,
                    "errorCode": error_code,
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
                    topic_manifest_path,
                    work_dir / "backup" / "topic_manifest.json",
                    topic_manifest_existed,
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
                    "topic": topic,
                    "tableName": table_name,
                    "preflight": preflight,
                    "semanticGovernance": governance_result,
                    "activationVerification": activation_verification,
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
                "activationVerification": activation_verification,
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

    def _validate_preflight_binding(
        self,
        topic: str,
        table_name: str,
        preflight: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(preflight, dict):
            return self._preflight_validation_error("PREFLIGHT_BINDING_INVALID")
        if str(preflight.get("topic") or "") != topic or str(preflight.get("tableName") or "") != table_name:
            return self._preflight_validation_error("PREFLIGHT_SCOPE_MISMATCH")
        if preflight.get("success") is not True or preflight.get("publishable") is not True:
            return self._preflight_validation_error("PREFLIGHT_FAILED")
        if str(preflight.get("status") or "") != "PREFLIGHT_PASSED":
            return self._preflight_validation_error("PREFLIGHT_BINDING_INVALID")
        catalog = preflight.get("semanticCatalogVersion")
        catalog = catalog if isinstance(catalog, dict) else {}
        pending_hash = str(preflight.get("pendingSourceHash") or "").strip()
        catalog_hash = str(catalog.get("sourceHash") or catalog.get("source_hash") or "").strip()
        if pending_hash and catalog_hash and pending_hash != catalog_hash:
            return self._preflight_validation_error("PREFLIGHT_BINDING_INVALID")
        declared_hash = pending_hash or catalog_hash
        if len(declared_hash) != 64 or any(character not in "0123456789abcdef" for character in declared_hash.lower()):
            return self._preflight_validation_error("PREFLIGHT_BINDING_INVALID")
        pending_dir = self.settings.resolved_topic_path / topic / "pending" / table_name
        current_hash = semantic_candidate_source_hash(pending_dir)
        if not current_hash or current_hash != declared_hash:
            return {
                **self._preflight_validation_error("PREFLIGHT_STALE"),
                "expectedSourceHash": declared_hash,
                "actualSourceHash": current_hash,
            }
        return {
            "success": True,
            "status": "PREFLIGHT_VALID",
            "sourceHash": declared_hash,
        }

    @staticmethod
    def _preflight_validation_error(error_code: str) -> dict[str, Any]:
        return {
            "success": False,
            "status": error_code,
            "errorCode": error_code,
        }

    @staticmethod
    def _post_publish_governance_issue(
        topic: str,
        table_name: str,
        governance_result: dict[str, Any],
    ) -> str:
        if not isinstance(governance_result, dict) or governance_result.get("success") is not True:
            return "SEMANTIC_GOVERNANCE_COMMIT_FAILED"
        if str(governance_result.get("topic") or "") != topic:
            return "POST_PUBLISH_SCOPE_MISMATCH"
        if str(governance_result.get("tableName") or "") != table_name:
            return "POST_PUBLISH_SCOPE_MISMATCH"
        release_gate = governance_result.get("releaseGate")
        if not isinstance(release_gate, dict) or release_gate.get("publishable") is not True:
            return "POST_PUBLISH_RELEASE_GATE_FAILED"
        return ""

    @staticmethod
    def _preflight_failure_result(
        topic: str,
        table_name: str,
        preflight: dict[str, Any],
        validation: dict[str, Any],
        transaction_id: str = "",
        record_path: Path | None = None,
        recall_index: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        error_code = str(validation.get("errorCode") or validation.get("status") or "PREFLIGHT_FAILED")
        result: dict[str, Any] = {
            "success": False,
            "status": error_code,
            "publishState": "PENDING",
            "topic": topic,
            "tableName": table_name,
            "preflight": preflight,
            "preflightValidation": validation,
            "errorCode": error_code,
        }
        if validation.get("error"):
            result["error"] = str(validation.get("error"))[:500]
        if transaction_id:
            result["transactionId"] = transaction_id
        if record_path is not None:
            result["transactionPath"] = str(record_path)
        if recall_index is not None:
            result["recallIndex"] = recall_index
            result["esUpsert"] = recall_index.get("es", {})
        return result

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

    @staticmethod
    def _snapshot_file(source: Path, destination: Path) -> bool:
        if not source.exists() or not source.is_file():
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return True

    def _restore_active_filesystem(
        self,
        target_dir: Path,
        target_backup: Path,
        target_existed: bool,
        governance_dir: Path,
        governance_backup: Path,
        governance_existed: bool,
        topic_manifest_path: Path,
        topic_manifest_backup: Path,
        topic_manifest_existed: bool,
    ) -> dict[str, Any]:
        try:
            self._restore_directory(target_dir, target_backup, target_existed)
            self._restore_directory(governance_dir, governance_backup, governance_existed)
            self._restore_file(topic_manifest_path, topic_manifest_backup, topic_manifest_existed)
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
        self.topic_assets._topic_contract_cache.clear()
        self.topic_assets._semantic_source_hash_cache.clear()

    @staticmethod
    def _restore_directory(target: Path, backup: Path, existed: bool) -> None:
        if target.exists():
            shutil.rmtree(target)
        if existed:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(backup, target)

    @staticmethod
    def _restore_file(target: Path, backup: Path, existed: bool) -> None:
        if target.exists():
            target.unlink()
        if existed:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, target)

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
