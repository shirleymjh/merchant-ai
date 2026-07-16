from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import requests

from merchant_ai.config import Settings
from merchant_ai.models import RecallItem
from merchant_ai.services.cache import build_ttl_cache, stable_cache_key


class RecallDocumentProvider(Protocol):
    settings: Settings

    def _load_documents(self) -> list[RecallItem]:
        ...

    def clear_cache(self) -> None:
        ...


@dataclass
class RecallIndexPreparation:
    success: bool
    previous_manifest: dict[str, Any]
    candidate_manifest: dict[str, Any]
    updated_refs: list[str]
    changed_only: bool
    topic: str
    table_name: str
    replace_all: bool
    external_documents: bool
    transaction_id: str
    transaction_path: Path
    staged_manifest_path: Path | None = None
    es_stage: dict[str, Any] = field(default_factory=dict)
    es_result: dict[str, Any] = field(default_factory=dict)
    error_code: str = ""
    error: str = ""

    @property
    def active_manifest(self) -> dict[str, Any]:
        return self.previous_manifest


def source_hash(docs: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for doc in docs:
        digest.update(doc["ref"].encode("utf-8"))
        digest.update(doc["hash"].encode("utf-8"))
    return digest.hexdigest()


def load_index_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def changed_refs(docs: list[dict[str, str]], previous: dict[str, Any], changed_only: bool) -> list[str]:
    previous_docs = {
        str(item.get("ref") or ""): str(item.get("hash") or "")
        for item in previous.get("docs", [])
        if isinstance(item, dict)
    }
    if not changed_only or not previous_docs:
        return [doc["ref"] for doc in docs]
    current_refs = {doc["ref"] for doc in docs}
    changed = [doc["ref"] for doc in docs if previous_docs.get(doc["ref"]) != doc["hash"]]
    removed = sorted(set(previous_docs) - current_refs)
    return changed + ["deleted:%s" % ref for ref in removed]


def build_index_manifest(
    docs: list[dict[str, str]],
    previous: dict[str, Any],
    changed_only: bool,
    root: str = "",
) -> dict[str, Any]:
    docs.sort(key=lambda item: item["ref"])
    semantic_source_hash = source_hash(docs)
    updated_refs = changed_refs(docs, previous, changed_only)
    return {
        "indexVersion": semantic_source_hash[:16],
        "semanticSourceHash": semantic_source_hash,
        "docCount": len(docs),
        "updatedRefs": updated_refs,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "docs": [
            {
                "ref": doc["ref"],
                "hash": doc["hash"],
                **({"docId": doc["docId"]} if doc.get("docId") else {}),
                **({"sourceType": doc["sourceType"]} if doc.get("sourceType") else {}),
                **({"semanticPath": doc["semanticPath"]} if doc.get("semanticPath") else {}),
            }
            for doc in docs
        ],
    }


def recall_documents_for_manifest(items: list[RecallItem]) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    for item in items:
        metadata = item.metadata or {}
        semantic_path = str(metadata.get("semanticPath") or "")
        semantic_ref = str(metadata.get("semanticRefId") or item.doc_id or "")
        payload = {
            "docId": item.doc_id,
            "title": item.title,
            "content": item.content,
            "sourceType": item.source_type,
            "topic": item.topic,
            "table": item.table,
            "answerMode": item.answer_mode,
            "semanticRefId": semantic_ref,
            "semanticPath": semantic_path,
            "metadata": metadata,
        }
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        docs.append(
            {
                "ref": manifest_ref_for_recall_item(item),
                "hash": digest,
                "docId": str(item.doc_id or ""),
                "sourceType": str(item.source_type or ""),
                "semanticPath": semantic_path,
            }
        )
    docs.sort(key=lambda doc: (doc["ref"], doc.get("docId", "")))
    return docs


def manifest_ref_for_recall_item(item: RecallItem) -> str:
    metadata = item.metadata or {}
    semantic_path = str(metadata.get("semanticPath") or "").strip()
    if semantic_path.startswith("topics/"):
        return semantic_path.removeprefix("topics/")
    if semantic_path:
        return semantic_path
    return str(metadata.get("semanticRefId") or item.doc_id or "").strip()


class EsRecallIndexAdapter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._embedding_cache = build_ttl_cache("recall_index_embedding", settings, settings.cache_recall_ttl_seconds)

    def sync(self, docs: list[RecallItem], deleted_refs: list[str], replace_all: bool = False) -> dict[str, Any]:
        if not self.settings.es_base_url:
            return {"success": False, "mode": "es", "errorCode": "ES_BASE_URL_MISSING"}
        if replace_all:
            self._delete_index()
        self._ensure_index()
        deleted = self._delete_refs(deleted_refs)
        upserted = self._bulk_upsert(docs)
        self._refresh_index()
        return {
            "success": bool(deleted.get("success", True) and upserted.get("success", True)),
            "mode": "es",
            "index": self.settings.es_index,
            "upserted": upserted.get("count", 0),
            "deleted": deleted.get("count", 0),
            "replaceAll": replace_all,
            "errors": [item for item in [deleted.get("error"), upserted.get("error")] if item],
        }

    def stage(self, docs: list[RecallItem], manifest: dict[str, Any]) -> dict[str, Any]:
        """Build and validate an immutable physical index without changing the read alias."""
        if not self.settings.es_base_url:
            return {"success": False, "mode": "es", "errorCode": "ES_BASE_URL_MISSING"}
        alias = self.settings.es_index
        version = str(manifest.get("indexVersion") or "unknown")
        physical_index = self._physical_index_name(alias, version)
        previous_state: dict[str, Any] = {}
        legacy_backup = ""
        try:
            previous_state = self._alias_state(alias)
            if previous_state.get("kind") == "concrete":
                legacy_backup = self._backup_concrete_index(alias, version)
            self._create_index(
                physical_index,
                {
                    "indexVersion": version,
                    "semanticSourceHash": str(manifest.get("semanticSourceHash") or ""),
                    "docCount": int(manifest.get("docCount") or 0),
                },
            )
            upserted = self._bulk_upsert(docs, index_name=physical_index)
            if not upserted.get("success", False):
                raise RuntimeError(str(upserted.get("error") or "ES_BULK_UPSERT_FAILED"))
            self._refresh_index(physical_index)
            validation = self._validate_staged_index(physical_index, manifest)
            if not validation.get("success", False):
                raise RuntimeError(str(validation.get("errorCode") or "ES_STAGE_VALIDATION_FAILED"))
            return {
                "success": True,
                "mode": "es",
                "transactionMode": "physical_index_alias_swap",
                "alias": alias,
                "physicalIndex": physical_index,
                "previousState": previous_state,
                "legacyBackupIndex": legacy_backup,
                "upserted": int(upserted.get("count") or 0),
                "validatedDocCount": validation.get("docCount", 0),
                "validatedSemanticSourceHash": validation.get("semanticSourceHash", ""),
                "committed": False,
            }
        except Exception as exc:
            self._delete_index(physical_index)
            if legacy_backup:
                self._delete_index(legacy_backup)
            return {
                "success": False,
                "mode": "es",
                "transactionMode": "physical_index_alias_swap",
                "alias": alias,
                "physicalIndex": physical_index,
                "previousState": previous_state,
                "errorCode": "ES_STAGE_FAILED",
                "error": str(exc)[:500],
            }

    def commit_stage(self, stage: dict[str, Any]) -> dict[str, Any]:
        alias = str(stage.get("alias") or self.settings.es_index)
        physical_index = str(stage.get("physicalIndex") or "")
        if not physical_index:
            return {"success": False, "errorCode": "ES_STAGE_MISSING"}
        previous_state = stage.get("previousState") if isinstance(stage.get("previousState"), dict) else {}
        actions: list[dict[str, Any]] = []
        if previous_state.get("kind") == "concrete":
            actions.append({"remove_index": {"index": alias}})
        else:
            for index_name in previous_state.get("indices") or []:
                actions.append({"remove": {"index": str(index_name), "alias": alias}})
        actions.append({"add": {"index": physical_index, "alias": alias, "is_write_index": True}})
        try:
            response = requests.post(
                self._url("_aliases"),
                headers=self._headers(),
                auth=self._auth(),
                json={"actions": actions},
                timeout=20,
            )
            if response.status_code not in {200, 201}:
                response.raise_for_status()
            stage["committed"] = True
            return {
                "success": True,
                "mode": "es",
                "transactionMode": "physical_index_alias_swap",
                "alias": alias,
                "physicalIndex": physical_index,
            }
        except Exception as exc:
            reconciliation: dict[str, Any] = {}
            try:
                current = self._alias_state(alias)
                if current.get("kind") == "alias" and physical_index in (current.get("indices") or []):
                    stage["committed"] = True
                    reconciliation = self.rollback_stage(stage)
            except Exception as reconcile_exc:
                reconciliation = {
                    "success": False,
                    "errorCode": "ES_ALIAS_STATE_UNCERTAIN",
                    "error": str(reconcile_exc)[:500],
                }
            return {
                "success": False,
                "mode": "es",
                "transactionMode": "physical_index_alias_swap",
                "errorCode": (
                    "ES_ALIAS_SWAP_FAILED"
                    if reconciliation.get("success", True)
                    else "ES_ALIAS_SWAP_STATE_UNCERTAIN"
                ),
                "error": str(exc)[:500],
                **({"reconciliation": reconciliation} if reconciliation else {}),
            }

    def rollback_stage(self, stage: dict[str, Any]) -> dict[str, Any]:
        if not stage.get("committed"):
            return {"success": True, "mode": "es", "rolledBack": False}
        alias = str(stage.get("alias") or self.settings.es_index)
        physical_index = str(stage.get("physicalIndex") or "")
        previous_state = stage.get("previousState") if isinstance(stage.get("previousState"), dict) else {}
        try:
            current = self._alias_state(alias)
            actions: list[dict[str, Any]] = []
            if current.get("kind") == "alias" and physical_index in (current.get("indices") or []):
                actions.append({"remove": {"index": physical_index, "alias": alias}})
            if previous_state.get("kind") == "alias":
                alias_options = previous_state.get("aliasOptions") if isinstance(previous_state.get("aliasOptions"), dict) else {}
                for index_name in previous_state.get("indices") or []:
                    options = alias_options.get(str(index_name)) if isinstance(alias_options.get(str(index_name)), dict) else {}
                    actions.append({"add": {"index": str(index_name), "alias": alias, **options}})
            elif previous_state.get("kind") == "concrete":
                backup = str(stage.get("legacyBackupIndex") or "")
                if not backup:
                    return {"success": False, "mode": "es", "errorCode": "ES_LEGACY_BACKUP_MISSING"}
                actions.append({"add": {"index": backup, "alias": alias, "is_write_index": True}})
            if actions:
                response = requests.post(
                    self._url("_aliases"),
                    headers=self._headers(),
                    auth=self._auth(),
                    json={"actions": actions},
                    timeout=20,
                )
                if response.status_code not in {200, 201}:
                    response.raise_for_status()
            stage["committed"] = False
            return {"success": True, "mode": "es", "rolledBack": True, "alias": alias}
        except Exception as exc:
            return {
                "success": False,
                "mode": "es",
                "errorCode": "ES_ALIAS_ROLLBACK_FAILED",
                "error": str(exc)[:500],
            }

    def finalize_stage(self, stage: dict[str, Any]) -> dict[str, Any]:
        retained: list[str] = []
        removed: list[str] = []
        previous_state = stage.get("previousState") if isinstance(stage.get("previousState"), dict) else {}
        candidates = (
            [str(item) for item in previous_state.get("indices") or []]
            if previous_state.get("kind") == "alias"
            else []
        )
        backup = str(stage.get("legacyBackupIndex") or "")
        if backup:
            candidates.append(backup)
        prefix = "%s__v_" % str(stage.get("alias") or self.settings.es_index)
        for index_name in candidates:
            if (index_name == backup or index_name.startswith(prefix)) and not self._index_has_aliases(index_name):
                try:
                    self._delete_index(index_name)
                    removed.append(index_name)
                except Exception:
                    retained.append(index_name)
            else:
                retained.append(index_name)
        return {"success": not retained, "removed": removed, "retained": retained}

    def abort_stage(self, stage: dict[str, Any]) -> dict[str, Any]:
        if stage.get("committed"):
            rollback = self.rollback_stage(stage)
            if not rollback.get("success", False):
                return rollback
        removed: list[str] = []
        retained: list[str] = []
        for key in ("physicalIndex", "legacyBackupIndex"):
            index_name = str(stage.get(key) or "")
            if not index_name:
                continue
            if self._index_has_aliases(index_name):
                retained.append(index_name)
                continue
            self._delete_index(index_name)
            removed.append(index_name)
        return {
            "success": True,
            "mode": "es",
            "aborted": True,
            "removed": removed,
            "retainedActive": retained,
        }

    def _delete_index(self, index_name: str = "") -> None:
        target = index_name or self.settings.es_index
        response = requests.delete(self._url(target), headers=self._headers(), auth=self._auth(), timeout=20)
        if response.status_code in {200, 202, 404}:
            return
        response.raise_for_status()

    def _refresh_index(self, index_name: str = "") -> None:
        target = index_name or self.settings.es_index
        response = requests.post(self._url("%s/_refresh" % target), headers=self._headers(), auth=self._auth(), timeout=20)
        if response.status_code in {200, 201}:
            return
        response.raise_for_status()

    def _ensure_index(self, index_name: str = "") -> None:
        target = index_name or self.settings.es_index
        url = self._url(target)
        response = requests.head(url, headers=self._headers(), auth=self._auth(), timeout=10)
        if response.status_code == 200:
            self._ensure_vector_mapping(target)
            return
        if response.status_code not in {404, 400}:
            response.raise_for_status()
        mapping = self._index_mapping()
        put_response = requests.put(url, headers=self._headers(), auth=self._auth(), json=mapping, timeout=20)
        if put_response.status_code not in {200, 201}:
            put_response.raise_for_status()

    def _create_index(self, index_name: str, metadata: dict[str, Any]) -> None:
        response = requests.put(
            self._url(index_name),
            headers=self._headers(),
            auth=self._auth(),
            json=self._index_mapping(metadata),
            timeout=20,
        )
        if response.status_code not in {200, 201}:
            response.raise_for_status()

    def _index_mapping(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        mappings: dict[str, Any] = {
            "properties": {
                "doc_id": {"type": "keyword"},
                "title": {"type": "text"},
                "content": {"type": "text"},
                "source_type": {"type": "keyword"},
                "topic": {"type": "keyword"},
                "table": {"type": "keyword"},
                "semantic_ref_id": {"type": "keyword"},
                "semantic_path": {"type": "keyword"},
                "merchant_uri": {"type": "keyword"},
                "context_layer": {"type": "keyword"},
                "metadata": {"type": "object", "enabled": True},
            }
        }
        if metadata:
            mappings["_meta"] = dict(metadata)
        if self.settings.es_vector_enabled:
            mappings["properties"][self.settings.es_vector_field] = self._vector_mapping()
        return {"mappings": mappings}

    def _ensure_vector_mapping(self, index_name: str = "") -> None:
        if not self.settings.es_vector_enabled or not self.settings.es_vector_field:
            return
        target = index_name or self.settings.es_index
        try:
            response = requests.put(
                self._url("%s/_mapping" % target),
                headers=self._headers(),
                auth=self._auth(),
                json={"properties": {self.settings.es_vector_field: self._vector_mapping()}},
                timeout=20,
            )
            if response.status_code in {200, 201}:
                return
            if response.status_code >= 400:
                response.raise_for_status()
        except Exception:
            return

    def _vector_mapping(self) -> dict[str, Any]:
        return {
            "type": "dense_vector",
            "dims": max(1, int(self.settings.embedding_dims or 1536)),
            "index": True,
            "similarity": "cosine",
        }

    def _bulk_upsert(self, docs: list[RecallItem], index_name: str = "") -> dict[str, Any]:
        if not docs:
            return {"success": True, "count": 0}
        target = index_name or self.settings.es_index
        lines: list[str] = []
        for doc in docs:
            lines.append(json.dumps({"index": {"_index": target, "_id": doc.doc_id}}, ensure_ascii=False))
            lines.append(json.dumps(self._recall_item_to_es_doc(doc), ensure_ascii=False))
        payload = "\n".join(lines) + "\n"
        response = requests.post(
            self._url("_bulk"),
            headers={**self._headers(), "Content-Type": "application/x-ndjson"},
            auth=self._auth(),
            data=payload.encode("utf-8"),
            timeout=30,
        )
        if response.status_code >= 400:
            return {"success": False, "count": 0, "error": response.text[:500]}
        body = response.json()
        return {"success": not bool(body.get("errors")), "count": len(docs), "error": json.dumps(body, ensure_ascii=False)[:500] if body.get("errors") else ""}

    def _alias_state(self, alias: str) -> dict[str, Any]:
        response = requests.get(self._url("_alias/%s" % alias), headers=self._headers(), auth=self._auth(), timeout=10)
        if response.status_code == 200:
            payload = response.json() if response.content else {}
            indices = sorted(str(name) for name in payload if name)
            options: dict[str, Any] = {}
            for index_name in indices:
                aliases = ((payload.get(index_name) or {}).get("aliases") or {}) if isinstance(payload, dict) else {}
                value = aliases.get(alias) if isinstance(aliases, dict) else {}
                options[index_name] = value if isinstance(value, dict) else {}
            return {"kind": "alias", "indices": indices, "aliasOptions": options}
        if response.status_code != 404:
            response.raise_for_status()
        head = requests.head(self._url(alias), headers=self._headers(), auth=self._auth(), timeout=10)
        if head.status_code == 200:
            return {"kind": "concrete", "indices": [alias]}
        if head.status_code not in {400, 404}:
            head.raise_for_status()
        return {"kind": "absent", "indices": []}

    def _backup_concrete_index(self, index_name: str, version: str) -> str:
        backup = self._physical_index_name(index_name, "legacy_%s" % version)
        mapping_response = requests.get(
            self._url("%s/_mapping" % index_name),
            headers=self._headers(),
            auth=self._auth(),
            timeout=20,
        )
        mapping_response.raise_for_status()
        mapping_payload = mapping_response.json() if mapping_response.content else {}
        mappings = ((mapping_payload.get(index_name) or {}).get("mappings") or {}) if isinstance(mapping_payload, dict) else {}
        create_response = requests.put(
            self._url(backup),
            headers=self._headers(),
            auth=self._auth(),
            json={"mappings": mappings},
            timeout=20,
        )
        create_response.raise_for_status()
        reindex_response = requests.post(
            self._url("_reindex?refresh=true&wait_for_completion=true"),
            headers=self._headers(),
            auth=self._auth(),
            json={"source": {"index": index_name}, "dest": {"index": backup, "op_type": "create"}},
            timeout=120,
        )
        reindex_response.raise_for_status()
        source_count = self._index_count(index_name)
        backup_count = self._index_count(backup)
        if source_count != backup_count:
            raise RuntimeError("ES_LEGACY_BACKUP_COUNT_MISMATCH:%s!=%s" % (backup_count, source_count))
        return backup

    def _validate_staged_index(self, index_name: str, manifest: dict[str, Any]) -> dict[str, Any]:
        actual_count = self._index_count(index_name)
        expected_count = int(manifest.get("docCount") or 0)
        mapping_response = requests.get(
            self._url("%s/_mapping" % index_name),
            headers=self._headers(),
            auth=self._auth(),
            timeout=20,
        )
        mapping_response.raise_for_status()
        payload = mapping_response.json() if mapping_response.content else {}
        metadata = ((payload.get(index_name) or {}).get("mappings") or {}).get("_meta") or {}
        expected_hash = str(manifest.get("semanticSourceHash") or "")
        actual_hash = str(metadata.get("semanticSourceHash") or "") if isinstance(metadata, dict) else ""
        actual_version = str(metadata.get("indexVersion") or "") if isinstance(metadata, dict) else ""
        if actual_count != expected_count:
            return {
                "success": False,
                "errorCode": "ES_STAGE_DOC_COUNT_MISMATCH",
                "docCount": actual_count,
                "expectedDocCount": expected_count,
            }
        if actual_hash != expected_hash or actual_version != str(manifest.get("indexVersion") or ""):
            return {
                "success": False,
                "errorCode": "ES_STAGE_SOURCE_HASH_MISMATCH",
                "semanticSourceHash": actual_hash,
                "expectedSemanticSourceHash": expected_hash,
            }
        return {"success": True, "docCount": actual_count, "semanticSourceHash": actual_hash}

    def _index_count(self, index_name: str) -> int:
        response = requests.get(
            self._url("%s/_count" % index_name),
            headers=self._headers(),
            auth=self._auth(),
            timeout=20,
        )
        response.raise_for_status()
        return int((response.json() or {}).get("count") or 0)

    def _index_has_aliases(self, index_name: str) -> bool:
        response = requests.get(
            self._url("%s/_alias" % index_name),
            headers=self._headers(),
            auth=self._auth(),
            timeout=10,
        )
        if response.status_code == 404:
            return False
        response.raise_for_status()
        payload = response.json() if response.content else {}
        aliases = ((payload.get(index_name) or {}).get("aliases") or {}) if isinstance(payload, dict) else {}
        return bool(aliases)

    @staticmethod
    def _physical_index_name(alias: str, version: str) -> str:
        safe_alias = re.sub(r"[^a-z0-9._-]+", "_", alias.lower()).strip("_-.+") or "merchant_ai_recall"
        safe_version = re.sub(r"[^a-z0-9_-]+", "_", version.lower()).strip("_-") or "unknown"
        return "%s__v_%s_%s" % (safe_alias[:180], safe_version[:40], uuid.uuid4().hex[:10])

    def _recall_item_to_es_doc(self, item: RecallItem) -> dict[str, Any]:
        payload = recall_item_to_es_doc(item)
        vector = self._embed_recall_item(item)
        if vector:
            payload[self.settings.es_vector_field] = vector
            metadata = dict(payload.get("metadata") or {})
            metadata["embeddingModel"] = self.settings.embedding_model
            metadata["embeddingDims"] = len(vector)
            payload["metadata"] = metadata
        return payload

    def _embed_recall_item(self, item: RecallItem) -> list[float]:
        if not self._vector_enabled():
            return []
        text = "\n".join(part for part in [item.title, item.content] if str(part or "").strip()).strip()
        if not text:
            return []
        cache_key = stable_cache_key(
            "recall_index_embedding",
            {
                "baseUrl": self.settings.embedding_base_url,
                "model": self.settings.embedding_model,
                "dims": self.settings.embedding_dims,
                "docId": item.doc_id,
                "textHash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            },
        )
        cached = self._embedding_cache.get(cache_key)
        if isinstance(cached, list):
            return [float(value) for value in cached]
        payload: dict[str, Any] = {"model": self.settings.embedding_model, "input": text}
        if int(self.settings.embedding_dims or 0) > 0:
            payload["dimensions"] = int(self.settings.embedding_dims)
        response = requests.post(
            "%s/embeddings" % self.settings.embedding_base_url.rstrip("/"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": "Bearer %s" % self._embedding_api_key(),
            },
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        body = response.json() or {}
        vector = (((body.get("data") or [{}])[0] or {}).get("embedding") or [])
        result = [float(value) for value in vector if isinstance(value, (int, float))]
        if result:
            self._embedding_cache.set(cache_key, result)
        return result

    def _vector_enabled(self) -> bool:
        return bool(self.settings.es_vector_enabled and self.settings.es_vector_field and self.settings.embedding_model and self._embedding_api_key())

    def _embedding_api_key(self) -> str:
        return str(self.settings.embedding_api_key or self.settings.llm_api_key or "").strip()

    def _delete_refs(self, deleted_refs: list[str], index_name: str = "") -> dict[str, Any]:
        count = 0
        errors: list[str] = []
        target = index_name or self.settings.es_index
        for ref in deleted_refs:
            deleted_ref = ref.removeprefix("deleted:")
            semantic_path = deleted_ref if deleted_ref.startswith(("rules/", "topics/")) else "topics/%s" % deleted_ref
            query = {"query": {"term": {"semantic_path": semantic_path}}}
            response = requests.post(
                self._url("%s/_delete_by_query" % target),
                headers=self._headers(),
                auth=self._auth(),
                json=query,
                timeout=20,
            )
            if response.status_code >= 400:
                errors.append(response.text[:300])
                continue
            count += int((response.json() or {}).get("deleted") or 0)
        return {"success": not errors, "count": count, "error": "; ".join(errors)}

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.settings.es_api_key:
            headers["Authorization"] = "Bearer %s" % self.settings.es_api_key
        return headers

    def _auth(self) -> tuple[str, str] | None:
        if self.settings.es_api_key:
            return None
        if self.settings.es_username:
            return (self.settings.es_username, self.settings.es_password)
        return None

    def _url(self, path: str) -> str:
        return "%s/%s" % (self.settings.es_base_url.rstrip("/"), path.lstrip("/"))


class RecallIndexManager:
    def __init__(
        self,
        settings: Settings,
        document_provider: RecallDocumentProvider,
        cache_clearers: Iterable[Callable[[], None]] | None = None,
        es_adapter: EsRecallIndexAdapter | None = None,
    ):
        self.settings = settings
        self.document_provider = document_provider
        self.cache_clearers = list(cache_clearers or [])
        self.es_adapter = es_adapter or EsRecallIndexAdapter(settings)
        self._transaction_lock = threading.RLock()

    @property
    def manifest_path(self) -> Path:
        return self.settings.resolved_workspace_path / "recall_index_manifest.json"

    @property
    def transaction_root(self) -> Path:
        return self.settings.resolved_workspace_path / "recall_index_transactions"

    @contextmanager
    def publication_transaction(self) -> Iterable[None]:
        """Serialize publish/index transitions across workers sharing the workspace."""
        with self._transaction_lock:
            lock_path = self.transaction_root / "publish.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a+", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def rebuild(
        self,
        changed_only: bool = True,
        topic: str = "",
        table_name: str = "",
        documents: list[RecallItem] | None = None,
    ) -> dict[str, Any]:
        with self.publication_transaction():
            preparation = self.prepare_rebuild(
                changed_only=changed_only,
                topic=topic,
                table_name=table_name,
                documents=documents,
            )
            if not preparation.success:
                return self.preparation_result(preparation)
            return self.commit_prepared(preparation)

    def prepare_rebuild(
        self,
        changed_only: bool = True,
        topic: str = "",
        table_name: str = "",
        documents: list[RecallItem] | None = None,
    ) -> RecallIndexPreparation:
        with self._transaction_lock:
            previous = load_index_manifest(self.manifest_path)
            self.clear_caches()
            # The candidate manifest and every physical ES generation are always
            # complete. Scope controls only change reporting and legacy adapters.
            manifest_docs = list(documents) if documents is not None else self.scoped_documents()
            manifest = build_index_manifest(
                recall_documents_for_manifest(manifest_docs),
                previous,
                changed_only=changed_only,
                root=str(self.settings.resolved_topic_path),
            )
            updated_refs = filter_changed_refs_for_scope(manifest.get("updatedRefs") or [], topic, table_name)
            replace_all = not changed_only and not topic and not table_name
            transaction_id = uuid.uuid4().hex
            transaction_path = self.transaction_root / "transactions" / (transaction_id + ".json")
            staged_manifest_path = self.transaction_root / "staged" / (transaction_id + ".manifest.json")
            preparation = RecallIndexPreparation(
                success=False,
                previous_manifest=previous,
                candidate_manifest=manifest,
                updated_refs=updated_refs,
                changed_only=changed_only,
                topic=topic,
                table_name=table_name,
                replace_all=replace_all,
                external_documents=documents is not None,
                transaction_id=transaction_id,
                transaction_path=transaction_path,
                staged_manifest_path=staged_manifest_path,
            )
            self._record_transaction(preparation, "PREPARING")
            self._atomic_write_json(staged_manifest_path, manifest)

            if not self.settings.es_enabled:
                preparation.success = True
                preparation.es_result = {"success": True, "mode": "disabled", "enabled": False}
                self._record_transaction(preparation, "LOCAL_STAGED")
                return preparation

            try:
                stage_method = getattr(self.es_adapter, "stage", None)
                if callable(stage_method):
                    es_stage = stage_method(manifest_docs, manifest)
                    preparation.es_stage = es_stage if isinstance(es_stage, dict) else {}
                    preparation.es_result = dict(preparation.es_stage)
                else:
                    scoped_docs = self._documents_for_legacy_sync(
                        manifest_docs,
                        updated_refs,
                        topic,
                        table_name,
                        replace_all,
                    )
                    deleted_refs = [ref for ref in updated_refs if str(ref).startswith("deleted:")]
                    sync_result = self.es_adapter.sync(scoped_docs, deleted_refs, replace_all=replace_all)
                    preparation.es_stage = {
                        **(sync_result if isinstance(sync_result, dict) else {}),
                        "transactionMode": "legacy_adapter",
                    }
                    preparation.es_result = dict(preparation.es_stage)
            except Exception as exc:
                preparation.es_result = {
                    "success": False,
                    "mode": "es",
                    "errorCode": "ES_STAGE_FAILED",
                    "error": str(exc)[:500],
                }

            if not preparation.es_result.get("success", False):
                preparation.error_code = str(preparation.es_result.get("errorCode") or "ES_STAGE_FAILED")
                preparation.error = str(preparation.es_result.get("error") or "")[:500]
                self._safe_unlink(staged_manifest_path)
                self._write_retry_record(preparation, "PENDING_RETRY")
                self._record_transaction(preparation, "PENDING_RETRY")
                return preparation

            preparation.success = True
            self._record_transaction(preparation, "ES_STAGED")
            return preparation

    def commit_prepared(self, preparation: RecallIndexPreparation) -> dict[str, Any]:
        with self._transaction_lock:
            if not preparation.success:
                return self.preparation_result(preparation)
            self._record_transaction(preparation, "COMMITTING")
            commit_result: dict[str, Any] = {"success": True, "mode": "disabled"}
            transactional_es = bool(
                self.settings.es_enabled
                and str(preparation.es_stage.get("transactionMode") or "") == "physical_index_alias_swap"
            )
            if transactional_es:
                try:
                    commit_result = self.es_adapter.commit_stage(preparation.es_stage)
                except Exception as exc:
                    commit_result = {
                        "success": False,
                        "mode": "es",
                        "errorCode": "ES_ALIAS_SWAP_FAILED",
                        "error": str(exc)[:500],
                    }
                if not commit_result.get("success", False):
                    preparation.success = False
                    preparation.error_code = str(commit_result.get("errorCode") or "ES_ALIAS_SWAP_FAILED")
                    preparation.error = str(commit_result.get("error") or "")[:500]
                    preparation.es_result = {**preparation.es_result, "commit": commit_result}
                    self._abort_es_stage(preparation)
                    self._safe_unlink(preparation.staged_manifest_path)
                    self._write_retry_record(preparation, "PENDING_RETRY")
                    self._record_transaction(preparation, "PENDING_RETRY")
                    return self.preparation_result(preparation)
                preparation.es_result = {**preparation.es_result, "commit": commit_result}
                self._record_transaction(preparation, "ES_ALIAS_ACTIVE_MANIFEST_PENDING")

            try:
                if preparation.staged_manifest_path is None:
                    raise RuntimeError("STAGED_MANIFEST_MISSING")
                self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(preparation.staged_manifest_path, self.manifest_path)
            except Exception as exc:
                rollback_result: dict[str, Any] = {"success": True, "rolledBack": False}
                if transactional_es:
                    try:
                        rollback_result = self.es_adapter.rollback_stage(preparation.es_stage)
                    except Exception as rollback_exc:
                        rollback_result = {
                            "success": False,
                            "errorCode": "ES_ALIAS_ROLLBACK_FAILED",
                            "error": str(rollback_exc)[:500],
                        }
                preparation.success = False
                preparation.error_code = (
                    "ACTIVE_MANIFEST_COMMIT_FAILED"
                    if rollback_result.get("success", False)
                    else "ACTIVE_MANIFEST_COMMIT_AND_ES_ROLLBACK_FAILED"
                )
                preparation.error = str(exc)[:500]
                preparation.es_result = {
                    **preparation.es_result,
                    "commit": commit_result,
                    "rollback": rollback_result,
                }
                if rollback_result.get("success", False):
                    self._abort_es_stage(preparation)
                self._safe_unlink(preparation.staged_manifest_path)
                self._write_retry_record(preparation, "PENDING_RETRY")
                self._record_transaction(preparation, "PENDING_RETRY")
                return self.preparation_result(preparation)

            cleanup_result: dict[str, Any] = {"success": True, "removed": [], "retained": []}
            if transactional_es:
                try:
                    cleanup_result = self.es_adapter.finalize_stage(preparation.es_stage)
                except Exception as exc:
                    cleanup_result = {"success": False, "error": str(exc)[:500], "retained": []}
            preparation.es_result = {
                **preparation.es_result,
                **({"commit": commit_result} if transactional_es else {}),
                **({"cleanup": cleanup_result} if transactional_es else {}),
            }
            preparation.success = True
            self._clear_retry_record(preparation)
            self._record_transaction(preparation, "ACTIVE")
            if preparation.external_documents:
                self.clear_caches()
            return self.preparation_result(preparation, active=True)

    def abort_prepared(self, preparation: RecallIndexPreparation, reason: str = "PUBLISH_ABORTED") -> dict[str, Any]:
        with self._transaction_lock:
            preparation.success = False
            preparation.error_code = reason
            self._abort_es_stage(preparation)
            self._safe_unlink(preparation.staged_manifest_path)
            self._write_retry_record(preparation, "PENDING_RETRY")
            self._record_transaction(preparation, "PENDING_RETRY")
            return self.preparation_result(preparation)

    def preparation_result(self, preparation: RecallIndexPreparation, active: bool = False) -> dict[str, Any]:
        active_manifest = preparation.candidate_manifest if active else preparation.previous_manifest
        status = "ACTIVE" if active else "PENDING_RETRY"
        result = {
            "success": bool(active),
            "status": status,
            "mode": "es" if self.settings.es_enabled else "local_recall",
            "indexVersion": active_manifest.get("indexVersion", ""),
            "semanticSourceHash": active_manifest.get("semanticSourceHash", ""),
            "docCount": active_manifest.get("docCount", 0),
            "candidateIndexVersion": preparation.candidate_manifest.get("indexVersion", ""),
            "candidateSemanticSourceHash": preparation.candidate_manifest.get("semanticSourceHash", ""),
            "candidateDocCount": preparation.candidate_manifest.get("docCount", 0),
            "activeManifestAdvanced": bool(active),
            "updatedRefs": preparation.updated_refs,
            "updatedRefCount": len(preparation.updated_refs),
            "cacheInvalidated": True,
            "rebuildMode": "full" if preparation.replace_all else "scoped_incremental",
            "rebuildScope": {"topic": preparation.topic or "", "table": preparation.table_name or ""},
            "transactionId": preparation.transaction_id,
            "transactionState": status,
            "transactionPath": str(preparation.transaction_path),
            "es": preparation.es_result,
            "manifestPath": str(self.manifest_path),
        }
        if not active:
            result["errorCode"] = preparation.error_code or "INDEX_PREPARE_FAILED"
            if preparation.error:
                result["error"] = preparation.error
            result["retryRecord"] = str(self._retry_record_path(preparation))
        return result

    def _documents_for_legacy_sync(
        self,
        docs: list[RecallItem],
        updated_refs: list[str],
        topic: str,
        table_name: str,
        replace_all: bool,
    ) -> list[RecallItem]:
        if replace_all:
            return list(docs)
        if topic or table_name:
            return [doc for doc in docs if recall_doc_in_scope(doc, topic, table_name)]
        changed_paths = {
            str(ref) if str(ref).startswith(("rules/", "topics/")) else "topics/%s" % str(ref)
            for ref in updated_refs
            if not str(ref).startswith("deleted:")
        }
        changed_scopes = recall_scopes_for_changed_refs(updated_refs)
        return [
            doc
            for doc in docs
            if str((doc.metadata or {}).get("semanticPath") or "") in changed_paths
            or str((doc.metadata or {}).get("semanticRefId") or doc.doc_id or "") in updated_refs
            or recall_doc_scope_key(doc) in changed_scopes
        ]

    def _abort_es_stage(self, preparation: RecallIndexPreparation) -> dict[str, Any]:
        if not self.settings.es_enabled:
            return {"success": True, "mode": "disabled"}
        if str(preparation.es_stage.get("transactionMode") or "") != "physical_index_alias_swap":
            return {"success": True, "mode": "legacy_adapter", "aborted": False}
        try:
            return self.es_adapter.abort_stage(preparation.es_stage)
        except Exception as exc:
            return {"success": False, "errorCode": "ES_STAGE_ABORT_FAILED", "error": str(exc)[:500]}

    def _record_transaction(self, preparation: RecallIndexPreparation, state: str) -> None:
        payload = {
            "transactionId": preparation.transaction_id,
            "state": state,
            "topic": preparation.topic,
            "tableName": preparation.table_name,
            "changedOnly": preparation.changed_only,
            "activeIndexVersion": preparation.previous_manifest.get("indexVersion", ""),
            "activeSemanticSourceHash": preparation.previous_manifest.get("semanticSourceHash", ""),
            "candidateIndexVersion": preparation.candidate_manifest.get("indexVersion", ""),
            "candidateSemanticSourceHash": preparation.candidate_manifest.get("semanticSourceHash", ""),
            "candidateDocCount": preparation.candidate_manifest.get("docCount", 0),
            "errorCode": preparation.error_code,
            "error": preparation.error,
            "esStage": preparation.es_stage,
            "updatedAt": datetime.now(timezone.utc).isoformat(),
        }
        self._atomic_write_json(preparation.transaction_path, payload)

    def _write_retry_record(self, preparation: RecallIndexPreparation, state: str) -> None:
        payload = {
            "state": state,
            "retryable": True,
            "transactionId": preparation.transaction_id,
            "topic": preparation.topic,
            "tableName": preparation.table_name,
            "changedOnly": preparation.changed_only,
            "activeManifest": {
                "indexVersion": preparation.previous_manifest.get("indexVersion", ""),
                "semanticSourceHash": preparation.previous_manifest.get("semanticSourceHash", ""),
                "docCount": preparation.previous_manifest.get("docCount", 0),
            },
            "candidateManifest": preparation.candidate_manifest,
            "errorCode": preparation.error_code,
            "error": preparation.error,
            "esStage": preparation.es_stage,
            "updatedAt": datetime.now(timezone.utc).isoformat(),
        }
        self._atomic_write_json(self._retry_record_path(preparation), payload)

    def _clear_retry_record(self, preparation: RecallIndexPreparation) -> None:
        self._safe_unlink(self._retry_record_path(preparation))

    def _retry_record_path(self, preparation: RecallIndexPreparation) -> Path:
        scope = "%s\n%s\n%s" % (
            preparation.topic,
            preparation.table_name,
            preparation.candidate_manifest.get("semanticSourceHash", ""),
        )
        key = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:24]
        return self.transaction_root / "pending" / (key + ".json")

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(".%s.%s.tmp" % (path.name, uuid.uuid4().hex))
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    @staticmethod
    def _safe_unlink(path: Path | None) -> None:
        if path is None:
            return
        try:
            path.unlink(missing_ok=True)
        except Exception:
            return

    def clear_caches(self) -> None:
        self.document_provider.clear_cache()
        for clearer in self.cache_clearers:
            clearer()

    def changed_documents(self, updated_refs: list[str], topic: str = "", table_name: str = "") -> list[RecallItem]:
        docs = self.scoped_documents(topic, table_name)
        if topic or table_name:
            return docs
        changed_paths = {
            str(ref) if str(ref).startswith(("rules/", "topics/")) else "topics/%s" % str(ref)
            for ref in updated_refs
            if not str(ref).startswith("deleted:")
        }
        if not changed_paths:
            return []
        changed_scopes = recall_scopes_for_changed_refs(updated_refs)
        selected = [
            doc
            for doc in docs
            if str((doc.metadata or {}).get("semanticPath") or "") in changed_paths
            or str((doc.metadata or {}).get("semanticRefId") or doc.doc_id or "") in updated_refs
            or recall_doc_scope_key(doc) in changed_scopes
        ]
        return selected

    def scoped_documents(self, topic: str = "", table_name: str = "") -> list[RecallItem]:
        docs = self.document_provider._load_documents()
        if topic or table_name:
            return [doc for doc in docs if recall_doc_in_scope(doc, topic, table_name)]
        return docs


def recall_item_to_es_doc(item: RecallItem) -> dict[str, Any]:
    metadata = item.metadata or {}
    return {
        "doc_id": item.doc_id,
        "title": item.title,
        "content": item.content,
        "source_type": item.source_type,
        "topic": item.topic,
        "table": item.table,
        "answer_mode": item.answer_mode,
        "semantic_ref_id": str(metadata.get("semanticRefId") or item.doc_id or ""),
        "semantic_path": str(metadata.get("semanticPath") or ""),
        "merchant_uri": str(metadata.get("merchantUri") or ""),
        "context_layer": str(metadata.get("contextLayer") or ""),
        "metadata": metadata,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def recall_doc_in_scope(item: RecallItem, topic: str, table_name: str) -> bool:
    if topic and item.topic != topic:
        return False
    if table_name and item.table != table_name:
        return False
    return bool(topic or table_name)


def recall_doc_scope_key(item: RecallItem) -> str:
    if item.topic and item.table:
        return "%s/tables/%s" % (item.topic, item.table)
    if item.topic and item.source_type == "SEMANTIC_RELATIONSHIP":
        return "%s/relationships" % item.topic
    return ""


def recall_scopes_for_changed_refs(refs: list[str]) -> set[str]:
    scopes: set[str] = set()
    for ref in refs:
        raw = str(ref).removeprefix("deleted:").removeprefix("topics/")
        raw = raw.split("#", 1)[0]
        parts = raw.split("/")
        if len(parts) >= 3 and parts[1] == "tables":
            scopes.add("%s/tables/%s" % (parts[0], parts[2]))
        elif len(parts) >= 2 and parts[1] == "relationships.json":
            scopes.add("%s/relationships" % parts[0])
    return scopes


def filter_changed_refs_for_scope(refs: list[str], topic: str = "", table_name: str = "") -> list[str]:
    if not topic and not table_name:
        return list(refs)
    prefix = "%s/tables/%s/" % (topic, table_name) if topic and table_name else topic
    scoped: list[str] = []
    for ref in refs:
        raw = str(ref)
        comparable = raw.removeprefix("deleted:").removeprefix("topics/")
        if comparable.startswith(prefix):
            scoped.append(raw)
    return scoped
