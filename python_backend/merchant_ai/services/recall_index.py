from __future__ import annotations

import hashlib
import json
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


def semantic_docs(root: Path, ref_prefix: str = "") -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    if not root.exists():
        return docs
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in {".json", ".md"}:
            continue
        relative = path.relative_to(root).as_posix()
        if ref_prefix:
            relative = "%s/%s" % (ref_prefix.strip("/"), relative)
        docs.append({"ref": relative, "path": str(path), "hash": hash_file(path)})
    return docs


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


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
    root: Path,
    previous: dict[str, Any],
    changed_only: bool,
    additional_roots: Iterable[tuple[str, Path]] | None = None,
) -> dict[str, Any]:
    docs = semantic_docs(root)
    for prefix, additional_root in additional_roots or []:
        docs.extend(semantic_docs(additional_root, ref_prefix=prefix))
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
        "docs": [{"ref": doc["ref"], "hash": doc["hash"]} for doc in docs],
    }


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

    def _delete_index(self) -> None:
        response = requests.delete(self._url(self.settings.es_index), headers=self._headers(), auth=self._auth(), timeout=20)
        if response.status_code in {200, 202, 404}:
            return
        response.raise_for_status()

    def _refresh_index(self) -> None:
        response = requests.post(self._url("%s/_refresh" % self.settings.es_index), headers=self._headers(), auth=self._auth(), timeout=20)
        if response.status_code in {200, 201}:
            return
        response.raise_for_status()

    def _ensure_index(self) -> None:
        url = self._url(self.settings.es_index)
        response = requests.head(url, headers=self._headers(), auth=self._auth(), timeout=10)
        if response.status_code == 200:
            self._ensure_vector_mapping()
            return
        if response.status_code not in {404, 400}:
            response.raise_for_status()
        mapping = {
            "mappings": {
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
        }
        if self.settings.es_vector_enabled:
            mapping["mappings"]["properties"][self.settings.es_vector_field] = self._vector_mapping()
        put_response = requests.put(url, headers=self._headers(), auth=self._auth(), json=mapping, timeout=20)
        if put_response.status_code not in {200, 201}:
            put_response.raise_for_status()

    def _ensure_vector_mapping(self) -> None:
        if not self.settings.es_vector_enabled or not self.settings.es_vector_field:
            return
        try:
            response = requests.put(
                self._url("%s/_mapping" % self.settings.es_index),
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

    def _bulk_upsert(self, docs: list[RecallItem]) -> dict[str, Any]:
        if not docs:
            return {"success": True, "count": 0}
        lines: list[str] = []
        for doc in docs:
            lines.append(json.dumps({"index": {"_index": self.settings.es_index, "_id": doc.doc_id}}, ensure_ascii=False))
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

    def _delete_refs(self, deleted_refs: list[str]) -> dict[str, Any]:
        count = 0
        errors: list[str] = []
        for ref in deleted_refs:
            deleted_ref = ref.removeprefix("deleted:")
            semantic_path = deleted_ref if deleted_ref.startswith("rules/") else "topics/%s" % deleted_ref
            query = {"query": {"term": {"semantic_path": semantic_path}}}
            response = requests.post(
                self._url("%s/_delete_by_query" % self.settings.es_index),
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

    @property
    def manifest_path(self) -> Path:
        return self.settings.resolved_workspace_path / "recall_index_manifest.json"

    def rebuild(self, changed_only: bool = True, topic: str = "", table_name: str = "") -> dict[str, Any]:
        previous = load_index_manifest(self.manifest_path)
        manifest = build_index_manifest(
            self.settings.resolved_topic_path,
            previous,
            changed_only=changed_only,
            additional_roots=[("rules", self.settings.resolved_rule_knowledge_path)],
        )
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        updated_refs = filter_changed_refs_for_scope(manifest.get("updatedRefs") or [], topic, table_name)
        self.clear_caches()
        es_result = {"success": True, "mode": "disabled", "enabled": False}
        replace_all = not changed_only and not topic and not table_name
        if self.settings.es_enabled:
            docs = self.scoped_documents(topic, table_name) if replace_all else self.changed_documents(updated_refs, topic, table_name)
            deleted_refs = [ref for ref in updated_refs if str(ref).startswith("deleted:")]
            try:
                es_result = self.es_adapter.sync(docs, deleted_refs, replace_all=replace_all)
            except Exception as exc:
                es_result = {"success": False, "mode": "es", "errorCode": "ES_SYNC_FAILED", "error": str(exc)[:500]}
        return {
            "success": bool(es_result.get("success", True)),
            "mode": "es" if self.settings.es_enabled else "local_recall",
            "indexVersion": manifest.get("indexVersion", ""),
            "semanticSourceHash": manifest.get("semanticSourceHash", ""),
            "docCount": manifest.get("docCount", 0),
            "updatedRefs": updated_refs,
            "updatedRefCount": len(updated_refs),
            "cacheInvalidated": True,
            "rebuildMode": "full" if replace_all else "scoped_incremental",
            "rebuildScope": {"topic": topic or "", "table": table_name or ""},
            "es": es_result,
            "manifestPath": str(self.manifest_path),
        }

    def clear_caches(self) -> None:
        self.document_provider.clear_cache()
        for clearer in self.cache_clearers:
            clearer()

    def changed_documents(self, updated_refs: list[str], topic: str = "", table_name: str = "") -> list[RecallItem]:
        docs = self.scoped_documents(topic, table_name)
        if topic or table_name:
            return docs
        changed_paths = {
            str(ref) if str(ref).startswith("rules/") else "topics/%s" % str(ref)
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
        raw = str(ref).removeprefix("deleted:")
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
        comparable = raw.removeprefix("deleted:")
        if comparable.startswith(prefix):
            scoped.append(raw)
    return scoped
