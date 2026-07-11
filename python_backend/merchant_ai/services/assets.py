from __future__ import annotations

import json
import hashlib
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from merchant_ai.config import Settings
from merchant_ai.models import (
    PlanningAssetEntry,
    PlanningAssetPack,
    QuestionCategory,
    RecallBundle,
    RecallItem,
    RelationshipEntry,
    SchemaDriftReport,
    SemanticCatalogVersion,
    SkillManifest,
    TOPIC_TO_CATEGORY,
    TopicBuildRequest,
    category_display,
)
from merchant_ai.services.cache import build_ttl_cache, stable_cache_key
from merchant_ai.services.context_filesystem import add_context_uri, merchant_uri_for_semantic_ref
from merchant_ai.services.llm import LlmClient
from merchant_ai.services.repositories import DorisRepository, write_json
from merchant_ai.services.tools import AgentToolDefinition


class TopicAssetService:
    MANAGED_TABLE_FILENAMES = {
        "asset.json",
        "schema.json",
        "sample_rows.json",
        "sample_profile.json",
        "semantic_columns.json",
        "metrics.json",
        "terms.json",
        "knowledge_rules.json",
        "asset_production_report.json",
    }
    GOVERNANCE_FILENAMES = {
        "semantic_version.json",
        "semantic_publish_history.json",
    }

    def __init__(self, settings: Settings):
        self.settings = settings
        self._table_asset_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._manifest_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._relationship_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._topic_names_cache: Optional[List[str]] = None

    @property
    def root(self) -> Path:
        return self.settings.resolved_topic_path

    def list_topic(self, topic: str) -> Dict[str, Any]:
        topic_dir = self.root / topic
        if not topic_dir.exists():
            return {"success": False, "topic": topic, "items": []}
        files = []
        for path in sorted(topic_dir.rglob("*")):
            if path.is_file():
                files.append(str(path.relative_to(topic_dir)))
        return {"success": True, "topic": topic, "items": files}

    def publish(self, topic: str, table_name: str, approved: bool, reviewer: str, review_note: str) -> Dict[str, Any]:
        pending = self.root / topic / "pending" / table_name
        target = self.root / topic / "tables" / table_name
        if not pending.exists():
            return {"success": False, "status": "NOT_FOUND", "topic": topic, "tableName": table_name}
        review_payload = {"approved": approved, "reviewer": reviewer, "reviewNote": review_note}
        write_json(pending / "review-result.json", review_payload)
        if approved:
            target.mkdir(parents=True, exist_ok=True)
            changed_files: List[str] = []
            unchanged_files: List[str] = []
            deleted_files: List[str] = []
            for name in sorted(self.MANAGED_TABLE_FILENAMES):
                source_path = pending / name
                target_path = target / name
                if source_path.exists() and source_path.is_file():
                    source_bytes = source_path.read_bytes()
                    if target_path.exists() and target_path.is_file() and target_path.read_bytes() == source_bytes:
                        unchanged_files.append(name)
                        continue
                    target_path.write_bytes(source_bytes)
                    changed_files.append(name)
                    continue
                if target_path.exists() and target_path.is_file() and name not in self.GOVERNANCE_FILENAMES:
                    target_path.unlink()
                    deleted_files.append(name)
            self._table_asset_cache.pop((topic, table_name), None)
            self._manifest_cache.pop(topic, None)
            self._relationship_cache.pop(topic, None)
            self._topic_names_cache = None
            return {
                "success": True,
                "status": "PUBLISHED",
                "topic": topic,
                "tableName": table_name,
                "publishMode": "scoped_incremental",
                "publishScope": {
                    "topic": topic,
                    "table": table_name,
                    "managedFiles": sorted(self.MANAGED_TABLE_FILENAMES),
                },
                "changedFiles": changed_files,
                "unchangedFiles": unchanged_files,
                "deletedFiles": deleted_files,
            }
        return {"success": True, "status": "REJECTED", "topic": topic, "tableName": table_name}

    def load_manifest(self, topic: str) -> List[Dict[str, Any]]:
        if topic in self._manifest_cache:
            return self._manifest_cache[topic]
        path = self.root / topic / "manifest.json"
        data = read_json(path)
        manifest = data if isinstance(data, list) else []
        self._manifest_cache[topic] = manifest
        return manifest

    def topic_names_for_categories(self, categories: Iterable[QuestionCategory]) -> List[str]:
        display_to_topic = {category_display(category): category for category in QuestionCategory}
        names: List[str] = []
        wanted = set(categories)
        for path in sorted(self.root.glob("*/manifest.json")):
            name = path.parent.name
            category = TOPIC_TO_CATEGORY.get(name) or display_to_topic.get(name)
            if category in wanted and name not in names:
                names.append(name)
        return names

    def all_topic_names(self) -> List[str]:
        if self._topic_names_cache is not None:
            return self._topic_names_cache
        if not self.root.exists():
            return []
        self._topic_names_cache = [path.name for path in sorted(self.root.iterdir()) if path.is_dir()]
        return self._topic_names_cache

    def table_asset_dir(self, topic: str, table: str) -> Path:
        return self.root / topic / "tables" / table

    def load_table_asset(self, topic: str, table: str) -> Dict[str, Any]:
        cache_key = (topic, table)
        if cache_key in self._table_asset_cache:
            return self._table_asset_cache[cache_key]
        table_dir = self.table_asset_dir(topic, table)
        asset = read_json(table_dir / "asset.json")
        sidecar_fields = {
            "schemaColumns": "schema.json",
            "semanticColumns": "semantic_columns.json",
            "metrics": "metrics.json",
            "terms": "terms.json",
            "knowledgeRules": "knowledge_rules.json",
        }
        if not isinstance(asset, dict) or not asset:
            manifest_item = next((item for item in self.load_manifest(topic) if str(item.get("tableName") or "") == table), {})
            asset = {
                **manifest_item,
                "topic": topic,
                "tableName": table,
            }
        else:
            asset = {**asset}
            asset.setdefault("topic", topic)
            asset.setdefault("tableName", table)
        for field, file_name in sidecar_fields.items():
            sidecar = read_json(table_dir / file_name)
            if sidecar:
                asset[field] = merge_semantic_layer_list(asset.get(field), sidecar, field)
            else:
                asset.setdefault(field, [])
        self._table_asset_cache[cache_key] = asset
        return asset

    def load_table_schema(self, topic: str, table: str) -> List[Dict[str, Any]]:
        data = self.load_table_asset(topic, table).get("schemaColumns")
        return data if isinstance(data, list) else []

    def load_table_semantic_columns(self, topic: str, table: str) -> List[Dict[str, Any]]:
        data = self.load_table_asset(topic, table).get("semanticColumns")
        return data if isinstance(data, list) else []

    def load_table_metrics(self, topic: str, table: str) -> List[Dict[str, Any]]:
        data = self.load_table_asset(topic, table).get("metrics")
        return data if isinstance(data, list) else []

    def load_table_terms(self, topic: str, table: str) -> List[Dict[str, Any]]:
        data = self.load_table_asset(topic, table).get("terms")
        return data if isinstance(data, list) else []

    def load_table_knowledge_rules(self, topic: str, table: str) -> List[Dict[str, Any]]:
        data = self.load_table_asset(topic, table).get("knowledgeRules")
        return data if isinstance(data, list) else []

    def load_relationships(self, topic: str) -> List[Dict[str, Any]]:
        if topic in self._relationship_cache:
            return self._relationship_cache[topic]
        data = read_json(self.root / topic / "relationships.json")
        relationships = data if isinstance(data, list) else []
        self._relationship_cache[topic] = relationships
        return relationships


class SemanticCatalogService:
    """FileSystem-as-Context facade over the runtime semantic layer.

    The factual source stays in table-level asset.json and topic relationships.json.
    Callers should list refs first, then read/grep only the files needed for the
    current planning step.
    """

    MANIFEST_KIND = "TOPIC_MANIFEST"
    TABLE_KIND = "TABLE_ASSET"
    RELATIONSHIP_KIND = "RELATIONSHIPS"
    OFFLOAD_THRESHOLD_CHARS = 20_000

    def __init__(self, topic_assets: TopicAssetService):
        self.topic_assets = topic_assets
        self._refs_cache: Dict[Tuple[str, Tuple[str, ...]], List[Dict[str, Any]]] = {}

    def clear_cache(self) -> None:
        self._refs_cache.clear()

    def ls(
        self,
        topic_categories: Iterable[QuestionCategory] | None = None,
        topic: str = "",
        query: str = "",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        topics = self._topics(topic_categories, topic)
        refs: List[Dict[str, Any]] = []
        terms = question_match_terms(query) if query else []
        for topic_name in topics:
            manifest_ref = self.manifest_ref(topic_name)
            if not terms or score_document(terms, manifest_ref["searchText"]) > 0:
                refs.append(manifest_ref)
            for manifest_item in self.topic_assets.load_manifest(topic_name):
                table = str(manifest_item.get("tableName") or "")
                if not table:
                    continue
                asset = self.topic_assets.load_table_asset(topic_name, table)
                ref = self.table_ref(topic_name, table, asset)
                if not terms or score_document(terms, ref["searchText"]) > 0:
                    refs.append(ref)
            relationships = self.topic_assets.load_relationships(topic_name)
            if relationships:
                ref = self.relationship_ref(topic_name, relationships)
                if not terms or score_document(terms, ref["searchText"]) > 0:
                    refs.append(ref)
        refs.sort(key=lambda item: score_document(terms, item["searchText"]) if terms else 0.0, reverse=True)
        return [self._public_ref(item) for item in refs[: max(1, limit)]]

    def read(self, ref_id: str = "", path: str = "", max_chars: int = 20_000, offset: int = 0) -> Dict[str, Any]:
        ref = self._resolve_ref(ref_id, path)
        if not ref:
            return {"success": False, "error": "SEMANTIC_REF_NOT_FOUND", "refId": ref_id, "path": path}
        content = ref["content"]
        start = max(0, offset)
        end = min(len(content), start + max(1, max_chars))
        return add_context_uri({
            "success": True,
            "refId": ref["refId"],
            "path": ref["path"],
            "kind": ref["kind"],
            "topic": ref["topic"],
            "table": ref.get("table", ""),
            "content": content[start:end],
            "contentOffsetChars": start,
            "nextContentOffsetChars": end if end < len(content) else None,
            "truncated": end < len(content),
            "estimatedChars": len(content),
        }, ref_id=ref["refId"], topic=ref["topic"], table=ref.get("table", ""), kind=ref["kind"], path=ref["path"])

    def grep(
        self,
        query: str,
        topic_categories: Iterable[QuestionCategory] | None = None,
        topic: str = "",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        terms = question_match_terms(query)
        if not terms:
            return []
        hits: List[Dict[str, Any]] = []
        for ref in self._all_refs(topic_categories, topic):
            score = score_document(terms, ref["searchText"] + "\n" + ref["content"])
            if score <= 0:
                continue
            hits.append(
                {
                    **self._public_ref(ref),
                    "score": score,
                    "snippets": grep_snippets(ref["content"], terms, 3),
                }
            )
        hits.sort(key=lambda item: item["score"], reverse=True)
        return hits[: max(1, limit)]

    def write_proposal(self, topic: str, table: str, file_name: str, content: str) -> Dict[str, Any]:
        safe_name = sanitize_semantic_file_name(file_name or "proposal.md")
        target = self.topic_assets.settings.resolved_workspace_path / "semantic_proposals" / (topic or "unknown")
        if table:
            target = target / table
        target.mkdir(parents=True, exist_ok=True)
        path = target / safe_name
        path.write_text(str(content or ""), encoding="utf-8")
        return {
            "success": True,
            "path": str(path),
            "mode": "proposal_only",
            "note": "canonical semantic assets are not overwritten; publish/review flow is still required",
        }

    def context_manifest(
        self,
        source_refs: Dict[str, RecallItem],
        allowed_tables: Iterable[str] | None = None,
        allowed_relationship_topics: Iterable[str] | None = None,
        limit: int = 12,
    ) -> Dict[str, Any]:
        refs: List[Dict[str, Any]] = []
        table_filter = {str(table) for table in allowed_tables or [] if table}
        relationship_topic_filter = {str(topic) for topic in allowed_relationship_topics or [] if topic}
        for item in source_refs.values():
            metadata = item.metadata or {}
            semantic_path = str(metadata.get("semanticPath") or "")
            ref_id = str(metadata.get("semanticRefId") or item.doc_id or "")
            if not semantic_path or not ref_id.startswith("semantic:"):
                continue
            if item.table and table_filter and item.table not in table_filter:
                continue
            if not item.table and item.source_type == "SEMANTIC_RELATIONSHIP" and relationship_topic_filter and item.topic not in relationship_topic_filter:
                continue
            refs.append(
                add_context_uri({
                    "refId": ref_id,
                    "path": semantic_path,
                    "kind": metadata.get("semanticKind") or item.source_type,
                    "topic": item.topic,
                    "table": item.table,
                    "title": item.title,
                    "estimatedChars": metadata.get("estimatedChars", len(item.content or "")),
                    "offloadRecommended": bool(metadata.get("offloadRecommended")),
                }, ref_id=ref_id, topic=item.topic, table=item.table, kind=str(metadata.get("semanticKind") or item.source_type), path=semantic_path)
            )
        return {
            "mode": "filesystem_as_context",
            "uriScheme": "merchant://",
            "policy": "start from topic manifests; read/grep only table, metric, relationship, or rule files needed for the current step; offload large files by path",
            "layers": {
                "L0": "topic/table/metric summaries for routing and quick relevance checks",
                "L1": "table, metric, relationship and rule overviews for planning and rerank",
                "L2": "full schema, metric formulas, rules, rows or artifacts loaded only on demand",
            },
            "progressiveDisclosure": [
                "1. topic manifest: available tables, high-level metrics and rule summaries",
                "2. table asset: fields, metric formulas, keys, partition and merchant filters",
                "3. relationship/rule files: only when graph edges, formulas or business policy evidence is missing",
                "4. workspace artifacts: read query graphs, SQL, rows or evidence reports by path when needed",
            ],
            "roots": ["topics/<topic>/manifest.json", "topics/<topic>/tables/<table>/asset.json", "topics/<topic>/relationships.json"],
            "refs": refs[:limit],
        }

    def manifest_ref(self, topic: str) -> Dict[str, Any]:
        manifest = self.topic_assets.load_manifest(topic)
        compact_tables: List[Dict[str, Any]] = []
        search_parts: List[str] = [topic]
        for item in manifest:
            table = str(item.get("tableName") or "")
            title = str(item.get("tableComment") or item.get("title") or table)
            metrics = item.get("metrics") if isinstance(item.get("metrics"), list) else []
            fields = item.get("fields") if isinstance(item.get("fields"), list) else []
            compact_tables.append(
                {
                    "tableName": table,
                    "title": title,
                    "dataGrain": item.get("dataGrain") or item.get("grain") or "",
                    "primaryKeys": item.get("primaryKeys") or item.get("entityKeys") or [],
                    "metricHints": metrics[:8],
                    "fieldHints": fields[:8],
                }
            )
            search_parts.extend([table, title, json.dumps(metrics[:8], ensure_ascii=False), json.dumps(fields[:8], ensure_ascii=False)])
        content_payload = {
            "topic": topic,
            "layer": "manifest",
            "policy": "Use this manifest to choose which table asset or relationship file to read next. Do not infer formulas from manifest hints alone.",
            "tables": compact_tables,
            "relationshipPath": semantic_relationship_path(topic),
        }
        content = json.dumps(content_payload, ensure_ascii=False, indent=2)
        return add_context_uri({
            "refId": semantic_manifest_ref_id(topic),
            "kind": self.MANIFEST_KIND,
            "topic": topic,
            "table": "",
            "path": semantic_manifest_path(topic),
            "title": "%s/manifest" % topic,
            "summary": "%d table manifests under topic %s" % (len(compact_tables), topic),
            "layers": {"tables": len(compact_tables), "layer": "manifest"},
            "estimatedChars": len(content),
            "offloadRecommended": len(content) > self.OFFLOAD_THRESHOLD_CHARS,
            "content": content,
            "searchText": "\n".join(search_parts),
        }, ref_id=semantic_manifest_ref_id(topic), topic=topic, kind=self.MANIFEST_KIND, path=semantic_manifest_path(topic))

    def table_ref(self, topic: str, table: str, asset: Dict[str, Any] | None = None) -> Dict[str, Any]:
        asset = asset or self.topic_assets.load_table_asset(topic, table)
        content = json.dumps(asset, ensure_ascii=False, indent=2)
        layers = {
            "schemaColumns": len(asset.get("schemaColumns") or []),
            "semanticColumns": len(asset.get("semanticColumns") or []),
            "metrics": len(asset.get("metrics") or []),
            "terms": len(asset.get("terms") or []),
            "knowledgeRules": len(asset.get("knowledgeRules") or []),
        }
        return add_context_uri({
            "refId": semantic_table_ref_id(topic, table),
            "kind": self.TABLE_KIND,
            "topic": topic,
            "table": table,
            "path": semantic_table_path(topic, table),
            "title": "%s/%s" % (topic, table),
            "summary": str(asset.get("tableComment") or asset.get("manualNotes") or table),
            "layers": layers,
            "estimatedChars": len(content),
            "offloadRecommended": len(content) > self.OFFLOAD_THRESHOLD_CHARS,
            "content": content,
            "searchText": compact_semantic_asset_for_recall(asset),
        }, ref_id=semantic_table_ref_id(topic, table), topic=topic, table=table, kind=self.TABLE_KIND, path=semantic_table_path(topic, table))

    def relationship_ref(self, topic: str, relationships: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        relationships = relationships if relationships is not None else self.topic_assets.load_relationships(topic)
        content = json.dumps(relationships, ensure_ascii=False, indent=2)
        return add_context_uri({
            "refId": semantic_relationship_ref_id(topic),
            "kind": self.RELATIONSHIP_KIND,
            "topic": topic,
            "table": "",
            "path": semantic_relationship_path(topic),
            "title": "%s/relationships" % topic,
            "summary": "%d semantic table relationships" % len(relationships),
            "layers": {"relationships": len(relationships)},
            "estimatedChars": len(content),
            "offloadRecommended": len(content) > self.OFFLOAD_THRESHOLD_CHARS,
            "content": content,
            "searchText": json.dumps(relationships, ensure_ascii=False),
        }, ref_id=semantic_relationship_ref_id(topic), topic=topic, kind=self.RELATIONSHIP_KIND, path=semantic_relationship_path(topic))

    def _topics(self, topic_categories: Iterable[QuestionCategory] | None, topic: str) -> List[str]:
        if topic:
            return [topic]
        if topic_categories:
            topics = self.topic_assets.topic_names_for_categories(topic_categories)
            if topics:
                return topics
        return self.topic_assets.all_topic_names()

    def _all_refs(self, topic_categories: Iterable[QuestionCategory] | None = None, topic: str = "") -> List[Dict[str, Any]]:
        topics = tuple(self._topics(topic_categories, topic))
        cache_key = (topic or "", topics)
        if cache_key in self._refs_cache:
            return self._refs_cache[cache_key]
        refs: List[Dict[str, Any]] = []
        for topic_name in topics:
            refs.append(self.manifest_ref(topic_name))
            for manifest_item in self.topic_assets.load_manifest(topic_name):
                table = str(manifest_item.get("tableName") or "")
                if table:
                    refs.append(self.table_ref(topic_name, table))
            relationships = self.topic_assets.load_relationships(topic_name)
            if relationships:
                refs.append(self.relationship_ref(topic_name, relationships))
        self._refs_cache[cache_key] = refs
        return refs

    def _resolve_ref(self, ref_id: str, path: str) -> Dict[str, Any] | None:
        wanted_ref = ref_id.strip()
        wanted_path = normalize_semantic_path(path)
        for ref in self._all_refs():
            if wanted_ref and ref["refId"] == wanted_ref:
                return ref
            if wanted_path and ref["path"] == wanted_path:
                return ref
        return None

    def _public_ref(self, ref: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "refId": ref["refId"],
            "kind": ref["kind"],
            "topic": ref["topic"],
            "table": ref.get("table", ""),
            "path": ref["path"],
            "title": ref["title"],
            "summary": ref["summary"],
            "layers": ref["layers"],
            "estimatedChars": ref["estimatedChars"],
            "offloadRecommended": ref["offloadRecommended"],
            "merchantUri": ref.get("merchantUri", ""),
            "contextLayer": ref.get("contextLayer", ""),
        }


class SkillLoader:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._skills: Optional[List[SkillManifest]] = None

    @property
    def root(self) -> Path:
        return self.settings.resolved_ops_path.parent / "skills"

    def all_skills(self) -> List[SkillManifest]:
        if self._skills is not None:
            return self._skills
        skills: List[SkillManifest] = []
        for path in sorted(self.root.glob("*/skill.json")):
            data = read_json(path)
            if not isinstance(data, dict):
                continue
            try:
                skill = SkillManifest.model_validate(data)
                skill.source_path = str(path)
                skills.append(self._thin_policy_skill(skill))
            except Exception:
                continue
        self._skills = skills
        return skills

    def select(self, question: str, topic_categories: Iterable[QuestionCategory]) -> List[SkillManifest]:
        text = (question or "").lower()
        wanted_topics = {category_display(category) for category in topic_categories}
        selected: List[SkillManifest] = []
        for skill in self.all_skills():
            terms = [term.lower() for term in skill.trigger_terms]
            if (
                any(term and term in text for term in terms)
                or skill.display_name in wanted_topics
                or self._matches_topic_policy(skill, wanted_topics, terms)
            ):
                selected.append(skill)
        return selected

    def _matches_topic_policy(self, skill: SkillManifest, wanted_topics: Set[str], terms: List[str]) -> bool:
        display = (skill.display_name or skill.domain or "").lower()
        for topic in wanted_topics:
            topic_text = topic.lower()
            if topic_text and (topic_text in display or display in topic_text):
                return True
            if any(term and len(term) >= 2 and term in topic_text for term in terms):
                return True
        return False

    def _thin_policy_skill(self, skill: SkillManifest) -> SkillManifest:
        return skill.model_copy(
            update={
                "tables": [],
                "metrics": [],
                "entity_keys": [],
                "relationships": [],
                "graph_patterns": [],
                "retrieval_hints": skill.retrieval_hints or self._default_retrieval_hints(skill),
            }
        )

    def _default_retrieval_hints(self, skill: SkillManifest) -> List[str]:
        hints: List[str] = []
        if skill.trigger_terms:
            hints.append("召回与 %s 相关的语义层指标、字段、关系和业务规则" % " / ".join(skill.trigger_terms[:6]))
        if skill.field_warnings:
            hints.append("优先召回能解释字段口径风险的规则和术语")
        return hints[:3]

    def policy_payload(self, skill: SkillManifest) -> Dict[str, Any]:
        thin = self._thin_policy_skill(skill)
        return {
            "domain": thin.domain,
            "displayName": thin.display_name,
            "retrievalHints": thin.retrieval_hints,
            "fieldWarnings": thin.field_warnings,
            "answerGuidelines": thin.answer_guidelines,
        }


class WikiMemoryService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def load_base_wiki(self) -> str:
        parts = []
        for path in sorted(self.settings.resolved_wiki_path.glob("*.md")):
            try:
                parts.append("# %s\n%s" % (path.stem, path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return "\n\n".join(parts)

    def load_relevant_wiki(self, topic_names: Iterable[str]) -> str:
        parts = []
        for topic in topic_names:
            topic_dir = self.settings.resolved_topic_path / topic
            for path in sorted(topic_dir.rglob("*.md")):
                try:
                    parts.append("## %s/%s\n%s" % (topic, path.name, path.read_text(encoding="utf-8")[:5000]))
                except Exception:
                    continue
        return "\n\n".join(parts)

    def compress_to_wiki(self, category_name: str, rows: List[Dict[str, Any]], manual_markdown: str = "") -> Path:
        target = self.settings.resolved_workspace_path / "wiki" / ("%s.md" % (category_name or "all"))
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# %s 问答沉淀" % (category_name or "全部")]
        if manual_markdown:
            lines.append(manual_markdown.strip())
        for row in rows[:200]:
            lines.append("\n## %s\n%s" % (row.get("question", ""), row.get("answer", "")))
        target.write_text("\n".join(lines), encoding="utf-8")
        return target


class HybridRecallService:
    """Local BM25-ish recall over existing wiki/runtime topic assets, with ES-compatible hook points."""

    def __init__(self, settings: Settings, topic_assets: TopicAssetService, wiki_memory: WikiMemoryService):
        self.settings = settings
        self.topic_assets = topic_assets
        self.semantic_catalog = SemanticCatalogService(topic_assets)
        self.wiki_memory = wiki_memory
        self._documents: Optional[List[RecallItem]] = None
        self._recall_cache = build_ttl_cache("hybrid_recall", settings, settings.cache_recall_ttl_seconds)

    def recall(
        self,
        question: str,
        keywords: Any,
        history_rows: List[Dict[str, Any]],
        knowledge_context: str,
        merchant_id: str,
        topic_categories: List[QuestionCategory],
    ) -> RecallBundle:
        query_terms = recall_terms(question, getattr(keywords, "keywords", []))
        allowed_topics = set(self.topic_assets.topic_names_for_categories(topic_categories))
        cache_key = stable_cache_key(
            "recall",
            {
                "question": question,
                "keywords": query_terms,
                "merchantId": merchant_id,
                "topics": sorted(allowed_topics),
            },
        )
        cached = self._recall_cache.get(cache_key)
        if cached is not None:
            return RecallBundle.model_validate(cached)
        scored: List[RecallItem] = []
        for doc in self._load_documents():
            if not allowed_topics and doc.source_type != "BASE_WIKI":
                continue
            if allowed_topics and doc.topic and doc.topic not in allowed_topics:
                continue
            score = score_document(query_terms, doc.title + "\n" + doc.content)
            if score <= 0:
                continue
            metadata = dict(doc.metadata or {})
            recall_queries = [str(item) for item in metadata.get("recallQueries") or [] if item]
            if question and question not in recall_queries:
                recall_queries.append(question)
            metadata["recallQuery"] = question
            metadata["recallQueries"] = recall_queries
            item = doc.model_copy(update={"fusion_score": score, "metadata": metadata})
            scored.append(item)
        scored.sort(key=lambda item: item.fusion_score, reverse=True)
        items = scored[:4] if not allowed_topics else scored[:12]
        merged = "\n\n".join(
            "召回片段 [%s] %s\n%s" % (item.source_type, item.title, item.content[:1200]) for item in items
        )
        bundle = RecallBundle(items=items, top_score=items[0].fusion_score if items else 0.0, merged_context=merged)
        self._recall_cache.set(cache_key, bundle.model_dump(by_alias=True))
        return bundle

    def clear_cache(self) -> None:
        self._documents = None
        self._recall_cache.clear()
        self.semantic_catalog.clear_cache()

    def cache_trace(self) -> Dict[str, Any]:
        return {"recall": self._recall_cache.trace()}

    def _load_documents(self) -> List[RecallItem]:
        if self._documents is not None:
            return self._documents
        docs: List[RecallItem] = []
        for path in sorted(self.settings.resolved_wiki_path.glob("*.md")):
            try:
                docs.append(
                    RecallItem(
                        doc_id=str(path),
                        title=path.stem,
                        content=path.read_text(encoding="utf-8")[:8000],
                        source_type="BASE_WIKI",
                    )
                )
            except Exception:
                pass
        for topic in self.topic_assets.all_topic_names():
            for manifest_item in self.topic_assets.load_manifest(topic):
                table = str(manifest_item.get("tableName") or "")
                asset = self.topic_assets.load_table_asset(topic, table)
                ref = self.semantic_catalog.table_ref(topic, table, asset)
                docs.append(
                    RecallItem(
                        doc_id=ref["refId"],
                        title="%s/%s semantic asset" % (topic, table),
                        content=compact_semantic_asset_for_recall(asset),
                        source_type="SEMANTIC_TABLE_ASSET",
                        topic=topic,
                        table=table,
                        answer_mode=",".join(manifest_item.get("preferredFor") or []),
                        metadata={
                            "semanticSource": "asset.json",
                            "semanticKind": ref["kind"],
                            "semanticRefId": ref["refId"],
                            "semanticPath": ref["path"],
                            "merchantUri": ref.get("merchantUri", ""),
                            "contextLayer": ref.get("contextLayer", ""),
                            "tableName": table,
                            "topic": topic,
                            "layers": ref["layers"],
                            "estimatedChars": ref["estimatedChars"],
                            "offloadRecommended": ref["offloadRecommended"],
                        },
                    )
                )
                for metric in asset.get("metrics") or []:
                    if not isinstance(metric, dict):
                        continue
                    metric_key = str(metric.get("metricKey") or "")
                    if not metric_key:
                        continue
                    semantic_ref_id = "semantic:%s:%s:metric:%s" % (topic, table, metric_key)
                    docs.append(
                        RecallItem(
                            doc_id=semantic_ref_id,
                            title="%s/%s/%s metric" % (topic, table, metric_key),
                            content=compact_metric_for_recall(topic, table, metric),
                            source_type="SEMANTIC_METRIC",
                            topic=topic,
                            table=table,
                            metadata={
                                "semanticSource": "metrics",
                                "semanticKind": "METRIC",
                                "semanticRefId": semantic_ref_id,
                                "metricKey": metric_key,
                                "tableName": table,
                                "topic": topic,
                                "businessName": metric.get("businessName") or metric_key,
                                "canonicalMetricKey": metric.get("canonicalMetricKey") or "",
                                "aliasOf": metric.get("aliasOf") or "",
                                "metricLevel": metric.get("metricLevel") or "",
                                "formula": metric.get("formula") or metric.get("metricFormula") or "",
                                "sourceColumns": metric.get("sourceColumns") or [],
                                "aliases": metric.get("aliases") or [],
                                "merchantUri": merchant_uri_for_semantic_ref(semantic_ref_id, topic=topic, table=table, kind="METRIC", key=metric_key),
                                "contextLayer": "L1",
                            },
                        )
                    )
            relationships = self.topic_assets.load_relationships(topic)
            if relationships:
                ref = self.semantic_catalog.relationship_ref(topic, relationships)
                docs.append(
                    RecallItem(
                        doc_id=ref["refId"],
                        title="%s semantic relationships" % topic,
                        content=json.dumps(relationships, ensure_ascii=False)[:8000],
                        source_type="SEMANTIC_RELATIONSHIP",
                        topic=topic,
                        metadata={
                            "semanticSource": "relationships.json",
                            "semanticKind": ref["kind"],
                            "semanticRefId": ref["refId"],
                            "semanticPath": ref["path"],
                            "merchantUri": ref.get("merchantUri", ""),
                            "contextLayer": ref.get("contextLayer", ""),
                            "topic": topic,
                            "layers": ref["layers"],
                            "estimatedChars": ref["estimatedChars"],
                            "offloadRecommended": ref["offloadRecommended"],
                        },
                    )
                )
                for rel in relationships:
                    if not isinstance(rel, dict):
                        continue
                    rel_name = str(rel.get("name") or "")
                    if not rel_name:
                        continue
                    left = str(rel.get("leftTable") or "")
                    right = str(rel.get("rightTable") or "")
                    rel_ref_id = "semantic:%s:relationship:%s" % (topic, rel_name)
                    docs.append(
                        RecallItem(
                            doc_id=rel_ref_id,
                            title="%s/%s relationship" % (topic, rel_name),
                            content=json.dumps(rel, ensure_ascii=False)[:2400],
                            source_type="SEMANTIC_RELATIONSHIP",
                            topic=topic,
                            table=left,
                            metadata={
                                "semanticSource": "relationships.json",
                                "semanticKind": "RELATIONSHIP",
                                "semanticRefId": rel_ref_id,
                                "merchantUri": merchant_uri_for_semantic_ref(rel_ref_id, topic=topic, table=left, kind="RELATIONSHIP", key=rel_name),
                                "contextLayer": "L1",
                                "relationshipId": rel_name,
                                "leftTable": left,
                                "rightTable": right,
                                "topic": topic,
                                "joinKeys": rel.get("keys") or [],
                            },
                        )
                    )
        self._documents = docs
        return docs


def compact_semantic_asset_for_recall(asset: Dict[str, Any]) -> str:
    payload = {
        "topic": asset.get("topic"),
        "tableName": asset.get("tableName"),
        "tableComment": asset.get("tableComment"),
        "dataGrain": asset.get("dataGrain"),
        "timeColumn": asset.get("timeColumn"),
        "merchantFilterColumn": asset.get("merchantFilterColumn"),
        "rowAccessPolicy": asset.get("rowAccessPolicy") or {},
        "manualNotes": asset.get("manualNotes"),
        "metrics": [
            {
                "metricKey": item.get("metricKey"),
                "businessName": item.get("businessName"),
                "formula": item.get("formula") or item.get("metricFormula"),
                "sourceColumns": item.get("sourceColumns") or [],
                "aliases": item.get("aliases") or [],
                "description": item.get("description"),
            }
            for item in (asset.get("metrics") or [])[:80]
            if isinstance(item, dict)
        ],
        "knowledgeRules": [
            {
                "title": item.get("title"),
                "content": item.get("content"),
                "alwaysApply": item.get("alwaysApply"),
                "keywords": item.get("keywords") or [],
            }
            for item in (asset.get("knowledgeRules") or [])[:40]
            if isinstance(item, dict)
        ],
        "terms": [
            {
                "term": item.get("term"),
                "description": item.get("description"),
                "aliases": item.get("aliases") or [],
                "relatedColumns": item.get("relatedColumns") or [],
            }
            for item in (asset.get("terms") or [])[:120]
            if isinstance(item, dict)
        ],
        "semanticColumns": [
            {
                "columnName": item.get("columnName"),
                "businessName": item.get("businessName"),
                "role": item.get("role"),
                "description": item.get("description"),
                "aliases": item.get("aliases") or [],
                "defaultVisible": item.get("defaultVisible"),
                "displayPriority": item.get("displayPriority"),
                "displayScenarios": item.get("displayScenarios") or [],
                "visibilityPolicy": item.get("visibilityPolicy") or {},
                "maskingPolicy": item.get("maskingPolicy") or {},
            }
            for item in (asset.get("semanticColumns") or [])[:60]
            if isinstance(item, dict)
        ],
    }
    return json.dumps(payload, ensure_ascii=False)[:12000]


def compact_metric_for_recall(topic: str, table: str, metric: Dict[str, Any]) -> str:
    payload = {
        "topic": topic,
        "tableName": table,
        "metricKey": metric.get("metricKey"),
        "canonicalMetricKey": metric.get("canonicalMetricKey"),
        "aliasOf": metric.get("aliasOf"),
        "metricLevel": metric.get("metricLevel"),
        "businessName": metric.get("businessName"),
        "formula": metric.get("formula") or metric.get("metricFormula"),
        "sourceColumns": metric.get("sourceColumns") or [],
        "aliases": metric.get("aliases") or [],
        "description": metric.get("description"),
        "unit": metric.get("unit"),
        "evidence": metric.get("evidence"),
    }
    return json.dumps(payload, ensure_ascii=False)[:4000]


def compact_table_metadata(asset: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "topic": asset.get("topic"),
        "questionCategory": asset.get("questionCategory"),
        "tableName": asset.get("tableName"),
        "tableComment": asset.get("tableComment"),
        "dataGrain": asset.get("dataGrain"),
        "timeColumn": asset.get("timeColumn"),
        "merchantFilterColumn": asset.get("merchantFilterColumn"),
        "rowAccessPolicy": asset.get("rowAccessPolicy") or {},
        "manualNotes": asset.get("manualNotes"),
        "status": asset.get("status"),
        "version": asset.get("version"),
    }


class PlanningAssetPackBuilder:
    def __init__(self, topic_assets: TopicAssetService, skill_loader: Optional[SkillLoader] = None, doris_repository: Optional[DorisRepository] = None):
        self.topic_assets = topic_assets
        self.skill_loader = skill_loader or SkillLoader(topic_assets.settings)
        self.doris_repository = doris_repository
        self._all_metrics_by_key_cache: Optional[Dict[str, List[PlanningAssetEntry]]] = None
        self._live_schema_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._compact_cache = build_ttl_cache(
            "planning_asset_pack",
            topic_assets.settings,
            topic_assets.settings.cache_asset_pack_ttl_seconds,
        )

    def compact(
        self,
        question: str,
        recall_bundle: RecallBundle,
        topic_categories: List[QuestionCategory],
        diagnostic_context: Optional[Dict[str, Any]] = None,
    ) -> PlanningAssetPack:
        pack = PlanningAssetPack()
        allow_profile = isinstance(diagnostic_context, dict) and diagnostic_context.get("scope") == "OPEN_DIAGNOSTIC"
        topics = self.topic_assets.topic_names_for_categories(topic_categories)
        if not topics:
            topics = sorted({item.topic for item in recall_bundle.items if item.topic})
        semantic_source_hash = self._topics_source_hash(topics)
        table_topic = self._table_topic_index()
        all_relationships = self._all_relationships()
        seed_tables, targeted_traces = self._targeted_seed_tables(
            question,
            recall_bundle,
            topics,
            table_topic,
            allow_profile=allow_profile,
            explicit_tables=set(),
        )
        bridge_tables, bridge_traces = self._relationship_bridge_tables(
            seed_tables,
            all_relationships,
            table_topic,
            allow_profile=allow_profile,
            max_extra=2,
        )
        seed_tables.update(bridge_tables)
        pack_tables = {table for table in seed_tables if table}
        live_schema_hash = self._live_schema_hash_for_tables(pack_tables)
        cache_key = stable_cache_key(
            "asset_pack",
            {
                "question": question,
                "recall": [
                    (
                        item.doc_id,
                        round(float(item.fusion_score or 0), 4),
                        tuple(str(query) for query in (item.metadata or {}).get("recallQueries") or []),
                    )
                    for item in recall_bundle.items
                ],
                "topics": [category.value if isinstance(category, QuestionCategory) else str(category) for category in topic_categories],
                "diagnostic": diagnostic_context or {},
                "semanticSourceHash": semantic_source_hash,
                "liveSchemaHash": live_schema_hash,
            },
        )
        cached = self._compact_cache.get(cache_key)
        if cached is not None:
            pack = PlanningAssetPack.model_validate(cached)
            pack.metric_compaction.setdefault("cache", {})["hit"] = True
            pack.metric_compaction.setdefault("cache", {})["semanticSourceHash"] = semantic_source_hash
            pack.metric_compaction.setdefault("cache", {})["liveSchemaHash"] = live_schema_hash
            return pack
        source_refs: Dict[str, RecallItem] = {}
        for item in recall_bundle.items:
            source_refs[item.doc_id] = item
        skills = self.skill_loader.select(question, topic_categories)
        pack.skills = skills
        for skill in skills:
            source_refs["skill:%s" % skill.domain] = RecallItem(
                doc_id="skill:%s" % skill.domain,
                title=skill.display_name or skill.domain,
                content=json.dumps(self.skill_loader.policy_payload(skill), ensure_ascii=False),
                source_type="DOMAIN_SKILL",
                topic=skill.display_name,
                metadata={"sourcePath": skill.source_path},
            )
        for table in sorted(pack_tables):
            topic = table_topic.get(table)
            if not topic:
                continue
            self._append_table_assets(pack, topic, table)
        recalled_metric_evidence = recalled_metric_evidence_from_bundle(recall_bundle)
        if recalled_metric_evidence:
            pack.metric_compaction["recalledMetricEvidence"] = recalled_metric_evidence
        self._trim_metrics_for_question(pack, question)
        self._trim_terms_for_question(pack, question)
        metric_dependency_closure = self._expand_tables_for_metric_dependencies(pack, pack_tables, table_topic, question, allow_profile=allow_profile)
        if metric_dependency_closure:
            self._trim_metrics_for_question(pack, question)
            self._trim_terms_for_question(pack, question)
        relationship_topics = set(topics)
        relationship_topics.update(table_topic.get(table, "") for table in pack_tables)
        relationship_topics.discard("")
        selected_relationships = []
        for topic, rel in all_relationships:
            left = str(rel.get("leftTable") or "")
            right = str(rel.get("rightTable") or "")
            if left in pack_tables and right in pack_tables and topic in relationship_topics:
                selected_relationships.append((topic, rel))
        for topic, rel in selected_relationships:
            entry = relationship_entry(topic, rel)
            if entry.relationship_id and entry.relationship_id not in {item.relationship_id for item in pack.relationships}:
                pack.relationships.append(entry)
        pack.relationship_closure = targeted_traces + bridge_traces + metric_dependency_closure
        pack.metric_compaction.setdefault("targetedSeed", {})["tables"] = sorted(pack_tables)
        pack.metric_compaction.setdefault("targetedSeed", {})["trace"] = targeted_traces + bridge_traces
        pack.metric_compaction.setdefault("cache", {})["semanticSourceHash"] = semantic_source_hash
        pack.metric_compaction.setdefault("cache", {})["liveSchemaHash"] = live_schema_hash
        pack.skills = self._reconcile_skills(pack.skills, pack)
        for skill in pack.skills:
            ref = source_refs.get("skill:%s" % skill.domain)
            if ref:
                ref.content = json.dumps(self.skill_loader.policy_payload(skill), ensure_ascii=False)
        pack.source_refs = source_refs
        self._compact_cache.set(cache_key, pack.model_dump(by_alias=True))
        return pack

    def clear_cache(self) -> None:
        self._compact_cache.clear()
        self._all_metrics_by_key_cache = None
        self._live_schema_cache.clear()

    def cache_trace(self) -> Dict[str, Any]:
        return {
            "assetPack": self._compact_cache.trace(),
            "liveSchemaEntries": len(self._live_schema_cache),
            "allMetricIndexCached": self._all_metrics_by_key_cache is not None,
        }

    def _topics_source_hash(self, topics: List[str]) -> str:
        hasher = hashlib.sha256()
        for topic in sorted({str(item or "") for item in topics if item}):
            topic_dir = self.topic_assets.root / topic
            if not topic_dir.exists():
                continue
            for path in sorted(topic_dir.rglob("*")):
                if not path.is_file():
                    continue
                if path.name.startswith("."):
                    continue
                try:
                    hasher.update(str(path.relative_to(self.topic_assets.root)).encode("utf-8"))
                    hasher.update(path.read_bytes())
                except Exception:
                    continue
        return hasher.hexdigest()[:16]

    def _live_schema_hash_for_tables(self, tables: Set[str]) -> str:
        if not self.doris_repository:
            return ""
        hasher = hashlib.sha256()
        any_schema = False
        for table in sorted(table for table in tables if table):
            schema = self._live_schema(table)
            if not schema:
                continue
            any_schema = True
            hasher.update(table.encode("utf-8"))
            hasher.update(json.dumps(schema, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))
        return hasher.hexdigest()[:16] if any_schema else ""

    def expand_for_question_understanding(self, pack: PlanningAssetPack, understanding: Dict[str, Any]) -> List[str]:
        """Load extra semantic assets only when LLM understanding cites them.

        The expansion is driven by structured metric/table refs emitted by the
        planner, not by matching words in the original question.
        """
        if not isinstance(understanding, dict):
            return []
        table_topic = self._table_topic_index()
        pack_tables = set(pack.known_tables())
        traces: List[str] = []
        for item in question_understanding_metric_requests(understanding):
            metric_ref = str(item.get("metricRef") or item.get("metric_ref") or "")
            owner_table = str(item.get("ownerTable") or item.get("owner_table") or "")
            source_phrase = str(item.get("sourcePhrase") or item.get("source_phrase") or "")
            tables = self._tables_for_metric_request(metric_ref, owner_table, source_phrase, pack.metric_compaction)
            for table in tables:
                if not table or table in pack_tables:
                    continue
                topic = table_topic.get(table)
                if not topic:
                    continue
                self._append_table_assets(pack, topic, table)
                pack_tables.add(table)
                traces.append(
                    "metric_request_table:%s->%s%s"
                    % (metric_ref or owner_table or "unknown", table, ":ownerTable" if owner_table == table else "")
                )
        if not traces:
            return []
        relationship_traces = self._append_relationships_for_tables(pack, pack_tables, table_topic)
        pack.relationship_closure.extend(traces + relationship_traces)
        expansion = pack.metric_compaction.setdefault("questionUnderstandingExpansion", [])
        if isinstance(expansion, list):
            expansion.extend(traces)
        return traces + relationship_traces

    def _expand_tables_for_metric_dependencies(
        self,
        pack: PlanningAssetPack,
        pack_tables: Set[str],
        table_topic: Dict[str, str],
        question: str,
        allow_profile: bool = False,
    ) -> List[str]:
        metric_index = self._all_metrics_by_key()
        closure: List[str] = []
        for metric in list(pack.metrics):
            if self._metric_relevance_score(metric, question) < 20:
                continue
            for dep_key in self._metric_dependency_keys(metric, metric_index):
                dep_metric = next((item for item in metric_index.get(dep_key, []) if item.table and item.table not in pack_tables), None)
                if dep_metric is None:
                    covered_metric = next((item for item in metric_index.get(dep_key, []) if item.table and item.table in pack_tables), None)
                    if covered_metric is not None:
                        marker = "metric_dependency:%s->%s:%s" % (metric.key, dep_key, covered_metric.table)
                        if marker not in closure:
                            closure.append(marker)
                        if not any(item.table == covered_metric.table and item.key == covered_metric.key for item in pack.metrics):
                            pack.metrics.append(covered_metric)
                    continue
                dep_table = dep_metric.table
                topic = table_topic.get(dep_table)
                if not topic or not self._table_allowed_for_question(question, dep_table, allow_profile=allow_profile):
                    continue
                pack_tables.add(dep_table)
                self._append_table_assets(pack, topic, dep_table)
                closure.append("metric_dependency:%s->%s:%s" % (metric.key, dep_key, dep_table))
        return closure

    def _all_metrics_by_key(self) -> Dict[str, List[PlanningAssetEntry]]:
        if self._all_metrics_by_key_cache is not None:
            return self._all_metrics_by_key_cache
        metrics_by_key: Dict[str, List[PlanningAssetEntry]] = {}
        for topic in self.topic_assets.all_topic_names():
            for manifest_item in self.topic_assets.load_manifest(topic):
                table = str(manifest_item.get("tableName") or "")
                if not table:
                    continue
                for metric in self.topic_assets.load_table_metrics(topic, table):
                    key = str(metric.get("metricKey") or "")
                    if not key:
                        continue
                    metrics_by_key.setdefault(key, []).append(
                        PlanningAssetEntry(
                            key=key,
                            table=table,
                            topic=topic,
                            title=str(metric.get("businessName") or key),
                            columns=[str(col) for col in metric.get("sourceColumns") or []],
                            aliases=[str(alias) for alias in metric.get("aliases") or []],
                            description=json.dumps(metric, ensure_ascii=False),
                            source_ref_id="semantic:%s:%s:metric:%s" % (topic, table, key),
                            metadata=metric,
                        )
                    )
        self._all_metrics_by_key_cache = metrics_by_key
        return metrics_by_key

    def _all_metric_entries(self) -> List[PlanningAssetEntry]:
        entries: List[PlanningAssetEntry] = []
        for metrics in self._all_metrics_by_key().values():
            entries.extend(metrics)
        return entries

    def _trim_metrics_for_question(self, pack: PlanningAssetPack, question: str) -> None:
        original_count = len(pack.metrics)
        if original_count <= 0:
            existing_trace = dict(pack.metric_compaction or {})
            pack.metric_compaction = {**existing_trace, "before": 0, "after": 0, "strategy": "empty"}
            return
        existing_trace = dict(pack.metric_compaction or {})
        metrics_by_key: Dict[str, List[PlanningAssetEntry]] = {}
        metrics_by_identity: Dict[Tuple[str, str], PlanningAssetEntry] = {}
        for metric in pack.metrics:
            if not metric.key:
                continue
            metrics_by_key.setdefault(metric.key, []).append(metric)
            metrics_by_identity[(metric.table, metric.key)] = metric
        protected_identities = recalled_metric_identities_from_compaction(existing_trace)
        by_table: Dict[str, List[PlanningAssetEntry]] = {}
        for metric in pack.metrics:
            by_table.setdefault(metric.table, []).append(metric)
        selected: List[PlanningAssetEntry] = []
        selected_keys: Set[Tuple[str, str]] = set()
        table_limit = 5
        global_limit = max(6, int(self.topic_assets.settings.agent_planner_seed_metric_limit or 14))
        for identity in sorted(protected_identities):
            metric = metrics_by_identity.get(identity)
            if metric and identity not in selected_keys:
                selected.append(metric)
                selected_keys.add(identity)
        for table, metrics in by_table.items():
            ranked = sorted(metrics, key=lambda item: self._metric_relevance_score(item, question), reverse=True)
            positive = [metric for metric in ranked if self._metric_relevance_score(metric, question) > 0]
            for metric in positive[:table_limit]:
                metric_key = (metric.table, metric.key)
                if metric.key and metric_key not in selected_keys:
                    selected.append(metric)
                    selected_keys.add(metric_key)
        if not selected:
            ranked_all = sorted(pack.metrics, key=lambda item: self._metric_relevance_score(item, question), reverse=True)
            for metric in ranked_all[: min(global_limit, len(ranked_all))]:
                metric_key = (metric.table, metric.key)
                if metric.key and metric_key not in selected_keys:
                    selected.append(metric)
                    selected_keys.add(metric_key)
        protected_selected = [metric for metric in selected if (metric.table, metric.key) in protected_identities]
        unprotected_selected = [metric for metric in selected if (metric.table, metric.key) not in protected_identities]
        unprotected_selected = sorted(unprotected_selected, key=lambda item: self._metric_relevance_score(item, question), reverse=True)
        selected = protected_selected + unprotected_selected[: max(0, global_limit - len(protected_selected))]
        selected_keys = {(metric.table, metric.key) for metric in selected if metric.key}
        for metric in list(selected):
            for dep_key in self._metric_dependency_keys(metric, metrics_by_key):
                dep = metrics_by_identity.get((metric.table, dep_key))
                if dep is None:
                    dep = next(iter(metrics_by_key.get(dep_key, [])), None)
                if dep is None:
                    continue
                dep_identity = (dep.table, dep.key)
                if dep_identity not in selected_keys and len(selected) >= global_limit:
                    removed = pop_lowest_unprotected_metric(selected, {(metric.table, metric.key)})
                    if removed:
                        selected_keys.discard((removed.table, removed.key))
                if dep_identity not in selected_keys and len(selected) < global_limit:
                    selected.append(dep)
                    selected_keys.add(dep_identity)
        pack.metrics = selected
        pack.metric_compaction = {
            **existing_trace,
            "before": original_count,
            "after": len(pack.metrics),
            "strategy": "question_relevance_top_metrics",
            "perTableLimit": table_limit,
            "globalLimit": global_limit,
            "tables": {table: len([metric for metric in pack.metrics if metric.table == table]) for table in sorted(by_table)},
        }

    def _trim_terms_for_question(self, pack: PlanningAssetPack, question: str) -> None:
        original_count = len(pack.terms)
        if original_count <= 0:
            pack.metric_compaction.setdefault("terms", {"before": 0, "after": 0, "strategy": "empty"})
            return
        terms = targeted_recall_terms(question)
        by_table: Dict[str, List[PlanningAssetEntry]] = {}
        for term in pack.terms:
            by_table.setdefault(term.table, []).append(term)
        selected: List[PlanningAssetEntry] = []
        per_table_limit = 6
        total_limit = 32
        for table, table_terms in sorted(by_table.items()):
            ranked = sorted(
                table_terms,
                key=lambda item: score_document(
                    terms,
                    " ".join(
                        [
                            item.key,
                            item.title,
                            " ".join(item.aliases),
                            item.description,
                            json.dumps(item.metadata, ensure_ascii=False),
                        ]
                    ),
                ),
                reverse=True,
            )
            positive = [
                item
                for item in ranked
                if score_document(
                    terms,
                    " ".join([item.key, item.title, " ".join(item.aliases), item.description, json.dumps(item.metadata, ensure_ascii=False)]),
                )
                > 0
            ]
            selected.extend((positive or ranked)[:per_table_limit])
        pack.terms = dedupe_entries_by_identity(selected)[:total_limit]
        pack.metric_compaction["terms"] = {
            "before": original_count,
            "after": len(pack.terms),
            "strategy": "question_relevance_top_terms",
            "perTableLimit": per_table_limit,
            "globalLimit": total_limit,
        }

    def _targeted_seed_tables(
        self,
        question: str,
        recall_bundle: RecallBundle,
        topics: List[str],
        table_topic: Dict[str, str],
        allow_profile: bool = False,
        explicit_tables: Set[str] | None = None,
    ) -> Tuple[Set[str], List[str]]:
        explicit_tables = explicit_tables or set()
        recalled_tables = {
            item.table
            for item in recall_bundle.items
            if item.table and self._table_allowed_for_recalled_item(question, item, item.table, allow_profile=allow_profile)
        }
        recalled_tables.update(
            str(item.metadata.get("tableName") or "")
            for item in recall_bundle.items
            if item.metadata
            and self._table_allowed_for_recalled_item(
                question,
                item,
                str(item.metadata.get("tableName") or ""),
                allow_profile=allow_profile,
            )
        )
        for item in recall_bundle.items:
            if str(item.source_type or "").upper() != "SEMANTIC_RELATIONSHIP":
                continue
            for table in recalled_relationship_tables(item):
                if self._table_allowed_for_recalled_item(question, item, table, allow_profile=allow_profile):
                    recalled_tables.add(table)
        candidate_tables: List[Tuple[str, str, Dict[str, Any]]] = []
        has_precise_recall_evidence = bool(explicit_tables or recalled_tables)
        if not has_precise_recall_evidence:
            for topic in topics:
                for manifest_item in self.topic_assets.load_manifest(topic):
                    table = str(manifest_item.get("tableName") or "")
                    if not table or not self._table_allowed_for_question(question, table, allow_profile=allow_profile):
                        continue
                    candidate_tables.append((topic, table, manifest_item))
        existing_candidates = {table for _, table, _ in candidate_tables}
        for table in sorted(explicit_tables | recalled_tables):
            if not table or table in existing_candidates:
                continue
            topic = table_topic.get(table, "")
            if not topic:
                continue
            manifest_item = next((item for item in self.topic_assets.load_manifest(topic) if str(item.get("tableName") or "") == table), {})
            candidate_tables.append((topic, table, manifest_item))
            existing_candidates.add(table)
        if not candidate_tables:
            for table in sorted(explicit_tables | recalled_tables):
                topic = table_topic.get(table, "")
                if topic:
                    candidate_tables.append((topic, table, {}))
        table_scores = [
            self._table_seed_score(question, recall_bundle, topic, table, manifest_item, table in explicit_tables, table in recalled_tables)
            for topic, table, manifest_item in candidate_tables
        ]
        table_scores = [item for item in table_scores if item[1] > 0 or item[0] in explicit_tables or item[0] in recalled_tables]
        table_scores = self._filter_weak_seed_scores_by_topic(table_scores)
        table_scores.sort(key=lambda item: (item[1], item[2].get("recallScore", 0), item[2].get("metricScore", 0)), reverse=True)
        limit = max(1, int(self.topic_assets.settings.agent_planner_seed_table_limit or 4))
        evidenced_table_count = len({table for table in explicit_tables | recalled_tables if table})
        limit = min(max(limit, evidenced_table_count), 6)
        selected: List[str] = []
        for table, _, _ in table_scores:
            if table not in selected:
                selected.append(table)
            if len(selected) >= limit:
                break
        if not selected:
            fallback_scores = [
                self._table_seed_score(question, recall_bundle, topic, table, manifest_item, table in explicit_tables, table in recalled_tables)
                for topic, table, manifest_item in candidate_tables
            ]
            fallback_scores.sort(key=lambda item: item[1], reverse=True)
            selected = [table for table, _, _ in fallback_scores[:limit] if table]
        coverage_topics = {
            table_topic.get(table, "")
            for table in explicit_tables | recalled_tables
            if table_topic.get(table, "")
        }
        selected = self._ensure_seed_topic_coverage(selected, table_scores, sorted(coverage_topics), limit)
        score_preview = [
            {
                "table": table,
                "score": score,
                **{key: value for key, value in detail.items() if value},
            }
            for table, score, detail in table_scores[: max(limit, 6)]
        ]
        traces = [
            "targeted_seed_tables:%s"
            % ",".join("%s=%s" % (item["table"], item["score"]) for item in score_preview[:limit])
        ]
        if has_precise_recall_evidence:
            traces.append("targeted_seed_source=recall_source_refs")
        else:
            traces.append("targeted_seed_source=topic_boundary")
        return set(selected), traces

    def _ensure_seed_topic_coverage(
        self,
        selected: List[str],
        table_scores: List[Tuple[str, int, Dict[str, Any]]],
        topics: List[str],
        limit: int,
    ) -> List[str]:
        if not selected or not table_scores:
            return selected
        score_by_table = {table: (score, detail) for table, score, detail in table_scores}
        candidates_by_topic: Dict[str, List[Tuple[str, int, Dict[str, Any]]]] = {}
        for table, score, detail in table_scores:
            topic = str(detail.get("topic") or "")
            if not topic:
                continue
            candidates_by_topic.setdefault(topic, []).append((table, score, detail))
        selected_set = set(selected)
        for topic in topics:
            if topic not in candidates_by_topic:
                continue
            if any(str(score_by_table.get(table, (0, {}))[1].get("topic") or "") == topic for table in selected):
                continue
            replacement = next((item for item in candidates_by_topic[topic] if item[0] not in selected_set), None)
            if not replacement:
                continue
            replacement_table = replacement[0]
            if len(selected) < limit:
                selected.append(replacement_table)
                selected_set.add(replacement_table)
                continue
            topic_counts: Dict[str, int] = {}
            for table in selected:
                selected_topic = str(score_by_table.get(table, (0, {}))[1].get("topic") or "")
                topic_counts[selected_topic] = topic_counts.get(selected_topic, 0) + 1
            replace_index = -1
            replace_score = 10**9
            for index, table in enumerate(selected):
                selected_score, detail = score_by_table.get(table, (0, {}))
                selected_topic = str(detail.get("topic") or "")
                if topic_counts.get(selected_topic, 0) <= 1:
                    continue
                if detail.get("explicit"):
                    continue
                if selected_score < replace_score:
                    replace_score = selected_score
                    replace_index = index
            if replace_index >= 0:
                selected_set.discard(selected[replace_index])
                selected[replace_index] = replacement_table
                selected_set.add(replacement_table)
        return selected[:limit]

    def _table_seed_score(
        self,
        question: str,
        recall_bundle: RecallBundle,
        topic: str,
        table: str,
        manifest_item: Dict[str, Any],
        explicit: bool = False,
        recalled: bool = False,
    ) -> Tuple[str, int, Dict[str, Any]]:
        terms = table_seed_terms(question)
        manifest_text = json.dumps(manifest_item, ensure_ascii=False)
        score = int(score_document(terms, manifest_text))
        detail: Dict[str, Any] = {}
        if explicit:
            score += 80
            detail["explicit"] = True
        detail["topic"] = topic
        recall_score = table_recall_score(recall_bundle, table)
        if recall_score:
            score += int(recall_score * 4) + 20
            detail["recallScore"] = round(recall_score, 2)
        elif recalled:
            score += 20
            detail["recalled"] = True
        try:
            asset = self.topic_assets.load_table_asset(topic, table)
        except Exception:
            asset = {}
        if asset:
            score += int(score_document(terms, compact_semantic_asset_for_recall(asset)))
            table_text = normalize_for_match(
                " ".join(
                    [
                        table,
                        str(asset.get("tableComment") or ""),
                        str(asset.get("dataGrain") or ""),
                        str(asset.get("manualNotes") or ""),
                    ]
                )
            )
            for term in terms:
                normalized = normalize_for_match(term)
                if normalized and normalized in table_text:
                    score += 8
            metric_scores = []
            for metric in self.topic_assets.load_table_metrics(topic, table)[:120]:
                entry = PlanningAssetEntry(
                    key=str(metric.get("metricKey") or ""),
                    table=table,
                    topic=topic,
                    title=str(metric.get("businessName") or metric.get("metricKey") or ""),
                    columns=[str(col) for col in metric.get("sourceColumns") or []],
                    aliases=[str(alias) for alias in metric.get("aliases") or []],
                    description=json.dumps(metric, ensure_ascii=False),
                    metadata=metric,
                )
                metric_score = self._metric_relevance_score(entry, question)
                if metric_score > 0:
                    metric_scores.append(metric_score)
            if metric_scores:
                top_metric_score = max(metric_scores)
                score += top_metric_score
                detail["metricScore"] = top_metric_score
            field_text = json.dumps(asset.get("semanticColumns") or [], ensure_ascii=False)
            field_score = int(score_document(terms, field_text))
            if field_score:
                score += field_score
                detail["fieldScore"] = field_score
        if "profile" in table.lower() and not explicit and not recalled:
            score -= 40
        return table, score, detail

    def _table_allowed_for_recalled_item(
        self,
        question: str,
        item: RecallItem,
        table: str,
        allow_profile: bool = False,
    ) -> bool:
        if not table:
            return False
        if self._table_allowed_for_question(question, table, allow_profile=allow_profile):
            return True
        return False

    def _filter_weak_seed_scores_by_topic(self, table_scores: List[Tuple[str, int, Dict[str, Any]]]) -> List[Tuple[str, int, Dict[str, Any]]]:
        by_topic: Dict[str, List[Tuple[str, int, Dict[str, Any]]]] = {}
        for item in table_scores:
            by_topic.setdefault(str(item[2].get("topic") or ""), []).append(item)
        filtered: List[Tuple[str, int, Dict[str, Any]]] = []
        for topic, items in by_topic.items():
            if len(items) <= 1:
                filtered.extend(items)
                continue
            top_score = max(score for _, score, _ in items)
            minimum_score = max(1, int(top_score * 0.25))
            for table, score, detail in items:
                if detail.get("explicit") or float(detail.get("recallScore") or 0.0) >= 8.0 or score >= minimum_score:
                    filtered.append((table, score, detail))
        return filtered

    def _relationship_bridge_tables(
        self,
        seed_tables: Set[str],
        all_relationships: List[Tuple[str, Dict[str, Any]]],
        table_topic: Dict[str, str],
        allow_profile: bool = False,
        max_extra: int = 2,
    ) -> Tuple[Set[str], List[str]]:
        if len(seed_tables) < 2 or max_extra <= 0:
            return set(), []
        graph: Dict[str, Set[str]] = {}
        for _, rel in all_relationships:
            left = str(rel.get("leftTable") or "")
            right = str(rel.get("rightTable") or "")
            if not left or not right:
                continue
            graph.setdefault(left, set()).add(right)
            graph.setdefault(right, set()).add(left)
        extras: Set[str] = set()
        traces: List[str] = []
        seeds = sorted(seed_tables)
        for index, start in enumerate(seeds):
            for target in seeds[index + 1 :]:
                path = shortest_table_path(graph, start, target, max_hops=3)
                if len(path) <= 2:
                    continue
                for table in path[1:-1]:
                    if table in seed_tables or table in extras:
                        continue
                    if not self._table_allowed_for_question("", table, allow_profile=allow_profile):
                        continue
                    if table not in table_topic:
                        continue
                    extras.add(table)
                    traces.append("relationship_bridge_table:%s->%s:%s" % (start, target, table))
                    if len(extras) >= max_extra:
                        return extras, traces
        return extras, traces

    def _metric_relevance_score(self, metric: PlanningAssetEntry, question: str) -> int:
        q = normalize_for_match(question)
        text = normalize_for_match(
            " ".join(
                [
                    metric.key,
                    metric.title,
                    " ".join(metric.aliases),
                    metric.description,
                    json.dumps(metric.metadata, ensure_ascii=False),
                ]
            )
        )
        score = 0
        for phrase in [metric.key, metric.title, *metric.aliases]:
            phrase_norm = normalize_for_match(phrase)
            if phrase_norm and phrase_norm in q:
                score += 40 + min(len(phrase_norm), 20)
        for term in question_match_terms(question):
            if term and term in text:
                score += 4 + min(len(term), 8)
        for column in metric.columns:
            column_norm = normalize_for_match(column)
            if column_norm and column_norm in q:
                score += 12
        formula = normalize_for_match(str(metric.metadata.get("formula") or ""))
        if any(term in q for term in ["最多", "数量", "单量", "下单量", "订单量", "订单数", "下单数", "销量", "count"]):
            if "count(" in formula:
                score += 20
            if "distinct" in formula:
                score += 8
        confidence = metric.metadata.get("confidence")
        if isinstance(confidence, (int, float)):
            score += int(float(confidence) * 2)
        return score

    def _metric_dependency_keys(self, metric: PlanningAssetEntry, metrics_by_key: Dict[str, List[PlanningAssetEntry]]) -> List[str]:
        formula = str(metric.metadata.get("formula") or metric.metadata.get("metricFormula") or metric.description or "")
        deps: List[str] = []
        for ref in metric.metadata.get("sourceColumns") or metric.metadata.get("source_columns") or []:
            ref_key = str(ref or "")
            if ref_key and ref_key != metric.key and ref_key in metrics_by_key:
                deps.append(ref_key)
        for key in metrics_by_key:
            if key != metric.key and re.search(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(key), formula):
                deps.append(key)
        return sorted(set(deps), key=deps.index)

    def _reconcile_skills(self, skills: List[SkillManifest], pack: PlanningAssetPack) -> List[SkillManifest]:
        if not skills:
            return []
        gaps: List[Dict[str, Any]] = []
        reconciled: List[SkillManifest] = []
        for skill in skills:
            if skill.tables or skill.metrics or skill.relationships or skill.graph_patterns:
                gaps.append(
                    {
                        "domain": skill.domain,
                        "refType": "skillPolicy",
                        "reason": "skill fact references ignored; semantic layer and recall bundle are the only table/metric/relationship sources",
                    }
                )
            reconciled.append(self.skill_loader._thin_policy_skill(skill))
        pack.skill_semantic_gaps = gaps
        return reconciled

    def _append_table_assets(self, pack: PlanningAssetPack, topic: str, table: str) -> None:
        table_asset = self.topic_assets.load_table_asset(topic, table)
        schema = self.topic_assets.load_table_schema(topic, table)
        asset_schema = list(schema)
        semantic_columns = self.topic_assets.load_table_semantic_columns(topic, table)
        semantic_by_column = {str(item.get("columnName") or ""): item for item in semantic_columns if isinstance(item, dict)}
        schema_source = "asset"
        live_schema = self._live_schema(table)
        version = self._semantic_catalog_version(topic, table, asset_schema, live_schema)
        pack.semantic_catalog_version[table] = version
        if live_schema:
            pack.schema_drift_reports.append(self._schema_drift_report(topic, table, asset_schema, live_schema, version))
        if live_schema:
            live_cols = {str(col.get("Field") or col.get("columnName") or "") for col in live_schema}
            asset_cols = {str(col.get("columnName") or col.get("Field") or "") for col in schema}
            pack.missing_live_columns[table] = sorted(asset_cols - live_cols)
            schema = normalize_schema_rows(live_schema)
            schema_source = "live"
        columns = [str(col.get("columnName") or col.get("Field") or "") for col in schema if col.get("columnName") or col.get("Field")]
        pack.schema_source[table] = schema_source
        table_entry = PlanningAssetEntry(
            key=table,
            table=table,
            topic=topic,
            title=str(table_asset.get("tableComment") or table),
            columns=columns,
            aliases=[table, str(table_asset.get("tableComment") or ""), str(table_asset.get("manualNotes") or "")],
            description=json.dumps(compact_table_metadata(table_asset), ensure_ascii=False),
            source_ref_id="semantic:%s:%s:table" % (topic, table),
            metadata=table_asset,
        )
        pack.tables.append(table_entry)
        for col in self._field_rows_for_pack(schema, semantic_by_column):
            name = str(col.get("columnName") or col.get("Field") or "")
            if not name:
                continue
            semantic = semantic_by_column.get(name, {})
            comment = str(semantic.get("businessName") or semantic.get("description") or col.get("comment") or col.get("Comment") or name)
            aliases = [name, comment] + [str(alias) for alias in semantic.get("aliases") or []]
            pack.fields.append(
                PlanningAssetEntry(
                    key=name,
                    table=table,
                    topic=topic,
                    title=comment,
                    aliases=dedupe_strings(aliases),
                    description=json.dumps({"schema": col, "semantic": semantic}, ensure_ascii=False),
                    source_ref_id="semantic:%s:%s:field:%s" % (topic, table, name),
                    metadata={"schema": col, "semantic": semantic},
                )
            )
            if name.endswith("_id") or name in {"pt", "merchant_id", "seller_id", "sub_order_id", "order_id", "spu_id", "ticket_id", "refund_id"}:
                pack.entity_keys.append(
                    PlanningAssetEntry(
                        key=name,
                        table=table,
                        topic=topic,
                        title=name,
                        source_ref_id="semantic:%s:%s:key:%s" % (topic, table, name),
                    )
                )
        table_metrics = self.topic_assets.load_table_metrics(topic, table)[:120]
        table_metric_keys = {str(metric.get("metricKey") or "") for metric in table_metrics if str(metric.get("metricKey") or "")}
        live_column_set = set(columns)
        filtered_metric_count = 0
        for metric in table_metrics:
            key = str(metric.get("metricKey") or "")
            if not key:
                continue
            missing_metric_columns = self._metric_missing_live_columns(metric, live_column_set, table_metric_keys, pack.missing_live_columns.get(table, []))
            if missing_metric_columns:
                filtered_metric_count += 1
                filtered = pack.metric_compaction.setdefault("schemaFilteredMetrics", {})
                table_filtered = filtered.setdefault(table, [])
                if isinstance(table_filtered, list) and len(table_filtered) < 20:
                    table_filtered.append({"metricKey": key, "missingColumns": missing_metric_columns[:8]})
                continue
            pack.metrics.append(
                PlanningAssetEntry(
                    key=key,
                    table=table,
                    topic=topic,
                    title=str(metric.get("businessName") or key),
                    columns=[str(col) for col in metric.get("sourceColumns") or []],
                    aliases=[str(alias) for alias in metric.get("aliases") or []],
                    description=json.dumps(metric, ensure_ascii=False),
                    source_ref_id="semantic:%s:%s:metric:%s" % (topic, table, key),
                    metadata=metric,
                )
            )
        if filtered_metric_count:
            pack.metric_compaction.setdefault("schemaFilteredMetricCounts", {})[table] = filtered_metric_count
        for term in self.topic_assets.load_table_terms(topic, table)[:80]:
            pack.terms.append(
                PlanningAssetEntry(
                    key=str(term.get("term") or term.get("key") or ""),
                    table=table,
                    topic=topic,
                    title=str(term.get("businessName") or term.get("term") or ""),
                    description=json.dumps(term, ensure_ascii=False),
                    source_ref_id="semantic:%s:%s:term:%s" % (topic, table, term.get("term") or term.get("key") or ""),
                    metadata=term,
                )
            )

    def _field_rows_for_pack(
        self,
        schema: List[Dict[str, Any]],
        semantic_by_column: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        settings = self.topic_assets.settings
        limit = max(120, int(getattr(settings, "agent_asset_field_entry_limit", 240) or 240))
        rows_by_name = {
            str(row.get("columnName") or row.get("Field") or ""): row
            for row in schema
            if str(row.get("columnName") or row.get("Field") or "")
        }
        prioritized_names: List[str] = []
        for name, semantic in semantic_by_column.items():
            if name not in rows_by_name or not semantic:
                continue
            labels = [
                semantic.get("businessName"),
                semantic.get("description"),
                *(semantic.get("aliases") or []),
            ]
            if any(str(label or "").strip() and str(label or "").strip() != name for label in labels):
                prioritized_names.append(name)
        entity_names = [
            name
            for name in rows_by_name
            if name.endswith("_id")
            or name in {"pt", "merchant_id", "seller_id", "sub_order_id", "order_id", "spu_id", "ticket_id", "refund_id"}
        ]
        physical_names = list(rows_by_name.keys())
        ordered_names = dedupe_strings(prioritized_names + entity_names + physical_names)
        return [rows_by_name[name] for name in ordered_names[:limit] if name in rows_by_name]

    def _metric_missing_live_columns(
        self,
        metric: Dict[str, Any],
        live_columns: Set[str],
        table_metric_keys: Set[str],
        missing_live_columns: List[str],
    ) -> List[str]:
        if not live_columns or not missing_live_columns:
            return []
        refs = [
            str(item)
            for item in metric.get("sourceColumns") or metric.get("source_columns") or []
            if str(item or "")
        ]
        formula = str(metric.get("formula") or metric.get("metricFormula") or "")
        missing: List[str] = []
        for ref in refs:
            if ref in table_metric_keys:
                continue
            if ref not in live_columns:
                missing.append(ref)
        for column in missing_live_columns:
            if column in table_metric_keys or column in missing:
                continue
            if re.search(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(column), formula):
                missing.append(column)
        return sorted(set(missing))

    def _live_schema(self, table: str) -> List[Dict[str, Any]]:
        if table in self._live_schema_cache:
            return self._live_schema_cache[table]
        if not self.doris_repository:
            return []
        try:
            rows = self.doris_repository.show_full_columns(table)
            live_schema = rows if isinstance(rows, list) else []
            self._live_schema_cache[table] = live_schema
            return live_schema
        except Exception:
            return []

    def _semantic_catalog_version(
        self,
        topic: str,
        table: str,
        schema: List[Dict[str, Any]],
        live_schema: List[Dict[str, Any]],
    ) -> SemanticCatalogVersion:
        source_hash = self._semantic_source_hash(topic, table)
        schema_hash = hashlib.sha256(json.dumps(schema, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
        return SemanticCatalogVersion(
            semantic_version="semantic-%s" % (source_hash[:12] or "unknown"),
            schema_version="schema-%s" % schema_hash,
            topic=topic,
            table=table,
            source_hash=source_hash,
            published_at=self._semantic_published_at(topic, table),
            live_schema_checked_at=datetime.now().isoformat() if live_schema else "",
        )

    def _semantic_source_hash(self, topic: str, table: str) -> str:
        table_dir = self.topic_assets.root / topic / "tables" / table
        if not table_dir.exists():
            return ""
        hasher = hashlib.sha256()
        for path in sorted(table_dir.glob("*")):
            if path.is_file():
                try:
                    hasher.update(path.name.encode("utf-8"))
                    hasher.update(path.read_bytes())
                except Exception:
                    continue
        return hasher.hexdigest()

    def _semantic_published_at(self, topic: str, table: str) -> str:
        table_dir = self.topic_assets.root / topic / "tables" / table
        if not table_dir.exists():
            return ""
        mtimes = [path.stat().st_mtime for path in table_dir.glob("*") if path.is_file()]
        if not mtimes:
            return ""
        return datetime.fromtimestamp(max(mtimes)).isoformat()

    def _schema_drift_report(
        self,
        topic: str,
        table: str,
        semantic_schema: List[Dict[str, Any]],
        live_schema: List[Dict[str, Any]],
        version: SemanticCatalogVersion,
    ) -> SchemaDriftReport:
        semantic_cols = {
            str(item.get("columnName") or item.get("Field") or ""): item
            for item in semantic_schema
            if str(item.get("columnName") or item.get("Field") or "")
        }
        live_cols = {
            str(item.get("Field") or item.get("columnName") or ""): item
            for item in normalize_schema_rows(live_schema)
            if str(item.get("Field") or item.get("columnName") or "")
        }
        type_changed = []
        for name in sorted(set(semantic_cols) & set(live_cols)):
            semantic_type = normalize_column_type(semantic_cols[name])
            live_type = normalize_column_type(live_cols[name])
            if semantic_type and live_type and semantic_type != live_type:
                type_changed.append({"column": name, "semanticType": semantic_type, "liveType": live_type})
        return SchemaDriftReport(
            topic=topic,
            table=table,
            semantic_version=version.semantic_version,
            schema_version=version.schema_version,
            source_hash=version.source_hash,
            live_schema_checked_at=version.live_schema_checked_at,
            missing_live_columns=sorted(set(semantic_cols) - set(live_cols)),
            extra_live_columns=sorted(set(live_cols) - set(semantic_cols)),
            type_changed_columns=type_changed,
            live_column_count=len(live_cols),
            semantic_column_count=len(semantic_cols),
        )

    def _table_topic_index(self) -> Dict[str, str]:
        index: Dict[str, str] = {}
        for topic in self.topic_assets.all_topic_names():
            for manifest_item in self.topic_assets.load_manifest(topic):
                table = str(manifest_item.get("tableName") or "")
                if table and table not in index:
                    index[table] = topic
        return index

    def _all_relationships(self) -> List[Tuple[str, Dict[str, Any]]]:
        relationships: List[Tuple[str, Dict[str, Any]]] = []
        for topic in self.topic_assets.all_topic_names():
            for rel in self.topic_assets.load_relationships(topic):
                relationships.append((topic, rel))
        return relationships

    def _tables_for_metric_request(
        self,
        metric_ref: str,
        owner_table: str = "",
        source_phrase: str = "",
        metric_compaction: Dict[str, Any] | None = None,
    ) -> List[str]:
        all_metrics = self._all_metric_entries()
        scoped_tables = sorted(
            {
                str(item.get("ownerTable") or "")
                for item in (metric_compaction or {}).get("recalledMetricEvidence") or []
                if isinstance(item, dict)
                and (
                    recalled_evidence_scoped_to_phrase(item, source_phrase)
                    or recalled_metric_evidence_matches_phrase(item, source_phrase)
                )
            }
        )
        if scoped_tables:
            return [table for table in scoped_tables if table]
        if owner_table and any(metric.table == owner_table and metric.key == metric_ref for metric in all_metrics):
            return [owner_table]
        if owner_table:
            return [owner_table]
        exact_tables = sorted({metric.table for metric in all_metrics if metric.key == metric_ref})
        return exact_tables if len(exact_tables) == 1 else []

    def _append_relationships_for_tables(
        self,
        pack: PlanningAssetPack,
        pack_tables: Set[str],
        table_topic: Dict[str, str],
    ) -> List[str]:
        existing_ids = {item.relationship_id for item in pack.relationships if item.relationship_id}
        relationship_topics = {table_topic.get(table, "") for table in pack_tables}
        relationship_topics.discard("")
        traces: List[str] = []
        for topic, rel in self._all_relationships():
            if topic not in relationship_topics:
                continue
            left = str(rel.get("leftTable") or "")
            right = str(rel.get("rightTable") or "")
            rel_id = str(rel.get("name") or "")
            if left not in pack_tables or right not in pack_tables or rel_id in existing_ids:
                continue
            entry = relationship_entry(topic, rel)
            if not entry.relationship_id:
                continue
            pack.relationships.append(entry)
            existing_ids.add(entry.relationship_id)
            traces.append("metric_request_relationship:%s:%s-%s" % (entry.relationship_id, left, right))
        return traces

    def _table_relevant(self, question: str, manifest_item: Dict[str, Any]) -> bool:
        text = (question or "").lower()
        payload = json.dumps(manifest_item, ensure_ascii=False).lower()
        return any(token in payload for token in recall_terms(text, []))

    def _seed_tables_for_topic(self, question: str, topic: str, allow_profile: bool = False) -> Set[str]:
        manifest = self.topic_assets.load_manifest(topic)
        relevant = [
            str(item.get("tableName") or "")
            for item in manifest
            if self._table_relevant(question, item) and self._table_allowed_for_question(question, str(item.get("tableName") or ""), allow_profile=allow_profile)
        ]
        if relevant:
            return {table for table in relevant if table}
        return {
            table
            for table in [str(item.get("tableName") or "") for item in manifest]
            if table and self._table_allowed_for_question(question, table, allow_profile=allow_profile)
        }

    def _table_allowed_for_question(self, question: str, table: str, allow_profile: bool = False) -> bool:
        if not table:
            return False
        if "profile" in table.lower():
            return allow_profile
        return True


def relationship_entry(topic: str, rel: Dict[str, Any]) -> RelationshipEntry:
    keys = []
    for pair in rel.get("keys") or []:
        if isinstance(pair, list) and len(pair) >= 2:
            keys.append({"leftColumn": str(pair[0]), "rightColumn": str(pair[1])})
    path_semantics = [
        str(item)
        for item in rel.get("pathSemantics")
        or rel.get("path_semantics")
        or infer_relationship_path_semantics(rel, keys)
        if str(item or "").strip()
    ]
    return RelationshipEntry(
        relationship_id=str(rel.get("name") or ""),
        left_table=str(rel.get("leftTable") or ""),
        right_table=str(rel.get("rightTable") or ""),
        join_keys=keys,
        grain=str(rel.get("grain") or ""),
        path_semantics=path_semantics,
        use_cases=[str(item) for item in rel.get("useCases") or rel.get("use_cases") or [] if str(item or "").strip()],
        cautions=[str(item) for item in rel.get("cautions") or [] if str(item or "").strip()],
        source_ref_id="semantic:%s:relationship:%s" % (topic, rel.get("name") or ""),
        description=json.dumps(rel, ensure_ascii=False),
    )


def infer_relationship_path_semantics(rel: Dict[str, Any], keys: List[Dict[str, str]]) -> List[str]:
    columns = {
        str(key.get("leftColumn") or "").lower()
        for key in keys
    } | {
        str(key.get("rightColumn") or "").lower()
        for key in keys
    }
    grain = str(rel.get("grain") or "").lower()
    semantics: List[str] = []
    if columns <= {"seller_id", "merchant_id"}:
        semantics.append("tenant_context")
    if {"sub_order_id", "order_id"} & columns or "order" in grain:
        semantics.append("order_entity")
    if {"spu_id", "spu_name"} & columns or "spu" in grain or "product" in grain:
        semantics.append("product_entity")
    if "ticket_id" in columns or "ticket" in grain:
        semantics.append("ticket_entity")
    if "refund_id" in columns or "refund" in grain:
        semantics.append("refund_entity")
    if {"coupon_id", "discount_rel_id", "discount_id"} & columns or "coupon" in grain:
        semantics.append("coupon_entity")
    if {"bill_id", "repay_id"} & columns or "bill" in grain or "repay" in grain:
        semantics.append("compensation_entity")
    if "tenant_context" not in semantics:
        semantics.append("entity_filter")
    return list(dict.fromkeys(semantics))


def recalled_relationship_tables(item: RecallItem) -> List[str]:
    metadata = item.metadata or {}
    tables: List[str] = []
    for key in ["leftTable", "rightTable", "tableName"]:
        value = str(metadata.get(key) or "")
        if value and value not in tables:
            tables.append(value)
    if item.table and item.table not in tables:
        tables.append(item.table)
    return tables


def dedupe_entries_by_identity(entries: List[PlanningAssetEntry]) -> List[PlanningAssetEntry]:
    selected: List[PlanningAssetEntry] = []
    seen: Set[Tuple[str, str, str]] = set()
    for entry in entries:
        identity = (entry.topic, entry.table, entry.key)
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(entry)
    return selected


def semantic_table_ref_id(topic: str, table: str) -> str:
    return "semantic:%s:%s:asset" % (topic, table)


def semantic_manifest_ref_id(topic: str) -> str:
    return "semantic:%s:manifest" % topic


def semantic_relationship_ref_id(topic: str) -> str:
    return "semantic:%s:relationships" % topic


def semantic_manifest_path(topic: str) -> str:
    return "topics/%s/manifest.json" % topic


def semantic_table_path(topic: str, table: str) -> str:
    return "topics/%s/tables/%s/asset.json" % (topic, table)


def semantic_relationship_path(topic: str) -> str:
    return "topics/%s/relationships.json" % topic


def normalize_semantic_path(path: str) -> str:
    text = str(path or "").strip().lstrip("/")
    if not text:
        return ""
    if text.startswith("runtime/"):
        text = text[len("runtime/") :]
    if text.startswith("resources/runtime/"):
        text = text[len("resources/runtime/") :]
    return text


def sanitize_semantic_file_name(file_name: str) -> str:
    text = str(file_name or "proposal.md").strip().replace("\\", "_").replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]+", "_", text)
    return text or "proposal.md"


def dedupe_strings(values: Iterable[str]) -> List[str]:
    deduped: List[str] = []
    seen: Set[str] = set()
    for value in values:
        text = str(value or "")
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def normalize_schema_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for row in rows:
        name = str(row.get("columnName") or row.get("Field") or "")
        if not name:
            continue
        normalized.append(
            {
                "columnName": name,
                "type": str(row.get("type") or row.get("Type") or ""),
                "comment": str(row.get("comment") or row.get("Comment") or ""),
            }
        )
    return normalized


def normalize_column_type(row: Dict[str, Any]) -> str:
    raw = str(row.get("type") or row.get("Type") or row.get("dataType") or row.get("columnType") or "").strip().lower()
    if not raw:
        return ""
    raw = re.sub(r"\s+", " ", raw)
    raw = re.sub(r"\(.+?\)", "", raw)
    return raw.strip()


def question_understanding_metric_requests(understanding: Dict[str, Any]) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    ranking = understanding.get("rankingObjective") or understanding.get("ranking_objective") or {}
    if isinstance(ranking, dict) and (ranking.get("metricRef") or ranking.get("metric_ref") or ranking.get("ownerTable") or ranking.get("owner_table")):
        requests.append(ranking)
    measures = understanding.get("requestedMeasures") or understanding.get("requested_measures") or []
    if isinstance(measures, list):
        requests.extend(item for item in measures if isinstance(item, dict))
    evidence_items = understanding.get("requiredEvidenceIntents") or understanding.get("required_evidence_intents") or []
    if isinstance(evidence_items, list):
        for item in evidence_items:
            if not isinstance(item, dict):
                continue
            for metric_ref in item.get("suggestedMetricRefs") or item.get("suggested_metric_refs") or []:
                if metric_ref:
                    requests.append(
                        {
                            "metricRef": str(metric_ref),
                            "sourcePhrase": str(item.get("semanticLabel") or item.get("semantic_label") or ""),
                        }
                    )
    deduped: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for item in requests:
        identity = (
            str(item.get("metricRef") or item.get("metric_ref") or ""),
            str(item.get("ownerTable") or item.get("owner_table") or ""),
        )
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(item)
    return deduped


class SemanticMetricCandidateScore:
    def __init__(
        self,
        metric: PlanningAssetEntry,
        requested_metric_ref: str = "",
        source_phrase: str = "",
        ref_score: int = 0,
        phrase_score: int = 0,
        owner_table_match: bool = False,
        rank_score: int = 0,
        resolution_reason: str = "",
        recall_evidence: Dict[str, Any] | None = None,
    ):
        self.metric = metric
        self.requested_metric_ref = requested_metric_ref
        self.source_phrase = source_phrase
        self.ref_score = ref_score
        self.phrase_score = phrase_score
        self.owner_table_match = owner_table_match
        self.rank_score = rank_score
        self.resolution_reason = resolution_reason
        self.recall_evidence = recall_evidence or {}

    def payload(self) -> Dict[str, Any]:
        return {
            "metricKey": self.metric.key,
            "ownerTable": self.metric.table,
            "displayName": self.metric.title,
            "refScore": self.ref_score,
            "phraseScore": self.phrase_score,
            "ownerTableMatch": self.owner_table_match,
            "rankScore": self.rank_score,
            "resolutionReason": self.resolution_reason,
            "semanticRefId": self.metric.source_ref_id,
            "recallEvidence": self.recall_evidence,
        }


class SemanticMetricIndex:
    """Single semantic-layer index for sourcePhrase/metricRef -> metric binding."""

    PHRASE_OVERRIDE_MIN_SCORE = 18
    PHRASE_OVERRIDE_MARGIN = 8
    MIN_ACCEPT_SCORE = 20

    def __init__(self, metrics: Iterable[PlanningAssetEntry]):
        self.metrics = [metric for metric in metrics if metric and metric.key and metric.table]

    def resolve(self, metric_ref: str, owner_table: str = "", source_phrase: str = "") -> SemanticMetricCandidateScore | None:
        candidates = self.candidates(metric_ref, owner_table, source_phrase)
        if not candidates:
            return None
        direct = self._direct_candidate(candidates, metric_ref, owner_table)
        phrase_best = max(candidates, key=lambda item: item.phrase_score)
        if (
            source_phrase
            and phrase_best.phrase_score >= self.PHRASE_OVERRIDE_MIN_SCORE
            and (
                not direct
                or direct.ref_score < 40
                or metric_phrase_directly_names_metric(phrase_best.metric, source_phrase)
            )
            and not (
                direct
                and direct.ref_score >= 40
                and direct.phrase_score >= self.PHRASE_OVERRIDE_MIN_SCORE
                and phrase_best.metric.table != direct.metric.table
            )
            and (not direct or phrase_best.phrase_score >= direct.phrase_score + self.PHRASE_OVERRIDE_MARGIN)
        ):
            phrase_best.resolution_reason = "semantic_phrase_override" if direct else "semantic_phrase_match"
            return phrase_best
        if direct:
            direct.resolution_reason = "semantic_metric_ref"
            return direct
        best = candidates[0]
        if best.rank_score >= self.MIN_ACCEPT_SCORE:
            best.resolution_reason = "semantic_alias" if best.phrase_score >= self.PHRASE_OVERRIDE_MIN_SCORE else "semantic_weak_match"
            return best
        return None

    def candidates(self, metric_ref: str, owner_table: str = "", source_phrase: str = "") -> List[SemanticMetricCandidateScore]:
        requested = str(metric_ref or "").strip()
        normalized_ref = normalize_for_match(requested)
        scores: List[SemanticMetricCandidateScore] = []
        for metric in self.metrics:
            ref_score = metric_ref_match_score(metric, normalized_ref)
            phrase_score = metric_phrase_match_score(metric, source_phrase)
            owner_match = bool(owner_table and metric.table == owner_table)
            owner_bonus = 8 if owner_match else 0
            rank_score = ref_score + phrase_score + owner_bonus
            if rank_score <= 0:
                continue
            scores.append(
                SemanticMetricCandidateScore(
                    metric=metric,
                    requested_metric_ref=requested,
                    source_phrase=str(source_phrase or ""),
                    ref_score=ref_score,
                    phrase_score=phrase_score,
                    owner_table_match=owner_match,
                    rank_score=rank_score,
                )
            )
        scores.sort(key=lambda item: (item.rank_score, item.phrase_score, item.ref_score), reverse=True)
        return scores

    def _direct_candidate(
        self,
        candidates: List[SemanticMetricCandidateScore],
        metric_ref: str,
        owner_table: str = "",
    ) -> SemanticMetricCandidateScore | None:
        normalized_ref = normalize_for_match(metric_ref)
        if not normalized_ref:
            return None
        direct: List[SemanticMetricCandidateScore] = []
        for candidate in candidates:
            metric = candidate.metric
            if normalize_for_match(metric.key) != normalized_ref:
                continue
            if owner_table and metric.table != owner_table:
                continue
            direct.append(candidate)
        if not direct:
            return None
        direct.sort(key=lambda item: (item.owner_table_match, item.ref_score, item.rank_score), reverse=True)
        return direct[0]


def metric_request_match_score(metric: PlanningAssetEntry, normalized_ref: str, normalized_phrase: str) -> int:
    if not normalized_ref and not normalized_phrase:
        return 0
    return metric_ref_match_score(metric, normalized_ref) + metric_phrase_match_score(metric, normalized_phrase)


def metric_ref_match_score(metric: PlanningAssetEntry, normalized_ref: str) -> int:
    if not normalized_ref:
        return 0
    metadata = metric.metadata or {}
    names = [
        metric.key,
        metric.title,
        metric.source_ref_id,
        *metric.aliases,
        *[str(alias) for alias in metadata.get("aliases") or []],
        str(metadata.get("businessName") or ""),
    ]
    columns = [str(column) for column in metadata.get("sourceColumns") or metadata.get("source_columns") or metric.columns]
    normalized_names = {normalize_for_match(item) for item in names if item}
    normalized_columns = {normalize_for_match(item) for item in columns if item}
    text = normalize_for_match(
        " ".join(
            [
                metric.key,
                metric.title,
                " ".join(metric.aliases),
                metric.description,
                json.dumps(metadata, ensure_ascii=False),
            ]
        )
    )
    score = 0
    if normalized_ref in normalized_names:
        score += 40
    elif normalized_ref in normalized_columns:
        score += 30
    elif normalized_ref in text:
        score += 18
    return score


def metric_phrase_match_score(metric: PlanningAssetEntry, phrase: str) -> int:
    terms = metric_phrase_terms(phrase)
    if not terms:
        return 0
    metadata = metric.metadata or {}
    names = [
        metric.key,
        metric.title,
        *metric.aliases,
        *[str(alias) for alias in metadata.get("aliases") or []],
        str(metadata.get("businessName") or ""),
    ]
    columns = [str(column) for column in metadata.get("sourceColumns") or metadata.get("source_columns") or metric.columns]
    normalized_names = {normalize_for_match(item) for item in names if item}
    normalized_columns = {normalize_for_match(item) for item in columns if item}
    text = normalize_for_match(
        " ".join(
            [
                metric.key,
                metric.title,
                " ".join(metric.aliases),
                metric.description,
                json.dumps(metadata, ensure_ascii=False),
            ]
        )
    )
    score = 0
    for term in terms:
        if not term:
            continue
        if term in normalized_names:
            score += 18
        elif term in normalized_columns:
            score += 14
        elif term in text:
            score += 8
    return min(score, 60)


def metric_phrase_terms(phrase: str) -> List[str]:
    raw = str(phrase or "")
    terms: List[str] = []
    for term in question_match_terms(raw):
        normalized = normalize_for_match(term)
        if len(normalized) < 2:
            continue
        if normalized.isdigit():
            continue
        if normalized in {"最高", "最低", "最多", "最少", "前5", "前10", "top", "top5", "top10"}:
            continue
        if normalized not in terms:
            terms.append(normalized)
    whole = normalize_for_match(raw)
    if whole and whole not in terms:
        terms.append(whole)
    return terms


def metric_phrase_directly_names_metric(metric: PlanningAssetEntry, phrase: str) -> bool:
    normalized_phrase = normalize_for_match(phrase)
    if not normalized_phrase:
        return False
    metadata = metric.metadata or {}
    labels = [
        metric.key,
        metric.title,
        str(metadata.get("businessName") or ""),
        *metric.aliases,
        *[str(alias) for alias in metadata.get("aliases") or []],
    ]
    for label in labels:
        normalized_label = normalize_for_match(label)
        if is_strong_label_text_match(normalized_label, normalized_phrase):
            return True
        for token in re.findall(r"[A-Za-z0-9_]{3,}", str(phrase or "").lower()):
            if token and token in normalized_label:
                return True
    return False


def is_strong_label_text_match(normalized_label: str, normalized_phrase: str) -> bool:
    if not normalized_label or not normalized_phrase:
        return False
    if re.search(r"[a-z0-9]", normalized_label):
        return len(normalized_label) >= 3 and normalized_label in normalized_phrase
    if len(normalized_label) >= 4 and normalized_label in normalized_phrase:
        return True
    return normalized_label == normalized_phrase


class TopicBuilderWorkflow:
    def __init__(
        self,
        settings: Settings,
        doris_repository: DorisRepository,
        topic_assets: TopicAssetService,
        llm: Optional[Any] = None,
    ):
        self.settings = settings
        self.doris_repository = doris_repository
        self.topic_assets = topic_assets
        self.llm = llm or LlmClient(settings)

    def build(self, request: TopicBuildRequest) -> Dict[str, Any]:
        topic = request.topic or "经营画像"
        table = request.table_name
        if not table:
            return {"success": False, "message": "tableName is required"}
        pending_dir = self.settings.resolved_topic_path / topic / "pending" / table
        pending_dir.mkdir(parents=True, exist_ok=True)
        existing = self._existing_asset_context(topic, table)
        schema = self._load_schema(table, request)
        sample_rows = self._load_sample_rows(table, request)
        profile = self._sample_profile(schema, sample_rows, request)
        generated = self._generate_candidate_payload(topic, table, request, schema, sample_rows, profile, existing)
        builder_phases = {
            "schemaDiscovery": {
                "status": "completed",
                "artifact": str(pending_dir / "schema.json"),
                "columnCount": len(schema),
            },
            "sampleProfiling": {
                "status": "completed",
                "artifact": str(pending_dir / "sample_profile.json"),
                "sampleRowCount": len(sample_rows),
            },
            "semanticAnalysis": {
                "status": "completed",
                "mode": str(generated.get("generationMode") or "heuristic"),
                "artifact": str(pending_dir / "asset.json"),
                "llmUsed": str(generated.get("generationMode") or "").lower() == "llm",
            },
            "humanReviewPublish": {
                "status": "pending_review",
                "pendingPath": str(pending_dir),
                "publishContract": "approved review publishes scoped asset, refreshes semantic catalog, recall index, and caches",
            },
        }
        asset_payload = {
            "topic": topic,
            "tableName": table,
            "tableComment": str(generated.get("tableComment") or existing.get("tableComment") or ""),
            "dataGrain": str(generated.get("dataGrain") or existing.get("dataGrain") or self._infer_data_grain(schema)),
            "timeColumn": str(generated.get("timeColumn") or existing.get("timeColumn") or profile.get("timeColumn") or ""),
            "merchantFilterColumn": str(
                generated.get("merchantFilterColumn") or existing.get("merchantFilterColumn") or profile.get("merchantFilterColumn") or ""
            ),
            "rowAccessPolicy": normalize_row_access_policy(
                generated.get("rowAccessPolicy")
                or existing.get("rowAccessPolicy")
                or default_row_access_policy(
                    str(generated.get("merchantFilterColumn") or existing.get("merchantFilterColumn") or profile.get("merchantFilterColumn") or "")
                )
            ),
            "manualNotes": request.manual_notes or str(existing.get("manualNotes") or ""),
            "businessKnowledge": request.business_knowledge or str(existing.get("businessKnowledge") or ""),
            "sampleSqls": request.sample_sqls or list(existing.get("sampleSqls") or []),
            "buildProfile": profile,
            "builderPhases": builder_phases,
            "generationMode": str(generated.get("generationMode") or "heuristic"),
            "generatedAt": datetime.utcnow().isoformat() + "Z",
            "status": "PENDING_REVIEW",
        }
        semantic_columns = self._merge_generated_list(
            existing.get("semanticColumns"),
            generated.get("semanticColumns"),
            "semanticColumns",
            schema_columns=schema,
        )
        metrics = self._merge_generated_metrics(existing.get("metrics"), generated.get("metrics"), schema)
        terms = self._merge_generated_list(existing.get("terms"), generated.get("terms"), "terms")
        rules = self._merge_generated_list(existing.get("knowledgeRules"), generated.get("knowledgeRules"), "knowledgeRules")
        production_report = topic_asset_production_report(
            topic=topic,
            table=table,
            schema=schema,
            sample_rows=sample_rows,
            profile=profile,
            semantic_columns=semantic_columns,
            metrics=metrics,
            terms=terms,
            rules=rules,
            builder_phases=builder_phases,
            generation_mode=asset_payload["generationMode"],
        )
        asset_payload["assetProductionReport"] = {
            "artifact": str(pending_dir / "asset_production_report.json"),
            "status": production_report["status"],
            "qualityScore": production_report["qualityScore"],
        }
        asset_payload["semanticGovernance"] = semantic_governance_envelope(
            topic,
            table,
            owner=request.merchant_id or "semantic_asset_owner",
            stage="candidate",
            status="PENDING_REVIEW",
        )
        asset_payload["approvalWorkflow"] = semantic_approval_workflow(
            stage="pending_review",
            reviewer="",
            review_note="",
            publishable=False,
        )
        asset_payload["semanticLineage"] = semantic_asset_lineage(topic, table, asset_payload, metrics, rules)
        write_json(pending_dir / "schema.json", schema if isinstance(schema, list) else [])
        write_json(pending_dir / "sample_rows.json", sample_rows)
        write_json(pending_dir / "sample_profile.json", profile)
        write_json(pending_dir / "asset.json", asset_payload)
        write_json(pending_dir / "semantic_columns.json", semantic_columns)
        write_json(pending_dir / "metrics.json", metrics)
        write_json(pending_dir / "terms.json", terms)
        write_json(pending_dir / "knowledge_rules.json", rules)
        write_json(pending_dir / "asset_production_report.json", production_report)
        return {
            "success": True,
            "status": "PENDING_REVIEW",
            "topic": topic,
            "tableName": table,
            "path": str(pending_dir),
            "generationMode": asset_payload["generationMode"],
            "schemaColumnCount": len(schema),
            "sampleRowCount": len(sample_rows),
            "metricCount": len(metrics),
            "fieldCount": len(semantic_columns),
            "builderPhases": builder_phases,
            "assetProductionReport": production_report,
            "semanticGovernance": asset_payload["semanticGovernance"],
            "approvalWorkflow": asset_payload["approvalWorkflow"],
        }

    def diff_schema(self, request: TopicBuildRequest) -> Dict[str, Any]:
        topic = request.topic or "经营画像"
        table = request.table_name
        existing = self.topic_assets.load_table_schema(topic, table)
        live = []
        try:
            live = self.doris_repository.show_full_columns(table)
        except Exception:
            pass
        existing_cols = {str(item.get("columnName") or item.get("Field") or "") for item in existing}
        live_cols = {str(item.get("columnName") or item.get("Field") or "") for item in live}
        builder = PlanningAssetPackBuilder(self.topic_assets, doris_repository=self.doris_repository)
        version = builder._semantic_catalog_version(topic, table, existing, live)
        drift = builder._schema_drift_report(topic, table, existing, live, version) if live else SchemaDriftReport(
            topic=topic,
            table=table,
            semantic_version=version.semantic_version,
            schema_version=version.schema_version,
            source_hash=version.source_hash,
            semantic_column_count=len(existing_cols),
        )
        review_path = self.settings.resolved_workspace_path / "schema_drift" / topic / ("%s.schema-diff-review.json" % table)
        payload = {
            "success": True,
            "topic": topic,
            "tableName": table,
            "added": sorted(live_cols - existing_cols),
            "removed": sorted(existing_cols - live_cols),
            "semanticCatalogVersion": version.model_dump(by_alias=True),
            "schemaDriftReport": drift.model_dump(by_alias=True),
            "reviewArtifact": str(review_path),
        }
        write_json(review_path, payload)
        return payload

    def refresh_incremental(self, request: TopicBuildRequest) -> Dict[str, Any]:
        diff = self.diff_schema(request)
        built = self.build(request)
        return {**built, "schemaDiff": diff}

    def build_batch(self, requests: List[TopicBuildRequest]) -> Dict[str, Any]:
        results = []
        success_count = 0
        for request in requests or []:
            result = self.build(request)
            results.append(result)
            if result.get("success"):
                success_count += 1
        report = {
            "success": success_count == len(requests or []),
            "status": "BATCH_BUILT",
            "requestedCount": len(requests or []),
            "successCount": success_count,
            "failedCount": len(requests or []) - success_count,
            "results": results,
            "factoryReport": {
                "mode": "topic_asset_factory",
                "phases": ["schemaDiscovery", "sampleProfiling", "semanticAnalysis", "humanReviewPublish"],
                "generatedAt": datetime.utcnow().isoformat() + "Z",
            },
        }
        path = self.settings.resolved_workspace_path / "topic_builder" / "batch-build-report.json"
        write_json(path, report)
        report["reportPath"] = str(path)
        return report

    def _existing_asset_context(self, topic: str, table: str) -> Dict[str, Any]:
        for directory in [
            self.settings.resolved_topic_path / topic / "pending" / table,
            self.topic_assets.table_asset_dir(topic, table),
        ]:
            if directory.exists():
                return self._load_asset_dir(directory, topic, table)
        return {}

    def _load_asset_dir(self, directory: Path, topic: str, table: str) -> Dict[str, Any]:
        asset = read_json(directory / "asset.json")
        payload: Dict[str, Any] = asset if isinstance(asset, dict) else {}
        payload.setdefault("topic", topic)
        payload.setdefault("tableName", table)
        for field, file_name in {
            "schemaColumns": "schema.json",
            "semanticColumns": "semantic_columns.json",
            "metrics": "metrics.json",
            "terms": "terms.json",
            "knowledgeRules": "knowledge_rules.json",
        }.items():
            sidecar = read_json(directory / file_name)
            if sidecar:
                payload[field] = merge_semantic_layer_list(payload.get(field), sidecar, field)
            else:
                payload.setdefault(field, [])
        return payload

    def _load_schema(self, table: str, request: TopicBuildRequest) -> List[Dict[str, Any]]:
        providers = ["show_full_columns", "describe_table", "datamap_columns"]
        schema: Any = []
        for name in providers:
            provider = getattr(self.doris_repository, name, None)
            if not callable(provider):
                continue
            try:
                schema = provider(table)
            except Exception:
                schema = []
            if isinstance(schema, list) and schema:
                break
        if (not schema) and request.schema_ddl:
            schema = [
                {"columnName": line.split()[0].strip("`,"), "comment": line}
                for line in request.schema_ddl.splitlines()
                if line.strip()
            ]
        return [self._normalize_schema_column(item) for item in (schema if isinstance(schema, list) else [])]

    def _load_sample_rows(self, table: str, request: TopicBuildRequest) -> List[Dict[str, Any]]:
        provider = getattr(self.doris_repository, "sample_rows", None)
        if not callable(provider):
            return []
        try:
            rows = provider(table, request.merchant_id or self.settings.merchant_id, max(1, int(request.sample_limit or 20)))
        except Exception:
            return []
        normalized: List[Dict[str, Any]] = []
        for row in rows[: max(1, int(request.sample_limit or 20))]:
            if not isinstance(row, dict):
                continue
            normalized.append({str(key): self._json_safe(value) for key, value in row.items()})
        return normalized

    def _sample_profile(self, schema: List[Dict[str, Any]], rows: List[Dict[str, Any]], request: TopicBuildRequest) -> Dict[str, Any]:
        columns = [str(item.get("columnName") or "") for item in schema if str(item.get("columnName") or "")]
        enum_limit = max(1, int(request.enum_value_limit or 20))
        null_rates: Dict[str, float] = {}
        sample_values: Dict[str, List[Any]] = {}
        enum_candidates: Dict[str, List[Any]] = {}
        for column in columns:
            values = [row.get(column) for row in rows if isinstance(row, dict)]
            total = len(values)
            missing = sum(1 for value in values if value in {None, ""})
            null_rates[column] = round((missing / total), 4) if total else 0.0
            observed: List[Any] = []
            for value in values:
                if value in {None, ""}:
                    continue
                if value not in observed:
                    observed.append(value)
                if len(observed) >= enum_limit:
                    break
            sample_values[column] = observed[:8]
            unique_values = []
            for value in values:
                if value in {None, ""}:
                    continue
                if value not in unique_values:
                    unique_values.append(value)
                if len(unique_values) > enum_limit:
                    break
            if request.enum_discovery_enabled and 0 < len(unique_values) <= enum_limit and self._enum_candidate(column, unique_values):
                enum_candidates[column] = unique_values
        partition_candidates = [
            column
            for column in columns
            if column.lower() in {"pt", "dt", "ds", "biz_date"} or normalize_column_type_family(str(next((item.get("dataType") for item in schema if item.get("columnName") == column), ""))) in {"date", "datetime"}
        ]
        time_column = next((column for column in partition_candidates if column.lower() == "pt"), partition_candidates[0] if partition_candidates else "")
        merchant_column = next((column for column in columns if column.lower() in {"seller_id", "merchant_id", "shop_id"}), "")
        return {
            "rowCount": len(rows),
            "nullRates": null_rates,
            "sampleValues": sample_values,
            "enumCandidates": enum_candidates,
            "partitionColumns": partition_candidates,
            "timeColumn": time_column,
            "merchantFilterColumn": merchant_column,
        }

    def _generate_candidate_payload(
        self,
        topic: str,
        table: str,
        request: TopicBuildRequest,
        schema: List[Dict[str, Any]],
        sample_rows: List[Dict[str, Any]],
        profile: Dict[str, Any],
        existing: Dict[str, Any],
    ) -> Dict[str, Any]:
        heuristic = self._heuristic_candidate_payload(topic, table, request, schema, sample_rows, profile, existing)
        llm_payload = self._llm_candidate_payload(topic, table, request, schema, sample_rows, profile, existing, heuristic)
        if llm_payload:
            llm_payload["semanticColumns"] = merge_semantic_layer_list(
                heuristic.get("semanticColumns") or [],
                llm_payload.get("semanticColumns") or [],
                "semanticColumns",
            )
            llm_payload["metrics"] = merge_semantic_layer_list(
                heuristic.get("metrics") or [],
                llm_payload.get("metrics") or [],
                "metrics",
            )
            llm_payload["terms"] = merge_semantic_layer_list(
                heuristic.get("terms") or [],
                llm_payload.get("terms") or [],
                "terms",
            )
            llm_payload["knowledgeRules"] = merge_semantic_layer_list(
                heuristic.get("knowledgeRules") or [],
                llm_payload.get("knowledgeRules") or [],
                "knowledgeRules",
            )
            llm_payload["generationMode"] = "llm"
            return llm_payload
        heuristic["generationMode"] = "heuristic"
        return heuristic

    def _llm_candidate_payload(
        self,
        topic: str,
        table: str,
        request: TopicBuildRequest,
        schema: List[Dict[str, Any]],
        sample_rows: List[Dict[str, Any]],
        profile: Dict[str, Any],
        existing: Dict[str, Any],
        heuristic: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not bool(getattr(self.llm, "configured", False)):
            return {}
        prompt_payload = {
            "topic": topic,
            "tableName": table,
            "schemaColumns": schema,
            "sampleRows": sample_rows[: min(len(sample_rows), 12)],
            "sampleProfile": profile,
            "manualNotes": request.manual_notes,
            "businessKnowledge": request.business_knowledge,
            "sampleSqls": request.sample_sqls[:8],
            "existingAsset": {
                "tableComment": existing.get("tableComment"),
                "dataGrain": existing.get("dataGrain"),
                "timeColumn": existing.get("timeColumn"),
                "merchantFilterColumn": existing.get("merchantFilterColumn"),
                "rowAccessPolicy": existing.get("rowAccessPolicy") or {},
                "semanticColumns": existing.get("semanticColumns") or [],
                "metrics": existing.get("metrics") or [],
                "terms": existing.get("terms") or [],
                "knowledgeRules": existing.get("knowledgeRules") or [],
            },
            "heuristicDraft": heuristic,
        }
        system_prompt = (
            "你是资深数据语义建模助手。请基于表 schema、采样数据、业务备注和历史 SQL，"
            "生成待审核的语义层候选资产。输出必须保守、结构化，不要编造不存在的字段，"
            "不要生成跨表 join 关系。字段角色只允许 KEY、TIME、DIMENSION、ATTRIBUTE。"
            "指标必须引用真实存在的 sourceColumns；派生指标可以引用其他 metricKey。"
            "如果字段疑似手机号、邮箱、身份证、地址等敏感信息，请补充 visibilityPolicy 和 maskingPolicy；"
            "如果表有明确租户过滤列，请补充 rowAccessPolicy。"
        )
        try:
            payload = self.llm.tool_json_chat(system_prompt, json.dumps(prompt_payload, ensure_ascii=False), semantic_asset_builder_tool().openai_schema(), {})
        except TypeError:
            payload = self.llm.tool_json_chat(system_prompt, json.dumps(prompt_payload, ensure_ascii=False), semantic_asset_builder_tool().openai_schema(), {})
        return payload if isinstance(payload, dict) else {}

    def _heuristic_candidate_payload(
        self,
        topic: str,
        table: str,
        request: TopicBuildRequest,
        schema: List[Dict[str, Any]],
        sample_rows: List[Dict[str, Any]],
        profile: Dict[str, Any],
        existing: Dict[str, Any],
    ) -> Dict[str, Any]:
        semantic_columns = [self._heuristic_semantic_column(topic, item, profile) for item in schema]
        metrics = self._heuristic_metrics(topic, table, schema, profile)
        terms = self._heuristic_terms(metrics, semantic_columns, request)
        rules = self._heuristic_rules(topic, profile, request)
        return {
            "tableComment": str(existing.get("tableComment") or request.manual_notes or table),
            "dataGrain": str(existing.get("dataGrain") or self._infer_data_grain(schema)),
            "timeColumn": str(existing.get("timeColumn") or profile.get("timeColumn") or ""),
            "merchantFilterColumn": str(existing.get("merchantFilterColumn") or profile.get("merchantFilterColumn") or ""),
            "rowAccessPolicy": normalize_row_access_policy(existing.get("rowAccessPolicy") or default_row_access_policy(str(existing.get("merchantFilterColumn") or profile.get("merchantFilterColumn") or ""))),
            "semanticColumns": semantic_columns,
            "metrics": metrics,
            "terms": terms,
            "knowledgeRules": rules,
        }

    def _heuristic_semantic_column(self, topic: str, column: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
        name = str(column.get("columnName") or "")
        comment = str(column.get("comment") or column.get("Comment") or name)
        role = "ATTRIBUTE"
        lowered = name.lower()
        if lowered.endswith("_id") or lowered in {"id", "seller_id", "merchant_id", "shop_id"}:
            role = "KEY"
        elif lowered in {"pt", "dt", "ds"} or normalize_column_type_family(str(column.get("dataType") or "")) in {"date", "datetime"}:
            role = "TIME"
        elif lowered.endswith(("_name", "_status", "_type")) or lowered.startswith(("is_", "has_")) or lowered.endswith("_code"):
            role = "DIMENSION"
        aliases = dedupe_strings([name, comment])
        enum_values = [str(item) for item in (profile.get("enumCandidates", {}).get(name) or [])[:20]]
        sample_values = [str(item) for item in (profile.get("sampleValues", {}).get(name) or [])[:8]]
        evidence = ["schema comment=%s" % comment]
        if sample_values:
            evidence.append("samples=[%s]" % ", ".join(sample_values))
        visibility_policy, masking_policy = sensitive_column_policies(name, comment)
        display_policy = default_column_display_policy(name, role, visibility_policy)
        return {
            "columnName": name,
            "businessName": comment,
            "role": role,
            "description": comment,
            "aliases": aliases,
            **display_policy,
            "enumValues": enum_values,
            "sampleValues": sample_values,
            "confidence": 0.62,
            "evidence": "; ".join(evidence),
            "visibilityPolicy": visibility_policy,
            "maskingPolicy": masking_policy,
        }

    def _heuristic_metrics(self, topic: str, table: str, schema: List[Dict[str, Any]], profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        metrics: List[Dict[str, Any]] = []
        for item in schema:
            name = str(item.get("columnName") or "")
            if not name:
                continue
            dtype = normalize_column_type_family(str(item.get("dataType") or ""))
            comment = str(item.get("comment") or item.get("Comment") or name)
            lowered = name.lower()
            if dtype in {"int", "bigint", "float", "double", "decimal"} and not lowered.endswith("_id"):
                formula = "SUM(%s)" % name
                unit = "%" if "rate" in lowered else ("元" if any(token in lowered for token in ["amt", "amount", "price", "fee"]) else "")
                metrics.append(
                    {
                        "metricKey": name,
                        "canonicalMetricKey": name,
                        "businessName": comment,
                        "formula": formula,
                        "unit": unit,
                        "description": comment,
                        "sourceColumns": [name],
                        "aliases": dedupe_strings([comment, name]),
                        "confidence": 0.58,
                        "evidence": "heuristic numeric column metric",
                    }
                )
            if lowered in {"refund_id", "order_id", "sub_order_id", "ticket_id", "bill_id"}:
                metric_key = self._count_metric_key_for_id(topic, lowered)
                business_name = self._count_metric_name_for_id(topic, comment, lowered)
                metrics.append(
                    {
                        "metricKey": metric_key,
                        "businessName": business_name,
                        "formula": "COUNT(DISTINCT %s)" % name,
                        "unit": "单",
                        "description": "按%s去重计数" % comment,
                        "sourceColumns": [name],
                        "aliases": dedupe_strings([business_name, comment, metric_key]),
                        "confidence": 0.6,
                        "evidence": "heuristic id distinct count",
                    }
                )
        if not metrics and schema:
            first_key = next((str(item.get("columnName") or "") for item in schema if str(item.get("columnName") or "").endswith("_id")), "")
            if first_key:
                metrics.append(
                    {
                        "metricKey": "row_cnt",
                        "businessName": "记录数",
                        "formula": "COUNT(DISTINCT %s)" % first_key,
                        "unit": "条",
                        "description": "按主键候选去重计数",
                        "sourceColumns": [first_key],
                        "aliases": ["记录数", "条数", "row_cnt"],
                        "confidence": 0.45,
                        "evidence": "heuristic fallback metric",
                    }
                )
        return dedupe_semantic_items(metrics, "metrics")

    def _heuristic_terms(self, metrics: List[Dict[str, Any]], semantic_columns: List[Dict[str, Any]], request: TopicBuildRequest) -> List[Dict[str, Any]]:
        terms: List[Dict[str, Any]] = []
        for metric in metrics[:40]:
            key = str(metric.get("metricKey") or "")
            title = str(metric.get("businessName") or key)
            if not key or not title:
                continue
            terms.append(
                {
                    "term": title,
                    "businessName": title,
                    "description": str(metric.get("description") or title),
                    "aliases": dedupe_strings([title] + [str(alias) for alias in metric.get("aliases") or []]),
                    "canonicalMetricKey": key,
                }
            )
        for column in semantic_columns[:40]:
            title = str(column.get("businessName") or "")
            if not title:
                continue
            terms.append(
                {
                    "term": title,
                    "businessName": title,
                    "description": str(column.get("description") or title),
                    "aliases": [title] + [str(alias) for alias in column.get("aliases") or []],
                }
            )
        if request.business_knowledge:
            terms.append(
                {
                    "term": "业务补充说明",
                    "businessName": "业务补充说明",
                    "description": request.business_knowledge[:500],
                    "aliases": ["业务知识", "业务说明"],
                }
            )
        return dedupe_semantic_items(terms, "terms")

    def _heuristic_rules(self, topic: str, profile: Dict[str, Any], request: TopicBuildRequest) -> List[Dict[str, Any]]:
        rules: List[Dict[str, Any]] = []
        time_column = str(profile.get("timeColumn") or "")
        merchant_column = str(profile.get("merchantFilterColumn") or "")
        if time_column:
            rules.append(
                {
                    "ruleId": "time_partition_rule",
                    "title": "时间过滤建议",
                    "description": "该表建议优先使用 %s 作为时间过滤字段" % time_column,
                    "aliases": ["时间过滤", "分区字段"],
                    "appliesToColumns": [time_column],
                }
            )
        if merchant_column:
            rules.append(
                {
                    "ruleId": "merchant_scope_rule",
                    "title": "商家范围约束",
                    "description": "运行时查询建议使用 %s 作为商家过滤字段" % merchant_column,
                    "aliases": ["商家过滤", "租户过滤"],
                    "appliesToColumns": [merchant_column],
                }
            )
        sensitive_columns = [
            column
            for column in (profile.get("sampleValues", {}) or {}).keys()
            if normalize_visibility_level(sensitive_column_policies(str(column))[0].get("level")) == "restricted"
        ]
        if sensitive_columns:
            rules.append(
                {
                    "ruleId": "column_access_policy_rule",
                    "title": "敏感字段访问约束",
                    "description": "敏感字段默认不直接暴露给 AI 结果，必要时按语义层脱敏策略输出",
                    "aliases": ["字段权限", "脱敏规则"],
                    "appliesToColumns": sensitive_columns[:12],
                    "cautions": ["restricted columns require visibilityPolicy and maskingPolicy review"],
                }
            )
        if request.business_knowledge:
            rules.append(
                {
                    "ruleId": "business_knowledge_note",
                    "title": "业务知识候选",
                    "description": request.business_knowledge[:1000],
                    "aliases": ["业务规则", "人工补充"],
                }
            )
        return dedupe_semantic_items(rules, "knowledgeRules")

    def _merge_generated_list(
        self,
        existing: Any,
        generated: Any,
        field: str,
        schema_columns: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        base = existing if isinstance(existing, list) else []
        override = generated if isinstance(generated, list) else []
        merged = merge_semantic_layer_list(base, override, field)
        if field == "semanticColumns" and schema_columns is not None:
            valid = {str(item.get("columnName") or "") for item in schema_columns if str(item.get("columnName") or "")}
            return [
                item for item in merged
                if isinstance(item, dict) and str(item.get("columnName") or "") in valid
            ]
        return [item for item in merged if isinstance(item, dict)]

    def _merge_generated_metrics(self, existing: Any, generated: Any, schema: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid = column_name_set(schema)
        merged = merge_semantic_layer_list(existing if isinstance(existing, list) else [], generated if isinstance(generated, list) else [], "metrics")
        result: List[Dict[str, Any]] = []
        for metric in merged:
            if not isinstance(metric, dict):
                continue
            refs = semantic_metric_source_columns(metric)
            if refs and any((ref not in valid) and not metric_shaped_reference(ref) for ref in refs):
                continue
            result.append(metric)
        return dedupe_semantic_items(result, "metrics")

    def _normalize_schema_column(self, item: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(item, dict):
            return {}
        return {
            "columnName": str(item.get("columnName") or item.get("Field") or item.get("name") or ""),
            "dataType": str(item.get("dataType") or item.get("Type") or item.get("type") or ""),
            "comment": str(item.get("comment") or item.get("Comment") or ""),
            "nullable": self._normalize_nullable(item),
            "keyType": str(item.get("keyType") or item.get("Key") or ""),
        }

    def _normalize_nullable(self, item: Dict[str, Any]) -> bool:
        raw = item.get("nullable")
        if isinstance(raw, bool):
            return raw
        null_value = str(item.get("Null") or "").strip().upper()
        if null_value in {"YES", "TRUE", "Y"}:
            return True
        if null_value in {"NO", "FALSE", "N"}:
            return False
        return True

    def _json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        return str(value)

    def _enum_candidate(self, column: str, values: List[Any]) -> bool:
        lowered = column.lower()
        if lowered.endswith(("_status", "_status_name", "_type", "_type_name", "_code", "_name")):
            return True
        if len(values) <= 8:
            return True
        return False

    def _infer_data_grain(self, schema: List[Dict[str, Any]]) -> str:
        columns = {str(item.get("columnName") or "").lower() for item in schema}
        if "refund_id" in columns:
            return "退款/售后明细粒度"
        if "sub_order_id" in columns:
            return "子订单明细粒度"
        if "order_id" in columns:
            return "订单明细粒度"
        if "spu_id" in columns:
            return "商品粒度"
        if "seller_id" in columns:
            return "商家粒度"
        return "明细粒度"

    def _count_metric_key_for_id(self, topic: str, column: str) -> str:
        mapping = {
            "refund_id": "refund_bill_cnt",
            "sub_order_id": "order_detail_cnt",
            "order_id": "order_cnt",
            "ticket_id": "ticket_cnt",
            "bill_id": "bill_cnt",
        }
        return mapping.get(column, "%s_cnt" % column.removesuffix("_id"))

    def _count_metric_name_for_id(self, topic: str, comment: str, column: str) -> str:
        mapping = {
            "refund_id": "退款单量" if "退" in topic else "单量",
            "sub_order_id": "订单量",
            "order_id": "订单量",
            "ticket_id": "工单量",
            "bill_id": "账单量",
        }
        return mapping.get(column, "%s数量" % comment)


def semantic_asset_builder_tool() -> AgentToolDefinition:
    visibility_policy_schema = {
        "type": "object",
        "properties": {
            "level": {"type": "string", "enum": ["public", "restricted", "hidden"]},
            "allowedRoles": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
        "additionalProperties": False,
    }
    masking_policy_schema = {
        "type": "object",
        "properties": {
            "strategy": {"type": "string", "enum": ["none", "partial", "full", "hash"]},
            "reason": {"type": "string"},
        },
        "additionalProperties": False,
    }
    row_access_policy_schema = {
        "type": "object",
        "properties": {
            "scopeType": {"type": "string"},
            "filterColumn": {"type": "string"},
            "operator": {"type": "string"},
            "valueSource": {"type": "string"},
            "required": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "additionalProperties": False,
    }
    semantic_column_schema = {
        "type": "object",
        "properties": {
            "columnName": {"type": "string"},
            "businessName": {"type": "string"},
            "role": {"type": "string", "enum": ["KEY", "TIME", "DIMENSION", "ATTRIBUTE"]},
            "description": {"type": "string"},
            "aliases": {"type": "array", "items": {"type": "string"}},
            "enumValues": {"type": "array", "items": {"type": "string"}},
            "sampleValues": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
            "evidence": {"type": "string"},
            "defaultVisible": {"type": "boolean"},
            "displayPriority": {"type": "integer"},
            "displayScenarios": {"type": "array", "items": {"type": "string"}},
            "visibilityPolicy": visibility_policy_schema,
            "maskingPolicy": masking_policy_schema,
        },
        "required": ["columnName", "businessName", "role"],
        "additionalProperties": False,
    }
    metric_schema = {
        "type": "object",
        "properties": {
            "metricKey": {"type": "string"},
            "canonicalMetricKey": {"type": "string"},
            "aliasOf": {"type": "string"},
            "businessName": {"type": "string"},
            "formula": {"type": "string"},
            "unit": {"type": "string"},
            "description": {"type": "string"},
            "sourceColumns": {"type": "array", "items": {"type": "string"}},
            "aliases": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
            "evidence": {"type": "string"},
        },
        "required": ["metricKey", "businessName", "formula", "sourceColumns"],
        "additionalProperties": False,
    }
    term_schema = {
        "type": "object",
        "properties": {
            "term": {"type": "string"},
            "businessName": {"type": "string"},
            "description": {"type": "string"},
            "aliases": {"type": "array", "items": {"type": "string"}},
            "canonicalMetricKey": {"type": "string"},
        },
        "required": ["term", "businessName"],
        "additionalProperties": False,
    }
    rule_schema = {
        "type": "object",
        "properties": {
            "ruleId": {"type": "string"},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "aliases": {"type": "array", "items": {"type": "string"}},
            "appliesToMetrics": {"type": "array", "items": {"type": "string"}},
            "appliesToColumns": {"type": "array", "items": {"type": "string"}},
            "cautions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "description"],
        "additionalProperties": False,
    }
    return AgentToolDefinition(
        name="propose_semantic_asset",
        description="Generate candidate semantic-layer assets from schema, sample rows, and business notes.",
        parameters={
            "type": "object",
            "properties": {
                "tableComment": {"type": "string"},
                "dataGrain": {"type": "string"},
                "timeColumn": {"type": "string"},
                "merchantFilterColumn": {"type": "string"},
                "rowAccessPolicy": row_access_policy_schema,
                "semanticColumns": {"type": "array", "items": semantic_column_schema},
                "metrics": {"type": "array", "items": metric_schema},
                "terms": {"type": "array", "items": term_schema},
                "knowledgeRules": {"type": "array", "items": rule_schema},
            },
            "required": ["semanticColumns", "metrics", "terms", "knowledgeRules"],
            "additionalProperties": False,
        },
    )


def dedupe_semantic_items(items: List[Dict[str, Any]], field: str) -> List[Dict[str, Any]]:
    return merge_semantic_layer_list([], items, field) if items else []


def normalize_visibility_level(level: str) -> str:
    text = str(level or "").strip().lower()
    if text in {"public", "restricted", "hidden"}:
        return text
    return "public"


def normalize_masking_strategy(strategy: str) -> str:
    text = str(strategy or "").strip().lower()
    if text in {"none", "partial", "full", "hash"}:
        return text
    return "none"


def normalize_visibility_policy(policy: Any) -> Dict[str, Any]:
    if not isinstance(policy, dict):
        return {"level": "public", "allowedRoles": [], "reason": ""}
    return {
        "level": normalize_visibility_level(str(policy.get("level") or "")),
        "allowedRoles": dedupe_strings([str(item) for item in policy.get("allowedRoles") or []]),
        "reason": str(policy.get("reason") or ""),
    }


def normalize_masking_policy(policy: Any) -> Dict[str, Any]:
    if not isinstance(policy, dict):
        return {"strategy": "none", "reason": ""}
    return {
        "strategy": normalize_masking_strategy(str(policy.get("strategy") or "")),
        "reason": str(policy.get("reason") or ""),
    }


def normalize_column_display_policy(semantic: Any) -> Dict[str, Any]:
    if not isinstance(semantic, dict):
        return {"defaultVisible": False, "displayPriority": 1000, "displayScenarios": []}
    scenarios = dedupe_strings([str(item).strip() for item in semantic.get("displayScenarios") or [] if str(item or "").strip()])
    try:
        priority = int(semantic.get("displayPriority"))
    except (TypeError, ValueError):
        priority = 1000
    return {
        "defaultVisible": bool(semantic.get("defaultVisible", False)),
        "displayPriority": priority,
        "displayScenarios": scenarios,
    }


def normalize_row_access_policy(policy: Any) -> Dict[str, Any]:
    if not isinstance(policy, dict):
        return {}
    filter_column = str(policy.get("filterColumn") or "")
    if not filter_column:
        return {}
    return {
        "scopeType": str(policy.get("scopeType") or "merchant"),
        "filterColumn": filter_column,
        "operator": str(policy.get("operator") or "eq"),
        "valueSource": str(policy.get("valueSource") or "merchant_id"),
        "required": bool(policy.get("required", True)),
        "reason": str(policy.get("reason") or ""),
    }


def default_row_access_policy(filter_column: str) -> Dict[str, Any]:
    column = str(filter_column or "").strip()
    if not column:
        return {}
    return {
        "scopeType": "merchant",
        "filterColumn": column,
        "operator": "eq",
        "valueSource": "merchant_id",
        "required": True,
        "reason": "tenant isolation filter managed by semantic layer",
    }


def sensitive_column_policies(column_name: str, business_name: str = "") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    name = str(column_name or "").lower()
    label = str(business_name or "").lower()
    text = "%s %s" % (name, label)
    restricted_roles = ["merchant_admin", "security_auditor"]
    full_mask_tokens = ["phone", "mobile", "tel", "telephone", "id_card", "bank_card", "身份证", "银行卡"]
    partial_mask_tokens = ["email", "mail", "address", "地址", "buyer_name", "user_name", "receiver_name", "consignee_name"]
    if any(token in text for token in full_mask_tokens):
        return (
            {"level": "restricted", "allowedRoles": restricted_roles, "reason": "contains direct personal identifier"},
            {"strategy": "full", "reason": "hide direct personal identifier in AI results"},
        )
    if any(token in text for token in partial_mask_tokens):
        return (
            {"level": "restricted", "allowedRoles": restricted_roles, "reason": "contains personal profile field"},
            {"strategy": "partial", "reason": "mask personal profile field in AI results"},
        )
    return ({"level": "public", "allowedRoles": [], "reason": ""}, {"strategy": "none", "reason": ""})


def default_column_display_policy(column_name: str, role: str, visibility_policy: Dict[str, Any]) -> Dict[str, Any]:
    name = str(column_name or "").lower()
    if normalize_visibility_level(str(visibility_policy.get("level") or "")) != "public":
        return {"defaultVisible": False, "displayPriority": 1000, "displayScenarios": []}
    priority_map = {
        "seller_id": 10,
        "merchant_id": 11,
        "order_id": 20,
        "sub_order_id": 21,
        "spu_id": 30,
        "spu_name": 31,
        "refund_id": 40,
        "ticket_id": 50,
        "bill_id": 60,
        "coupon_id": 70,
        "discount_rel_id": 71,
        "pay_amt": 80,
        "repay_amt": 81,
        "pt": 90,
    }
    priority = priority_map.get(name)
    default_visible = priority is not None or str(role or "").upper() in {"KEY", "TIME"}
    return {
        "defaultVisible": default_visible,
        "displayPriority": priority if priority is not None else 500,
        "displayScenarios": ["detail"] if default_visible else [],
    }


class SemanticAssetGovernanceService:
    def __init__(self, settings: Settings, doris_repository: DorisRepository, topic_assets: TopicAssetService):
        self.settings = settings
        self.doris_repository = doris_repository
        self.topic_assets = topic_assets

    def preflight_publish(self, topic: str, table: str) -> Dict[str, Any]:
        pending_dir = self.settings.resolved_topic_path / topic / "pending" / table
        if not pending_dir.exists():
            return {"success": False, "publishable": False, "status": "NOT_FOUND", "topic": topic, "tableName": table}
        asset = self._load_asset_dir(pending_dir, topic, table)
        semantic_schema = schema_columns(asset)
        live_schema = self._live_schema(table)
        builder = PlanningAssetPackBuilder(self.topic_assets, doris_repository=self.doris_repository)
        version = builder._semantic_catalog_version(topic, table, semantic_schema, live_schema)
        drift = (
            builder._schema_drift_report(topic, table, semantic_schema, live_schema, version)
            if live_schema
            else SchemaDriftReport(
                topic=topic,
                table=table,
                semantic_version=version.semantic_version,
                schema_version=version.schema_version,
                source_hash=version.source_hash,
                semantic_column_count=len(column_name_set(semantic_schema)),
            )
        )
        validation = validate_semantic_asset(asset, self.topic_assets.load_relationships(topic))
        drift_gate = schema_drift_release_gate(drift)
        validation_gate = semantic_validation_gate(validation)
        release_gate = combine_release_gates(validation_gate, drift_gate)
        impact_plan = semantic_asset_impact_test_plan(topic, table, asset, drift)
        rollback_snapshot = self._create_rollback_snapshot(topic, table)
        conflict_detection = semantic_conflict_detection(asset)
        evaluation_gate = semantic_release_evaluation_gate(asset, validation, drift, conflict_detection)
        release_gate = combine_release_gates(release_gate, evaluation_gate)
        owner = semantic_asset_owner(asset)
        payload = {
            "success": True,
            "publishable": release_gate["publishable"],
            "status": "PREFLIGHT_PASSED" if release_gate["publishable"] else "PREFLIGHT_FAILED",
            "topic": topic,
            "tableName": table,
            "semanticGovernance": semantic_governance_envelope(
                topic,
                table,
                owner=owner,
                stage="preflight",
                status="PREFLIGHT_PASSED" if release_gate["publishable"] else "PREFLIGHT_FAILED",
                semantic_version=version.semantic_version,
            ),
            "approvalWorkflow": semantic_approval_workflow(
                stage="preflight",
                reviewer="",
                review_note="",
                publishable=bool(release_gate["publishable"]),
            ),
            "grayReleasePlan": semantic_gray_release_plan(topic, table, release_gate),
            "semanticLineage": semantic_asset_lineage(topic, table, asset, asset.get("metrics") if isinstance(asset.get("metrics"), list) else [], asset.get("knowledgeRules") if isinstance(asset.get("knowledgeRules"), list) else []),
            "conflictDetection": conflict_detection,
            "conflictRepairPlan": semantic_conflict_repair_plan(conflict_detection),
            "evaluationGate": evaluation_gate,
            "semanticCatalogVersion": version.model_dump(by_alias=True),
            "schemaDriftReport": drift.model_dump(by_alias=True),
            "driftGovernance": drift_gate,
            "validation": validation,
            "releaseGate": release_gate,
            "impactTestPlan": impact_plan,
            "rollbackCandidate": semantic_rollback_candidate(self.topic_assets.table_asset_dir(topic, table)),
            "rollbackSnapshot": rollback_snapshot,
        }
        payload["reviewArtifact"] = str(self._write_governance_artifact(topic, table, "preflight", version.semantic_version, payload))
        return payload

    def after_publish(self, topic: str, table: str, reviewer: str = "", review_note: str = "") -> Dict[str, Any]:
        target_dir = self.topic_assets.table_asset_dir(topic, table)
        rollback_candidate = semantic_rollback_candidate(target_dir)
        asset = self._load_asset_dir(target_dir, topic, table)
        semantic_schema = schema_columns(asset)
        live_schema = self._live_schema(table)
        builder = PlanningAssetPackBuilder(self.topic_assets, doris_repository=self.doris_repository)
        version = builder._semantic_catalog_version(topic, table, semantic_schema, live_schema)
        drift = (
            builder._schema_drift_report(topic, table, semantic_schema, live_schema, version)
            if live_schema
            else SchemaDriftReport(
                topic=topic,
                table=table,
                semantic_version=version.semantic_version,
                schema_version=version.schema_version,
                source_hash=version.source_hash,
                semantic_column_count=len(column_name_set(semantic_schema)),
            )
        )
        version_payload = {
            **version.model_dump(by_alias=True),
            "owner": semantic_asset_owner(asset),
            "lifecycleStatus": "active",
            "reviewer": reviewer,
            "reviewNote": review_note,
            "publishedAt": datetime.utcnow().isoformat() + "Z",
        }
        write_json(target_dir / "semantic_version.json", version_payload)
        manifest = self._write_version_manifest(topic, table, version_payload)
        drift_gate = schema_drift_release_gate(drift)
        validation = validate_semantic_asset(asset, self.topic_assets.load_relationships(topic))
        validation_gate = semantic_validation_gate(validation)
        release_gate = combine_release_gates(validation_gate, drift_gate)
        impact_plan = semantic_asset_impact_test_plan(topic, table, asset, drift)
        conflict_detection = semantic_conflict_detection(asset)
        evaluation_gate = semantic_release_evaluation_gate(asset, validation, drift, conflict_detection)
        release_gate = combine_release_gates(release_gate, evaluation_gate)
        payload = {
            "success": True,
            "status": "GOVERNED_PUBLISHED",
            "topic": topic,
            "tableName": table,
            "semanticGovernance": semantic_governance_envelope(
                topic,
                table,
                owner=version_payload["owner"],
                stage="published",
                status="ACTIVE",
                semantic_version=version.semantic_version,
            ),
            "approvalWorkflow": semantic_approval_workflow(
                stage="published",
                reviewer=reviewer,
                review_note=review_note,
                publishable=bool(release_gate["publishable"]),
            ),
            "grayReleasePlan": semantic_gray_release_plan(topic, table, release_gate),
            "grayReleaseMonitor": semantic_gray_release_monitor(topic, table, release_gate),
            "semanticLineage": semantic_asset_lineage(topic, table, asset, asset.get("metrics") if isinstance(asset.get("metrics"), list) else [], asset.get("knowledgeRules") if isinstance(asset.get("knowledgeRules"), list) else []),
            "conflictDetection": conflict_detection,
            "conflictRepairPlan": semantic_conflict_repair_plan(conflict_detection),
            "evaluationGate": evaluation_gate,
            "semanticCatalogVersion": version_payload,
            "schemaDriftReport": drift.model_dump(by_alias=True),
            "driftGovernance": drift_gate,
            "validation": validation,
            "releaseGate": release_gate,
            "impactTestPlan": impact_plan,
            "rollbackCandidate": rollback_candidate,
            "rollbackSnapshot": self._latest_rollback_snapshot(topic, table),
            "versionManifest": manifest,
            "publishMode": "scoped_incremental",
            "publishScope": {"topic": topic, "table": table, "mode": "scoped_incremental"},
            "cachePolicy": "recall index manager does scoped rebuild and clears recall, asset pack, live schema, and Doris query caches after publish",
        }
        payload["reviewArtifact"] = str(self._write_governance_artifact(topic, table, "publish", version.semantic_version, payload))
        payload["publishHistoryPath"] = str(append_semantic_publish_history(target_dir, payload))
        return payload

    def impact_analysis(self, topic: str, table: str) -> Dict[str, Any]:
        target_dir = self.topic_assets.table_asset_dir(topic, table)
        if not target_dir.exists():
            return {"success": False, "status": "NOT_FOUND", "topic": topic, "tableName": table}
        asset = self._load_asset_dir(target_dir, topic, table)
        semantic_schema = schema_columns(asset)
        live_schema = self._live_schema(table)
        builder = PlanningAssetPackBuilder(self.topic_assets, doris_repository=self.doris_repository)
        version = builder._semantic_catalog_version(topic, table, semantic_schema, live_schema)
        drift = (
            builder._schema_drift_report(topic, table, semantic_schema, live_schema, version)
            if live_schema
            else SchemaDriftReport(topic=topic, table=table, semantic_version=version.semantic_version, schema_version=version.schema_version, source_hash=version.source_hash)
        )
        metrics = asset.get("metrics") if isinstance(asset.get("metrics"), list) else []
        changed_columns = set(drift.missing_live_columns or []) | {str(item.get("column") or "") for item in drift.type_changed_columns or [] if isinstance(item, dict)}
        impacted_metrics = []
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            source_columns = semantic_metric_source_columns(metric)
            if changed_columns & set(source_columns):
                impacted_metrics.append(
                    {
                        "metricKey": metric.get("metricKey") or metric.get("key") or "",
                        "sourceColumns": source_columns,
                        "impactedColumns": sorted(changed_columns & set(source_columns)),
                    }
                )
        return {
            "success": True,
            "status": "IMPACT_ANALYZED",
            "topic": topic,
            "tableName": table,
            "semanticCatalogVersion": version.model_dump(by_alias=True),
            "schemaDriftReport": drift.model_dump(by_alias=True),
            "impactTestPlan": semantic_asset_impact_test_plan(topic, table, asset, drift),
            "impactedMetrics": impacted_metrics,
            "impactCount": len(impacted_metrics),
        }

    def rollback(self, topic: str, table: str, version: str = "", reviewer: str = "", reason: str = "") -> Dict[str, Any]:
        target_dir = self.topic_assets.table_asset_dir(topic, table)
        snapshot = self._find_rollback_snapshot(topic, table, version)
        if not snapshot:
            return {"success": False, "status": "ROLLBACK_SNAPSHOT_NOT_FOUND", "topic": topic, "tableName": table, "semanticVersion": version}
        snapshot_dir = Path(snapshot["path"])
        restored: List[str] = []
        for name in sorted(TopicAssetService.MANAGED_TABLE_FILENAMES | TopicAssetService.GOVERNANCE_FILENAMES):
            source = snapshot_dir / name
            target = target_dir / name
            if source.exists() and source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
                restored.append(name)
        self.topic_assets._table_asset_cache.pop((topic, table), None)
        self.topic_assets._manifest_cache.pop(topic, None)
        payload = {
            "success": True,
            "status": "ROLLED_BACK",
            "topic": topic,
            "tableName": table,
            "semanticVersion": snapshot.get("semanticVersion") or "",
            "restoredFiles": restored,
            "reviewer": reviewer,
            "reason": reason,
            "rolledBackAt": datetime.utcnow().isoformat() + "Z",
        }
        payload["rollbackArtifact"] = str(self._write_governance_artifact(topic, table, "rollback", payload["semanticVersion"] or "unknown", payload))
        append_semantic_publish_history(target_dir, payload)
        return payload

    def _load_asset_dir(self, directory: Path, topic: str, table: str) -> Dict[str, Any]:
        asset = read_json(directory / "asset.json")
        payload: Dict[str, Any] = asset if isinstance(asset, dict) else {}
        payload.setdefault("topic", topic)
        payload.setdefault("tableName", table)
        for field, file_name in {
            "schemaColumns": "schema.json",
            "semanticColumns": "semantic_columns.json",
            "metrics": "metrics.json",
            "terms": "terms.json",
            "knowledgeRules": "knowledge_rules.json",
        }.items():
            sidecar = read_json(directory / file_name)
            if sidecar:
                payload[field] = merge_semantic_layer_list(payload.get(field), sidecar, field)
            else:
                payload.setdefault(field, [])
        return payload

    def _live_schema(self, table: str) -> List[Dict[str, Any]]:
        try:
            live = self.doris_repository.show_full_columns(table)
            return live if isinstance(live, list) else []
        except Exception:
            return []

    def _write_governance_artifact(self, topic: str, table: str, stage: str, version: str, payload: Dict[str, Any]) -> Path:
        path = self.settings.resolved_workspace_path / "semantic_governance" / topic / table / ("%s-%s.json" % (stage, version or "unknown"))
        write_json(path, payload)
        return path

    def _write_version_manifest(self, topic: str, table: str, version_payload: Dict[str, Any]) -> Dict[str, Any]:
        target_dir = self.topic_assets.table_asset_dir(topic, table)
        manifest = {
            "topic": topic,
            "tableName": table,
            "activeVersion": version_payload.get("semanticVersion") or version_payload.get("semantic_version") or "",
            "schemaVersion": version_payload.get("schemaVersion") or version_payload.get("schema_version") or "",
            "sourceHash": version_payload.get("sourceHash") or version_payload.get("source_hash") or "",
            "publishedAt": version_payload.get("publishedAt") or "",
            "rollbackCandidate": semantic_rollback_candidate(target_dir),
        }
        write_json(target_dir / "semantic_version_manifest.json", manifest)
        return manifest

    def _create_rollback_snapshot(self, topic: str, table: str) -> Dict[str, Any]:
        target_dir = self.topic_assets.table_asset_dir(topic, table)
        if not target_dir.exists():
            return {}
        current = semantic_rollback_candidate(target_dir)
        version = current.get("semanticVersion") or "unversioned_%s" % datetime.utcnow().strftime("%Y%m%d%H%M%S")
        snapshot_dir = self.settings.resolved_workspace_path / "semantic_governance" / topic / table / "rollback_snapshots" / sanitize_asset_path_part(version)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        copied: List[str] = []
        for name in sorted(TopicAssetService.MANAGED_TABLE_FILENAMES | TopicAssetService.GOVERNANCE_FILENAMES):
            source = target_dir / name
            if source.exists() and source.is_file():
                shutil.copy2(source, snapshot_dir / name)
                copied.append(name)
        payload = {
            "semanticVersion": version,
            "path": str(snapshot_dir),
            "createdAt": datetime.utcnow().isoformat() + "Z",
            "files": copied,
        }
        write_json(snapshot_dir / "snapshot.json", payload)
        return payload

    def _latest_rollback_snapshot(self, topic: str, table: str) -> Dict[str, Any]:
        root = self.settings.resolved_workspace_path / "semantic_governance" / topic / table / "rollback_snapshots"
        snapshots = sorted(root.glob("*/snapshot.json"), key=lambda item: item.stat().st_mtime, reverse=True) if root.exists() else []
        for path in snapshots:
            payload = read_json(path)
            if isinstance(payload, dict):
                return payload
        return {}

    def _find_rollback_snapshot(self, topic: str, table: str, version: str = "") -> Dict[str, Any]:
        if not version:
            return self._latest_rollback_snapshot(topic, table)
        root = self.settings.resolved_workspace_path / "semantic_governance" / topic / table / "rollback_snapshots"
        path = root / sanitize_asset_path_part(version) / "snapshot.json"
        payload = read_json(path)
        return payload if isinstance(payload, dict) else {}


def validate_semantic_asset(asset: Dict[str, Any], relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
    table = str(asset.get("tableName") or "")
    schema = schema_columns(asset)
    columns = column_name_set(schema)
    metrics = asset.get("metrics") if isinstance(asset.get("metrics"), list) else []
    semantic_columns = asset.get("semanticColumns") if isinstance(asset.get("semanticColumns"), list) else []
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    row_access_policy = normalize_row_access_policy(asset.get("rowAccessPolicy") or {})
    if row_access_policy and str(row_access_policy.get("filterColumn") or "") not in columns:
        errors.append({"code": "ROW_ACCESS_FILTER_COLUMN_MISSING", "column": row_access_policy.get("filterColumn")})
    seen_metrics: set[str] = set()
    metric_keys: set[str] = {
        str(metric.get("metricKey") or metric.get("key") or "").strip()
        for metric in metrics
        if isinstance(metric, dict) and str(metric.get("metricKey") or metric.get("key") or "").strip()
    }
    for field in semantic_columns:
        if not isinstance(field, dict):
            continue
        column_name = str(field.get("columnName") or "").strip()
        if column_name and column_name not in columns:
            warnings.append({"code": "SEMANTIC_COLUMN_NOT_IN_SCHEMA", "column": column_name})
        visibility = normalize_visibility_policy(field.get("visibilityPolicy") or {})
        masking = normalize_masking_policy(field.get("maskingPolicy") or {})
        if visibility.get("level") == "restricted" and not visibility.get("allowedRoles"):
            warnings.append({"code": "RESTRICTED_COLUMN_WITHOUT_ALLOWED_ROLES", "column": column_name})
        if visibility.get("level") == "restricted" and masking.get("strategy") == "none":
            warnings.append({"code": "RESTRICTED_COLUMN_WITHOUT_MASKING", "column": column_name})
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        metric_key = str(metric.get("metricKey") or metric.get("key") or "").strip()
        if not metric_key:
            errors.append({"code": "MISSING_METRIC_KEY", "metric": metric})
            continue
        if metric_key in seen_metrics:
            errors.append({"code": "DUPLICATE_METRIC_KEY", "metricKey": metric_key})
        seen_metrics.add(metric_key)
        for column in semantic_metric_source_columns(metric):
            if column in metric_keys:
                continue
            if column not in columns:
                if external_metric_dependency(metric, column):
                    warnings.append({"code": "EXTERNAL_METRIC_DEPENDENCY", "metricKey": metric_key, "metricRef": column})
                else:
                    errors.append({"code": "METRIC_SOURCE_COLUMN_MISSING", "metricKey": metric_key, "column": column})
        canonical = str(metric.get("canonicalMetricKey") or "").strip()
        alias_of = str(metric.get("aliasOf") or "").strip()
        if alias_of and alias_of not in metric_keys:
            warnings.append({"code": "ALIAS_TARGET_NOT_IN_SAME_ASSET", "metricKey": metric_key, "aliasOf": alias_of})
        if canonical and canonical != metric_key:
            warnings.append({"code": "CANONICAL_METRIC_EXTERNAL_OR_ALIAS", "metricKey": metric_key, "canonicalMetricKey": canonical})
    for relationship in relationships:
        if not isinstance(relationship, dict):
            continue
        left = str(relationship.get("leftTable") or relationship.get("left_table") or "")
        right = str(relationship.get("rightTable") or relationship.get("right_table") or "")
        if table not in {left, right}:
            continue
        for left_key, right_key in relationship_key_pairs(relationship):
            key = left_key if table == left else right_key
            if key and key not in columns:
                errors.append({"code": "RELATIONSHIP_KEY_COLUMN_MISSING", "relationship": relationship.get("name") or relationship.get("relationshipId"), "column": key})
    return {"errors": errors, "warnings": warnings, "errorCount": len(errors), "warningCount": len(warnings)}


def semantic_validation_gate(validation: Dict[str, Any]) -> Dict[str, Any]:
    error_count = int(validation.get("errorCount") or len(validation.get("errors") or []))
    warning_count = int(validation.get("warningCount") or len(validation.get("warnings") or []))
    if error_count:
        return {
            "publishable": False,
            "severity": "blocking",
            "status": "BLOCKED",
            "blockingReasons": ["SEMANTIC_VALIDATION_ERRORS"],
            "warningReasons": [],
            "errorCount": error_count,
            "warningCount": warning_count,
        }
    return {
        "publishable": True,
        "severity": "warning" if warning_count else "passed",
        "status": "REVIEW_REQUIRED" if warning_count else "PASSED",
        "blockingReasons": [],
        "warningReasons": ["SEMANTIC_VALIDATION_WARNINGS"] if warning_count else [],
        "errorCount": error_count,
        "warningCount": warning_count,
    }


def schema_drift_release_gate(drift: SchemaDriftReport) -> Dict[str, Any]:
    missing = list(drift.missing_live_columns or [])
    extra = list(drift.extra_live_columns or [])
    type_changed = list(drift.type_changed_columns or [])
    incompatible_type_changes = [
        item
        for item in type_changed
        if not compatible_schema_type_change(str(item.get("semanticType") or ""), str(item.get("liveType") or ""))
    ]
    blocking: List[str] = []
    warnings: List[str] = []
    if missing:
        blocking.append("MISSING_LIVE_COLUMNS")
    if incompatible_type_changes:
        blocking.append("INCOMPATIBLE_TYPE_CHANGES")
    compatible_type_changes = max(0, len(type_changed) - len(incompatible_type_changes))
    if compatible_type_changes:
        warnings.append("COMPATIBLE_TYPE_CHANGES")
    if extra:
        warnings.append("EXTRA_LIVE_COLUMNS")
    if not drift.live_column_count:
        warnings.append("LIVE_SCHEMA_UNAVAILABLE")
    severity = "blocking" if blocking else "warning" if warnings else "passed"
    return {
        "publishable": not blocking,
        "severity": severity,
        "status": "BLOCKED" if blocking else "REVIEW_REQUIRED" if warnings else "PASSED",
        "blockingReasons": blocking,
        "warningReasons": warnings,
        "missingLiveColumnCount": len(missing),
        "extraLiveColumnCount": len(extra),
        "typeChangedColumnCount": len(type_changed),
        "incompatibleTypeChangedColumnCount": len(incompatible_type_changes),
    }


def compatible_schema_type_change(semantic_type: str, live_type: str) -> bool:
    semantic = normalize_column_type_family(semantic_type)
    live = normalize_column_type_family(live_type)
    if not semantic or not live:
        return True
    if semantic == live:
        return True
    compatible = {
        ("bigint", "decimal"),
        ("int", "bigint"),
        ("int", "decimal"),
        ("float", "decimal"),
        ("double", "decimal"),
        ("text", "varchar"),
        ("varchar", "text"),
        ("string", "varchar"),
        ("varchar", "string"),
        ("datetime", "varchar"),
        ("date", "varchar"),
    }
    return (semantic, live) in compatible


def normalize_column_type_family(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if text.startswith("decimal"):
        return "decimal"
    if text.startswith("varchar") or text.startswith("char"):
        return "varchar"
    if text.startswith("datetime"):
        return "datetime"
    if text.startswith("date"):
        return "date"
    if text.startswith("bigint"):
        return "bigint"
    if text.startswith("int"):
        return "int"
    if text.startswith("tinyint"):
        return "int"
    if text.startswith("double"):
        return "double"
    if text.startswith("float"):
        return "float"
    if text in {"text", "string", "json", "boolean", "bool"}:
        return text
    return text.split("(", 1)[0]


def combine_release_gates(*gates: Dict[str, Any]) -> Dict[str, Any]:
    blocking: List[str] = []
    warnings: List[str] = []
    for gate in gates:
        blocking.extend(str(item) for item in gate.get("blockingReasons") or [])
        warnings.extend(str(item) for item in gate.get("warningReasons") or [])
    publishable = not blocking
    return {
        "publishable": publishable,
        "status": "BLOCKED" if blocking else "REVIEW_REQUIRED" if warnings else "PASSED",
        "severity": "blocking" if blocking else "warning" if warnings else "passed",
        "blockingReasons": sorted(set(blocking)),
        "warningReasons": sorted(set(warnings)),
    }


def semantic_asset_impact_test_plan(topic: str, table: str, asset: Dict[str, Any], drift: SchemaDriftReport) -> Dict[str, Any]:
    metrics = asset.get("metrics") if isinstance(asset.get("metrics"), list) else []
    metric_keys = [
        str(metric.get("metricKey") or metric.get("key") or "")
        for metric in metrics
        if isinstance(metric, dict) and str(metric.get("metricKey") or metric.get("key") or "")
    ][:8]
    impacted_columns = sorted(
        set(drift.missing_live_columns or [])
        | {str(item.get("column") or "") for item in drift.type_changed_columns or [] if item.get("column")}
    )
    commands = [
        "python_backend/.venv/bin/python -m ruff check python_backend scripts",
        "cd python_backend && .venv/bin/python -m pytest tests/test_harness.py -q -k 'semantic or drift or recall or planner'",
    ]
    return {
        "topic": topic,
        "tableName": table,
        "impactedMetrics": metric_keys,
        "impactedColumns": impacted_columns[:16],
        "suggestedCommands": commands,
        "reviewFocus": [
            "schema drift blocking columns",
            "metric canonical/alias consistency",
            "recall index changed refs",
            "planner asset pack compactness",
        ],
    }


def topic_asset_production_report(
    *,
    topic: str,
    table: str,
    schema: List[Dict[str, Any]],
    sample_rows: List[Dict[str, Any]],
    profile: Dict[str, Any],
    semantic_columns: List[Dict[str, Any]],
    metrics: List[Dict[str, Any]],
    terms: List[Dict[str, Any]],
    rules: List[Dict[str, Any]],
    builder_phases: Dict[str, Any],
    generation_mode: str,
) -> Dict[str, Any]:
    schema_count = len(schema or [])
    semantic_count = len(semantic_columns or [])
    coverage = float(semantic_count) / float(schema_count or 1)
    quality_score = min(
        1.0,
        0.25
        + (0.25 if sample_rows else 0.0)
        + (0.2 if metrics else 0.0)
        + (0.15 if terms else 0.0)
        + (0.15 if rules else 0.0),
    )
    return {
        "status": "ready_for_human_review",
        "topic": topic,
        "tableName": table,
        "generatedAt": datetime.utcnow().isoformat() + "Z",
        "generationMode": generation_mode or "heuristic",
        "schemaDiscovery": {
            "columnCount": schema_count,
            "columns": [str(item.get("columnName") or item.get("Field") or "") for item in (schema or [])[:80]],
        },
        "sampleProfiling": {
            "sampleRowCount": len(sample_rows or []),
            "timeColumn": profile.get("timeColumn") or "",
            "merchantFilterColumn": profile.get("merchantFilterColumn") or "",
            "enumCandidates": sorted((profile.get("enumCandidates") or {}).keys())[:40]
            if isinstance(profile.get("enumCandidates"), dict)
            else [],
        },
        "semanticDraft": {
            "fieldCount": semantic_count,
            "metricCount": len(metrics or []),
            "termCount": len(terms or []),
            "ruleCount": len(rules or []),
            "fieldCoverage": round(coverage, 4),
        },
        "builderPhases": builder_phases,
        "qualityScore": round(quality_score, 4),
        "humanReview": {
            "required": True,
            "reviewFocus": [
                "field business meaning and sensitive columns",
                "metric formula and owner table",
                "term alias conflicts",
                "row access policy and merchant filter",
            ],
        },
    }


def semantic_asset_owner(asset: Dict[str, Any]) -> str:
    governance = asset.get("semanticGovernance") if isinstance(asset.get("semanticGovernance"), dict) else {}
    owner = governance.get("owner") or asset.get("owner") or asset.get("merchantId") or "semantic_asset_owner"
    return str(owner or "semantic_asset_owner")


def semantic_governance_envelope(
    topic: str,
    table: str,
    *,
    owner: str,
    stage: str,
    status: str,
    semantic_version: str = "",
) -> Dict[str, Any]:
    return {
        "topic": topic,
        "tableName": table,
        "owner": owner or "semantic_asset_owner",
        "semanticVersion": semantic_version,
        "lifecycleStatus": status,
        "stage": stage,
        "versioning": {
            "enabled": True,
            "versionFile": "semantic_version.json",
            "historyFile": "semantic_publish_history.json",
        },
        "approval": {
            "required": True,
            "states": ["pending_review", "preflight", "published", "rolled_back"],
        },
        "rollback": {
            "enabled": True,
            "snapshotPolicy": "create snapshot before publish",
        },
    }


def semantic_approval_workflow(stage: str, reviewer: str, review_note: str, publishable: bool) -> Dict[str, Any]:
    return {
        "stage": stage,
        "required": True,
        "reviewer": reviewer or "",
        "reviewNote": review_note or "",
        "publishable": bool(publishable),
        "nextActions": ["approve_publish", "reject", "request_changes"] if stage != "published" else ["monitor", "rollback_if_needed"],
    }


def semantic_gray_release_plan(topic: str, table: str, release_gate: Dict[str, Any]) -> Dict[str, Any]:
    publishable = bool((release_gate or {}).get("publishable"))
    return {
        "enabled": publishable,
        "topic": topic,
        "tableName": table,
        "strategy": "scoped_incremental",
        "stages": [
            {"name": "preflight", "traffic": 0, "required": True},
            {"name": "reviewed_publish", "traffic": 100 if publishable else 0, "required": True},
        ],
        "abortConditions": ["blocking_schema_drift", "semantic_validation_error", "owner_reject"],
    }


def semantic_gray_release_monitor(topic: str, table: str, release_gate: Dict[str, Any]) -> Dict[str, Any]:
    publishable = bool((release_gate or {}).get("publishable"))
    return {
        "enabled": publishable,
        "topic": topic,
        "tableName": table,
        "status": "monitoring" if publishable else "disabled",
        "metrics": [
            "planner_bind_success_rate",
            "query_validation_failure_rate",
            "answer_evidence_gap_rate",
            "recall_hit_rate",
        ],
        "rollbackTriggers": [
            "blocking_schema_drift",
            "golden_eval_failed",
            "evidence_gap_rate_above_threshold",
        ],
    }


def semantic_release_evaluation_gate(
    asset: Dict[str, Any],
    validation: Dict[str, Any],
    drift: SchemaDriftReport,
    conflict_detection: Dict[str, Any],
) -> Dict[str, Any]:
    metrics = asset.get("metrics") if isinstance(asset.get("metrics"), list) else []
    fields = asset.get("semanticColumns") if isinstance(asset.get("semanticColumns"), list) else []
    errors = list(validation.get("errors") or []) if isinstance(validation, dict) else []
    blocking_reasons: List[str] = []
    if errors:
        blocking_reasons.append("SEMANTIC_VALIDATION_ERRORS")
    if conflict_detection.get("conflictCount"):
        blocking_reasons.append("SEMANTIC_CONFLICTS")
    if getattr(drift, "missing_live_columns", None):
        blocking_reasons.append("SCHEMA_DRIFT_MISSING_COLUMNS")
    if not fields:
        blocking_reasons.append("NO_SEMANTIC_FIELDS")
    if not metrics:
        blocking_reasons.append("NO_METRICS")
    return {
        "publishable": not blocking_reasons,
        "severity": "blocking" if blocking_reasons else "passed",
        "blockingReasons": blocking_reasons,
        "goldenEval": {
            "enabled": True,
            "status": "passed" if not blocking_reasons else "failed",
            "caseCount": max(1, len(metrics)),
            "requiredBeforePublish": True,
        },
        "checks": [
            "semantic_validation",
            "schema_drift",
            "conflict_detection",
            "metric_presence",
        ],
    }


def semantic_conflict_detection(asset: Dict[str, Any]) -> Dict[str, Any]:
    metrics = asset.get("metrics") if isinstance(asset.get("metrics"), list) else []
    terms = asset.get("terms") if isinstance(asset.get("terms"), list) else []
    seen: Dict[str, str] = {}
    conflicts: List[Dict[str, Any]] = []
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        key = str(metric.get("canonicalMetricKey") or metric.get("metricKey") or metric.get("key") or "").strip()
        formula = str(metric.get("formula") or "").strip()
        if not key:
            continue
        if key in seen and seen[key] != formula:
            conflicts.append({"type": "metric_formula_conflict", "metricKey": key, "formulas": sorted({seen[key], formula})})
        seen.setdefault(key, formula)
    term_aliases: Dict[str, str] = {}
    for term in terms:
        if not isinstance(term, dict):
            continue
        canonical = str(term.get("canonicalMetricKey") or term.get("term") or "").strip()
        for alias in term.get("aliases") or []:
            alias_text = str(alias or "").strip().lower()
            if not alias_text:
                continue
            if alias_text in term_aliases and term_aliases[alias_text] != canonical:
                conflicts.append({"type": "term_alias_conflict", "alias": alias_text, "targets": sorted({term_aliases[alias_text], canonical})})
            term_aliases.setdefault(alias_text, canonical)
    return {
        "status": "passed" if not conflicts else "conflict_detected",
        "conflictCount": len(conflicts),
        "conflicts": conflicts[:20],
    }


def semantic_conflict_repair_plan(conflict_detection: Dict[str, Any]) -> Dict[str, Any]:
    conflicts = conflict_detection.get("conflicts") if isinstance(conflict_detection.get("conflicts"), list) else []
    actions = []
    for conflict in conflicts:
        kind = str(conflict.get("type") or "")
        if kind == "metric_formula_conflict":
            actions.append(
                {
                    "action": "choose_canonical_metric_formula",
                    "metricKey": conflict.get("metricKey", ""),
                    "candidates": conflict.get("formulas", []),
                    "requiresOwnerReview": True,
                }
            )
        elif kind == "term_alias_conflict":
            actions.append(
                {
                    "action": "split_or_reassign_alias",
                    "alias": conflict.get("alias", ""),
                    "targets": conflict.get("targets", []),
                    "requiresOwnerReview": True,
                }
            )
    return {
        "status": "no_conflict" if not actions else "repair_required",
        "autoRepairable": False,
        "actions": actions[:20],
    }


def semantic_asset_lineage(
    topic: str,
    table: str,
    asset: Dict[str, Any],
    metrics: List[Dict[str, Any]],
    rules: List[Dict[str, Any]],
) -> Dict[str, Any]:
    metric_lineage = []
    for metric in metrics or []:
        if not isinstance(metric, dict):
            continue
        metric_lineage.append(
            {
                "metricKey": str(metric.get("metricKey") or metric.get("key") or ""),
                "ownerTable": table,
                "sourceColumns": semantic_metric_source_columns(metric),
                "formula": str(metric.get("formula") or ""),
            }
        )
    return {
        "topic": topic,
        "tableName": table,
        "sourceTable": table,
        "timeColumn": str(asset.get("timeColumn") or ""),
        "merchantFilterColumn": str(asset.get("merchantFilterColumn") or ""),
        "metrics": metric_lineage[:50],
        "rules": [
            {
                "ruleId": str(rule.get("ruleId") or rule.get("key") or ""),
                "appliesToMetrics": [str(value) for value in (rule.get("appliesToMetrics") or [])[:20]],
                "appliesToColumns": [str(value) for value in (rule.get("appliesToColumns") or [])[:20]],
            }
            for rule in (rules or [])
            if isinstance(rule, dict)
        ][:50],
    }


def semantic_rollback_candidate(target_dir: Path) -> Dict[str, Any]:
    version_path = target_dir / "semantic_version.json"
    version = read_json(version_path)
    if not isinstance(version, dict):
        return {}
    return {
        "semanticVersion": version.get("semanticVersion") or version.get("semantic_version") or "",
        "schemaVersion": version.get("schemaVersion") or version.get("schema_version") or "",
        "sourceHash": version.get("sourceHash") or version.get("source_hash") or "",
        "publishedAt": version.get("publishedAt") or version.get("published_at") or "",
        "versionPath": str(version_path),
    }


def append_semantic_publish_history(target_dir: Path, payload: Dict[str, Any]) -> Path:
    path = target_dir / "semantic_publish_history.json"
    existing = read_json(path)
    history = existing if isinstance(existing, list) else []
    version = payload.get("semanticCatalogVersion") if isinstance(payload.get("semanticCatalogVersion"), dict) else {}
    history.append(
        {
            "semanticVersion": version.get("semanticVersion") or version.get("semantic_version") or "",
            "schemaVersion": version.get("schemaVersion") or version.get("schema_version") or "",
            "sourceHash": version.get("sourceHash") or version.get("source_hash") or "",
            "publishedAt": version.get("publishedAt") or version.get("published_at") or "",
            "owner": version.get("owner") or ((payload.get("semanticGovernance") or {}).get("owner") if isinstance(payload.get("semanticGovernance"), dict) else ""),
            "status": payload.get("status"),
            "releaseGate": payload.get("releaseGate"),
            "approvalWorkflow": payload.get("approvalWorkflow"),
            "grayReleasePlan": payload.get("grayReleasePlan"),
            "rollbackCandidate": payload.get("rollbackCandidate"),
            "reviewArtifact": payload.get("reviewArtifact"),
        }
    )
    write_json(path, history[-50:])
    return path


def sanitize_asset_path_part(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._")
    return text or "unknown"


def schema_columns(asset: Dict[str, Any]) -> List[Dict[str, Any]]:
    schema = asset.get("schemaColumns") or asset.get("schema") or []
    return schema if isinstance(schema, list) else []


def column_name_set(schema: List[Dict[str, Any]]) -> set[str]:
    return {str(item.get("columnName") or item.get("Field") or item.get("name") or "").strip() for item in schema if isinstance(item, dict) and str(item.get("columnName") or item.get("Field") or item.get("name") or "").strip()}


def semantic_metric_source_columns(metric: Dict[str, Any]) -> List[str]:
    result: List[str] = []
    for value in metric.get("sourceColumns") or metric.get("source_columns") or []:
        text = str(value or "").strip().strip("`")
        if text and text not in result:
            result.append(text)
    if result:
        return result
    formula = str(metric.get("formula") or metric.get("metricFormula") or "")
    for token in re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`|\\b([A-Za-z_][A-Za-z0-9_]*)\\b", formula):
        text = (token[0] or token[1] or "").strip()
        if text.lower() in {"sum", "count", "distinct", "avg", "min", "max", "case", "when", "then", "else", "end", "nullif"}:
            continue
        if text and text not in result:
            result.append(text)
    return result


def external_metric_dependency(metric: Dict[str, Any], ref: str) -> bool:
    formula = str(metric.get("formula") or metric.get("metricFormula") or "")
    unit = str(metric.get("unit") or "").strip()
    return bool(unit == "%" or "/" in formula) and metric_shaped_reference(ref)


def metric_shaped_reference(ref: str) -> bool:
    text = str(ref or "").strip().lower()
    return text.endswith(("_cnt", "_amt", "_rate", "_gmv")) or "gmv" in text


def relationship_key_pairs(relationship: Dict[str, Any]) -> List[Tuple[str, str]]:
    raw = relationship.get("keys") or relationship.get("joinKeys") or relationship.get("join_keys") or []
    pairs: List[Tuple[str, str]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pairs.append((str(item[0] or ""), str(item[1] or "")))
        elif isinstance(item, dict):
            pairs.append((str(item.get("left") or item.get("leftKey") or ""), str(item.get("right") or item.get("rightKey") or "")))
    return pairs


def read_json(path: Path) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def merge_semantic_layer_list(base: Any, override: Any, field: str) -> Any:
    if not isinstance(base, list) or not isinstance(override, list):
        return override
    key_fields = {
        "schemaColumns": ["columnName", "name", "key"],
        "semanticColumns": ["columnName", "name", "key"],
        "metrics": ["metricKey", "key", "name"],
        "terms": ["term", "key", "name"],
        "knowledgeRules": ["ruleId", "id", "title", "name"],
    }.get(field, ["key", "name"])
    merged: List[Any] = []
    index: Dict[str, int] = {}
    for item in base:
        identity = semantic_layer_item_identity(item, key_fields)
        if identity:
            index[identity] = len(merged)
        merged.append(item)
    for item in override:
        identity = semantic_layer_item_identity(item, key_fields)
        if identity and identity in index:
            merged[index[identity]] = merge_semantic_layer_item(merged[index[identity]], item)
        else:
            if identity:
                index[identity] = len(merged)
            merged.append(item)
    return merged


def merge_semantic_layer_item(base: Any, override: Any) -> Any:
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    merged = {**base, **override}
    for key in ("businessName", "description", "comment", "evidence", "metricFormula", "formula"):
        base_value = str(base.get(key) or "")
        override_value = str(override.get(key) or "")
        if base_value and len(base_value) > len(override_value):
            merged[key] = base.get(key)
    for key in ("aliases", "relatedColumns", "sourceColumns", "enumValues", "sampleValues"):
        base_values = base.get(key)
        override_values = override.get(key)
        if isinstance(base_values, list) or isinstance(override_values, list):
            merged[key] = dedupe_strings(
                [str(item) for item in (base_values if isinstance(base_values, list) else [])]
                + [str(item) for item in (override_values if isinstance(override_values, list) else [])]
            )
    return merged


def semantic_layer_item_identity(item: Any, key_fields: List[str]) -> str:
    if not isinstance(item, dict):
        return ""
    for key in key_fields:
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return ""
    return None


def recall_terms(question: str, keywords: List[str]) -> List[str]:
    terms: List[str] = []
    for token in keywords:
        if token and token not in terms:
            terms.append(token)
    for token in [
        "gmv",
        "订单",
        "退款",
        "退货",
        "工单",
        "赔付",
        "优惠",
        "商品",
        "供应链",
        "保证金",
        "申诉",
        "处罚",
        "商家",
        "身份",
        "营业执照",
        "规则",
        "金额",
        "状态",
        "Top",
        "前",
    ]:
        if token.lower() in (question or "").lower() and token not in terms:
            terms.append(token)
    return terms or [question[:12]]


def targeted_recall_terms(question: str) -> List[str]:
    terms = recall_terms(question, [])
    for term in question_match_terms(question):
        if term and term not in terms:
            terms.append(term)
    return terms


def table_seed_terms(question: str) -> List[str]:
    generic_terms = {
        "最近",
        "过去",
        "近",
        "天",
        "日",
        "周",
        "月",
        "前",
        "top",
        "最高",
        "最多",
        "多少",
        "几个",
        "哪些",
        "情况",
        "怎么样",
        "是否",
        "有没有",
        "变化",
        "趋势",
        "走势",
        "同步",
        "上升",
        "下降",
        "波动",
        "异常",
        "金额",
        "数量",
        "单量",
        "占比",
        "比例",
        "关联",
        "对应",
        "同时",
        "分别",
        "分析",
        "判断",
        "原因",
    }
    terms: List[str] = []
    for term in targeted_recall_terms(question):
        normalized = str(term or "").strip().lower()
        if not normalized or normalized in generic_terms:
            continue
        if normalized.isdigit():
            continue
        if len(normalized) == 1 and not re.match(r"[a-z]", normalized):
            continue
        if normalized not in terms:
            terms.append(normalized)
    return terms or targeted_recall_terms(question)


def table_recall_score(recall_bundle: RecallBundle, table: str) -> float:
    score = 0.0
    if not recall_bundle or not table:
        return score
    for item in recall_bundle.items:
        item_table = item.table or str((item.metadata or {}).get("tableName") or "")
        if item_table != table:
            continue
        score = max(score, float(item.fusion_score or 0.0))
    return score


def recalled_metric_evidence_from_bundle(recall_bundle: RecallBundle) -> List[Dict[str, Any]]:
    if not recall_bundle:
        return []
    evidence_by_identity: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for item in recall_bundle.items:
        if str(item.source_type or "").upper() != "SEMANTIC_METRIC":
            continue
        metadata = item.metadata or {}
        table = item.table or str(metadata.get("tableName") or "")
        metric_key = str(metadata.get("metricKey") or "")
        semantic_ref_id = str(metadata.get("semanticRefId") or item.doc_id or "")
        if not table or not metric_key:
            continue
        identity = (table, metric_key, semantic_ref_id)
        recall_queries = [str(query) for query in metadata.get("recallQueries") or [] if query]
        if metadata.get("recallQuery") and str(metadata.get("recallQuery")) not in recall_queries:
            recall_queries.append(str(metadata.get("recallQuery")))
        current = evidence_by_identity.get(identity)
        if current is None:
            evidence_by_identity[identity] = {
                "ownerTable": table,
                "metricKey": metric_key,
                "semanticRefId": semantic_ref_id,
                "docId": item.doc_id,
                "title": item.title,
                "fusionScore": float(item.fusion_score or 0.0),
                "sourceType": item.source_type,
                "businessName": str(metadata.get("businessName") or ""),
                "canonicalMetricKey": str(metadata.get("canonicalMetricKey") or ""),
                "aliasOf": str(metadata.get("aliasOf") or ""),
                "metricLevel": str(metadata.get("metricLevel") or ""),
                "formula": str(metadata.get("formula") or ""),
                "sourceColumns": metadata.get("sourceColumns") or [],
                "aliases": metadata.get("aliases") or [],
                "recallQuery": recall_queries[-1] if recall_queries else "",
                "recallQueries": recall_queries,
                "matchedMetricLabel": str(metadata.get("matchedMetricLabel") or metadata.get("matchedExactMetricLabel") or ""),
                "metricResolutionType": str(metadata.get("metricResolutionType") or ""),
                "metricResolutionReason": str(metadata.get("metricResolutionReason") or ""),
                "metricResolutionConfidence": float(metadata.get("metricResolutionConfidence") or 0.0),
                "metricResolutionAmbiguous": bool(metadata.get("metricResolutionAmbiguous") or False),
            }
            continue
        merged_queries = list(current.get("recallQueries") or [])
        for query in recall_queries:
            if query and query not in merged_queries:
                merged_queries.append(query)
        if float(item.fusion_score or 0.0) > float(current.get("fusionScore") or 0.0):
            current.update(
                {
                    "docId": item.doc_id,
                    "title": item.title,
                    "fusionScore": float(item.fusion_score or 0.0),
                    "sourceType": item.source_type,
                    "businessName": str(metadata.get("businessName") or current.get("businessName") or ""),
                    "canonicalMetricKey": str(metadata.get("canonicalMetricKey") or current.get("canonicalMetricKey") or ""),
                    "aliasOf": str(metadata.get("aliasOf") or current.get("aliasOf") or ""),
                    "metricLevel": str(metadata.get("metricLevel") or current.get("metricLevel") or ""),
                    "formula": str(metadata.get("formula") or current.get("formula") or ""),
                    "sourceColumns": metadata.get("sourceColumns") or current.get("sourceColumns") or [],
                    "aliases": metadata.get("aliases") or current.get("aliases") or [],
                    "matchedMetricLabel": str(metadata.get("matchedMetricLabel") or metadata.get("matchedExactMetricLabel") or current.get("matchedMetricLabel") or ""),
                    "metricResolutionType": str(metadata.get("metricResolutionType") or current.get("metricResolutionType") or ""),
                    "metricResolutionReason": str(metadata.get("metricResolutionReason") or current.get("metricResolutionReason") or ""),
                    "metricResolutionConfidence": float(metadata.get("metricResolutionConfidence") or current.get("metricResolutionConfidence") or 0.0),
                    "metricResolutionAmbiguous": bool(metadata.get("metricResolutionAmbiguous") if metadata.get("metricResolutionAmbiguous") is not None else current.get("metricResolutionAmbiguous") or False),
                }
            )
        current["recallQueries"] = merged_queries
        current["recallQuery"] = merged_queries[-1] if merged_queries else str(current.get("recallQuery") or "")
    evidence = list(evidence_by_identity.values())
    evidence.sort(key=lambda item: float(item.get("fusionScore") or 0.0), reverse=True)
    return evidence


def recalled_metric_identities_from_compaction(metric_compaction: Dict[str, Any]) -> Set[Tuple[str, str]]:
    identities: Set[Tuple[str, str]] = set()
    for item in metric_compaction.get("recalledMetricEvidence") or []:
        if not isinstance(item, dict):
            continue
        table = str(item.get("ownerTable") or "")
        metric_key = str(item.get("metricKey") or "")
        if table and metric_key:
            identities.add((table, metric_key))
    return identities


def recalled_metric_evidence_map(metric_compaction: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    evidence_by_identity: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in metric_compaction.get("recalledMetricEvidence") or []:
        if not isinstance(item, dict):
            continue
        table = str(item.get("ownerTable") or "")
        metric_key = str(item.get("metricKey") or "")
        if not table or not metric_key:
            continue
        identity = (table, metric_key)
        current = evidence_by_identity.get(identity)
        if current is None:
            evidence_by_identity[identity] = dict(item)
            continue
        merged_queries = list(current.get("recallQueries") or [])
        for query in item.get("recallQueries") or []:
            query_text = str(query or "")
            if query_text and query_text not in merged_queries:
                merged_queries.append(query_text)
        if float(item.get("fusionScore") or 0.0) > float(current.get("fusionScore") or 0.0):
            evidence_by_identity[identity] = {**item, "recallQueries": merged_queries}
        else:
            current["recallQueries"] = merged_queries
            if merged_queries:
                current["recallQuery"] = merged_queries[-1]
    return evidence_by_identity


def recalled_evidence_scoped_to_phrase(evidence: Dict[str, Any], phrase: str) -> bool:
    normalized_phrase = normalize_for_match(phrase)
    if not normalized_phrase:
        return False
    queries = [str(query or "") for query in evidence.get("recallQueries") or [] if query]
    if evidence.get("recallQuery"):
        queries.append(str(evidence.get("recallQuery") or ""))
    for query in queries:
        normalized_query = normalize_for_match(query)
        if not normalized_query:
            continue
        if normalized_query == normalized_phrase:
            return True
        if "语义指标口径" in query and normalized_phrase in normalized_query:
            return True
    return False


def recalled_metric_evidence_matches_phrase(evidence: Dict[str, Any], phrase: str) -> bool:
    normalized_phrase = normalize_for_match(phrase)
    if not normalized_phrase:
        return False
    labels = [
        str(evidence.get("metricKey") or ""),
        str(evidence.get("businessName") or ""),
        *[str(alias) for alias in evidence.get("aliases") or []],
    ]
    canonical = str(evidence.get("canonicalMetricKey") or "")
    alias_of = str(evidence.get("aliasOf") or "")
    if canonical:
        labels.append(canonical)
    if alias_of:
        labels.append(alias_of)
    return any(is_strong_label_text_match(normalize_for_match(label), normalized_phrase) for label in labels if label)


def pop_lowest_unprotected_metric(metrics: List[PlanningAssetEntry], protected: Set[Tuple[str, str]]) -> PlanningAssetEntry | None:
    for index in range(len(metrics) - 1, -1, -1):
        identity = (metrics[index].table, metrics[index].key)
        if identity in protected:
            continue
        return metrics.pop(index)
    return None


def shortest_table_path(graph: Dict[str, Set[str]], start: str, target: str, max_hops: int = 3) -> List[str]:
    if not start or not target or start not in graph or target not in graph:
        return []
    if start == target:
        return [start]
    queue: List[List[str]] = [[start]]
    seen = {start}
    while queue:
        path = queue.pop(0)
        if len(path) - 1 >= max_hops:
            continue
        for neighbor in sorted(graph.get(path[-1], set())):
            if neighbor in path:
                continue
            next_path = path + [neighbor]
            if neighbor == target:
                return next_path
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(next_path)
    return []


def score_document(terms: List[str], document: str) -> float:
    doc = (document or "").lower()
    score = 0.0
    for term in terms:
        normalized = str(term).lower()
        if not normalized:
            continue
        count = doc.count(normalized)
        if count:
            score += 2.0 + min(count, 5) * 0.4
    return score


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def question_match_terms(question: str) -> List[str]:
    text = str(question or "")
    terms: List[str] = []
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text):
        normalized = token.lower()
        if normalized and normalized not in terms:
            terms.append(normalized)
    for seq in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        for size in range(2, min(6, len(seq)) + 1):
            for index in range(0, len(seq) - size + 1):
                gram = seq[index : index + size]
                if gram not in terms:
                    terms.append(gram)
    return terms


def grep_snippets(content: str, terms: List[str], limit: int) -> List[str]:
    snippets: List[str] = []
    text = str(content or "")
    lowered = text.lower()
    for term in terms:
        needle = str(term or "").lower()
        if not needle:
            continue
        pos = lowered.find(needle)
        if pos < 0:
            continue
        start = max(0, pos - 80)
        end = min(len(text), pos + len(needle) + 120)
        snippet = text[start:end].replace("\n", " ").strip()
        if snippet and snippet not in snippets:
            snippets.append(snippet)
        if len(snippets) >= limit:
            break
    return snippets
