from __future__ import annotations

import json
import re
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
    SkillManifest,
    TOPIC_TO_CATEGORY,
    TopicBuildRequest,
    category_display,
)
from merchant_ai.services.repositories import DorisRepository, write_json


class TopicAssetService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._table_asset_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._relationship_cache: Dict[str, List[Dict[str, Any]]] = {}

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
            for file_path in pending.iterdir():
                if file_path.is_file():
                    (target / file_path.name).write_bytes(file_path.read_bytes())
            self._table_asset_cache.pop((topic, table_name), None)
            return {"success": True, "status": "PUBLISHED", "topic": topic, "tableName": table_name}
        return {"success": True, "status": "REJECTED", "topic": topic, "tableName": table_name}

    def load_manifest(self, topic: str) -> List[Dict[str, Any]]:
        path = self.root / topic / "manifest.json"
        data = read_json(path)
        return data if isinstance(data, list) else []

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
        if not self.root.exists():
            return []
        return [path.name for path in sorted(self.root.iterdir()) if path.is_dir()]

    def table_asset_dir(self, topic: str, table: str) -> Path:
        return self.root / topic / "tables" / table

    def load_table_asset(self, topic: str, table: str) -> Dict[str, Any]:
        cache_key = (topic, table)
        if cache_key in self._table_asset_cache:
            return self._table_asset_cache[cache_key]
        table_dir = self.table_asset_dir(topic, table)
        asset = read_json(table_dir / "asset.json")
        if not isinstance(asset, dict) or not asset:
            manifest_item = next((item for item in self.load_manifest(topic) if str(item.get("tableName") or "") == table), {})
            asset = {
                **manifest_item,
                "topic": topic,
                "tableName": table,
                "schemaColumns": read_json(table_dir / "schema.json") or [],
                "semanticColumns": read_json(table_dir / "semantic_columns.json") or [],
                "metrics": read_json(table_dir / "metrics.json") or [],
                "terms": read_json(table_dir / "terms.json") or [],
                "knowledgeRules": read_json(table_dir / "knowledge_rules.json") or [],
            }
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

    TABLE_KIND = "TABLE_ASSET"
    RELATIONSHIP_KIND = "RELATIONSHIPS"
    OFFLOAD_THRESHOLD_CHARS = 20_000

    def __init__(self, topic_assets: TopicAssetService):
        self.topic_assets = topic_assets

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
        return {
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
        }

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
                {
                    "refId": ref_id,
                    "path": semantic_path,
                    "kind": metadata.get("semanticKind") or item.source_type,
                    "topic": item.topic,
                    "table": item.table,
                    "title": item.title,
                    "estimatedChars": metadata.get("estimatedChars", len(item.content or "")),
                    "offloadRecommended": bool(metadata.get("offloadRecommended")),
                }
            )
        return {
            "mode": "filesystem_as_context",
            "policy": "list semantic refs first; read/grep only refs needed for the current step; offload large files by path",
            "roots": ["topics/<topic>/manifest.json", "topics/<topic>/tables/<table>/asset.json", "topics/<topic>/relationships.json"],
            "refs": refs[:limit],
        }

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
        return {
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
        }

    def relationship_ref(self, topic: str, relationships: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        relationships = relationships if relationships is not None else self.topic_assets.load_relationships(topic)
        content = json.dumps(relationships, ensure_ascii=False, indent=2)
        return {
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
        }

    def _topics(self, topic_categories: Iterable[QuestionCategory] | None, topic: str) -> List[str]:
        if topic:
            return [topic]
        if topic_categories:
            topics = self.topic_assets.topic_names_for_categories(topic_categories)
            if topics:
                return topics
        return self.topic_assets.all_topic_names()

    def _all_refs(self, topic_categories: Iterable[QuestionCategory] | None = None, topic: str = "") -> List[Dict[str, Any]]:
        refs: List[Dict[str, Any]] = []
        for topic_name in self._topics(topic_categories, topic):
            for manifest_item in self.topic_assets.load_manifest(topic_name):
                table = str(manifest_item.get("tableName") or "")
                if table:
                    refs.append(self.table_ref(topic_name, table))
            relationships = self.topic_assets.load_relationships(topic_name)
            if relationships:
                refs.append(self.relationship_ref(topic_name, relationships))
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
        scored: List[RecallItem] = []
        for doc in self._load_documents():
            if not allowed_topics and doc.source_type != "BASE_WIKI":
                continue
            if allowed_topics and doc.topic and doc.topic not in allowed_topics:
                continue
            score = score_document(query_terms, doc.title + "\n" + doc.content)
            if score <= 0:
                continue
            item = doc.model_copy(update={"fusion_score": score})
            scored.append(item)
        scored.sort(key=lambda item: item.fusion_score, reverse=True)
        items = scored[:4] if not allowed_topics else scored[:12]
        merged = "\n\n".join(
            "召回片段 [%s] %s\n%s" % (item.source_type, item.title, item.content[:1200]) for item in items
        )
        return RecallBundle(items=items, top_score=items[0].fusion_score if items else 0.0, merged_context=merged)

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
                            "tableName": table,
                            "topic": topic,
                            "layers": ref["layers"],
                            "estimatedChars": ref["estimatedChars"],
                            "offloadRecommended": ref["offloadRecommended"],
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
                            "topic": topic,
                            "layers": ref["layers"],
                            "estimatedChars": ref["estimatedChars"],
                            "offloadRecommended": ref["offloadRecommended"],
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
            }
            for item in (asset.get("semanticColumns") or [])[:60]
            if isinstance(item, dict)
        ],
    }
    return json.dumps(payload, ensure_ascii=False)[:12000]


def compact_table_metadata(asset: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "topic": asset.get("topic"),
        "questionCategory": asset.get("questionCategory"),
        "tableName": asset.get("tableName"),
        "tableComment": asset.get("tableComment"),
        "dataGrain": asset.get("dataGrain"),
        "timeColumn": asset.get("timeColumn"),
        "merchantFilterColumn": asset.get("merchantFilterColumn"),
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
        source_refs: Dict[str, RecallItem] = {}
        tables_seen: Set[str] = set()
        for item in recall_bundle.items:
            source_refs[item.doc_id] = item
            if item.table and self._table_allowed_for_question(question, item.table, allow_profile=allow_profile):
                tables_seen.add(item.table)
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
        table_topic = self._table_topic_index()
        all_relationships = self._all_relationships()
        seed_tables = set(tables_seen)
        for topic in topics:
            if any(table_topic.get(table) == topic for table in seed_tables):
                continue
            seed_tables.update(self._seed_tables_for_topic(question, topic, allow_profile=allow_profile))
        pack_tables = {table for table in seed_tables if table}
        for table in sorted(pack_tables):
            topic = table_topic.get(table)
            if not topic:
                continue
            self._append_table_assets(pack, topic, table)
        self._trim_metrics_for_question(pack, question)
        metric_dependency_closure = self._expand_tables_for_metric_dependencies(pack, pack_tables, table_topic, question, allow_profile=allow_profile)
        if metric_dependency_closure:
            self._trim_metrics_for_question(pack, question)
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
        pack.relationship_closure = metric_dependency_closure
        pack.skills = self._reconcile_skills(pack.skills, pack)
        for skill in pack.skills:
            ref = source_refs.get("skill:%s" % skill.domain)
            if ref:
                ref.content = json.dumps(self.skill_loader.policy_payload(skill), ensure_ascii=False)
        pack.source_refs = source_refs
        return pack

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
            table = self._table_for_metric_request(metric_ref, owner_table, source_phrase)
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
            pack.metric_compaction = {"before": 0, "after": 0, "strategy": "empty"}
            return
        metrics_by_key: Dict[str, List[PlanningAssetEntry]] = {}
        metrics_by_identity: Dict[Tuple[str, str], PlanningAssetEntry] = {}
        for metric in pack.metrics:
            if not metric.key:
                continue
            metrics_by_key.setdefault(metric.key, []).append(metric)
            metrics_by_identity[(metric.table, metric.key)] = metric
        by_table: Dict[str, List[PlanningAssetEntry]] = {}
        for metric in pack.metrics:
            by_table.setdefault(metric.table, []).append(metric)
        selected: List[PlanningAssetEntry] = []
        selected_keys: Set[Tuple[str, str]] = set()
        table_limit = 5
        global_limit = 24
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
        selected = sorted(selected, key=lambda item: self._metric_relevance_score(item, question), reverse=True)[:global_limit]
        selected_keys = {(metric.table, metric.key) for metric in selected if metric.key}
        for metric in list(selected):
            for dep_key in self._metric_dependency_keys(metric, metrics_by_key):
                dep = metrics_by_identity.get((metric.table, dep_key))
                if dep is None:
                    dep = next(iter(metrics_by_key.get(dep_key, [])), None)
                if dep is None:
                    continue
                dep_identity = (dep.table, dep.key)
                if dep_identity not in selected_keys and len(selected) < global_limit:
                    selected.append(dep)
                    selected_keys.add(dep_identity)
        pack.metrics = selected
        pack.metric_compaction = {
            "before": original_count,
            "after": len(pack.metrics),
            "strategy": "question_relevance_top_metrics",
            "perTableLimit": table_limit,
            "globalLimit": global_limit,
            "tables": {table: len([metric for metric in pack.metrics if metric.table == table]) for table in sorted(by_table)},
        }

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
        semantic_columns = self.topic_assets.load_table_semantic_columns(topic, table)
        semantic_by_column = {str(item.get("columnName") or ""): item for item in semantic_columns if isinstance(item, dict)}
        schema_source = "asset"
        live_schema = self._live_schema(table)
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
        for col in schema[:120]:
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
        for metric in self.topic_assets.load_table_metrics(topic, table)[:120]:
            key = str(metric.get("metricKey") or "")
            if not key:
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

    def _live_schema(self, table: str) -> List[Dict[str, Any]]:
        if not self.doris_repository:
            return []
        try:
            rows = self.doris_repository.show_full_columns(table)
            return rows if isinstance(rows, list) else []
        except Exception:
            return []

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

    def _table_for_metric_request(self, metric_ref: str, owner_table: str = "", source_phrase: str = "") -> str:
        resolved = SemanticMetricIndex(self._all_metric_entries()).resolve(metric_ref, owner_table, source_phrase)
        return resolved.metric.table if resolved and resolved.metric else ""

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
    return RelationshipEntry(
        relationship_id=str(rel.get("name") or ""),
        left_table=str(rel.get("leftTable") or ""),
        right_table=str(rel.get("rightTable") or ""),
        join_keys=keys,
        source_ref_id="semantic:%s:relationship:%s" % (topic, rel.get("name") or ""),
        description=json.dumps(rel, ensure_ascii=False),
    )


def semantic_table_ref_id(topic: str, table: str) -> str:
    return "semantic:%s:%s:asset" % (topic, table)


def semantic_relationship_ref_id(topic: str) -> str:
    return "semantic:%s:relationships" % topic


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
    ):
        self.metric = metric
        self.requested_metric_ref = requested_metric_ref
        self.source_phrase = source_phrase
        self.ref_score = ref_score
        self.phrase_score = phrase_score
        self.owner_table_match = owner_table_match
        self.rank_score = rank_score
        self.resolution_reason = resolution_reason

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
            and not (
                direct
                and direct.owner_table_match
                and direct.phrase_score >= self.PHRASE_OVERRIDE_MIN_SCORE
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


class TopicBuilderWorkflow:
    def __init__(self, settings: Settings, doris_repository: DorisRepository, topic_assets: TopicAssetService):
        self.settings = settings
        self.doris_repository = doris_repository
        self.topic_assets = topic_assets

    def build(self, request: TopicBuildRequest) -> Dict[str, Any]:
        topic = request.topic or "经营画像"
        table = request.table_name
        if not table:
            return {"success": False, "message": "tableName is required"}
        pending_dir = self.settings.resolved_topic_path / topic / "pending" / table
        pending_dir.mkdir(parents=True, exist_ok=True)
        schema = []
        try:
            schema = self.doris_repository.show_full_columns(table)
        except Exception:
            if request.schema_ddl:
                schema = [{"columnName": line.split()[0].strip("`,"), "comment": line} for line in request.schema_ddl.splitlines() if line.strip()]
        write_json(pending_dir / "schema.json", schema if isinstance(schema, list) else [])
        write_json(
            pending_dir / "asset.json",
            {
                "topic": topic,
                "tableName": table,
                "manualNotes": request.manual_notes,
                "businessKnowledge": request.business_knowledge,
                "sampleSqls": request.sample_sqls,
            },
        )
        return {"success": True, "status": "PENDING_REVIEW", "topic": topic, "tableName": table, "path": str(pending_dir)}

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
        return {
            "success": True,
            "topic": topic,
            "tableName": table,
            "added": sorted(live_cols - existing_cols),
            "removed": sorted(existing_cols - live_cols),
        }

    def refresh_incremental(self, request: TopicBuildRequest) -> Dict[str, Any]:
        return self.build(request)


def read_json(path: Path) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
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
