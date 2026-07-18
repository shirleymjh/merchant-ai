from __future__ import annotations

import json
import hashlib
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

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
    TopicBuildRequest,
    category_display,
    register_topic_contract,
)
from merchant_ai.services.cache import build_ttl_cache, stable_cache_key
from merchant_ai.services.context_filesystem import add_context_uri, merchant_uri_for_semantic_ref
from merchant_ai.services.llm import LlmClient
from merchant_ai.services.repositories import DorisRepository, write_json
from merchant_ai.services.semantic_request import semantic_request_cache_key
from merchant_ai.services.semantic_joins import plan_governed_joins
from merchant_ai.services.time_semantics import resolve_time_range
from merchant_ai.services.tools import AgentToolDefinition


SUPPORTED_METRIC_AGGREGATION_POLICIES = frozenset(
    {
        "period_rollup",
        "period_recompute",
        "latest_value_only",
        "daily_value_only",
        "ratio_of_sums",
    }
)

SUPPORTED_METRIC_AS_OF_POLICIES = frozenset(
    {
        "calendar",
        "latest_available_partition",
        "latest_observation",
        "not_applicable",
        "undeclared",
    }
)

SUPPORTED_METRIC_TIME_SELECTION_POLICIES = frozenset(
    {
        "period_window",
        "latest_as_of",
        "per_time_grain",
        "undeclared",
    }
)

SUPPORTED_METRIC_MISSING_DATA_POLICIES = frozenset(
    {
        "fail_closed",
        "disclose_unknown",
        "skip_missing",
        "zero_fill",
        "not_applicable",
        "undeclared",
    }
)

SUPPORTED_METRIC_ZERO_VALUE_POLICIES = frozenset(
    {
        "preserve_observed_zero",
        "treat_as_missing",
        "not_applicable",
        "undeclared",
    }
)

SUPPORTED_CALCULATION_TIME_ROLLUP_POLICIES = frozenset(
    {
        "ADDITIVE",
        "NOT_COMPOSABLE",
        "RECOMPUTE_FROM_DETAIL",
        "LAST_VALUE",
        "RATIO_OF_SUMS",
        "WEIGHTED_AVERAGE",
    }
)

SUPPORTED_CALCULATION_WINDOW_POLICIES = frozenset(
    {
        "ANY",
        "EXACT_ONLY",
        "NATIVE_GRAIN_ONLY",
    }
)

SUPPORTED_CALCULATION_AGGREGATIONS = frozenset(
    {
        "SUM",
        "COUNT",
        "COUNT_DISTINCT",
        "AVG",
        "MIN",
        "MAX",
        "MAX_BY",
        "MIN_BY",
        "NDV",
        "RATIO",
        "EXPRESSION",
    }
)

# Only policies with a concrete, shared Quick/QueryGraph execution behavior may
# pass publication. The broader enums above remain useful for proposal/review,
# but unsupported declarations must not become governed runtime contracts.
EXECUTABLE_METRIC_AS_OF_POLICIES_BY_SELECTION = {
    "period_window": frozenset({"calendar", "latest_available_partition"}),
    "latest_as_of": frozenset({"calendar", "latest_available_partition", "latest_observation"}),
    "per_time_grain": frozenset({"calendar", "latest_available_partition"}),
}
EXECUTABLE_METRIC_MISSING_DATA_POLICIES = frozenset({"disclose_unknown", "fail_closed"})
EXECUTABLE_METRIC_ZERO_VALUE_POLICIES = frozenset({"preserve_observed_zero"})
SUPPORTED_ENTITY_LOOKUP_POLICY_MODES = frozenset(
    {
        "clarify",
        "not_required",
        "global",
        "unbounded",
        "all_partitions",
        "bounded_default",
        "default_window",
    }
)
SUPPORTED_ENTITY_COMPARISON_POLICIES = frozenset(
    {
        "exact",
        "case_insensitive",
        "casefold",
        "trimmed",
        "trim",
        "trimmed_case_insensitive",
        "trim_casefold",
        "integer",
        "decimal",
        "number",
        "numeric",
    }
)
SUPPORTED_ENTITY_FILTER_OPERATORS = frozenset(
    {"EQ", "IN", "NE", "GT", "GTE", "LT", "LTE"}
)
ENTITY_SEMANTIC_ROLES = frozenset(
    {"KEY", "ENTITY", "ENTITY_KEY", "PRIMARY_KEY", "IDENTIFIER", "UNIQUE_KEY"}
)
NO_TIME_ENTITY_LOOKUP_MODES = frozenset(
    {"not_required", "global", "unbounded", "all_partitions"}
)

# These are the table-level semantic files that are actually overlaid by every
# active asset reader. Keep the runtime loader and activation identity derived
# from one structural contract so a new sidecar cannot silently bypass version
# invalidation.
ACTIVE_SEMANTIC_SIDECAR_FIELDS = {
    "schemaColumns": "schema.json",
    "semanticColumns": "semantic_columns.json",
    "metrics": "metrics.json",
    "terms": "terms.json",
    "knowledgeRules": "knowledge_rules.json",
}
ACTIVE_TOPIC_SEMANTIC_FILENAMES = frozenset({"manifest.json", "relationships.json"})
ACTIVE_TABLE_SEMANTIC_FILENAMES = frozenset({"asset.json", *ACTIVE_SEMANTIC_SIDECAR_FIELDS.values()})

SemanticActivationSignature = Tuple[Tuple[str, int, int, int, int], ...]


class TopicAssetService:
    SEMANTIC_LIST_FIELDS = set(ACTIVE_SEMANTIC_SIDECAR_FIELDS)
    SEMANTIC_SIDECAR_FILES = {
        "schema.json",
        "semantic_columns.json",
        "metrics.json",
        "terms.json",
        "knowledge_rules.json",
        "schema.md",
        "semantic_columns.md",
        "metrics.md",
        "terms.md",
        "knowledge_rules.md",
        "review.md",
    }
    MANAGED_TABLE_FILENAMES = {
        "asset.json",
        "sample_rows.json",
        "sample_profile.json",
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
        self._topic_contract_cache: Dict[str, Dict[str, Any]] = {}
        self._semantic_source_hash_cache: Dict[
            Tuple[str, ...],
            Tuple[SemanticActivationSignature, str],
        ] = {}

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
            active_asset = self._canonicalize_pending_asset(pending, topic, table_name)
            active_asset.update(
                {
                    "status": "PUBLISHED",
                    "reviewedAt": datetime.utcnow().isoformat() + "Z",
                    "reviewer": reviewer,
                    "reviewNote": review_note,
                }
            )
            changed_files: List[str] = []
            unchanged_files: List[str] = []
            deleted_files: List[str] = []
            for name in sorted(self.MANAGED_TABLE_FILENAMES):
                source_path = pending / name
                target_path = target / name
                if source_path.exists() and source_path.is_file():
                    # Keep the reviewed candidate immutable. Publication
                    # materializes a governed active copy; if a later index or
                    # governance step fails, pending cannot falsely look live.
                    source_bytes = (
                        json.dumps(active_asset, ensure_ascii=False, indent=2).encode("utf-8")
                        if name == "asset.json"
                        else source_path.read_bytes()
                    )
                    if target_path.exists() and target_path.is_file() and target_path.read_bytes() == source_bytes:
                        unchanged_files.append(name)
                        continue
                    target_path.write_bytes(source_bytes)
                    changed_files.append(name)
                    continue
                if target_path.exists() and target_path.is_file() and name not in self.GOVERNANCE_FILENAMES:
                    target_path.unlink()
                    deleted_files.append(name)
            for name in sorted(self.SEMANTIC_SIDECAR_FILES):
                target_path = target / name
                if target_path.exists() and target_path.is_file():
                    target_path.unlink()
                    deleted_files.append(name)
            manifest_changed = self._upsert_published_manifest_entry(topic, active_asset)
            self._table_asset_cache.pop((topic, table_name), None)
            self._manifest_cache.pop(topic, None)
            self._relationship_cache.pop(topic, None)
            self._topic_names_cache = None
            self._topic_contract_cache.pop(topic, None)
            self._semantic_source_hash_cache.clear()
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
                "manifestChanged": manifest_changed,
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

    def load_topic_contract(self, topic: str) -> Dict[str, Any]:
        """Load the open topic/category contract declared by published assets."""

        topic_name = str(topic or "").strip()
        if topic_name in self._topic_contract_cache:
            return dict(self._topic_contract_cache[topic_name])
        manifest = self.load_manifest(topic_name)
        sources: List[Dict[str, Any]] = [item for item in manifest if isinstance(item, dict)]
        for item in manifest:
            table = str(item.get("tableName") or "") if isinstance(item, dict) else ""
            if table:
                sources.append(self.load_table_asset(topic_name, table))
        category = ""
        display_name = ""
        aliases: List[str] = []
        metadata: Dict[str, Any] = {}
        for source in sources:
            declared = source.get("topicContract") if isinstance(source.get("topicContract"), dict) else {}
            if not category:
                category = str(
                    declared.get("categoryId")
                    or source.get("questionCategory")
                    or source.get("categoryId")
                    or source.get("category")
                    or ""
                ).strip()
            if not display_name:
                display_name = str(
                    declared.get("displayName")
                    or source.get("topicDisplayName")
                    or source.get("displayName")
                    or ""
                ).strip()
            for values in (
                declared.get("aliases") or [],
                source.get("topicAliases") or [],
                source.get("categoryAliases") or [],
            ):
                candidates = values if isinstance(values, list) else [values]
                aliases.extend(str(value).strip() for value in candidates if str(value or "").strip())
            for key in (
                "topicRole",
                "routingRole",
                "riskLevel",
                "openDiagnostic",
                "diagnosticProfiles",
                "diagnosticGoals",
                "diagnosticIntents",
                "clarificationContracts",
                "clarificationLabel",
                "linkedTopics",
            ):
                value = declared.get(key) if key in declared else source.get(key)
                if value not in (None, "", []) and key not in metadata:
                    metadata[key] = value
        declared_category = category
        effective_category = topic_name if not category or category == str(QuestionCategory.UNKNOWN) else category
        category_id = register_topic_contract(
            topic_name,
            effective_category,
            display_name or topic_name,
            list(dict.fromkeys(aliases)),
        )
        contract = {
            "topic": topic_name,
            "categoryId": category_id,
            "declaredCategoryId": declared_category,
            "displayName": display_name or topic_name,
            "aliases": list(dict.fromkeys(aliases)),
            "metadata": metadata,
        }
        self._topic_contract_cache[topic_name] = contract
        return dict(contract)

    def topic_contracts(self) -> List[Dict[str, Any]]:
        return [self.load_topic_contract(topic) for topic in self.all_topic_names()]

    def resolve_topic_category(self, value: Any) -> QuestionCategory:
        """Resolve an asset topic/display/category value without a closed map."""

        raw = str(getattr(value, "value", value) or "").strip()
        if not raw:
            return QuestionCategory.UNKNOWN
        for contract in self.topic_contracts():
            names = {
                str(contract.get("topic") or ""),
                str(contract.get("categoryId") or ""),
                str(contract.get("displayName") or ""),
                *[str(item) for item in contract.get("aliases") or []],
            }
            if raw in names:
                return QuestionCategory(contract.get("categoryId") or raw)
        return QuestionCategory(raw)

    def topic_names_for_categories(self, categories: Iterable[QuestionCategory]) -> List[str]:
        wanted = {str(getattr(item, "value", item) or "").strip() for item in categories}
        names: List[str] = []
        for contract in self.topic_contracts():
            candidates = {
                str(contract.get("topic") or ""),
                str(contract.get("categoryId") or ""),
                str(contract.get("displayName") or ""),
                *[str(item) for item in contract.get("aliases") or []],
            }
            topic_name = str(contract.get("topic") or "")
            if wanted.intersection(candidates) and topic_name and topic_name not in names:
                names.append(topic_name)
        return names

    def all_topic_names(self) -> List[str]:
        if self._topic_names_cache is not None:
            return self._topic_names_cache
        if not self.root.exists():
            return []
        self._topic_names_cache = [path.name for path in sorted(self.root.iterdir()) if path.is_dir()]
        return self._topic_names_cache

    def semantic_source_hash(self, topics: Iterable[str]) -> str:
        cache_key = tuple(sorted({str(item or "") for item in topics if item}))
        files = self.canonical_semantic_files(cache_key)
        signature = semantic_activation_signature(self.root, files)
        if signature is None:
            return ""
        cached = self._semantic_source_hash_cache.get(cache_key)
        if cached and cached[0] == signature:
            return cached[1]
        digest = active_semantic_activation_digest(self.root, files)
        # Do not cache a digest assembled while activation files were changing.
        # A following call will retry from a coherent filesystem snapshot.
        if not digest or semantic_activation_signature(self.root, files) != signature:
            return ""
        self._semantic_source_hash_cache[cache_key] = (signature, digest)
        return digest

    def semantic_table_source_hash(self, topic: str, table: str) -> str:
        """Return the active semantic identity relevant to one governed table."""

        candidates = [
            *(self.root / topic / name for name in ACTIVE_TOPIC_SEMANTIC_FILENAMES),
            *(
                self.table_asset_dir(topic, table) / name
                for name in ACTIVE_TABLE_SEMANTIC_FILENAMES
            ),
        ]
        files = [
            path
            for path in candidates
            if path.exists()
            and path.is_file()
            and is_active_semantic_activation_file(path, self.root)
        ]
        return active_semantic_activation_digest(self.root, files)

    def canonical_semantic_files(self, topics: Iterable[str]) -> List[Path]:
        files: List[Path] = []
        for topic in sorted({str(item or "") for item in topics if item}):
            topic_dir = self.root / topic
            if not topic_dir.exists():
                continue
            for name in ACTIVE_TOPIC_SEMANTIC_FILENAMES:
                path = topic_dir / name
                if path.exists() and path.is_file() and is_active_semantic_activation_file(path, self.root):
                    files.append(path)
            tables_dir = topic_dir / "tables"
            if not tables_dir.exists():
                continue
            for table_dir in sorted((path for path in tables_dir.iterdir() if path.is_dir()), key=lambda path: path.name):
                for name in ACTIVE_TABLE_SEMANTIC_FILENAMES:
                    path = table_dir / name
                    if path.exists() and path.is_file() and is_active_semantic_activation_file(path, self.root):
                        files.append(path)
        return sorted(files, key=lambda path: path.relative_to(self.root).as_posix())

    def load_topic_context(self, topic_names: Iterable[str]) -> str:
        parts: List[str] = []
        for topic in topic_names:
            manifest = self.load_manifest(str(topic))
            if manifest:
                parts.append("## %s/manifest.json\n%s" % (topic, json.dumps(manifest, ensure_ascii=False)[:5000]))
            for item in manifest:
                table = str(item.get("tableName") or "")
                if not table:
                    continue
                asset = self.load_table_asset(str(topic), table)
                parts.append("## %s/%s/asset.json\n%s" % (topic, table, compact_semantic_asset_for_recall(asset)[:5000]))
        return "\n\n".join(parts)

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
            }
        else:
            asset = {**asset}
            asset.setdefault("topic", topic)
            asset.setdefault("tableName", table)
        for field, file_name in ACTIVE_SEMANTIC_SIDECAR_FIELDS.items():
            sidecar = read_json(table_dir / file_name)
            if isinstance(sidecar, list):
                asset[field] = sidecar
        for field in self.SEMANTIC_LIST_FIELDS:
            if not isinstance(asset.get(field), list):
                asset[field] = []
        asset = enforce_sample_evidence_governance(asset)
        self._table_asset_cache[cache_key] = asset
        return asset

    def _canonicalize_pending_asset(self, pending_dir: Path, topic: str, table: str) -> Dict[str, Any]:
        asset = read_json(pending_dir / "asset.json")
        payload: Dict[str, Any] = asset if isinstance(asset, dict) else {}
        payload.setdefault("topic", topic)
        payload.setdefault("tableName", table)
        for field, file_name in ACTIVE_SEMANTIC_SIDECAR_FIELDS.items():
            if isinstance(payload.get(field), list):
                continue
            sidecar = read_json(pending_dir / file_name)
            payload[field] = sidecar if isinstance(sidecar, list) else []
        return payload

    def _upsert_published_manifest_entry(self, topic: str, asset: Dict[str, Any]) -> bool:
        table = str(asset.get("tableName") or "").strip()
        if not table:
            raise ValueError("published semantic asset has no tableName")
        manifest_path = self.root / topic / "manifest.json"
        current = read_json(manifest_path)
        items = [dict(item) for item in current if isinstance(item, dict)] if isinstance(current, list) else []
        existing = next((item for item in items if str(item.get("tableName") or "") == table), {})
        metrics = [item for item in asset.get("metrics") or [] if isinstance(item, dict)]
        rules = [item for item in asset.get("knowledgeRules") or [] if isinstance(item, dict)]
        schema = [item for item in asset.get("schemaColumns") or [] if isinstance(item, dict)]
        usage = normalize_table_usage_profile(asset.get("tableUsageProfile") or {}, table)
        preferred_for = list(existing.get("preferredFor") or usage.get("defaultForIntents") or [])
        if not preferred_for:
            if schema:
                preferred_for.append("DETAIL")
            if metrics:
                preferred_for.extend(["METRIC", "TOPN", "GROUP_AGG"])
            if rules:
                preferred_for.append("RULE_REFERENCE")
        updated = {
            **existing,
            "tableName": table,
            "tableComment": str(asset.get("tableComment") or existing.get("tableComment") or ""),
            # L0 summaries are deliberately curated and opt-in.  Do not derive
            # them from schema, metrics or free-form semantic sidecars: the
            # Topic manifest is the only context available before the model
            # chooses a table and must not disclose lower-layer definitions.
            "businessSummary": str(
                asset.get("businessSummary")
                or existing.get("businessSummary")
                or asset.get("tableComment")
                or existing.get("tableComment")
                or table
            ),
            "dataGrain": str(asset.get("dataGrain") or existing.get("dataGrain") or ""),
            "timeColumn": str(asset.get("timeColumn") or existing.get("timeColumn") or ""),
            "merchantFilterColumn": str(
                asset.get("merchantFilterColumn") or existing.get("merchantFilterColumn") or ""
            ),
            "freshnessType": str(asset.get("freshnessType") or existing.get("freshnessType") or ""),
            "supportsDetail": bool(schema),
            "supportsMetrics": bool(metrics),
            "preferredFor": dedupe_strings(preferred_for),
            "metricCount": len(metrics),
            "ruleCount": len(rules),
            "status": "PUBLISHED",
        }
        rewritten: List[Dict[str, Any]] = []
        replaced = False
        for item in items:
            if str(item.get("tableName") or "") == table:
                rewritten.append(updated)
                replaced = True
            else:
                rewritten.append(item)
        if not replaced:
            rewritten.append(updated)
        changed = rewritten != items
        if changed or not manifest_path.exists():
            write_json(manifest_path, rewritten)
        return changed

    def load_table_schema(self, topic: str, table: str) -> List[Dict[str, Any]]:
        data = self.load_table_asset(topic, table).get("schemaColumns")
        return data if isinstance(data, list) else []

    def load_table_semantic_columns(self, topic: str, table: str) -> List[Dict[str, Any]]:
        data = self.load_table_asset(topic, table).get("semanticColumns")
        return data if isinstance(data, list) else []

    def load_table_metrics(self, topic: str, table: str) -> List[Dict[str, Any]]:
        asset = self.load_table_asset(topic, table)
        data = asset.get("metrics")
        if not isinstance(data, list):
            return []
        table_time_column = str(asset.get("timeColumn") or "").strip()
        return [
            {
                **metric,
                "timeColumn": str(metric.get("timeColumn") or table_time_column or "").strip(),
            }
            for metric in data
            if isinstance(metric, dict)
        ]

    def load_table_terms(self, topic: str, table: str) -> List[Dict[str, Any]]:
        data = self.load_table_asset(topic, table).get("terms")
        return data if isinstance(data, list) else []

    def load_table_knowledge_rules(self, topic: str, table: str) -> List[Dict[str, Any]]:
        data = self.load_table_asset(topic, table).get("knowledgeRules")
        return data if isinstance(data, list) else []

    def always_apply_rules(
        self,
        categories: Iterable[QuestionCategory],
        user_scope: Optional[Dict[str, Any]] = None,
        merchant_id: str = "",
    ) -> List[Dict[str, Any]]:
        rules: List[Dict[str, Any]] = []
        seen: set[str] = set()
        relevant_topics = set(self.topic_names_for_categories(categories))
        scope = user_scope or {}
        region = str(scope.get("region") or "")
        store_ids = {str(item) for item in scope.get("storeIds") or scope.get("store_ids") or []}
        now = datetime.utcnow()
        for topic in self.all_topic_names():
            for manifest_item in self.load_manifest(topic):
                table = str(manifest_item.get("tableName") or "")
                if not table:
                    continue
                for item in self.load_table_knowledge_rules(topic, table):
                    if not isinstance(item, dict) or not bool(item.get("alwaysApply")):
                        continue
                    rule_scope = str(item.get("scope") or "topic").lower()
                    if rule_scope != "global" and topic not in relevant_topics:
                        continue
                    if not always_apply_rule_active(item, now):
                        continue
                    regions = {str(value) for value in item.get("regions") or item.get("regionIds") or []}
                    merchants = {str(value) for value in item.get("merchantIds") or []}
                    stores = {str(value) for value in item.get("storeIds") or []}
                    if regions and region not in regions:
                        continue
                    if merchants and str(merchant_id or "") not in merchants:
                        continue
                    if stores and not (stores & store_ids):
                        continue
                    content = str(item.get("content") or item.get("description") or "").strip()
                    title = str(item.get("title") or item.get("name") or "强制业务规则").strip()
                    fingerprint = "%s:%s:%s" % (topic, table, content)
                    if not content or fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    rules.append({**item, "topic": topic, "tableName": table, "title": title, "content": content, "priority": int(item.get("priority") or 0)})
        rules.sort(key=lambda item: (-int(item.get("priority") or 0), str(item.get("topic") or ""), str(item.get("title") or "")))
        winners: List[Dict[str, Any]] = []
        conflicts: Dict[str, Dict[str, Any]] = {}
        for rule in rules:
            conflict_key = str(rule.get("conflictKey") or rule.get("ruleKey") or rule.get("title") or "")
            current = conflicts.get(conflict_key)
            if current and str(current.get("content") or "") != str(rule.get("content") or ""):
                current.setdefault("suppressedConflicts", []).append({"ruleId": rule.get("ruleId"), "priority": rule.get("priority"), "content": rule.get("content")})
                continue
            conflicts[conflict_key] = rule
            winners.append(rule)
        budget = max(1, int(getattr(self.settings, "always_apply_rule_budget", 20) or 20))
        return winners[:budget]

    def stage_knowledge_suggestion_patch(self, topic: str, table: str, suggestion: Dict[str, Any]) -> Dict[str, Any]:
        """Materialize an approved suggestion into a reviewable pending semantic asset."""
        target = self.table_asset_dir(topic, table)
        pending = self.root / topic / "pending" / table
        if not target.exists() and not pending.exists():
            return {"success": False, "status": "TARGET_ASSET_NOT_FOUND", "topic": topic, "tableName": table}
        if not pending.exists():
            shutil.copytree(target, pending)
        asset_path = pending / "asset.json"
        asset = read_json(asset_path)
        if not isinstance(asset, dict):
            asset = self.load_table_asset(topic, table)
        asset = {**asset, "topic": topic, "tableName": table}
        suggestion_id = str(suggestion.get("suggestionId") or suggestion.get("suggestion_id") or "")
        suggestion_type = str(suggestion.get("suggestionType") or suggestion.get("suggestion_type") or "rule").lower()
        payload = suggestion.get("payload") if isinstance(suggestion.get("payload"), dict) else {}
        content = str(payload.get("correctionText") or payload.get("content") or payload.get("question") or "").strip()
        metric_name = str(suggestion.get("metricName") or suggestion.get("metric_name") or "").strip()
        aliases = [str(item) for item in suggestion.get("aliases") or [] if str(item).strip()]
        before = {
            "metrics": len(asset.get("metrics") or []),
            "terms": len(asset.get("terms") or []),
            "knowledgeRules": len(asset.get("knowledgeRules") or []),
        }
        patched_kind = "knowledgeRules"
        if suggestion_type == "term":
            patched_kind = "terms"
            item = {
                "term": metric_name or (aliases[0] if aliases else suggestion_id),
                "description": content,
                "aliases": aliases,
                "sourceSuggestionId": suggestion_id,
                "reviewStatus": "approved",
            }
            asset["terms"] = upsert_semantic_suggestion_item(asset.get("terms") or [], item, "sourceSuggestionId")
        elif suggestion_type == "metric" and (suggestion.get("sourceFields") or suggestion.get("aggregation")):
            patched_kind = "metrics"
            metric_key = stable_cache_key("suggested_metric", {"id": suggestion_id, "name": metric_name})[:24]
            item = {
                "metricKey": metric_key,
                "businessName": metric_name or suggestion_id,
                "aliases": aliases,
                "sourceColumns": list(suggestion.get("sourceFields") or []),
                "aggregation": str(suggestion.get("aggregation") or ""),
                "filterConditions": list(suggestion.get("filterConditions") or []),
                "description": content,
                "sourceSuggestionId": suggestion_id,
                "reviewStatus": "approved",
            }
            asset["metrics"] = upsert_semantic_suggestion_item(asset.get("metrics") or [], item, "sourceSuggestionId")
        else:
            item = {
                "ruleId": "suggestion_%s" % re.sub(r"[^a-zA-Z0-9_]+", "_", suggestion_id).strip("_"),
                "title": metric_name or "商家确认经营规则",
                "content": content or "商家已确认该经营口径。",
                "keywords": list(dict.fromkeys([metric_name, *aliases]))[:12],
                "alwaysApply": bool(payload.get("alwaysApply", True)),
                "sourceSuggestionId": suggestion_id,
                "reviewStatus": "approved",
            }
            asset["knowledgeRules"] = upsert_semantic_suggestion_item(asset.get("knowledgeRules") or [], item, "sourceSuggestionId")
        asset["status"] = "PENDING_REVIEW"
        asset.setdefault("semanticGovernance", {})["lastKnowledgeSuggestionId"] = suggestion_id
        write_json(asset_path, asset)
        after = {
            "metrics": len(asset.get("metrics") or []),
            "terms": len(asset.get("terms") or []),
            "knowledgeRules": len(asset.get("knowledgeRules") or []),
        }
        diff = {
            "success": True,
            "status": "PATCH_STAGED",
            "topic": topic,
            "tableName": table,
            "suggestionId": suggestion_id,
            "patchedKind": patched_kind,
            "before": before,
            "after": after,
            "pendingPath": str(pending),
        }
        write_json(pending / "knowledge_suggestion_patch.json", diff)
        return diff

    def verify_published_suggestion(self, topic: str, table: str, suggestion_id: str) -> Dict[str, Any]:
        self._table_asset_cache.pop((topic, table), None)
        asset = self.load_table_asset(topic, table)
        matches = []
        for kind in ["metrics", "terms", "knowledgeRules"]:
            for item in asset.get(kind) or []:
                if isinstance(item, dict) and str(item.get("sourceSuggestionId") or "") == suggestion_id:
                    matches.append({"kind": kind, "item": item})
        return {
            "success": bool(matches),
            "status": "READBACK_VERIFIED" if matches else "READBACK_MISSING",
            "suggestionId": suggestion_id,
            "matches": matches,
        }

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
    TOPIC_INDEX_KIND = "TOPIC_INDEX"
    TABLE_KIND = "TABLE_ASSET"
    METRIC_KIND = "METRIC"
    RELATIONSHIP_KIND = "RELATIONSHIPS"
    RELATIONSHIP_INDEX_KIND = "RELATIONSHIP_CATALOG"
    RELATIONSHIP_ENTRY_KIND = "RELATIONSHIP"
    TABLE_DETAIL_KIND = "TABLE_DETAIL"
    TABLE_SECTION_KINDS = {
        "metrics": "METRIC_CATALOG",
        "columns": "COLUMN_DETAILS",
        "schema": "SCHEMA",
        "terms": "TERMINOLOGY",
        "rules": "BUSINESS_RULES",
    }
    TABLE_SECTION_FIELDS = {
        "metrics": "metrics",
        "columns": "semanticColumns",
        "schema": "schemaColumns",
        "terms": "terms",
        "rules": "knowledgeRules",
    }
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
        path: str = "",
        query: str = "",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        normalized_path = normalize_semantic_path(path)
        if normalized_path:
            return self._ls_path(normalized_path, query=query, limit=limit)
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
                # L0 listing must be question-independent and must never load
                # asset.json.  It exposes only the next browse coordinate.
                ref = self.table_detail_ref(topic_name, table, manifest_item)
                if not terms or score_document(terms, ref["searchText"]) > 0:
                    refs.append(ref)
        refs.sort(key=lambda item: score_document(terms, item["searchText"]) if terms else 0.0, reverse=True)
        return [self._public_ref(item) for item in refs[: max(1, limit)]]

    def read(
        self,
        ref_id: str = "",
        path: str = "",
        max_chars: int = 20_000,
        offset: int = 0,
    ) -> Dict[str, Any]:
        wanted_ref = str(ref_id or "").strip()
        wanted_path = normalize_semantic_path(path)
        if wanted_ref and wanted_path:
            ref_by_id = self._resolve_ref(wanted_ref, "")
            ref_by_path = self._resolve_ref("", wanted_path)
            if (
                not ref_by_id
                or not ref_by_path
                or str(ref_by_id.get("refId") or "") != str(ref_by_path.get("refId") or "")
                or str(ref_by_id.get("path") or "") != str(ref_by_path.get("path") or "")
            ):
                return {
                    "success": False,
                    "error": "SEMANTIC_REF_PATH_CONFLICT",
                    "refId": wanted_ref,
                    "path": wanted_path,
                }
            ref = ref_by_id
        else:
            ref = self._resolve_ref(wanted_ref, wanted_path)
        if not ref:
            return {"success": False, "error": "SEMANTIC_REF_NOT_FOUND", "refId": ref_id, "path": path}
        if (
            str(ref.get("kind") or "").upper() == self.TABLE_KIND
            or str(ref.get("path") or "").endswith("/asset.json")
        ):
            return {
                "success": False,
                "error": "FULL_TABLE_ASSET_DENIED",
                "refId": str(ref.get("refId") or ref_id),
                "path": str(ref.get("path") or path),
                "instruction": "Read the table detail.json and then only required child refs.",
            }
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
        path: str = "",
    ) -> List[Dict[str, Any]]:
        terms = question_match_terms(query)
        if not terms:
            return []
        hits: List[Dict[str, Any]] = []
        refs = self._grep_refs_for_path(path) if normalize_semantic_path(path) else self._all_refs(topic_categories, topic)
        for ref in refs:
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

    def _grep_refs_for_path(self, path: str) -> List[Dict[str, Any]]:
        """Resolve a bounded filesystem subtree for targeted semantic grep.

        Topic-wide grep remains index-only so a broad search cannot materialize
        every table asset.  Once Core chooses a table or section, grep may
        inspect that bounded subtree and returns exact/index paths which still
        require a subsequent trusted read_file call before binding.
        """

        normalized = normalize_semantic_path(path).rstrip("/")
        exact = self._resolve_ref("", normalized)
        if exact:
            return [exact]

        section_match = re.fullmatch(
            r"topics/([^/]+)/tables/([^/]+)/(metrics|columns|terms|rules)",
            normalized,
        )
        if section_match:
            topic, table, section = section_match.groups()
            section_ref = self.table_section_ref(topic, table, section)
            if not section_ref:
                return []
            try:
                payload = json.loads(section_ref.get("content") or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                payload = {}
            refs: List[Dict[str, Any]] = []
            for item in payload.get("entries") or []:
                if not isinstance(item, dict):
                    continue
                entry = self.table_entry_ref(topic, table, section, str(item.get("key") or ""))
                if entry:
                    refs.append(entry)
            return refs

        table_match = re.fullmatch(r"topics/([^/]+)/tables/([^/]+)", normalized)
        if table_match:
            topic, table = table_match.groups()
            if not self._manifest_item(topic, table):
                return []
            refs = [self.table_detail_ref(topic, table)]
            refs.extend(
                ref
                for section in self.TABLE_SECTION_FIELDS
                if (ref := self.table_section_ref(topic, table, section)) is not None
            )
            return refs

        tables_match = re.fullmatch(r"topics/([^/]+)/tables", normalized)
        if tables_match:
            topic = tables_match.group(1)
            return [
                self.table_detail_ref(topic, str(item.get("tableName") or ""), item)
                for item in self.topic_assets.load_manifest(topic)
                if isinstance(item, dict) and str(item.get("tableName") or "")
            ]

        topic_match = re.fullmatch(r"topics/([^/]+)", normalized)
        if topic_match:
            return self._all_refs(topic=topic_match.group(1))
        return []

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
        seen: Set[Tuple[str, str]] = set()
        table_filter = {str(table) for table in allowed_tables or [] if table}
        relationship_topic_filter = {str(topic) for topic in allowed_relationship_topics or [] if topic}
        for item in source_refs.values():
            metadata = item.metadata or {}
            semantic_path = str(metadata.get("semanticPath") or "")
            ref_id = str(metadata.get("semanticRefId") or item.doc_id or "")
            kind = str(metadata.get("semanticKind") or item.source_type or "").upper()
            topic = str(item.topic or metadata.get("topic") or "")
            table = str(item.table or metadata.get("tableName") or "")
            if str(item.source_type or "").upper() == "GOVERNED_RULE":
                continue
            if kind in {"TABLE_ASSET", "SEMANTIC_TABLE_ASSET"} and topic and table:
                ref_id = semantic_table_detail_ref_id(topic, table)
                semantic_path = semantic_table_detail_path(topic, table)
            elif kind in {"METRIC", "SEMANTIC_METRIC"} and topic and table and ":metric:" in ref_id:
                semantic_path = semantic_metric_path(topic, table, ref_id.split(":metric:", 1)[1])
            elif kind in {
                "RELATIONSHIP",
                "RELATIONSHIPS",
                "RELATIONSHIP_CATALOG",
                "SEMANTIC_RELATIONSHIP",
            } and topic:
                relationship_id = str(metadata.get("relationshipId") or "").strip()
                relationship_key = semantic_relationship_key_for_name(
                    self.topic_assets.load_relationships(topic),
                    relationship_id,
                )
                if relationship_key:
                    ref_id = semantic_relationship_entry_ref_id(topic, relationship_key)
                    semantic_path = semantic_relationship_entry_path(topic, relationship_key)
                else:
                    ref_id = semantic_relationship_index_ref_id(topic)
                    semantic_path = semantic_relationship_index_path(topic)
            if not semantic_path or not ref_id.startswith("semantic:"):
                continue
            resolved_by_ref = self._resolve_ref(ref_id, "")
            resolved_by_path = self._resolve_ref("", semantic_path)
            if not resolved_by_ref or not resolved_by_path:
                continue
            if (
                str(resolved_by_ref.get("refId") or "") != str(resolved_by_path.get("refId") or "")
                or str(resolved_by_ref.get("path") or "") != str(resolved_by_path.get("path") or "")
            ):
                continue
            resolved = resolved_by_ref
            resolved_table = str(resolved.get("table") or table)
            resolved_topic = str(resolved.get("topic") or topic)
            if resolved_table and table_filter and resolved_table not in table_filter:
                continue
            if not resolved_table and relationship_topic_filter and resolved_topic not in relationship_topic_filter:
                continue
            identity = (str(resolved.get("refId") or ""), str(resolved.get("path") or ""))
            if identity in seen:
                continue
            seen.add(identity)
            refs.append(
                add_context_uri({
                    "refId": identity[0],
                    "path": identity[1],
                    "kind": str(resolved.get("kind") or kind),
                    "topic": resolved_topic,
                    "table": resolved_table,
                    "title": item.title,
                    "estimatedChars": int(resolved.get("estimatedChars") or metadata.get("estimatedChars") or len(item.content or "")),
                    "offloadRecommended": bool(resolved.get("offloadRecommended") or metadata.get("offloadRecommended")),
                }, ref_id=identity[0], topic=resolved_topic, table=resolved_table, kind=str(resolved.get("kind") or kind), path=identity[1])
            )
        return {
            "mode": "filesystem_as_context",
            "uriScheme": "merchant://",
            "policy": "start from topic manifests and table detail files; open section indexes and exact entries only as needed; full assets remain an optional fallback",
            "layers": {
                "L0": "topic/table/metric summaries for routing and quick relevance checks",
                "L1": "table, metric, relationship and rule overviews for planning and rerank",
                "L2": "full schema, metric formulas, rules, rows or artifacts loaded only on demand",
            },
            "progressiveDisclosure": [
                "1. topic manifest: available tables and business summaries",
                "2. table detail: grain, time/scope columns and child section coordinates",
                "3. section index then exact metric/column/rule entry; relationships only when an edge is needed",
                "4. workspace artifacts: read query graphs, SQL, rows or evidence reports by path when needed",
            ],
            "roots": [
                "topics/<topic>/manifest.json",
                "topics/<topic>/tables/<table>/detail.json",
                "topics/<topic>/tables/<table>/<section>/index.json",
                "topics/<topic>/relationships.json",
            ],
            "refs": refs[:limit],
        }

    def manifest_ref(self, topic: str) -> Dict[str, Any]:
        manifest = self.topic_assets.load_manifest(topic)
        compact_tables: List[Dict[str, Any]] = []
        search_parts: List[str] = [topic]
        for item in manifest:
            table = str(item.get("tableName") or "")
            title = str(item.get("tableComment") or item.get("title") or table)
            business_summary = str(item.get("businessSummary") or "").strip()
            compact_table = {
                "topic": topic,
                "table": table,
                "title": title,
                "detailRefId": semantic_table_detail_ref_id(topic, table),
                "detailPath": semantic_table_detail_path(topic, table),
            }
            if business_summary:
                compact_table["businessSummary"] = business_summary
            compact_tables.append(compact_table)
            search_parts.extend([table, title, business_summary])
        content_payload = {
            "topic": topic,
            "layer": "manifest",
            "policy": "Choose a table, then read only its detailRefId. This layer contains no metric, column, schema, relationship, or rule definitions.",
            "tables": compact_tables,
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

    def topic_index_ref(self) -> Dict[str, Any]:
        """Expose a thin global Topic directory without loading table assets."""

        topics: List[Dict[str, Any]] = []
        search_parts: List[str] = []
        for topic in self.topic_assets.all_topic_names():
            contract = self.topic_assets.load_topic_contract(topic)
            summaries = dedupe_strings(
                [
                    str(item.get("businessSummary") or "").strip()
                    for item in self.topic_assets.load_manifest(topic)
                    if isinstance(item, dict) and str(item.get("businessSummary") or "").strip()
                ]
            )
            item = {
                "topic": topic,
                "displayName": str(contract.get("displayName") or topic),
                "aliases": [str(value) for value in contract.get("aliases") or []],
                "description": "；".join(summaries[:3]),
                "manifestRefId": semantic_manifest_ref_id(topic),
                "manifestPath": semantic_manifest_path(topic),
            }
            topics.append(item)
            search_parts.extend(
                [
                    topic,
                    item["displayName"],
                    *item["aliases"],
                    item["description"],
                ]
            )
        payload = {
            "layer": "topic_index",
            "policy": (
                "Use this index only when the seed Topic does not cover the question. "
                "Read one candidate manifest before searching or opening files in that Topic."
            ),
            "topics": topics,
        }
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        return add_context_uri(
            {
                "refId": "semantic:topics:index",
                "kind": self.TOPIC_INDEX_KIND,
                "topic": "",
                "table": "",
                "path": "topics/index.json",
                "title": "Topic Index",
                "summary": "%d published Topics" % len(topics),
                "layers": {"topics": len(topics), "layer": "topic_index"},
                "estimatedChars": len(content),
                "offloadRecommended": False,
                "content": content,
                "searchText": "\n".join(search_parts),
            },
            ref_id="semantic:topics:index",
            kind=self.TOPIC_INDEX_KIND,
            path="topics/index.json",
        )

    def table_detail_ref(
        self,
        topic: str,
        table: str,
        manifest_item: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Build the L1 table detail strictly from the published L0 manifest."""

        item = dict(manifest_item or self._manifest_item(topic, table) or {})
        child_refs = [
            {
                "kind": kind,
                "refId": semantic_table_section_ref_id(topic, table, section),
                "path": semantic_table_section_path(topic, table, section),
            }
            for section, kind in self.TABLE_SECTION_KINDS.items()
        ]
        child_refs.append(
            {
                "kind": self.RELATIONSHIP_INDEX_KIND,
                "refId": semantic_relationship_index_ref_id(topic),
                "path": semantic_relationship_index_path(topic),
                "use": "read only when the plan needs a cross-table edge",
            }
        )
        navigation_hints = item.get("navigationHints") or {}
        semantic_navigation: Dict[str, Any] = {}
        if isinstance(navigation_hints, dict):
            for source_key, section, target_key in (
                ("metrics", "metrics", "metricLeaves"),
                ("columns", "columns", "columnLeaves"),
            ):
                leaves: List[Dict[str, Any]] = []
                raw_leaves = navigation_hints.get(source_key) or []
                if not isinstance(raw_leaves, list):
                    continue
                for raw_leaf in raw_leaves[:16]:
                    if isinstance(raw_leaf, str):
                        key = raw_leaf.strip()
                        aliases: List[str] = []
                    elif isinstance(raw_leaf, dict):
                        key = str(raw_leaf.get("key") or "").strip()
                        aliases = dedupe_strings(
                            [str(value) for value in raw_leaf.get("aliases") or []]
                        )[:8]
                    else:
                        continue
                    if not key or not re.fullmatch(r"[A-Za-z0-9_]+", key):
                        continue
                    leaf = {
                        "key": key,
                        "refId": semantic_table_entry_ref_id(
                            topic,
                            table,
                            section,
                            key,
                        ),
                        "path": semantic_table_entry_path(
                            topic,
                            table,
                            section,
                            key,
                        ),
                    }
                    if aliases:
                        leaf["aliases"] = aliases
                    leaves.append(leaf)
                if leaves:
                    semantic_navigation[target_key] = leaves
        if semantic_navigation:
            semantic_navigation["policy"] = (
                "Navigation only. Read each exact leaf before binding it into a Contract; "
                "do not scan the broad index when one of these aliases matches."
            )
        payload = {
            "topic": topic,
            "tableName": table,
            "title": str(item.get("tableComment") or item.get("title") or table),
            "businessSummary": str(item.get("businessSummary") or ""),
            "dataGrain": str(item.get("dataGrain") or item.get("grain") or ""),
            "timeColumn": str(item.get("timeColumn") or ""),
            "merchantFilterColumn": str(item.get("merchantFilterColumn") or item.get("scopeFilterColumn") or ""),
            "freshnessType": str(item.get("freshnessType") or ""),
            "supportsDetail": bool(item.get("supportsDetail")),
            "supportsMetrics": bool(item.get("supportsMetrics")),
            "preferredFor": [str(value) for value in item.get("preferredFor") or []],
            "metricCount": int(item.get("metricCount") or 0),
            "ruleCount": int(item.get("ruleCount") or 0),
            "children": child_refs,
            "policy": "Choose only the metric/column/schema/rule index needed next. Full asset.json is not exposed to the Planner.",
        }
        if semantic_navigation:
            payload["semanticNavigation"] = semantic_navigation
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        ref_id = semantic_table_detail_ref_id(topic, table)
        path = semantic_table_detail_path(topic, table)
        return add_context_uri(
            {
                "refId": ref_id,
                "kind": self.TABLE_DETAIL_KIND,
                "topic": topic,
                "table": table,
                "path": path,
                "title": "%s/%s/detail" % (topic, table),
                "summary": payload["title"],
                "layers": {"childCount": len(child_refs)},
                "estimatedChars": len(content),
                "offloadRecommended": False,
                "content": content,
                "searchText": "\n".join(
                    [
                        topic,
                        table,
                        payload["title"],
                        payload["businessSummary"],
                        payload["dataGrain"],
                        *payload["preferredFor"],
                    ]
                ),
            },
            ref_id=ref_id,
            topic=topic,
            table=table,
            kind=self.TABLE_DETAIL_KIND,
            path=path,
        )

    def table_section_ref(self, topic: str, table: str, section: str) -> Dict[str, Any] | None:
        section_name = str(section or "").strip().lower()
        field = self.TABLE_SECTION_FIELDS.get(section_name)
        kind = self.TABLE_SECTION_KINDS.get(section_name)
        if not field or not kind or not self._manifest_item(topic, table):
            return None
        asset = self.topic_assets.load_table_asset(topic, table)
        values = asset.get(field)
        values = values if isinstance(values, list) else []
        if section_name == "schema":
            payload = {"topic": topic, "tableName": table, "section": section_name, field: values}
        else:
            entries = []
            entry_keys = semantic_table_entry_keys(section_name, values)
            for index, value in enumerate(values):
                if not isinstance(value, dict):
                    continue
                key = entry_keys[index]
                entries.append(
                    {
                        "key": key,
                        "title": semantic_table_entry_title(section_name, value, key),
                        "refId": semantic_table_entry_ref_id(topic, table, section_name, key),
                        "path": semantic_table_entry_path(topic, table, section_name, key),
                    }
                )
            payload = {
                "topic": topic,
                "tableName": table,
                "section": section_name,
                "entries": entries,
                "policy": "This is an index only. Read an exact entry ref before binding it into a plan.",
            }
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        ref_id = semantic_table_section_ref_id(topic, table, section_name)
        path = semantic_table_section_path(topic, table, section_name)
        return add_context_uri(
            {
                "refId": ref_id,
                "kind": kind,
                "topic": topic,
                "table": table,
                "path": path,
                "title": "%s/%s/%s" % (topic, table, section_name),
                "summary": "%d governed %s entries" % (len(values), section_name),
                "layers": {"section": section_name, "entryCount": len(values)},
                "estimatedChars": len(content),
                "offloadRecommended": len(content) > self.OFFLOAD_THRESHOLD_CHARS,
                "content": content,
                "searchText": json.dumps(values, ensure_ascii=False),
            },
            ref_id=ref_id,
            topic=topic,
            table=table,
            kind=kind,
            path=path,
        )

    def table_entry_ref(
        self,
        topic: str,
        table: str,
        section: str,
        entry_key: str,
    ) -> Dict[str, Any] | None:
        section_name = str(section or "").strip().lower()
        field = self.TABLE_SECTION_FIELDS.get(section_name)
        if not field or section_name == "schema" or not self._manifest_item(topic, table):
            return None
        asset = self.topic_assets.load_table_asset(topic, table)
        values = asset.get(field) if isinstance(asset.get(field), list) else []
        entry_keys = semantic_table_entry_keys(section_name, values)
        selected: Dict[str, Any] | None = None
        for index, value in enumerate(values):
            if not isinstance(value, dict):
                continue
            if entry_keys[index] == entry_key:
                selected = value
                break
        if selected is None:
            return None
        if section_name == "metrics":
            return self.metric_ref(topic, table, entry_key, selected)
        if section_name == "columns":
            selected = progressive_semantic_column_definition(asset, selected)
        kind = {"columns": "COLUMN", "terms": "TERM", "rules": "BUSINESS_RULE"}.get(section_name, "SEMANTIC_ENTRY")
        payload = {
            "topic": topic,
            "tableName": table,
            "section": section_name,
            "key": entry_key,
            "definition": selected,
        }
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        ref_id = semantic_table_entry_ref_id(topic, table, section_name, entry_key)
        path = semantic_table_entry_path(topic, table, section_name, entry_key)
        title = semantic_table_entry_title(section_name, selected, entry_key)
        return add_context_uri(
            {
                "refId": ref_id,
                "kind": kind,
                "topic": topic,
                "table": table,
                "path": path,
                "title": title,
                "summary": title,
                "layers": {"section": section_name, "entry": entry_key},
                "estimatedChars": len(content),
                "offloadRecommended": len(content) > self.OFFLOAD_THRESHOLD_CHARS,
                "content": content,
                "searchText": json.dumps(selected, ensure_ascii=False),
            },
            ref_id=ref_id,
            topic=topic,
            table=table,
            kind=kind,
            path=path,
        )

    def table_ref(self, topic: str, table: str, asset: Dict[str, Any] | None = None) -> Dict[str, Any]:
        asset = enforce_sample_evidence_governance(asset or self.topic_assets.load_table_asset(topic, table))
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

    def relationship_index_ref(
        self,
        topic: str,
        relationships: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        relationships = (
            relationships
            if relationships is not None
            else self.topic_assets.load_relationships(topic)
        )
        keys = semantic_table_entry_keys("relationships", relationships)
        entries: List[Dict[str, Any]] = []
        for index, relationship in enumerate(relationships):
            if not isinstance(relationship, dict):
                continue
            key = keys[index]
            entries.append(
                {
                    "key": key,
                    "name": str(relationship.get("name") or key),
                    "leftTable": str(relationship.get("leftTable") or ""),
                    "rightTable": str(relationship.get("rightTable") or ""),
                    "refId": semantic_relationship_entry_ref_id(topic, key),
                    "path": semantic_relationship_entry_path(topic, key),
                    "useCases": [
                        str(item)
                        for item in relationship.get("useCases") or []
                        if str(item or "").strip()
                    ],
                }
            )
        payload = {
            "topic": topic,
            "section": "relationships",
            "entries": entries,
            "policy": (
                "This is an index only. Read exactly one relationship entry before "
                "binding or requesting a Topic expansion."
            ),
        }
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        ref_id = semantic_relationship_index_ref_id(topic)
        path = semantic_relationship_index_path(topic)
        return add_context_uri(
            {
                "refId": ref_id,
                "kind": self.RELATIONSHIP_INDEX_KIND,
                "topic": topic,
                "table": "",
                "path": path,
                "title": "%s/relationships/index" % topic,
                "summary": "%d governed relationship entries" % len(entries),
                "layers": {"relationships": len(entries), "layer": "index"},
                "estimatedChars": len(content),
                "offloadRecommended": False,
                "content": content,
                "searchText": json.dumps(entries, ensure_ascii=False),
            },
            ref_id=ref_id,
            topic=topic,
            kind=self.RELATIONSHIP_INDEX_KIND,
            path=path,
        )

    def relationship_entry_ref(
        self,
        topic: str,
        entry_key: str,
        relationships: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any] | None:
        relationships = (
            relationships
            if relationships is not None
            else self.topic_assets.load_relationships(topic)
        )
        keys = semantic_table_entry_keys("relationships", relationships)
        selected: Dict[str, Any] | None = None
        for index, relationship in enumerate(relationships):
            if isinstance(relationship, dict) and keys[index] == entry_key:
                selected = dict(relationship)
                break
        if selected is None:
            return None
        payload = {
            "topic": topic,
            "section": "relationships",
            "key": entry_key,
            "relationships": [selected],
        }
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        ref_id = semantic_relationship_entry_ref_id(topic, entry_key)
        path = semantic_relationship_entry_path(topic, entry_key)
        left = str(selected.get("leftTable") or "")
        right = str(selected.get("rightTable") or "")
        return add_context_uri(
            {
                "refId": ref_id,
                "kind": self.RELATIONSHIP_ENTRY_KIND,
                "topic": topic,
                "table": left,
                "path": path,
                "title": "%s/%s relationship" % (
                    topic,
                    str(selected.get("name") or entry_key),
                ),
                "summary": "%s -> %s" % (left, right),
                "layers": {"relationship": entry_key, "layer": "entry"},
                "estimatedChars": len(content),
                "offloadRecommended": False,
                "content": content,
                "searchText": json.dumps(selected, ensure_ascii=False),
            },
            ref_id=ref_id,
            topic=topic,
            table=left,
            kind=self.RELATIONSHIP_ENTRY_KIND,
            path=path,
        )

    def metric_ref(
        self,
        topic: str,
        table: str,
        metric_key: str,
        metric: Dict[str, Any] | None = None,
    ) -> Dict[str, Any] | None:
        """Expose one published metric as an addressable virtual semantic file.

        Metric documents indexed for retrieval publish both a stable semantic
        ref and a real virtual file under ``metrics/<key>.json``.  The source
        remains the governed table asset, but callers never need a fragment
        path or the full asset to read one definition.
        """

        asset = self.topic_assets.load_table_asset(topic, table)
        selected = metric
        if not isinstance(selected, dict):
            matches = [
                item
                for item in asset.get("metrics") or []
                if isinstance(item, dict) and str(item.get("metricKey") or "") == metric_key
            ]
            selected = matches[0] if len(matches) == 1 else None
        if not isinstance(selected, dict):
            return None
        payload = {
            "topic": topic,
            "tableName": table,
            "metric": selected,
        }
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        ref_id = semantic_metric_ref_id(topic, table, metric_key)
        path = semantic_metric_path(topic, table, metric_key)
        title = str(selected.get("businessName") or selected.get("title") or metric_key)
        return add_context_uri(
            {
                "refId": ref_id,
                "kind": self.METRIC_KIND,
                "topic": topic,
                "table": table,
                "path": path,
                "title": "%s/%s/%s" % (topic, table, title),
                "summary": title,
                "layers": {"metric": metric_key, "layer": "metric_definition"},
                "estimatedChars": len(content),
                "offloadRecommended": len(content) > self.OFFLOAD_THRESHOLD_CHARS,
                "content": content,
                "searchText": compact_metric_for_recall(topic, table, selected),
            },
            ref_id=ref_id,
            topic=topic,
            table=table,
            kind=self.METRIC_KIND,
            path=path,
        )

    def _topics(self, topic_categories: Iterable[QuestionCategory] | None, topic: str) -> List[str]:
        if topic:
            return [topic]
        if topic_categories:
            topics = self.topic_assets.topic_names_for_categories(topic_categories)
            if topics:
                return topics
        return self.topic_assets.all_topic_names()

    def _all_refs(self, topic_categories: Iterable[QuestionCategory] | None = None, topic: str = "") -> List[Dict[str, Any]]:
        """Return only index-level refs.

        A global grep must not materialize every asset.json in a large catalog.
        Exact metric/column/rule definitions are resolved directly after the
        caller has selected a table or entry path.
        """
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
                    refs.append(self.table_detail_ref(topic_name, table, manifest_item))
                    for section in self.TABLE_SECTION_FIELDS:
                        refs.append(self._section_public_ref(topic_name, table, section, manifest_item))
            relationships = self.topic_assets.load_relationships(topic_name)
            if relationships:
                refs.append(self.relationship_index_ref(topic_name, relationships))
        self._refs_cache[cache_key] = refs
        return refs

    def _resolve_ref(self, ref_id: str, path: str) -> Dict[str, Any] | None:
        wanted_ref = ref_id.strip()
        wanted_path = normalize_semantic_path(path)
        if wanted_ref == "semantic:topics:index" or wanted_path == "topics/index.json":
            return self.topic_index_ref()
        relationship_entry_identity = parse_semantic_relationship_entry_identity(
            wanted_ref,
            wanted_path,
        )
        if relationship_entry_identity:
            relationship_kind, relationship_topic, relationship_key = (
                relationship_entry_identity
            )
            if relationship_topic not in self.topic_assets.all_topic_names():
                return None
            if relationship_kind == "index":
                return self.relationship_index_ref(
                    relationship_topic,
                    self.topic_assets.load_relationships(relationship_topic),
                )
            return self.relationship_entry_ref(
                relationship_topic,
                relationship_key,
                self.topic_assets.load_relationships(relationship_topic),
            )
        # Resolve directory/index files before exact entries.  Otherwise paths
        # such as ``metrics/index.json`` and ``columns/index.json`` are
        # accidentally parsed as an exact entry whose key is literally
        # ``index``.  That failed lookup used to return None immediately and
        # made an ls-advertised path unreadable through read_file.
        direct = parse_semantic_file_identity(wanted_ref, wanted_path)
        if direct:
            kind, topic, table, section = direct
            if kind == "manifest":
                return self.manifest_ref(topic) if topic in self.topic_assets.all_topic_names() else None
            if kind == "relationships":
                if topic not in self.topic_assets.all_topic_names():
                    return None
                return self.relationship_ref(topic, self.topic_assets.load_relationships(topic))
            if kind == "detail":
                return self.table_detail_ref(topic, table) if self._manifest_item(topic, table) else None
            if kind == "asset":
                return self.table_ref(topic, table) if self._manifest_item(topic, table) else None
            if kind == "section":
                return self.table_section_ref(topic, table, section)
        metric_identity = parse_semantic_metric_identity(wanted_ref, wanted_path)
        if metric_identity:
            metric_ref = self.metric_ref(*metric_identity)
            if metric_ref:
                return metric_ref
        entry_identity = parse_semantic_table_entry_identity(wanted_ref, wanted_path)
        if entry_identity:
            entry_ref = self.table_entry_ref(*entry_identity)
            if entry_ref:
                return entry_ref
        for ref in self._all_refs():
            if wanted_ref and ref["refId"] == wanted_ref:
                return ref
            if wanted_path and ref["path"] == wanted_path:
                return ref
        return None

    def _ls_path(self, path: str, query: str = "", limit: int = 50) -> List[Dict[str, Any]]:
        terms = question_match_terms(query) if query else []
        normalized = path.rstrip("/")
        table_match = re.fullmatch(r"topics/([^/]+)/tables/([^/]+)", normalized)
        if table_match:
            topic, table = table_match.groups()
            item = self._manifest_item(topic, table)
            if not item:
                return []
            refs = [self.table_detail_ref(topic, table, item)]
            for section in self.TABLE_SECTION_FIELDS:
                refs.append(self._section_public_ref(topic, table, section, item))
            return [
                self._public_ref(ref)
                for ref in refs
                if not terms or score_document(terms, ref.get("searchText", "")) > 0
            ][: max(1, limit)]
        section_match = re.fullmatch(r"topics/([^/]+)/tables/([^/]+)/(metrics|columns|terms|rules)", normalized)
        if section_match:
            topic, table, section = section_match.groups()
            section_ref = self.table_section_ref(topic, table, section)
            if not section_ref:
                return []
            try:
                payload = json.loads(section_ref.get("content") or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                payload = {}
            refs = []
            for item in payload.get("entries") or []:
                if not isinstance(item, dict):
                    continue
                entry = self.table_entry_ref(topic, table, section, str(item.get("key") or ""))
                if entry and (not terms or score_document(terms, entry.get("searchText", "")) > 0):
                    refs.append(self._public_ref(entry))
                if len(refs) >= max(1, limit):
                    break
            return refs
        relationship_match = re.fullmatch(r"topics/([^/]+)/relationships", normalized)
        if relationship_match:
            topic = relationship_match.group(1)
            index_ref = self.relationship_index_ref(topic)
            try:
                payload = json.loads(index_ref.get("content") or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                payload = {}
            refs = []
            for item in payload.get("entries") or []:
                if not isinstance(item, dict):
                    continue
                entry = self.relationship_entry_ref(topic, str(item.get("key") or ""))
                if entry and (
                    not terms or score_document(terms, entry.get("searchText", "")) > 0
                ):
                    refs.append(self._public_ref(entry))
                if len(refs) >= max(1, limit):
                    break
            return refs
        topic_match = re.fullmatch(r"topics/([^/]+)", normalized)
        if topic_match:
            return self.ls(topic=topic_match.group(1), query=query, limit=limit)
        if normalized in {"", ".", "topics"}:
            refs = [self.manifest_ref(topic) for topic in self.topic_assets.all_topic_names()]
            return [self._public_ref(ref) for ref in refs[: max(1, limit)]]
        return []

    def _manifest_item(self, topic: str, table: str) -> Dict[str, Any]:
        return next(
            (
                dict(item)
                for item in self.topic_assets.load_manifest(topic)
                if isinstance(item, dict) and str(item.get("tableName") or "") == table
            ),
            {},
        )

    def _section_public_ref(
        self,
        topic: str,
        table: str,
        section: str,
        manifest_item: Dict[str, Any],
    ) -> Dict[str, Any]:
        count = (
            int(manifest_item.get("metricCount") or 0)
            if section == "metrics"
            else int(manifest_item.get("ruleCount") or 0)
            if section == "rules"
            else 0
        )
        ref_id = semantic_table_section_ref_id(topic, table, section)
        path = semantic_table_section_path(topic, table, section)
        return add_context_uri(
            {
                "refId": ref_id,
                "kind": self.TABLE_SECTION_KINDS[section],
                "topic": topic,
                "table": table,
                "path": path,
                "title": "%s/%s/%s" % (topic, table, section),
                "summary": "governed %s entries" % section,
                "layers": {"section": section, "entryCountHint": count},
                "estimatedChars": 0,
                "offloadRecommended": False,
                "content": "",
                "searchText": "%s %s %s" % (topic, table, section),
            },
            ref_id=ref_id,
            topic=topic,
            table=table,
            kind=self.TABLE_SECTION_KINDS[section],
            path=path,
        )

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
            "contextDepth": ref.get("contextDepth", 0),
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


MARKDOWN_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4"), ("#####", "h5"), ("######", "h6")]
CHINESE_RECURSIVE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", ". ", "! ", "? ", "; ", ", ", " ", ""]


def split_markdown_for_recall(
    text: str,
    document_title: str,
    target_chars: int = 1600,
    max_chars: int = 2400,
    overlap_chars: int = 160,
) -> List[Dict[str, Any]]:
    """Split governed Markdown with LangChain's header and recursive splitters."""
    target = max(80, int(target_chars or 1600))
    maximum = max(target, 120, int(max_chars or 2400))
    overlap = max(0, min(int(overlap_chars or 0), maximum // 4))
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS,
        strip_headers=True,
    )
    sections = header_splitter.split_text(str(text or ""))
    chunks: List[Dict[str, Any]] = []
    section_counts: Dict[Tuple[str, ...], int] = {}
    previous_bodies: Dict[Tuple[str, ...], str] = {}
    for section in sections:
        effective_path = [
            str(section.metadata.get("h%d" % level) or "").strip()
            for level in range(1, 7)
            if str(section.metadata.get("h%d" % level) or "").strip()
        ] or [document_title]
        heading_text = " > ".join(effective_path)
        heading_prefix = "标题路径：%s" % heading_text
        body_limit = max(60, min(target, maximum - len(heading_prefix) - 2))
        section_overlap = max(0, min(overlap, body_limit // 4))
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=body_limit,
            chunk_overlap=section_overlap,
            separators=CHINESE_RECURSIVE_SEPARATORS,
            length_function=len,
            keep_separator=True,
            strip_whitespace=True,
        )
        section_chunks = recursive_splitter.split_documents([section])
        heading_key = tuple(effective_path)
        for split_doc in section_chunks:
            body = str(split_doc.page_content or "").strip()
            if not body:
                continue
            section_index = section_counts.get(heading_key, 0)
            previous_body = previous_bodies.get(heading_key, "")
            actual_overlap = common_chunk_overlap(previous_body, body, section_overlap) if section_index > 0 else 0
            content = "%s\n\n%s" % (heading_prefix, body)
            chunks.append(
                {
                    "headingPath": effective_path,
                    "headingText": heading_text,
                    "content": content,
                    "sectionChunkIndex": section_index,
                    "overlapChars": actual_overlap,
                }
            )
            section_counts[heading_key] = section_index + 1
            previous_bodies[heading_key] = body
    return chunks or [
        {
            "headingPath": [document_title],
            "headingText": document_title,
            "content": ("标题路径：%s\n\n%s" % (document_title, str(text or "").strip()))[:maximum],
            "sectionChunkIndex": 0,
            "overlapChars": 0,
        }
    ]


def common_chunk_overlap(previous: str, current: str, limit: int) -> int:
    maximum = min(max(0, int(limit or 0)), len(previous or ""), len(current or ""))
    for size in range(maximum, 0, -1):
        if previous[-size:] == current[:size]:
            return size
    return 0


class HybridRecallService:
    """Local BM25-ish recall over governed rules and runtime topic assets."""

    def __init__(self, settings: Settings, topic_assets: TopicAssetService):
        self.settings = settings
        self.topic_assets = topic_assets
        self.semantic_catalog = SemanticCatalogService(topic_assets)
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
        metric_refs = [
            str(getattr(item, "canonical_key", "") or getattr(item, "phrase", ""))
            for item in getattr(keywords, "mentions", []) or []
            if str(getattr(item, "kind", "") or "") == "metric"
        ] or list(getattr(keywords, "metric_keywords", []) or [])
        dimensions = [
            str(getattr(item, "canonical_key", "") or getattr(item, "phrase", ""))
            for item in getattr(keywords, "mentions", []) or []
            if str(getattr(item, "kind", "") or "") == "dimension"
        ] or list(getattr(keywords, "dimension_keywords", []) or [])
        cache_key = (
            semantic_request_cache_key(
                "recall",
                topics=sorted(allowed_topics),
                metrics=metric_refs,
                dimensions=dimensions,
                filters=[],
                time_range=resolve_time_range(question, self.settings.business_timezone),
                asset_version={"semanticSourceHash": self.topic_assets.semantic_source_hash(allowed_topics)},
                scope={"merchantId": merchant_id},
            )
            if metric_refs or dimensions
            else ""
        )
        cached = self._recall_cache.get(cache_key)
        if cached is not None:
            return RecallBundle.model_validate(cached)
        scored: List[RecallItem] = []
        for doc in self._load_documents():
            if not allowed_topics and doc.source_type != "GOVERNED_RULE":
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
        protected_metrics = exact_semantic_metric_recall_items(question, scored)
        recall_limit = 4 if not allowed_topics else 12
        items = dedupe_recall_items([*protected_metrics, *scored])[:recall_limit]
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
        for path in sorted(self.settings.resolved_rule_knowledge_path.glob("*.md")):
            try:
                chunks = split_markdown_for_recall(
                    path.read_text(encoding="utf-8"),
                    path.stem,
                    target_chars=self.settings.rule_chunk_target_chars,
                    max_chars=self.settings.rule_chunk_max_chars,
                    overlap_chars=self.settings.rule_chunk_overlap_chars,
                )
                for chunk_index, chunk in enumerate(chunks):
                    docs.append(
                        RecallItem(
                            doc_id="semantic:rules:%s:chunk:%04d" % (path.stem, chunk_index),
                            title="%s / %s" % (path.stem, chunk["headingText"]),
                            content=chunk["content"],
                            source_type="GOVERNED_RULE",
                            metadata={
                                "sourcePath": "rules/%s" % path.name,
                                "semanticPath": "rules/%s" % path.name,
                                "headingPath": chunk["headingPath"],
                                "headingText": chunk["headingText"],
                                "chunkIndex": chunk_index,
                                "sectionChunkIndex": chunk["sectionChunkIndex"],
                                "chunkChars": len(chunk["content"]),
                                "overlapChars": chunk["overlapChars"],
                                "chunkStrategy": "langchain_markdown_header_recursive",
                                "status": "PUBLISHED",
                                "visibilityPolicy": {"level": "public", "allowedRoles": []},
                            },
                        )
                    )
            except Exception:
                pass
        for topic in self.topic_assets.all_topic_names():
            for manifest_item in self.topic_assets.load_manifest(topic):
                table = str(manifest_item.get("tableName") or "")
                asset = self.topic_assets.load_table_asset(topic, table)
                ref = self.semantic_catalog.table_detail_ref(topic, table, manifest_item)
                docs.append(
                    RecallItem(
                        doc_id=ref["refId"],
                        title="%s/%s table candidate" % (topic, table),
                        content=compact_semantic_asset_for_recall(asset),
                        source_type="SEMANTIC_TABLE_ASSET",
                        topic=topic,
                        table=table,
                        answer_mode=",".join(manifest_item.get("preferredFor") or []),
                        metadata={
                            "semanticSource": "topic_manifest",
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
                            "status": asset.get("status") or manifest_item.get("status") or "PUBLISHED",
                            "version": asset.get("version") or "",
                            "merchantId": asset.get("merchantId") or "",
                            "allowedRoles": asset.get("allowedRoles") or [],
                            "requiredPermissions": asset.get("requiredPermissions") or [],
                            "visibilityPolicy": asset.get("visibilityPolicy") or {},
                            "expiresAt": asset.get("expiresAt") or "",
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
                    semantic_path = semantic_metric_path(topic, table, metric_key)
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
                                "semanticPath": semantic_path,
                                "metricKey": metric_key,
                                "tableName": table,
                                "topic": topic,
                                "businessName": metric.get("businessName") or metric_key,
                                "canonicalMetricKey": metric.get("canonicalMetricKey") or "",
                                "aliasOf": metric.get("aliasOf") or "",
                                "metricLevel": metric.get("metricLevel") or "",
                                "metricGrain": metric.get("metricGrain") or metric.get("grainHint") or "",
                                "metricIntent": metric.get("metricIntent") or "",
                                "aggregationPolicy": metric.get("aggregationPolicy") or "",
                                "applicableTimeGrain": metric.get("applicableTimeGrain") or "",
                                "timeColumn": metric.get("timeColumn") or "",
                                "timeSemantics": metric.get("timeSemantics") or {},
                                "missingValuePolicy": metric.get("missingValuePolicy") or "",
                                "zeroValueMeaning": metric.get("zeroValueMeaning") or "",
                                "selectionGuidance": metric.get("selectionGuidance") or "",
                                "preferredUseCases": metric.get("preferredUseCases") or [],
                                "notPreferredUseCases": metric.get("notPreferredUseCases") or [],
                                "temporalVariants": metric.get("temporalVariants") or {},
                                "formula": metric.get("formula") or metric.get("metricFormula") or "",
                                "sourceColumns": metric.get("sourceColumns") or [],
                                "aliases": metric.get("aliases") or [],
                                "merchantUri": merchant_uri_for_semantic_ref(semantic_ref_id, topic=topic, table=table, kind="METRIC", key=metric_key),
                                "contextLayer": "L1",
                                "status": metric.get("status") or asset.get("status") or "PUBLISHED",
                                "version": metric.get("version") or asset.get("version") or "",
                                "merchantId": metric.get("merchantId") or asset.get("merchantId") or "",
                                "allowedRoles": metric.get("allowedRoles") or asset.get("allowedRoles") or [],
                                "requiredPermissions": metric.get("requiredPermissions") or asset.get("requiredPermissions") or [],
                                "visibilityPolicy": metric.get("visibilityPolicy") or asset.get("visibilityPolicy") or {},
                                "expiresAt": metric.get("expiresAt") or asset.get("expiresAt") or "",
                            },
                        )
                    )
            relationships = self.topic_assets.load_relationships(topic)
            if relationships:
                ref = self.semantic_catalog.relationship_index_ref(topic, relationships)
                docs.append(
                    RecallItem(
                        doc_id=ref["refId"],
                        title="%s semantic relationship index" % topic,
                        content=str(ref.get("content") or "")[:8000],
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
                            "status": "PUBLISHED",
                            "visibilityPolicy": {"level": "public", "allowedRoles": []},
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
                    rel_key = semantic_relationship_key_for_name(
                        relationships,
                        rel_name,
                    )
                    rel_ref = self.semantic_catalog.relationship_entry_ref(
                        topic,
                        rel_key,
                        relationships,
                    )
                    if rel_ref is None:
                        continue
                    docs.append(
                        RecallItem(
                            doc_id=rel_ref["refId"],
                            title="%s/%s relationship" % (topic, rel_name),
                            content=json.dumps(rel, ensure_ascii=False)[:2400],
                            source_type="SEMANTIC_RELATIONSHIP",
                            topic=topic,
                            table=left,
                            metadata={
                                "semanticSource": "relationships.json",
                                "semanticKind": self.semantic_catalog.RELATIONSHIP_ENTRY_KIND,
                                "semanticRefId": rel_ref["refId"],
                                "semanticPath": rel_ref["path"],
                                "merchantUri": merchant_uri_for_semantic_ref(
                                    rel_ref["refId"],
                                    topic=topic,
                                    kind=self.semantic_catalog.RELATIONSHIP_ENTRY_KIND,
                                ),
                                "contextLayer": "L2",
                                "relationshipId": rel_name,
                                "leftTable": left,
                                "rightTable": right,
                                "topic": topic,
                                "joinKeys": rel.get("keys") or [],
                                "status": rel.get("status") or "PUBLISHED",
                                "version": rel.get("version") or "",
                                "merchantId": rel.get("merchantId") or "",
                                "allowedRoles": rel.get("allowedRoles") or [],
                                "requiredPermissions": rel.get("requiredPermissions") or [],
                                "visibilityPolicy": rel.get("visibilityPolicy") or {},
                                "expiresAt": rel.get("expiresAt") or "",
                            },
                        )
                    )
        self._documents = docs
        return docs


def always_apply_rule_active(rule: Dict[str, Any], now: datetime) -> bool:
    def parse(value: Any) -> Optional[datetime]:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None

    effective = parse(rule.get("effectiveFrom") or rule.get("effectiveAt"))
    expires = parse(rule.get("expiresAt") or rule.get("expiryAt"))
    return not ((effective and now < effective) or (expires and now >= expires))


def upsert_semantic_suggestion_item(items: List[Dict[str, Any]], candidate: Dict[str, Any], identity_key: str) -> List[Dict[str, Any]]:
    result = [dict(item) for item in items if isinstance(item, dict)]
    identity = str(candidate.get(identity_key) or "")
    for index, item in enumerate(result):
        if identity and str(item.get(identity_key) or "") == identity:
            result[index] = {**item, **candidate}
            return result
    result.append(candidate)
    return result


def compact_semantic_asset_for_recall(asset: Dict[str, Any]) -> str:
    payload = {
        "topic": asset.get("topic"),
        "tableName": asset.get("tableName"),
        "tableComment": asset.get("tableComment"),
        "dataGrain": asset.get("dataGrain"),
        "timeColumn": asset.get("timeColumn"),
        "merchantFilterColumn": asset.get("merchantFilterColumn"),
        "rowAccessPolicy": asset.get("rowAccessPolicy") or {},
        "resultAccessPolicies": normalize_result_access_policies(asset.get("resultAccessPolicies") or {}),
        "manualNotes": asset.get("manualNotes"),
        "tableUsageProfile": normalize_table_usage_profile(asset.get("tableUsageProfile") or {}, str(asset.get("tableName") or "")),
        "metrics": [
            {
                "metricKey": item.get("metricKey"),
                "businessName": item.get("businessName"),
                "formula": item.get("formula") or item.get("metricFormula"),
                "sourceColumns": item.get("sourceColumns") or [],
                "aliases": item.get("aliases") or [],
                "description": item.get("description"),
                "aggregationPolicy": item.get("aggregationPolicy"),
                "applicableTimeGrain": item.get("applicableTimeGrain"),
                "timeColumn": item.get("timeColumn"),
                "timeSemantics": item.get("timeSemantics") or {},
                "missingValuePolicy": item.get("missingValuePolicy"),
                "zeroValueMeaning": item.get("zeroValueMeaning"),
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
        "metricGrain": metric.get("metricGrain") or metric.get("grainHint"),
        "metricIntent": metric.get("metricIntent"),
        "aggregationPolicy": metric.get("aggregationPolicy"),
        "applicableTimeGrain": metric.get("applicableTimeGrain"),
        "timeSemantics": metric.get("timeSemantics") or {},
        "missingValuePolicy": metric.get("missingValuePolicy"),
        "zeroValueMeaning": metric.get("zeroValueMeaning"),
        "selectionGuidance": metric.get("selectionGuidance"),
        "preferredUseCases": metric.get("preferredUseCases") or [],
        "notPreferredUseCases": metric.get("notPreferredUseCases") or [],
        "temporalVariants": metric.get("temporalVariants") or {},
        "businessName": metric.get("businessName"),
        "formula": metric.get("formula") or metric.get("metricFormula"),
        "sourceColumns": metric.get("sourceColumns") or [],
        "aliases": metric.get("aliases") or [],
        "description": metric.get("description"),
        "unit": metric.get("unit"),
        "currency": metric.get("currency"),
        "aggregation": metric.get("aggregation"),
        "timeColumn": metric.get("timeColumn"),
        "requiredFilters": metric.get("requiredFilters") or [],
        "conflictsWith": metric.get("conflictsWith") or [],
        "clarificationQuestion": metric.get("clarificationQuestion"),
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
        "resultAccessPolicies": normalize_result_access_policies(asset.get("resultAccessPolicies") or {}),
        "manualNotes": asset.get("manualNotes"),
        "tableUsageProfile": normalize_table_usage_profile(asset.get("tableUsageProfile") or {}, str(asset.get("tableName") or "")),
        "status": asset.get("status"),
        "version": asset.get("version"),
    }


def semantic_enum_business_approved(column: Dict[str, Any]) -> bool:
    metadata = column.get("enumMetadata") if isinstance(column.get("enumMetadata"), dict) else {}
    return str(metadata.get("reviewStatus") or "UNREVIEWED").strip().upper() == "APPROVED"


def populated_semantic_enum_fields(column: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key in ("enumValues", "enumMappings", "enumMeanings", "valueLabels")
        if (value := column.get(key)) not in (None, [], {})
    }


def quarantine_unapproved_semantic_enums(column: Dict[str, Any]) -> Dict[str, Any]:
    """Remove unreviewed enum evidence from the runtime semantic view.

    Discovery output remains in the source asset for review.  Planner and
    semantic_read consumers receive only values explicitly approved by the
    asset contract, independent of any domain, table, or column name.
    """

    populated_fields = populated_semantic_enum_fields(column)
    if not populated_fields or semantic_enum_business_approved(column):
        return column
    sanitized = dict(column)
    suppressed_count = 0
    for key, value in populated_fields.items():
        suppressed_count += len(value) if isinstance(value, (list, tuple, set, dict)) else 1
        sanitized[key] = {} if isinstance(value, dict) else []
    sanitized.pop("sampleValues", None)
    evidence = sanitized.get("evidence")
    if isinstance(evidence, str):
        sanitized["evidence"] = re.sub(r";?\s*samples=\[[^\]]*\]", "", evidence).strip()
    metadata = dict(column.get("enumMetadata") or {})
    metadata.setdefault("reviewStatus", "UNREVIEWED")
    metadata.update(
        {
            "runtimePolicy": "QUARANTINED_UNTIL_APPROVED",
            "runtimeSuppressed": True,
            "suppressedValueCount": suppressed_count,
        }
    )
    sanitized["enumMetadata"] = metadata
    return sanitized


def enforce_sample_evidence_governance(asset: Dict[str, Any]) -> Dict[str, Any]:
    governance = asset.get("sampleEvidenceGovernance") if isinstance(asset.get("sampleEvidenceGovernance"), dict) else {}
    strip_sample_evidence = governance.get("usableForSemanticDecisions") is False
    sanitized = dict(asset)
    changed = False

    semantic_columns: List[Any] = []
    for raw in asset.get("semanticColumns") or []:
        if not isinstance(raw, dict):
            semantic_columns.append(raw)
            continue
        item = quarantine_unapproved_semantic_enums(raw)
        changed = changed or item is not raw
        semantic_columns.append(item)
    if changed:
        sanitized["semanticColumns"] = semantic_columns

    if not strip_sample_evidence:
        return sanitized if changed else asset

    sanitized["profiles"] = []
    for field in ("semanticColumns", "metrics", "terms", "knowledgeRules"):
        items: List[Dict[str, Any]] = []
        for raw in sanitized.get(field) or []:
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            item.pop("sampleValues", None)
            evidence = item.get("evidence")
            if isinstance(evidence, str):
                item["evidence"] = re.sub(r";?\s*samples=\[[^\]]*\]", "", evidence).strip()
            items.append(item)
        sanitized[field] = items
    sanitized["sampleEvidenceGovernance"] = {**governance, "enforcedAtLoad": True}
    return sanitized


def infer_business_layer(table: str) -> str:
    """Compatibility shim that deliberately does not infer from a table name.

    A physical naming convention is not a semantic contract.  Callers that need a
    layer must read ``tableUsageProfile.businessLayer`` from a reviewed asset.
    """

    del table
    return "UNDECLARED"


def default_table_queryable(table: str, business_layer: str = "") -> bool:
    """Fail closed when an asset has no reviewed queryability declaration."""

    del table, business_layer
    return False


def normalize_table_usage_profile(profile: Any, table: str = "") -> Dict[str, Any]:
    del table
    raw = profile if isinstance(profile, dict) else {}
    layer = str(raw.get("businessLayer") or "UNDECLARED").upper()
    try:
        authority = max(0, min(100, int(raw.get("authorityLevel", 0))))
    except (TypeError, ValueError):
        authority = 0
    queryable = raw.get("queryableByAgent")
    if not isinstance(queryable, bool):
        queryable = False
    topic_role = str(raw.get("topicRole") or "").upper()
    if topic_role not in {"ANCHOR", "DETAIL", "DIMENSION", "BRIDGE", "PROFILE", "AUXILIARY", "UNDECLARED"}:
        topic_role = "UNDECLARED"
    return {
        "contractStatus": str(raw.get("contractStatus") or "UNDECLARED").upper(),
        "businessLayer": layer,
        "queryableByAgent": queryable,
        "authorityLevel": authority,
        "topicRole": topic_role,
        "defaultForIntents": dedupe_strings([str(item).upper() for item in raw.get("defaultForIntents") or []]),
        "supportedIntents": dedupe_strings([str(item).upper() for item in raw.get("supportedIntents") or []]),
        "supportedMetrics": dedupe_strings([str(item) for item in raw.get("supportedMetrics") or []]),
        "supportedDimensions": dedupe_strings([str(item) for item in raw.get("supportedDimensions") or []]),
        "recommendedFor": dedupe_strings([str(item) for item in raw.get("recommendedFor") or []]),
        "notRecommendedFor": dedupe_strings([str(item) for item in raw.get("notRecommendedFor") or []]),
        "exclusionReason": str(raw.get("exclusionReason") or ("TABLE_USAGE_UNDECLARED" if not queryable else "")),
    }


def infer_aggregate_source_columns(
    schema: List[Dict[str, Any]],
    semantic_columns: List[Dict[str, Any]],
) -> List[str]:
    """Return only columns explicitly governed as measures.

    Numeric storage type alone does not make a field an additive metric.
    """

    del schema
    return dedupe_strings(
        [
            str(item.get("columnName") or "")
            for item in semantic_columns
            if isinstance(item, dict)
            and str(item.get("semanticRole") or item.get("role") or "").upper() in {"MEASURE", "METRIC"}
            and str(item.get("columnName") or "")
        ]
    )


def build_stable_topic_table_manifest(
    topic_assets: TopicAssetService,
    topics: List[str],
    source_hash: str = "",
) -> Dict[str, Any]:
    """Build the question-independent L0 table index for a Topic workspace."""

    topic_entries: List[Dict[str, Any]] = []
    flat_tables: List[Dict[str, Any]] = []
    for topic in dedupe_strings([str(item or "").strip() for item in topics]):
        if not topic:
            continue
        try:
            category = topic_assets.resolve_topic_category(topic)
        except AttributeError:
            category = None
        topic_id = str(getattr(category, "value", category) or topic)
        tables: List[Dict[str, Any]] = []
        for raw in topic_assets.load_manifest(topic):
            if not isinstance(raw, dict):
                continue
            table = str(raw.get("tableName") or "").strip()
            if not table:
                continue
            entry = {
                "topic": topic,
                "table": table,
                "title": str(raw.get("tableComment") or raw.get("title") or table),
                "detailRefId": semantic_table_detail_ref_id(topic, table),
                "detailPath": semantic_table_detail_path(topic, table),
            }
            business_summary = str(raw.get("businessSummary") or "").strip()
            if business_summary:
                entry["businessSummary"] = business_summary
            tables.append(entry)
            flat_tables.append(entry)
        topic_entries.append(
            {
                "topic": topic,
                "topicId": topic_id,
                "manifestRefId": semantic_manifest_ref_id(topic),
                "path": semantic_manifest_path(topic),
                "tableCount": len(tables),
                "tables": tables,
            }
        )
    return {
        "mode": "stable_topic_table_manifest",
        "sourceHash": str(source_hash or ""),
        "questionIndependent": True,
        "topics": topic_entries,
        "tables": flat_tables,
        "tableCount": len(flat_tables),
        "policy": "Published Topic manifest only; question text and RAG scores never add, remove, or reorder tables.",
    }


def stable_manifest_table_names(pack: PlanningAssetPack) -> Set[str]:
    """Return the executable table boundary declared by the Topic manifest."""

    return {
        str(item.get("table") or item.get("tableName") or "")
        for item in (pack.table_manifest or {}).get("tables") or []
        if isinstance(item, dict)
        and str(item.get("table") or item.get("tableName") or "")
    }


def table_allowed_by_stable_manifest(pack: PlanningAssetPack, table: str) -> bool:
    # Synthetic/legacy packs without a manifest retain their old behavior. A
    # present but empty stable manifest deliberately admits no executable table.
    if not pack.table_manifest:
        return True
    return bool(table and table in stable_manifest_table_names(pack))


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
        planning_hints: Optional[Dict[str, Any]] = None,
    ) -> PlanningAssetPack:
        pack = PlanningAssetPack()
        planning_hints = normalize_planning_hints(planning_hints)
        allow_profile = isinstance(diagnostic_context, dict) and diagnostic_context.get("scope") == "OPEN_DIAGNOSTIC"
        topics = self.topic_assets.topic_names_for_categories(topic_categories)
        if not topics:
            topics = sorted({item.topic for item in recall_bundle.items if item.topic})
        # A routed Topic is the stable workspace boundary. Recall hits may rank
        # assets inside that workspace, but must never enlarge it implicitly.
        # Cross-Topic access is admitted only after an explicit, typed expansion
        # updates topic_categories and rebuilds the manifest.
        topics = list(dict.fromkeys(topics))
        semantic_source_hash = self._topics_source_hash(topics)
        table_manifest = build_stable_topic_table_manifest(
            self.topic_assets,
            topics,
            source_hash=semantic_source_hash,
        )
        pack.table_manifest = table_manifest
        table_topic = self._table_topic_index()
        all_relationships = self._all_relationships()
        seed_tables, targeted_traces = self._targeted_seed_tables(
            question,
            recall_bundle,
            topics,
            table_topic,
            allow_profile=allow_profile,
            explicit_tables=set(),
            planning_hints=planning_hints,
            all_relationships=all_relationships,
        )
        if allow_profile:
            profile_tables = self._diagnostic_profile_seed_tables(table_topic, limit=1)
            for table in profile_tables:
                if table not in seed_tables:
                    seed_tables.add(table)
                    targeted_traces.append("open_diagnostic_profile_seed:%s" % table)
        bridge_tables, bridge_traces = self._relationship_bridge_tables(
            seed_tables,
            all_relationships,
            table_topic,
            allow_profile=allow_profile,
            max_extra=(
                0
                if any(str(item) == "targeted_seed_source=dimension_relationship_closure" for item in targeted_traces)
                else 2
            ),
        )
        manifest_tables = stable_manifest_table_names(pack)
        rejected_seed_tables = sorted(
            table for table in seed_tables if table not in manifest_tables
        )
        if rejected_seed_tables:
            targeted_traces.append(
                "topic_manifest_boundary_rejected:%s"
                % ",".join(rejected_seed_tables)
            )
        seed_tables = {table for table in seed_tables if table in manifest_tables}
        bridge_tables = {table for table in bridge_tables if table in manifest_tables}
        seed_tables.update(bridge_tables)
        pack_tables = {table for table in seed_tables if table}
        live_schema_hash = self._live_schema_hash_for_tables(pack_tables)
        recalled_metrics = [
            {
                "metricKey": str((item.metadata or {}).get("metricKey") or ""),
                "ownerTable": str(item.table or (item.metadata or {}).get("tableName") or ""),
            }
            for item in recall_bundle.items
            if str((item.metadata or {}).get("metricKey") or "")
        ]
        cache_key = semantic_request_cache_key(
            "asset_pack",
            topics=[category.value if isinstance(category, QuestionCategory) else str(category) for category in topic_categories],
            metrics=recalled_metrics,
            dimensions=list(planning_hints.get("dimensions") or []),
            filters=[
                *([diagnostic_context or {}] if diagnostic_context else []),
                *([{"ranking": planning_hints.get("ranking")}] if planning_hints.get("ranking") else []),
            ],
            time_range=resolve_time_range(question, self.topic_assets.settings.business_timezone),
            asset_version={
                "semanticSourceHash": semantic_source_hash,
                "liveSchemaHash": live_schema_hash,
                "sourceRefs": sorted(item.doc_id for item in recall_bundle.items if item.doc_id),
            },
            scope={"questionFingerprint": hashlib.sha256(normalize_for_match(question).encode("utf-8")).hexdigest()[:16]},
        )
        cached = self._compact_cache.get(cache_key)
        if cached is not None:
            pack = PlanningAssetPack.model_validate(cached)
            pack.table_manifest = table_manifest
            pack.metric_compaction.setdefault("cache", {})["hit"] = True
            pack.metric_compaction.setdefault("cache", {})["semanticSourceHash"] = semantic_source_hash
            pack.metric_compaction.setdefault("cache", {})["liveSchemaHash"] = live_schema_hash
            pack.metric_compaction["questionStructure"] = planning_hints
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
        pack.metric_compaction["questionStructure"] = planning_hints
        recalled_relationship_evidence = recalled_relationship_evidence_from_bundle(recall_bundle)
        if recalled_relationship_evidence:
            pack.metric_compaction["recalledRelationshipEvidence"] = recalled_relationship_evidence
        if not pack.metrics and not recalled_metric_evidence and self._deferred_structured_understanding(targeted_traces):
            metric_candidates, metric_candidate_traces = self._topic_metric_candidates_for_deferred_understanding(question, topics)
            if metric_candidates:
                pack.metrics.extend(metric_candidates)
                pack.metric_compaction["deferredMetricCandidates"] = {
                    "strategy": "topic_metric_candidates_only",
                    "count": len(metric_candidates),
                    "tables": sorted({item.table for item in metric_candidates if item.table}),
                }
                targeted_traces.extend(metric_candidate_traces)
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
        required_relationship_edges = relationship_edges_from_targeted_traces(targeted_traces)
        required_relationship_refs = {
            str(item).split(":", 1)[1]
            for item in targeted_traces
            if str(item).startswith("dimension_relationship_ref:")
        }
        recalled_relationship_refs = {
            str(item.get("semanticRefId") or "")
            for item in recalled_relationship_evidence_from_bundle(recall_bundle)
        }
        ordered_relationships = sorted(
            all_relationships,
            key=lambda item: (
                0
                if "semantic:%s:relationship:%s" % (item[0], item[1].get("name") or "")
                in recalled_relationship_refs
                else 1,
                str(item[0]),
                str(item[1].get("name") or ""),
            ),
        )
        for topic, rel in ordered_relationships:
            left = str(rel.get("leftTable") or "")
            right = str(rel.get("rightTable") or "")
            relationship_ref = "semantic:%s:relationship:%s" % (topic, rel.get("name") or "")
            if required_relationship_refs and relationship_ref not in required_relationship_refs:
                continue
            if not required_relationship_refs and required_relationship_edges and frozenset({left, right}) not in required_relationship_edges:
                continue
            if left in pack_tables and right in pack_tables and topic in relationship_topics:
                selected_relationships.append((topic, rel))
        for topic, rel in selected_relationships:
            entry = relationship_entry(topic, rel)
            if entry.relationship_id and entry.relationship_id not in {item.relationship_id for item in pack.relationships}:
                pack.relationships.append(entry)
        relationship_contracts, relationship_gaps = relationship_contract_assessment(
            pack,
            selected_relationships,
            planning_hints,
            targeted_traces,
        )
        if relationship_contracts:
            pack.metric_compaction["relationshipContracts"] = relationship_contracts
        trace_gaps = planning_gaps_from_asset_traces(targeted_traces)
        if trace_gaps or relationship_gaps:
            pack.metric_compaction["knowledgeRequestGaps"] = dedupe_planning_gaps(
                [*trace_gaps, *relationship_gaps]
            )
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
        pack.table_manifest = table_manifest
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
        return self.topic_assets.semantic_source_hash(topics)

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

    def _deferred_structured_understanding(self, traces: List[str]) -> bool:
        return any("targeted_seed_source=deferred_structured_understanding" in str(item) for item in traces)

    def _topic_metric_candidates_for_deferred_understanding(
        self,
        question: str,
        topics: List[str],
    ) -> Tuple[List[PlanningAssetEntry], List[str]]:
        limit = max(4, min(int(self.topic_assets.settings.agent_planner_seed_metric_limit or 8), 10))
        per_topic_limit = max(1, min(3, limit))
        selected: List[PlanningAssetEntry] = []
        traces: List[str] = []
        seen: Set[Tuple[str, str]] = set()
        for topic in topics:
            candidates: List[Tuple[int, PlanningAssetEntry]] = []
            for manifest_item in self.topic_assets.load_manifest(topic):
                table = str(manifest_item.get("tableName") or "")
                if not table:
                    continue
                for metric in self.topic_assets.load_table_metrics(topic, table):
                    key = str(metric.get("metricKey") or "")
                    if not key:
                        continue
                    entry = PlanningAssetEntry(
                        key=key,
                        table=table,
                        topic=topic,
                        title=str(metric.get("businessName") or key),
                        columns=[str(column) for column in metric.get("sourceColumns") or []],
                        aliases=[str(alias) for alias in metric.get("aliases") or []],
                        description=json.dumps(metric, ensure_ascii=False),
                        source_ref_id="semantic:%s:%s:metric:%s" % (topic, table, key),
                        metadata=metric,
                    )
                    score = self._metric_relevance_score(entry, question)
                    if score <= 0:
                        level = str(metric.get("metricLevel") or metric.get("metric_level") or "").lower()
                        score = int(metric.get("defaultCandidateScore") or (5 if "business" in level else 1))
                    canonical_key = str(metric.get("canonicalMetricKey") or metric.get("canonical_metric_key") or "")
                    alias_of = str(metric.get("aliasOf") or metric.get("alias_of") or "")
                    if canonical_key and canonical_key == key and not alias_of:
                        score += 2
                    if alias_of:
                        score -= 2
                    candidates.append((score, entry))
            candidates.sort(key=lambda item: item[0], reverse=True)
            topic_added = 0
            for score, entry in candidates:
                identity = (entry.table, entry.key)
                if identity in seen:
                    continue
                seen.add(identity)
                selected.append(entry)
                topic_added += 1
                traces.append("deferred_metric_candidate:%s:%s:%s" % (entry.table, entry.key, score))
                if topic_added >= per_topic_limit or len(selected) >= limit:
                    break
            if len(selected) >= limit:
                break
        return selected, traces

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
                if not table_allowed_by_stable_manifest(pack, table):
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

    def expand_for_metric_catalog_resolution(self, pack: PlanningAssetPack, question: str, limit: int = 6) -> List[str]:
        """Read exact metric candidates from the published semantic catalog.

        This does not select a business table by Topic or by hard-coded metric
        names.  It only loads owner tables for metrics whose published labels or
        aliases are explicitly named by the question, then lets the normal metric
        resolver decide whether the candidate set is unique enough to execute.
        """

        normalized_question = normalize_for_match(question)
        if not normalized_question:
            return []
        current_identities = {(metric.table, metric.key) for metric in pack.metrics if metric.table and metric.key}
        candidates: List[Tuple[int, str, PlanningAssetEntry]] = []
        for metric in self._all_metric_entries():
            if not metric.table or not metric.key:
                continue
            matched_label = metric_direct_match_label(metric, question)
            if not matched_label:
                continue
            score = self._metric_relevance_score(metric, question) or len(normalize_for_match(matched_label))
            candidates.append((score, matched_label, metric))
        candidates.sort(key=lambda item: (item[0], len(normalize_for_match(item[1]))), reverse=True)
        selected: List[Tuple[int, str, PlanningAssetEntry]] = []
        seen: Set[Tuple[str, str]] = set()
        for score, matched_label, metric in candidates:
            if not table_allowed_by_stable_manifest(pack, metric.table):
                continue
            identity = (metric.table, metric.key)
            if identity in seen:
                continue
            seen.add(identity)
            selected.append((score, matched_label, metric))
            if len(selected) >= max(1, limit):
                break
        if not selected:
            return []
        table_topic = self._table_topic_index()
        pack_tables = set(pack.known_tables())
        traces: List[str] = []
        evidence: List[Dict[str, Any]] = list(pack.metric_compaction.get("recalledMetricEvidence") or [])
        existing_evidence = {
            (str(item.get("ownerTable") or ""), str(item.get("metricKey") or ""))
            for item in evidence
            if isinstance(item, dict)
        }
        ambiguous_identities = ambiguous_catalog_metric_identities(selected)
        for score, matched_label, metric in selected:
            topic = table_topic.get(metric.table) or metric.topic
            if not table_allowed_by_stable_manifest(pack, metric.table):
                continue
            if metric.table not in pack_tables and topic:
                self._append_table_assets(pack, topic, metric.table)
                pack_tables.add(metric.table)
                traces.append("catalog_metric_table:%s:%s" % (metric.table, metric.key))
            elif (metric.table, metric.key) not in current_identities and not any(item.table == metric.table and item.key == metric.key for item in pack.metrics):
                pack.metrics.append(metric)
                traces.append("catalog_metric_entry:%s:%s" % (metric.table, metric.key))
            identity = (metric.table, metric.key)
            if identity not in existing_evidence:
                evidence.append(catalog_metric_evidence_payload(metric, matched_label, score, identity in ambiguous_identities))
                existing_evidence.add(identity)
        if evidence:
            pack.metric_compaction["recalledMetricEvidence"] = evidence
        pack.metric_compaction["catalogMetricCandidates"] = [
            {
                "ownerTable": metric.table,
                "metricKey": metric.key,
                "semanticRefId": metric.source_ref_id,
                "matchedMetricLabel": matched_label,
                "score": score,
                "ambiguous": (metric.table, metric.key) in ambiguous_identities,
            }
            for score, matched_label, metric in selected
        ]
        relationship_traces = self._append_relationships_for_tables(pack, pack_tables, table_topic)
        pack.relationship_closure.extend(traces + relationship_traces)
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
                if (
                    not table_allowed_by_stable_manifest(pack, dep_table)
                    or not topic
                    or not self._table_allowed_for_topic_question(
                        topic,
                        question,
                        dep_table,
                        allow_profile=allow_profile,
                    )
                ):
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
        planning_hints: Dict[str, Any] | None = None,
        all_relationships: List[Tuple[str, Dict[str, Any]]] | None = None,
    ) -> Tuple[Set[str], List[str]]:
        explicit_tables = explicit_tables or set()
        planning_hints = normalize_planning_hints(planning_hints)
        dimension_obligations = planning_dimension_obligations(planning_hints)
        all_relationships = list(all_relationships or [])
        workspace_topics = {str(topic) for topic in topics if str(topic)}

        def finalize(tables: Set[str], traces: List[str]) -> Tuple[Set[str], List[str]]:
            admitted = {
                table
                for table in tables
                if table and table_topic.get(table, "") in workspace_topics
            }
            rejected = sorted(set(tables) - admitted)
            if rejected:
                traces = [
                    *traces,
                    "topic_manifest_boundary_rejected:%s" % ",".join(rejected),
                ]
            return admitted, traces

        explicit_tables = {
            table
            for table in explicit_tables
            if table_topic.get(table, "") in workspace_topics
        }
        precise_recalled_tables: Set[str] = set()
        recalled_relationship_seed_tables: Set[str] = set()
        recalled_relationship_refs: Set[str] = set()
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
            if not self._recall_item_has_precise_table_evidence(item):
                continue
            item_tables = [item.table, str((item.metadata or {}).get("tableName") or "")]
            for table in item_tables:
                if table and self._table_allowed_for_recalled_item(question, item, table, allow_profile=allow_profile):
                    precise_recalled_tables.add(table)
        for item in recall_bundle.items:
            if str(item.source_type or "").upper() != "SEMANTIC_RELATIONSHIP":
                continue
            relationship_ref = str((item.metadata or {}).get("semanticRefId") or item.doc_id or "")
            if relationship_ref:
                recalled_relationship_refs.add(relationship_ref)
            for table in recalled_relationship_tables(item):
                if self._table_allowed_for_recalled_item(question, item, table, allow_profile=allow_profile):
                    recalled_tables.add(table)
                    recalled_relationship_seed_tables.add(table)
                    if self._recall_item_has_precise_table_evidence(item):
                        precise_recalled_tables.add(table)
        precise_metric_seed_tables, precise_metric_seed_traces = self._precise_metric_seed_tables(
            question,
            topics,
            table_topic,
            allow_profile=allow_profile,
        )
        precise_seed_tables = explicit_tables | precise_recalled_tables | precise_metric_seed_tables
        if precise_seed_tables:
            if dimension_obligations:
                dimension_seed_tables, dimension_traces = self._dimension_aware_seed_tables(
                    question=question,
                    planning_hints=planning_hints,
                    precise_seed_tables=precise_seed_tables,
                    precise_metric_seed_traces=precise_metric_seed_traces,
                    recalled_relationship_tables=recalled_relationship_seed_tables,
                    recalled_relationship_refs=recalled_relationship_refs,
                    table_topic=table_topic,
                    all_relationships=all_relationships,
                    allow_profile=allow_profile,
                )
                if dimension_seed_tables:
                    return finalize(dimension_seed_tables, dimension_traces)
            source_parts: List[str] = []
            if explicit_tables:
                source_parts.append("explicit_tables")
            if precise_recalled_tables:
                source_parts.append("recall_source_refs")
            if precise_metric_seed_tables:
                source_parts.append("semantic_metric_alias")
            traces = [
                "targeted_seed_tables:%s"
                % ",".join("%s=evidence_owner_table" % table for table in sorted(precise_seed_tables)),
                "table_selection_explanations:%s"
                % json.dumps(
                    [
                        {
                            "strategy": "evidence_owner_tables",
                            "reason": "precise_metric_or_recall_evidence",
                            "ownerTables": sorted(precise_seed_tables),
                        }
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "targeted_seed_source=%s" % "+".join(source_parts or ["evidence_owner_tables"]),
            ]
            return finalize(set(precise_seed_tables), traces + precise_metric_seed_traces)
        if self._broad_topic_question(question) and not allow_profile:
            traces = [
                "targeted_seed_tables:",
                "table_selection_explanations:%s"
                % json.dumps(
                    [
                        {
                            "strategy": "defer_table_selection",
                            "reason": "broad_topic_without_precise_recall",
                            "topics": topics,
                        }
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "targeted_seed_source=deferred_structured_understanding",
            ]
            return finalize(set(), traces)
        evidence_tables = {table for table in (explicit_tables | recalled_tables) if table}
        if evidence_tables:
            evidence_preview = [
                {
                    "table": table,
                    "strategy": "recalled_owner_table",
                    "topic": table_topic.get(table, ""),
                }
                for table in sorted(evidence_tables)
            ]
            traces = [
                "targeted_seed_tables:%s" % ",".join("%s=recalled_owner_table" % table for table in sorted(evidence_tables)),
                "table_selection_explanations:%s"
                % json.dumps(evidence_preview, ensure_ascii=False, separators=(",", ":")),
                "targeted_seed_source=recall_source_refs",
            ]
            return finalize(evidence_tables, traces)
        traces = [
            "targeted_seed_tables:",
            "table_selection_explanations:%s"
            % json.dumps(
                [
                    {
                        "strategy": "defer_table_selection",
                        "reason": "no_explicit_or_recalled_table_evidence",
                        "topics": topics,
                    }
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "targeted_seed_source=deferred_structured_understanding",
        ]
        return finalize(set(), traces)

    def _precise_metric_seed_tables(
        self,
        question: str,
        topics: List[str],
        table_topic: Dict[str, str],
        allow_profile: bool = False,
    ) -> Tuple[Set[str], List[str]]:
        q = normalize_for_match(question)
        if not q:
            return set(), []
        seed_tables: Set[str] = set()
        traces: List[str] = []
        for topic in topics:
            for manifest_item in self.topic_assets.load_manifest(topic):
                table = str(manifest_item.get("tableName") or "")
                if not table or table_topic.get(table) != topic:
                    continue
                if not self._table_queryable_for_topic(topic, table):
                    continue
                for metric in self.topic_assets.load_table_metrics(topic, table):
                    metric_key = str(metric.get("metricKey") or "")
                    labels = [
                        str(metric.get("displayName") or ""),
                        str(metric.get("businessName") or ""),
                        metric_key,
                        *[str(alias) for alias in metric.get("aliases") or []],
                    ]
                    matched = next((label for label in labels if label and normalize_for_match(label) in q), "")
                    if not matched:
                        continue
                    seed_tables.add(table)
                    traces.append("precise_metric_seed:%s:%s:%s" % (table, metric_key, matched))
        return seed_tables, traces

    def _dimension_aware_seed_tables(
        self,
        *,
        question: str,
        planning_hints: Dict[str, Any],
        precise_seed_tables: Set[str],
        precise_metric_seed_traces: List[str],
        recalled_relationship_tables: Set[str],
        recalled_relationship_refs: Set[str],
        table_topic: Dict[str, str],
        all_relationships: List[Tuple[str, Dict[str, Any]]],
        allow_profile: bool,
    ) -> Tuple[Set[str], List[str]]:
        """Replace dimension-incompatible summary seeds with executable facts.

        Selection is driven by governed metric/field metadata and the published
        relationship graph.  A relationship makes two owners reachable; it is
        not treated as proof that a raw multi-fact aggregation is fanout-safe.
        """

        obligations = planning_dimension_obligations(planning_hints)
        if not obligations:
            return set(), []
        graph = relationship_table_graph(all_relationships)
        metric_phrases = [str(item) for item in planning_hints.get("metricPhrases") or [] if str(item or "").strip()]
        preferred_dimension_owner = self._preferred_dimension_owner(
            obligations,
            table_topic,
            graph,
            candidate_metric_tables=set(),
            require_dimension_layer=True,
        )
        incompatible_tables = {
            table
            for table in precise_seed_tables
            if not self._table_satisfies_dimension_obligations(table, obligations, table_topic)
            or (
                preferred_dimension_owner
                and preferred_dimension_owner != table
                and self._table_business_layer(table, table_topic) in {"ADS", "SUMMARY"}
            )
        }
        if not incompatible_tables:
            # A governed owner already satisfies the requested grain.  Preserve
            # the compact single-owner behavior unless a relationship is needed.
            return set(precise_seed_tables), [
                "targeted_seed_tables:%s"
                % ",".join("%s=dimension_compatible_metric_owner" % table for table in sorted(precise_seed_tables)),
                *precise_metric_seed_traces,
                "targeted_seed_source=dimension_compatible_metric_owner",
            ]

        base_tables = set(precise_seed_tables) | set(recalled_relationship_tables)
        preferred_detail_targets = self._detail_metric_targets(
            precise_metric_seed_traces,
            table_topic,
        )
        selected_metrics: List[Dict[str, Any]] = []
        unresolved_phrases: List[str] = []
        for phrase in metric_phrases:
            candidates = self._dimension_metric_candidates(
                phrase,
                obligations,
                table_topic,
                graph,
                base_tables,
                incompatible_tables,
                preferred_detail_targets.get(phrase, set()),
            )
            if not candidates:
                unresolved_phrases.append(phrase)
                continue
            selected_metrics.append(candidates[0])

        if not selected_metrics:
            traces = downgrade_incompatible_metric_seed_traces(
                precise_metric_seed_traces,
                incompatible_tables,
            )
            for phrase in unresolved_phrases or metric_phrases:
                traces.append(
                    planning_gap_trace(
                        {
                            "code": "METRIC_DIMENSION_GRAIN_MISMATCH",
                            "type": "METRIC",
                            "query": "%s metric definition at requested dimension grain" % phrase,
                            "reason": "exact metric owner does not support the requested dimension and no compatible metric owner is published",
                            "metricRefs": metric_refs_for_phrase(planning_hints, phrase),
                            "dimensionRefs": dimension_refs_for_obligations(obligations),
                            "dimensionPhrase": dimension_phrase_for_obligations(obligations),
                            "sourceOwners": sorted(incompatible_tables),
                        }
                    )
                )
            return set(precise_seed_tables), traces + ["targeted_seed_source=dimension_incompatible_fail_closed"]

        metric_tables = {str(item.get("table") or "") for item in selected_metrics if item.get("table")}
        dimension_owner = self._preferred_dimension_owner(
            obligations,
            table_topic,
            graph,
            candidate_metric_tables=metric_tables,
            require_dimension_layer=False,
        )
        selected_tables: Set[str] = set(metric_tables)
        traces = downgrade_incompatible_metric_seed_traces(
            precise_metric_seed_traces,
            incompatible_tables,
        )
        for item in selected_metrics:
            traces.append(
                "precise_metric_seed:%s:%s:%s"
                % (item.get("table") or "", item.get("metricKey") or "", item.get("phrase") or "")
            )
            traces.append(
                "dimension_compatible_metric_seed:%s:%s:score=%s"
                % (item.get("table") or "", item.get("metricKey") or "", item.get("score") or 0)
            )
        if dimension_owner:
            selected_tables.add(dimension_owner)
            traces.append("dimension_authority_seed:%s" % dimension_owner)

        if not dimension_owner:
            traces.append(
                planning_gap_trace(
                    {
                        "code": "RELATIONSHIP_DIMENSION_OWNER_REQUIRED",
                        "type": "RELATIONSHIP",
                        "query": "published dimension owner and relationship path for %s"
                        % dimension_phrase_for_obligations(obligations),
                        "reason": "no governed table can own the requested dimension",
                        "metricRefs": [str(item.get("metricKey") or "") for item in selected_metrics],
                        "dimensionRefs": dimension_refs_for_obligations(obligations),
                        "dimensionPhrase": dimension_phrase_for_obligations(obligations),
                        "sourceOwners": sorted(metric_tables),
                        "targetOwner": "",
                        "requiredSemantics": ["join_path", "canonical_entity", "cardinality", "fanout_policy"],
                    }
                )
            )
        else:
            for metric in selected_metrics:
                source_owner = str(metric.get("table") or "")
                if not source_owner or source_owner == dimension_owner:
                    continue
                path = governed_dimension_path(
                    all_relationships,
                    source_owner,
                    dimension_owner,
                    obligations,
                    max_hops=3,
                )
                if not path:
                    traces.append(
                        planning_gap_trace(
                            {
                                "code": "RELATIONSHIP_PATH_REQUIRED",
                                "type": "RELATIONSHIP",
                                "query": "relationship path from %s to %s for metric %s by dimension %s"
                                % (
                                    source_owner,
                                    dimension_owner,
                                    metric.get("metricKey") or metric.get("phrase") or "",
                                    dimension_phrase_for_obligations(obligations),
                                ),
                                "reason": "metric owner lacks a published path to the requested dimension owner",
                                "metricRefs": [str(metric.get("metricKey") or "")],
                                "dimensionRefs": dimension_refs_for_obligations(obligations),
                                "dimensionPhrase": dimension_phrase_for_obligations(obligations),
                                "sourceOwner": source_owner,
                                "targetOwner": dimension_owner,
                                "requiredSemantics": ["join_path", "canonical_entity", "cardinality", "fanout_policy"],
                            }
                        )
                    )
                    continue
                selected_tables.update(path)
                if len(path) > 1:
                    traces.append("dimension_relationship_path:%s" % "->".join(path))
                    for left, right in zip(path, path[1:]):
                        relationship_ref = governed_relationship_ref_for_edge(
                            all_relationships,
                            left,
                            right,
                            dimension_owner,
                            obligations,
                            recalled_relationship_refs,
                        )
                        if relationship_ref:
                            traces.append("dimension_relationship_ref:%s" % relationship_ref)

        if unresolved_phrases:
            for phrase in unresolved_phrases:
                traces.append(
                    planning_gap_trace(
                        {
                            "code": "METRIC_DIMENSION_GRAIN_MISMATCH",
                            "type": "METRIC",
                            "query": "%s metric definition at requested dimension grain" % phrase,
                            "reason": "no governed dimension-compatible metric candidate was found",
                            "metricRefs": metric_refs_for_phrase(planning_hints, phrase),
                            "dimensionRefs": dimension_refs_for_obligations(obligations),
                            "dimensionPhrase": dimension_phrase_for_obligations(obligations),
                            "sourceOwners": sorted(incompatible_tables),
                        }
                    )
                )

        selected_tables = {
            table
            for table in selected_tables
            if table in table_topic
            and self._table_allowed_for_topic_question(
                table_topic.get(table, ""),
                question,
                table,
                allow_profile=allow_profile,
            )
        }
        traces.extend(
            [
                "targeted_seed_tables:%s"
                % ",".join("%s=dimension_relationship_closure" % table for table in sorted(selected_tables)),
                "table_selection_explanations:%s"
                % json.dumps(
                    [
                        {
                            "strategy": "dimension_relationship_closure",
                            "reason": "metric owners connected to requested dimension through published relationships",
                            "ownerTables": sorted(selected_tables),
                            "excludedSummaryOwners": sorted(incompatible_tables - selected_tables),
                        }
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "targeted_seed_source=dimension_relationship_closure",
            ]
        )
        return selected_tables, traces

    def _dimension_metric_candidates(
        self,
        phrase: str,
        obligations: List[Dict[str, Any]],
        table_topic: Dict[str, str],
        graph: Dict[str, Set[str]],
        base_tables: Set[str],
        incompatible_tables: Set[str],
        preferred_targets: Set[Tuple[str, str]],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for topic in self.topic_assets.all_topic_names():
            for manifest_item in self.topic_assets.load_manifest(topic):
                table = str(manifest_item.get("tableName") or "")
                if not table or table in incompatible_tables or table_topic.get(table) != topic:
                    continue
                if not self._table_queryable_for_topic(topic, table):
                    continue
                distance = shortest_distance_from_tables(graph, base_tables, table, max_hops=3)
                if base_tables and distance is None:
                    continue
                for metric in self.topic_assets.load_table_metrics(topic, table):
                    metric_key = str(metric.get("metricKey") or "")
                    is_preferred_target = (table, metric_key) in preferred_targets
                    similarity = semantic_exact_phrase_similarity(
                        phrase,
                        [
                            str(metric.get("businessName") or ""),
                            str(metric.get("displayName") or ""),
                            metric_key,
                            *[str(alias) for alias in metric.get("aliases") or []],
                        ],
                    )
                    if not is_preferred_target and similarity <= 0:
                        continue
                    if is_preferred_target:
                        similarity = max(similarity, 160)
                    metric_kind = str(metric.get("tableKind") or "").lower()
                    metric_intent = str(metric.get("metricIntent") or "").lower()
                    dimension_score = self._table_dimension_score(table, obligations, table_topic)
                    score = similarity + dimension_score
                    if "detail" in metric_kind or "detail" in metric_intent:
                        score += 24
                    formula = str(metric.get("formula") or "")
                    if re.search(r"\b(case|when|filter)\b", formula, re.I):
                        # Conditional variants remain eligible when the user's
                        # phrase names the condition, but do not outrank the
                        # unconditioned base measure merely because their label
                        # contains more words.
                        score -= 28
                    if distance is not None:
                        score += max(0, 12 - distance * 3)
                    candidates.append(
                        {
                            "table": table,
                            "topic": topic,
                            "metricKey": metric_key,
                            "phrase": phrase,
                            "score": score,
                            "distance": distance,
                        }
                    )
        return sorted(
            candidates,
            key=lambda item: (
                -int(item.get("score") or 0),
                int(item.get("distance")) if item.get("distance") is not None else 99,
                str(item.get("table") or ""),
                str(item.get("metricKey") or ""),
            ),
        )

    def _detail_metric_targets(
        self,
        precise_metric_seed_traces: List[str],
        table_topic: Dict[str, str],
    ) -> Dict[str, Set[Tuple[str, str]]]:
        targets: Dict[str, Set[Tuple[str, str]]] = {}
        for trace in precise_metric_seed_traces:
            text = str(trace or "")
            if not text.startswith("precise_metric_seed:"):
                continue
            parts = text.split(":", 3)
            if len(parts) < 4:
                continue
            _prefix, owner_table, metric_key, phrase = parts
            topic = table_topic.get(owner_table, "")
            if not topic:
                continue
            metric = next(
                (
                    item
                    for item in self.topic_assets.load_table_metrics(topic, owner_table)
                    if str(item.get("metricKey") or "") == metric_key
                ),
                {},
            )
            detail_ref = str(
                metric.get("detailMetricRef")
                or metric.get("drilldownMetricRef")
                or metric.get("detail_metric_ref")
                or ""
            )
            detail_identity = semantic_metric_ref_identity(detail_ref)
            if detail_identity:
                targets.setdefault(phrase, set()).add(detail_identity)
        return targets

    def _preferred_dimension_owner(
        self,
        obligations: List[Dict[str, Any]],
        table_topic: Dict[str, str],
        graph: Dict[str, Set[str]],
        candidate_metric_tables: Set[str],
        require_dimension_layer: bool,
    ) -> str:
        del graph, candidate_metric_tables, require_dimension_layer
        candidates: List[Tuple[int, str]] = []
        requested_topics = {str(item.get("topic") or "") for item in obligations if item.get("topic")}
        declared_owners = {
            str(item.get("ownerTable") or "")
            for item in obligations
            if str(item.get("ownerTable") or "")
        }
        # Field existence is not an authority contract. Only owners carried by
        # governed dimension mentions are eligible; otherwise the caller emits
        # RELATIONSHIP_DIMENSION_OWNER_REQUIRED instead of guessing from graph
        # centrality or a same-named fact column.
        for table in sorted(declared_owners):
            topic = table_topic.get(table, "")
            if not topic:
                continue
            score = self._table_dimension_score(table, obligations, table_topic)
            if score <= 0:
                continue
            layer = self._table_business_layer(table, table_topic)
            if layer == "DIM":
                score += 80
            if topic in requested_topics or str(self.topic_assets.resolve_topic_category(topic)) in requested_topics:
                score += 24
            asset = self.topic_assets.load_table_asset(topic, table)
            usage = asset.get("tableUsageProfile") if isinstance(asset, dict) else {}
            score += min(int((usage or {}).get("authorityLevel") or 0), 100)
            candidates.append((score, table))
        return max(candidates, default=(0, ""), key=lambda item: (item[0], item[1]))[1]

    def _table_satisfies_dimension_obligations(
        self,
        table: str,
        obligations: List[Dict[str, Any]],
        table_topic: Dict[str, str],
    ) -> bool:
        groups = dimension_obligation_groups(obligations)
        if not groups:
            return True
        topic = table_topic.get(table, "")
        fields = self.topic_assets.load_table_semantic_columns(topic, table) if topic else []
        columns = {
            str(item.get("columnName") or "")
            for item in fields
            if isinstance(item, dict) and str(item.get("columnName") or "")
        }
        return all(columns.intersection(columns_for_group) for columns_for_group in groups.values())

    def _table_dimension_score(
        self,
        table: str,
        obligations: List[Dict[str, Any]],
        table_topic: Dict[str, str],
    ) -> int:
        topic = table_topic.get(table, "")
        if not topic:
            return 0
        fields = self.topic_assets.load_table_semantic_columns(topic, table)
        by_column = {
            str(item.get("columnName") or ""): item
            for item in fields
            if isinstance(item, dict) and str(item.get("columnName") or "")
        }
        score = 0
        for columns in dimension_obligation_groups(obligations).values():
            matches = [by_column[column] for column in columns if column in by_column]
            if not matches:
                return 0
            match_score = max(
                20
                + (16 if str(item.get("role") or "").upper() == "KEY" else 4)
                for item in matches
            )
            score += match_score
        return score

    def _table_business_layer(self, table: str, table_topic: Dict[str, str]) -> str:
        topic = table_topic.get(table, "")
        if not topic:
            return ""
        asset = self.topic_assets.load_table_asset(topic, table)
        usage = asset.get("tableUsageProfile") if isinstance(asset, dict) else {}
        return str((usage or {}).get("businessLayer") or "").upper()

    def _recall_item_has_precise_table_evidence(self, item: RecallItem) -> bool:
        source_type = str(item.source_type or "").upper()
        metadata = item.metadata or {}
        semantic_kind = str(metadata.get("semanticKind") or "").upper()
        table = item.table or str(metadata.get("tableName") or "")
        metric_key = str(metadata.get("metricKey") or "")
        if not table or not metric_key:
            return False
        return source_type == "SEMANTIC_METRIC" or semantic_kind == "METRIC" or bool(metadata.get("metricKey"))

    def _should_defer_topic_seed_selection(self, question: str, has_precise_recall_evidence: bool) -> bool:
        del question
        return not has_precise_recall_evidence

    def _broad_topic_question(self, question: str) -> bool:
        del question
        return True

    def _table_allowed_for_recalled_item(
        self,
        question: str,
        item: RecallItem,
        table: str,
        allow_profile: bool = False,
    ) -> bool:
        if not table:
            return False
        if self._recall_item_has_precise_table_evidence(item) and self._recalled_metric_table_queryable(item, table):
            return True
        topic = str(item.topic or (item.metadata or {}).get("topic") or "")
        if self._table_allowed_for_topic_question(topic, question, table, allow_profile=allow_profile):
            return True
        return False

    def _recalled_metric_table_queryable(self, item: RecallItem, table: str) -> bool:
        topic = str(item.topic or (item.metadata or {}).get("topic") or "")
        return self._table_queryable_for_topic(topic, table) if topic else False

    def _table_queryable_for_topic(self, topic: str, table: str) -> bool:
        if not table:
            return False
        if not topic:
            return False
        try:
            asset = self.topic_assets.load_table_asset(topic, table)
        except Exception:
            return False
        if not asset or "tableUsageProfile" not in asset:
            return False
        usage = normalize_table_usage_profile(asset.get("tableUsageProfile") or {}, table)
        return bool(usage.get("queryableByAgent"))

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
                    if not self._table_allowed_for_topic_question(table_topic.get(table, ""), "", table, allow_profile=allow_profile):
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
            semantic_role = str(semantic.get("semanticRole") or semantic.get("role") or "").upper()
            if bool(col.get("isPrimaryKey")) or semantic_role in {"KEY", "ENTITY_KEY", "JOIN_KEY", "PRIMARY_KEY"}:
                pack.entity_keys.append(
                    PlanningAssetEntry(
                        key=name,
                        table=table,
                        topic=topic,
                        title=comment,
                        aliases=dedupe_strings(aliases),
                        description=json.dumps({"schema": col, "semantic": semantic}, ensure_ascii=False),
                        source_ref_id="semantic:%s:%s:key:%s" % (topic, table, name),
                        metadata={"schema": col, "semantic": semantic},
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
        prioritized_names = [name for name, semantic in semantic_by_column.items() if name in rows_by_name and semantic]
        contract_names = [
            name
            for name, row in rows_by_name.items()
            if bool(row.get("isPrimaryKey")) or bool(row.get("isPartitionColumn"))
        ]
        physical_names = list(rows_by_name.keys())
        ordered_names = dedupe_strings(prioritized_names + contract_names + physical_names)
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
        return self.topic_assets.semantic_table_source_hash(topic, table)

    def _semantic_published_at(self, topic: str, table: str) -> str:
        asset_path = self.topic_assets.root / topic / "tables" / table / "asset.json"
        if not asset_path.exists():
            return ""
        return datetime.fromtimestamp(asset_path.stat().st_mtime).isoformat()

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
            if self._table_relevant(question, item) and self._table_allowed_for_topic_question(topic, question, str(item.get("tableName") or ""), allow_profile=allow_profile)
        ]
        if relevant:
            return {table for table in relevant if table}
        return {
            table
            for table in [str(item.get("tableName") or "") for item in manifest]
            if table and self._table_allowed_for_topic_question(topic, question, table, allow_profile=allow_profile)
        }

    def _diagnostic_profile_seed_tables(self, table_topic: Dict[str, str], limit: int = 1) -> List[str]:
        candidates: List[Tuple[int, str]] = []
        for table, topic in table_topic.items():
            if not table:
                continue
            try:
                asset = self.topic_assets.load_table_asset(topic, table)
            except Exception:
                asset = {}
            usage = normalize_table_usage_profile(asset.get("tableUsageProfile") or {}, table)
            if not usage.get("queryableByAgent") or str(usage.get("topicRole") or "").upper() != "PROFILE":
                continue
            candidates.append((int(usage.get("authorityLevel") or 0), table))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return [table for _, table in candidates[: max(1, limit)]]

    def _table_allowed_for_question(self, question: str, table: str, allow_profile: bool = False) -> bool:
        del question, allow_profile
        if not table:
            return False
        return True

    def _table_allowed_for_topic_question(self, topic: str, question: str, table: str, allow_profile: bool = False) -> bool:
        if not self._table_allowed_for_question(question, table, allow_profile=allow_profile):
            return False
        if not topic:
            return False
        try:
            asset = self.topic_assets.load_table_asset(topic, table)
        except Exception:
            asset = {}
        if not asset or "tableUsageProfile" not in asset:
            return False
        usage = normalize_table_usage_profile(asset.get("tableUsageProfile") or {}, table)
        return bool(usage.get("queryableByAgent"))


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
    del rel, keys
    return ["UNDECLARED"]


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


def normalize_planning_hints(value: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    metric_phrases = dedupe_strings([str(item) for item in value.get("metricPhrases") or value.get("metric_phrases") or []])
    dimension_keywords = dedupe_strings(
        [str(item) for item in value.get("dimensionKeywords") or value.get("dimension_keywords") or []]
    )
    dimensions = [dict(item) for item in value.get("dimensions") or [] if isinstance(item, dict)]
    ranking = value.get("ranking") if isinstance(value.get("ranking"), dict) else {}
    return {
        key: item
        for key, item in {
            "metricPhrases": metric_phrases,
            "dimensionKeywords": dimension_keywords,
            "dimensions": dimensions,
            "ranking": dict(ranking),
            "analysisIntent": str(value.get("analysisIntent") or value.get("analysis_intent") or ""),
        }.items()
        if item not in (None, "", [], {})
    }


def planning_dimension_obligations(planning_hints: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    hints = normalize_planning_hints(planning_hints)
    obligations: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str, str]] = set()
    for item in hints.get("dimensions") or []:
        phrase = str(item.get("phrase") or "")
        column = str(item.get("column") or item.get("canonicalKey") or "")
        topic = str(item.get("topic") or "")
        owner = str(item.get("ownerTable") or "")
        identity = (phrase, column, topic, owner)
        if not column or identity in seen:
            continue
        seen.add(identity)
        obligations.append(
            {
                "phrase": phrase or column,
                "column": column,
                "topic": topic,
                "ownerTable": owner,
                "role": str(item.get("role") or ""),
                "source": str(item.get("source") or ""),
            }
        )
    return obligations


def dimension_obligation_groups(obligations: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    groups: Dict[str, Set[str]] = {}
    for item in obligations:
        phrase = str(item.get("phrase") or item.get("column") or "")
        column = str(item.get("column") or "")
        if phrase and column:
            groups.setdefault(phrase, set()).add(column)
    return groups


def dimension_refs_for_obligations(obligations: List[Dict[str, Any]]) -> List[str]:
    return dedupe_strings([str(item.get("column") or "") for item in obligations])


def dimension_phrase_for_obligations(obligations: List[Dict[str, Any]]) -> str:
    return " / ".join(dedupe_strings([str(item.get("phrase") or "") for item in obligations]))


def metric_refs_for_phrase(planning_hints: Dict[str, Any], phrase: str) -> List[str]:
    ranking = planning_hints.get("ranking") if isinstance(planning_hints.get("ranking"), dict) else {}
    if str(ranking.get("anchorMetricPhrase") or "") == str(phrase or ""):
        return dedupe_strings([str(item) for item in ranking.get("anchorMetricCandidates") or []])
    return []


def relationship_table_graph(
    relationships: List[Tuple[str, Dict[str, Any]]],
) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = {}
    for _topic, rel in relationships:
        left = str(rel.get("leftTable") or "")
        right = str(rel.get("rightTable") or "")
        if not left or not right:
            continue
        graph.setdefault(left, set()).add(right)
        graph.setdefault(right, set()).add(left)
    return graph


def relationship_edges_from_targeted_traces(traces: List[str]) -> Set[FrozenSet[str]]:
    edges: Set[FrozenSet[str]] = set()
    for trace in traces:
        text = str(trace or "")
        if not text.startswith("dimension_relationship_path:"):
            continue
        path = [item for item in text.split(":", 1)[1].split("->") if item]
        for left, right in zip(path, path[1:]):
            edges.add(frozenset({left, right}))
    return edges


def shortest_distance_from_tables(
    graph: Dict[str, Set[str]],
    starts: Set[str],
    target: str,
    max_hops: int = 3,
) -> int | None:
    if target in starts:
        return 0
    distances = [
        len(path) - 1
        for start in starts
        for path in [shortest_table_path(graph, start, target, max_hops=max_hops)]
        if path
    ]
    return min(distances) if distances else None


def governed_dimension_path(
    relationships: List[Tuple[str, Dict[str, Any]]],
    source: str,
    target: str,
    obligations: List[Dict[str, Any]],
    max_hops: int = 3,
) -> List[str]:
    """Choose a published path whose final edge reaches the governed key.

    A shorter name-based edge must not outrank a slightly longer path that lands
    on the KEY declared by the dimension obligation.  The function only ranks
    published edges; it never synthesizes a join.
    """

    graph = relationship_table_graph(relationships)
    if source == target:
        return [source]
    paths: List[List[str]] = []

    def visit(path: List[str]) -> None:
        if len(path) - 1 >= max_hops:
            return
        for neighbor in sorted(graph.get(path[-1], set())):
            if neighbor in path:
                continue
            candidate = [*path, neighbor]
            if neighbor == target:
                paths.append(candidate)
            else:
                visit(candidate)

    visit([source])
    if not paths:
        return []
    target_obligations = [
        item
        for item in obligations
        if not str(item.get("ownerTable") or "") or str(item.get("ownerTable") or "") == target
    ] or obligations
    preferred_columns = {
        str(item.get("column") or "")
        for item in target_obligations
        if str(item.get("role") or "").upper() == "KEY" and str(item.get("column") or "")
    }
    all_columns = {
        str(item.get("column") or "")
        for item in target_obligations
        if str(item.get("column") or "")
    }

    def path_score(path: List[str]) -> Tuple[int, int, str]:
        previous = path[-2]
        landing_columns: Set[str] = set()
        explicit_authority = 0
        for _topic, rel in relationships:
            left = str(rel.get("leftTable") or "")
            right = str(rel.get("rightTable") or "")
            if {left, right} != {previous, target}:
                continue
            for pair in rel.get("keys") or []:
                if not isinstance(pair, list) or len(pair) < 2:
                    continue
                landing_columns.add(str(pair[1] if right == target else pair[0]))
            if rel.get("canonicalEntityRef") or rel.get("entityMapping") or rel.get("fieldAuthority"):
                explicit_authority = 40
        authority_score = explicit_authority
        if preferred_columns.intersection(landing_columns):
            authority_score += 120
        elif all_columns.intersection(landing_columns):
            authority_score += 35
        return (authority_score, -(len(path) - 1), "->".join(path))

    return max(paths, key=path_score)


def governed_relationship_ref_for_edge(
    relationships: List[Tuple[str, Dict[str, Any]]],
    left_table: str,
    right_table: str,
    dimension_owner: str,
    obligations: List[Dict[str, Any]],
    recalled_refs: Set[str] | None = None,
) -> str:
    recalled_refs = recalled_refs or set()
    preferred_columns = {
        str(item.get("column") or "")
        for item in obligations
        if str(item.get("ownerTable") or "") == dimension_owner
        and str(item.get("role") or "").upper() == "KEY"
        and str(item.get("column") or "")
    }
    candidates: List[Tuple[int, str]] = []
    for topic, rel in relationships:
        rel_left = str(rel.get("leftTable") or "")
        rel_right = str(rel.get("rightTable") or "")
        if {rel_left, rel_right} != {left_table, right_table}:
            continue
        ref = "semantic:%s:relationship:%s" % (topic, rel.get("name") or "")
        score = 1000 if ref in recalled_refs else 0
        if dimension_owner in {rel_left, rel_right}:
            landing_columns = {
                str(pair[0] if rel_left == dimension_owner else pair[1])
                for pair in rel.get("keys") or []
                if isinstance(pair, list) and len(pair) >= 2
            }
            if preferred_columns.intersection(landing_columns):
                score += 120
        if rel.get("canonicalEntityRef") or rel.get("entityMapping") or rel.get("fieldAuthority"):
            score += 50
        candidates.append((score, ref))
    return max(candidates, default=(0, ""), key=lambda item: (item[0], item[1]))[1]


def semantic_phrase_similarity(phrase: str, labels: List[str]) -> int:
    target = normalize_for_match(phrase)
    if not target:
        return 0
    best = 0
    target_bigrams = semantic_bigrams(target)
    for raw_label in labels:
        label = normalize_for_match(raw_label)
        if not label:
            continue
        if target == label:
            best = max(best, 120)
            continue
        if target in label or label in target:
            best = max(best, 105)
        if semantic_subsequence(target, label):
            best = max(best, 88)
        label_bigrams = semantic_bigrams(label)
        if target_bigrams:
            overlap = len(target_bigrams.intersection(label_bigrams)) / len(target_bigrams)
            best = max(best, int(overlap * 72))
    return best


def semantic_exact_phrase_similarity(phrase: str, labels: List[str]) -> int:
    target = normalize_for_match(phrase)
    if not target:
        return 0
    return 120 if any(normalize_for_match(label) == target for label in labels if str(label or "").strip()) else 0


def semantic_metric_ref_identity(ref_id: str) -> Tuple[str, str] | None:
    parts = [part for part in str(ref_id or "").split(":") if part]
    if len(parts) < 5 or parts[-2] != "metric":
        return None
    table = parts[-3]
    metric_key = parts[-1]
    return (table, metric_key) if table and metric_key else None


def semantic_subsequence(needle: str, haystack: str) -> bool:
    if len(needle) < 2 or not haystack:
        return False
    index = 0
    for character in haystack:
        if index < len(needle) and needle[index] == character:
            index += 1
    return index == len(needle)


def semantic_bigrams(value: str) -> Set[str]:
    return {value[index : index + 2] for index in range(max(0, len(value) - 1))}


def downgrade_incompatible_metric_seed_traces(
    traces: List[str],
    incompatible_tables: Set[str],
) -> List[str]:
    downgraded: List[str] = []
    for trace in traces:
        text = str(trace or "")
        if not text.startswith("precise_metric_seed:"):
            downgraded.append(text)
            continue
        parts = text.split(":", 3)
        if len(parts) >= 2 and parts[1] in incompatible_tables:
            downgraded.append("dimension_incompatible_metric_seed:%s" % text[len("precise_metric_seed:") :])
        else:
            downgraded.append(text)
    return downgraded


def planning_gap_trace(gap: Dict[str, Any]) -> str:
    return "planning_gap:%s" % json.dumps(gap, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def planning_gaps_from_asset_traces(traces: List[str]) -> List[Dict[str, Any]]:
    gaps: List[Dict[str, Any]] = []
    for trace in traces:
        text = str(trace or "")
        if not text.startswith("planning_gap:"):
            continue
        try:
            payload = json.loads(text[len("planning_gap:") :])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            gaps.append(payload)
    return gaps


def recalled_relationship_evidence_from_bundle(recall_bundle: RecallBundle) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for item in recall_bundle.items:
        if str(item.source_type or "").upper() != "SEMANTIC_RELATIONSHIP":
            continue
        metadata = item.metadata or {}
        ref = str(metadata.get("semanticRefId") or item.doc_id or "")
        if not ref or ref in seen:
            continue
        seen.add(ref)
        evidence.append(
            {
                "semanticRefId": ref,
                "relationshipId": str(metadata.get("relationshipId") or metadata.get("name") or ""),
                "leftTable": str(metadata.get("leftTable") or ""),
                "rightTable": str(metadata.get("rightTable") or ""),
            }
        )
    return evidence


def relationship_contract_assessment(
    asset_pack: PlanningAssetPack,
    relationships: List[Tuple[str, Dict[str, Any]]],
    planning_hints: Dict[str, Any],
    targeted_traces: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dimensions = planning_dimension_obligations(planning_hints)
    metric_seed_count = len(
        {
            tuple(str(trace).split(":", 3)[1:3])
            for trace in targeted_traces
            if str(trace).startswith("precise_metric_seed:") and len(str(trace).split(":", 3)) >= 3
        }
    )
    metric_refs = dedupe_strings(
        [
            str(trace).split(":", 3)[2]
            for trace in targeted_traces
            if str(trace).startswith("precise_metric_seed:") and len(str(trace).split(":", 3)) >= 3
        ]
    )
    requires_cross_owner_aggregation = bool(dimensions and metric_seed_count > 0)
    contracts: List[Dict[str, Any]] = []
    gaps: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()
    canonical_refs: List[str] = []
    for topic, rel in relationships:
        identity = (
            str(rel.get("name") or ""),
            str(rel.get("leftTable") or ""),
            str(rel.get("rightTable") or ""),
        )
        if identity in seen:
            continue
        seen.add(identity)
        ref = "semantic:%s:relationship:%s" % (topic, rel.get("name") or "")
        cardinality = rel.get("cardinality") or rel.get("relationshipCardinality")
        fanout_policy = rel.get("fanoutPolicy") or rel.get("aggregationJoinPolicy") or rel.get("preAggregate")
        canonical_entity = rel.get("canonicalEntityRef") or rel.get("entityMapping")
        if canonical_entity:
            canonical_refs.append(ref)
        contracts.append(
            {
                "sourceRefId": ref,
                "relationshipId": identity[0],
                "leftTable": identity[1],
                "rightTable": identity[2],
                "joinKeysDeclared": bool(rel.get("keys")),
                "grainDeclared": bool(rel.get("grain")),
                "cardinality": cardinality or "undeclared",
                "fanoutPolicy": fanout_policy or "undeclared",
                "canonicalEntity": canonical_entity or "undeclared",
                "cautions": [str(item) for item in rel.get("cautions") or []],
            }
        )
    if requires_cross_owner_aggregation:
        dimension_owner = next(
            (
                str(trace).split(":", 1)[1]
                for trace in targeted_traces
                if str(trace).startswith("dimension_authority_seed:")
            ),
            "",
        )
        metric_owners = {
            str(trace).split(":", 3)[1]
            for trace in targeted_traces
            if str(trace).startswith("precise_metric_seed:") and len(str(trace).split(":", 3)) >= 3
        }
        if dimension_owner and metric_owners:
            # Validate each measure branch in its real direction: fact owner ->
            # authoritative dimension, then pre-aggregate.  A single connector
            # tree rooted at the dimension would model a raw multi-fact join and
            # could reverse many-to-one safety.
            for metric_owner in sorted(metric_owners - {dimension_owner}):
                join_result = plan_governed_joins(
                    asset_pack,
                    base_table=metric_owner,
                    required_tables=[dimension_owner],
                    usage="aggregate",
                )
                branch_gaps = list(join_result.gaps)
                if any(gap.code != "JOIN_PATH_NOT_FOUND" for gap in branch_gaps):
                    branch_gaps = [gap for gap in branch_gaps if gap.code != "JOIN_PATH_NOT_FOUND"]
                for gap in branch_gaps:
                    gaps.append(
                        {
                            "code": gap.code,
                            "type": "RELATIONSHIP",
                            "query": "governed aggregate projection from %s to %s"
                            % (metric_owner, dimension_owner),
                            "reason": gap.reason,
                            "metricRefs": metric_refs,
                            "dimensionRefs": dimension_refs_for_obligations(dimensions),
                            "dimensionPhrase": dimension_phrase_for_obligations(dimensions),
                            "relationshipRefs": [gap.relationship_ref_id] if gap.relationship_ref_id else [],
                            "sourceOwner": metric_owner,
                            "targetOwner": dimension_owner,
                            "requiredSemantics": [
                                "directional_cardinality",
                                "pre_aggregate_grain",
                                "dedup_key",
                                "fanout_policy",
                            ],
                            "evidence": gap.evidence,
                        }
                    )
    if requires_cross_owner_aggregation and relationships and not canonical_refs:
        gaps.append(
            {
                "code": "RELATIONSHIP_CANONICAL_ENTITY_UNDECLARED",
                "type": "RELATIONSHIP",
                "query": "canonical entity mapping for dimension %s" % dimension_phrase_for_obligations(dimensions),
                "reason": "published join paths do not declare which cross-table field is authoritative for the requested entity",
                "metricRefs": metric_refs,
                "dimensionRefs": dimension_refs_for_obligations(dimensions),
                "dimensionPhrase": dimension_phrase_for_obligations(dimensions),
                "relationshipRefs": [item["sourceRefId"] for item in contracts],
                "requiredSemantics": ["canonical_entity", "field_authority", "equivalence_policy"],
            }
        )
    return contracts, gaps


def dedupe_planning_gaps(gaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()
    for gap in gaps:
        if not isinstance(gap, dict):
            continue
        identity = (
            str(gap.get("code") or ""),
            str(gap.get("query") or ""),
            str(gap.get("sourceOwner") or gap.get("sourceOwners") or ""),
        )
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(gap)
    return selected


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


def semantic_table_detail_ref_id(topic: str, table: str) -> str:
    return "semantic:%s:%s:detail" % (topic, table)


def semantic_table_section_ref_id(topic: str, table: str, section: str) -> str:
    return "semantic:%s:%s:%s" % (topic, table, section)


def semantic_manifest_ref_id(topic: str) -> str:
    return "semantic:%s:manifest" % topic


def semantic_relationship_ref_id(topic: str) -> str:
    return "semantic:%s:relationships" % topic


def semantic_relationship_index_ref_id(topic: str) -> str:
    return "semantic:%s:relationship_index" % topic


def semantic_relationship_entry_ref_id(topic: str, entry_key: str) -> str:
    return "semantic:%s:relationship:%s" % (topic, entry_key)


def semantic_metric_ref_id(topic: str, table: str, metric_key: str) -> str:
    return "semantic:%s:%s:metric:%s" % (topic, table, metric_key)


def semantic_manifest_path(topic: str) -> str:
    return "topics/%s/manifest.json" % topic


def semantic_table_path(topic: str, table: str) -> str:
    return "topics/%s/tables/%s/asset.json" % (topic, table)


def semantic_table_detail_path(topic: str, table: str) -> str:
    return "topics/%s/tables/%s/detail.json" % (topic, table)


def semantic_table_section_path(topic: str, table: str, section: str) -> str:
    if section == "schema":
        return "topics/%s/tables/%s/schema.json" % (topic, table)
    return "topics/%s/tables/%s/%s/index.json" % (topic, table, section)


def semantic_metric_path(topic: str, table: str, metric_key: str) -> str:
    return semantic_table_entry_path(topic, table, "metrics", metric_key)


def semantic_table_entry_ref_id(topic: str, table: str, section: str, entry_key: str) -> str:
    singular = {"metrics": "metric", "columns": "field", "terms": "term", "rules": "rule"}.get(section, section)
    return "semantic:%s:%s:%s:%s" % (topic, table, singular, entry_key)


def semantic_table_entry_path(topic: str, table: str, section: str, entry_key: str) -> str:
    return "topics/%s/tables/%s/%s/%s.json" % (topic, table, section, entry_key)


def normalize_entity_filter_operator(value: Any) -> str:
    normalized = str(value or "").strip().upper().replace(" ", "_")
    return {
        "=": "EQ",
        "==": "EQ",
        "EQ": "EQ",
        "IN": "IN",
        "!=": "NE",
        "<>": "NE",
        "NE": "NE",
        ">": "GT",
        "GT": "GT",
        ">=": "GTE",
        "GTE": "GTE",
        "<": "LT",
        "LT": "LT",
        "<=": "LTE",
        "LTE": "LTE",
    }.get(normalized, "")


def normalize_entity_lookup_policy(
    raw: Any,
    *,
    inherited_time_column: str = "",
    source: str = "",
) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    policy = dict(raw)
    mode = str(policy.get("mode") or "").strip().lower()
    if not mode:
        return {}
    normalized: Dict[str, Any] = {
        **policy,
        "mode": mode,
        "timeColumn": str(
            policy.get("timeColumn")
            or policy.get("time_column")
            or inherited_time_column
            or ""
        ).strip(),
    }
    if "timeRequired" in policy or "time_required" in policy:
        normalized["timeRequired"] = bool(
            policy.get("timeRequired", policy.get("time_required"))
        )
    else:
        normalized["timeRequired"] = mode not in NO_TIME_ENTITY_LOOKUP_MODES
    if source:
        normalized["policySource"] = source
    return normalized


def progressive_semantic_column_definition(
    asset: Dict[str, Any],
    semantic_column: Dict[str, Any],
) -> Dict[str, Any]:
    """Publish one field's governed usage capabilities at the L2 boundary.

    This projection is deliberately generic. It translates declarations from
    the semantic asset into a stable field contract; it never recognizes a
    business domain, table name, column name, or example literal.
    """

    definition = dict(semantic_column or {})
    column_name = str(
        definition.get("columnName")
        or definition.get("Field")
        or definition.get("field")
        or ""
    ).strip()
    schema = next(
        (
            dict(item)
            for item in schema_columns(asset)
            if isinstance(item, dict)
            and str(item.get("columnName") or item.get("Field") or "").strip()
            == column_name
        ),
        {},
    )
    if schema:
        definition["schemaContract"] = {
            "dataType": str(
                schema.get("dataType") or schema.get("Type") or ""
            ).strip(),
            "nullable": schema.get("nullable", schema.get("Null")),
            "keyType": str(schema.get("keyType") or schema.get("Key") or "").strip(),
        }

    role = str(
        definition.get("entityRole")
        or definition.get("role")
        or definition.get("semanticRole")
        or ""
    ).strip().upper()
    if role not in ENTITY_SEMANTIC_ROLES:
        return definition

    definition["entityRole"] = role
    canonical_entity = str(
        definition.get("canonicalEntityRef")
        or definition.get("entityIdentity")
        or definition.get("canonicalEntityType")
        or definition.get("entityType")
        or ""
    ).strip()
    if canonical_entity:
        definition["canonicalEntityRef"] = canonical_entity

    definition["isUniqueEntityKey"] = bool(
        definition.get("isUniqueEntityKey")
        or definition.get("isUniqueKey")
        or definition.get("is_unique_entity_key")
    )

    raw_operators = (
        definition.get("filterOperators")
        or definition.get("filter_operators")
        or []
    )
    if isinstance(raw_operators, str):
        raw_operators = [raw_operators]
    operators = dedupe_strings(
        normalize_entity_filter_operator(item) for item in raw_operators
    )
    comparison_policy = str(definition.get("comparisonPolicy") or "").strip().lower()
    if not operators and comparison_policy in SUPPORTED_ENTITY_COMPARISON_POLICIES:
        # A governed equality comparison policy implies exact/same-entity
        # lookup operators. Range support remains opt-in.
        operators = ["EQ", "IN"]
    definition["filterOperators"] = operators

    field_policy = definition.get("lookupTimePolicy") or definition.get(
        "lookup_time_policy"
    )
    policy_source = "field"
    if not isinstance(field_policy, dict):
        field_policy = asset.get("entityLookupPolicy")
        policy_source = "table"
    definition["lookupTimePolicy"] = normalize_entity_lookup_policy(
        field_policy,
        inherited_time_column=str(asset.get("timeColumn") or ""),
        source=policy_source,
    )
    return definition


def semantic_table_entry_key(section: str, value: Dict[str, Any], index: int) -> str:
    candidates = {
        "metrics": ["metricKey", "key"],
        "columns": ["columnName", "field", "name"],
        "terms": ["term", "termKey", "key"],
        "rules": ["ruleKey", "ruleId", "id", "title"],
    }.get(section, ["key", "name"])
    raw = next((str(value.get(key) or "").strip() for key in candidates if str(value.get(key) or "").strip()), "")
    if raw:
        return sanitize_semantic_file_name(raw)[:120]
    digest = hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    return "%s-%s" % (section.rstrip("s"), digest)


def semantic_table_entry_keys(section: str, values: Sequence[Any]) -> List[str]:
    """Create stable entry keys and fail closed on true identity collisions."""

    bases = [
        semantic_table_entry_key(section, value, index) if isinstance(value, dict) else ""
        for index, value in enumerate(values)
    ]
    counts = Counter(base for base in bases if base)
    keys: List[str] = []
    seen: Set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            keys.append("")
            continue
        base = bases[index]
        key = base
        if counts[base] > 1:
            digest = hashlib.sha256(
                json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:10]
            key = "%s-%s" % (base[:109], digest)
        if key in seen:
            raise ValueError("SEMANTIC_ENTRY_KEY_COLLISION: %s/%s" % (section, key))
        seen.add(key)
        keys.append(key)
    return keys


def semantic_table_entry_title(section: str, value: Dict[str, Any], fallback: str) -> str:
    keys = {
        "metrics": ["businessName", "title", "metricKey"],
        "columns": ["businessName", "comment", "columnName"],
        "terms": ["term", "description"],
        "rules": ["title", "ruleKey"],
    }.get(section, ["title", "name", "key"])
    return next((str(value.get(key) or "").strip() for key in keys if str(value.get(key) or "").strip()), fallback)


def semantic_relationship_path(topic: str) -> str:
    return "topics/%s/relationships.json" % topic


def semantic_relationship_index_path(topic: str) -> str:
    return "topics/%s/relationships/index.json" % topic


def semantic_relationship_entry_path(topic: str, entry_key: str) -> str:
    return "topics/%s/relationships/%s.json" % (topic, entry_key)


def semantic_relationship_key_for_name(
    relationships: Sequence[Any],
    relationship_name: str,
) -> str:
    wanted = str(relationship_name or "").strip()
    if not wanted:
        return ""
    keys = semantic_table_entry_keys("relationships", relationships)
    for index, relationship in enumerate(relationships):
        if not isinstance(relationship, dict):
            continue
        if str(relationship.get("name") or "").strip() == wanted:
            return keys[index]
    return ""


def parse_semantic_relationship_entry_identity(
    ref_id: str,
    path: str,
) -> Tuple[str, str, str] | None:
    wanted_ref = str(ref_id or "").strip()
    wanted_path = normalize_semantic_path(path)
    index_ref = re.fullmatch(r"semantic:([^:]+):relationship_index", wanted_ref)
    if index_ref:
        return "index", str(index_ref.group(1)), ""
    index_path = re.fullmatch(r"topics/([^/]+)/relationships/index\.json", wanted_path)
    if index_path:
        return "index", str(index_path.group(1)), ""
    entry_ref = re.fullmatch(r"semantic:([^:]+):relationship:(.+)", wanted_ref)
    if entry_ref:
        return "entry", str(entry_ref.group(1)), str(entry_ref.group(2))
    entry_path = re.fullmatch(
        r"topics/([^/]+)/relationships/([^/]+)\.json",
        wanted_path,
    )
    if entry_path:
        return "entry", str(entry_path.group(1)), str(entry_path.group(2))
    return None


def parse_semantic_metric_identity(ref_id: str, path: str) -> Tuple[str, str, str] | None:
    ref_match = re.fullmatch(r"semantic:([^:]+):([^:]+):metric:(.+)", str(ref_id or "").strip())
    if ref_match:
        return tuple(str(item) for item in ref_match.groups())
    path_match = re.fullmatch(
        r"topics/([^/]+)/tables/([^/]+)/metrics/([^/]+)\.json",
        normalize_semantic_path(path),
    )
    if path_match:
        identity = tuple(str(item) for item in path_match.groups())
        return identity if identity[-1] != "index" else None
    return None


def parse_semantic_table_entry_identity(ref_id: str, path: str) -> Tuple[str, str, str, str] | None:
    wanted_ref = str(ref_id or "").strip()
    wanted_path = normalize_semantic_path(path)
    ref_match = re.fullmatch(r"semantic:([^:]+):([^:]+):(field|term|rule):(.+)", wanted_ref)
    if ref_match:
        topic, table, singular, key = ref_match.groups()
        section = {"field": "columns", "term": "terms", "rule": "rules"}[singular]
        return topic, table, section, key
    path_match = re.fullmatch(r"topics/([^/]+)/tables/([^/]+)/(columns|terms|rules)/([^/]+)\.json", wanted_path)
    if path_match:
        topic, table, section, key = path_match.groups()
        return (topic, table, section, key) if key != "index" else None
    return None


def parse_semantic_file_identity(ref_id: str, path: str) -> Tuple[str, str, str, str] | None:
    """Parse a canonical semantic virtual file without scanning all assets."""

    wanted_ref = str(ref_id or "").strip()
    wanted_path = normalize_semantic_path(path)
    manifest_match = re.fullmatch(r"semantic:([^:]+):manifest", wanted_ref)
    if manifest_match:
        return "manifest", manifest_match.group(1), "", ""
    relationship_match = re.fullmatch(r"semantic:([^:]+):relationships", wanted_ref)
    if relationship_match:
        return "relationships", relationship_match.group(1), "", ""
    table_match = re.fullmatch(
        r"semantic:([^:]+):([^:]+):(asset|detail|metrics|columns|schema|terms|rules)",
        wanted_ref,
    )
    if table_match:
        topic, table, suffix = table_match.groups()
        if suffix == "asset":
            return "asset", topic, table, ""
        if suffix == "detail":
            return "detail", topic, table, ""
        return "section", topic, table, suffix
    manifest_path_match = re.fullmatch(r"topics/([^/]+)/manifest\.json", wanted_path)
    if manifest_path_match:
        return "manifest", manifest_path_match.group(1), "", ""
    relationship_path_match = re.fullmatch(r"topics/([^/]+)/relationships\.json", wanted_path)
    if relationship_path_match:
        return "relationships", relationship_path_match.group(1), "", ""
    table_path_match = re.fullmatch(
        r"topics/([^/]+)/tables/([^/]+)/(asset|detail|schema)\.json",
        wanted_path,
    )
    if table_path_match:
        topic, table, suffix = table_path_match.groups()
        if suffix == "asset":
            return "asset", topic, table, ""
        if suffix == "detail":
            return "detail", topic, table, ""
        return "section", topic, table, suffix
    section_index_match = re.fullmatch(
        r"topics/([^/]+)/tables/([^/]+)/(metrics|columns|terms|rules)/index\.json",
        wanted_path,
    )
    if section_index_match:
        topic, table, section = section_index_match.groups()
        return "section", topic, table, section
    return None


def normalize_physical_table_metadata(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        if any(key in payload for key in ["primaryKeyColumns", "partitionColumns", "bucketColumns", "keyModel"]):
            return {
                "keyModel": str(payload.get("keyModel") or ""),
                "primaryKeyColumns": dedupe_strings([str(item) for item in payload.get("primaryKeyColumns") or []]),
                "partitionColumns": dedupe_strings([str(item) for item in payload.get("partitionColumns") or []]),
                "bucketColumns": dedupe_strings([str(item) for item in payload.get("bucketColumns") or []]),
                "source": str(payload.get("source") or "metadata_provider"),
            }
        ddl = show_create_table_ddl_from_payload(payload)
        return parse_doris_create_table_metadata(ddl) if ddl else {}
    if isinstance(payload, list):
        for item in payload:
            ddl = show_create_table_ddl_from_payload(item)
            if ddl:
                return parse_doris_create_table_metadata(ddl)
    if isinstance(payload, str):
        return parse_doris_create_table_metadata(payload)
    return {}


def show_create_table_ddl_from_payload(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        return ""
    for key, value in payload.items():
        normalized = str(key or "").strip().lower().replace("_", " ")
        if normalized in {"create table", "createtable", "ddl"} or "create table" in normalized:
            return str(value or "")
    return ""


def parse_doris_create_table_metadata(ddl: str) -> Dict[str, Any]:
    text = str(ddl or "")
    if not text.strip():
        return {}
    key_model = ""
    primary_key_columns: List[str] = []
    key_match = re.search(
        r"\b(?P<model>DUPLICATE|UNIQUE|AGGREGATE|PRIMARY)\s+KEY\s*\((?P<columns>[^)]*)\)",
        text,
        re.I | re.S,
    )
    if key_match:
        key_model = "%s KEY" % key_match.group("model").upper()
        primary_key_columns = parse_identifier_list(key_match.group("columns"))
    partition_columns: List[str] = []
    partition_match = re.search(
        r"\bPARTITION\s+BY\s+(?:RANGE|LIST)?\s*\((?P<columns>[^)]*)\)",
        text,
        re.I | re.S,
    )
    if partition_match:
        partition_columns = parse_identifier_list(partition_match.group("columns"))
    if not partition_columns:
        auto_partition_match = re.search(
            r"\bAUTO\s+PARTITION\s+BY\s+(?:RANGE|LIST)?\s*\((?P<expr>.*?)\)\s*\(",
            text,
            re.I | re.S,
        )
        if auto_partition_match:
            partition_columns = parse_identifier_list(auto_partition_match.group("expr"))
    bucket_columns: List[str] = []
    bucket_match = re.search(r"\bDISTRIBUTED\s+BY\s+HASH\s*\((?P<columns>[^)]*)\)", text, re.I | re.S)
    if bucket_match:
        bucket_columns = parse_identifier_list(bucket_match.group("columns"))
    result = {
        "keyModel": key_model,
        "primaryKeyColumns": dedupe_strings(primary_key_columns),
        "partitionColumns": dedupe_strings(partition_columns),
        "bucketColumns": dedupe_strings(bucket_columns),
        "source": "show_create_table",
    }
    return {key: value for key, value in result.items() if value}


def parse_identifier_list(expression: str) -> List[str]:
    identifiers = re.findall(r"`([^`]+)`", str(expression or ""))
    if identifiers:
        return dedupe_strings(identifiers)
    values = []
    for raw in re.split(r",", str(expression or "")):
        token = raw.strip()
        token = re.sub(r"\b(date_trunc|date_floor|date_ceil|to_date|cast)\s*\(", "(", token, flags=re.I)
        match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", token)
        if match:
            values.append(match.group(1))
    return dedupe_strings(values)


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
    selected_metrics = understanding.get("selectedMetrics") or understanding.get("selected_metrics") or []
    if isinstance(selected_metrics, list):
        requests.extend(item for item in selected_metrics if isinstance(item, dict))
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


def metric_business_label_texts(metric: PlanningAssetEntry) -> List[str]:
    metadata = metric.metadata or {}
    return [
        metric.key,
        metric.title,
        metric.source_ref_id,
        str(metadata.get("businessName") or ""),
        str(metadata.get("displayName") or ""),
        str(metadata.get("naturalName") or ""),
        str(metadata.get("originalBusinessName") or ""),
        *metric.aliases,
        *[str(alias) for alias in metadata.get("aliases") or []],
    ]


def metric_ref_match_score(metric: PlanningAssetEntry, normalized_ref: str) -> int:
    if not normalized_ref:
        return 0
    metadata = metric.metadata or {}
    names = metric_business_label_texts(metric)
    columns = [str(column) for column in metadata.get("sourceColumns") or metadata.get("source_columns") or metric.columns]
    normalized_names = {normalize_for_match(item) for item in names if item}
    normalized_columns = {normalize_for_match(item) for item in columns if item}
    score = 0
    if normalized_ref in normalized_names:
        score += 40
    elif normalized_ref in normalized_columns:
        score += 30
    return score


def metric_phrase_match_score(metric: PlanningAssetEntry, phrase: str) -> int:
    terms = metric_phrase_terms(phrase)
    normalized_phrase = normalize_for_match(phrase)
    if not terms or not normalized_phrase:
        return 0
    metadata = metric.metadata or {}
    names = metric_business_label_texts(metric)
    columns = [str(column) for column in metadata.get("sourceColumns") or metadata.get("source_columns") or metric.columns]
    normalized_names = {normalize_for_match(item) for item in names if item}
    normalized_columns = {normalize_for_match(item) for item in columns if item}
    score = 0
    for label in normalized_names:
        if is_strong_label_text_match(label, normalized_phrase):
            score = max(score, 40 + min(len(label), 20))
    for term in terms:
        if not term:
            continue
        if term in normalized_names:
            score += 18
        elif term in normalized_columns:
            score += 14
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
    return bool(metric_direct_match_label(metric, phrase))


def metric_direct_match_label(metric: PlanningAssetEntry, phrase: str) -> str:
    normalized_phrase = normalize_for_match(phrase)
    if not normalized_phrase:
        return ""
    metadata = metric.metadata or {}
    labels = [
        metric.key,
        metric.title,
        str(metadata.get("businessName") or ""),
        str(metadata.get("displayName") or ""),
        str(metadata.get("naturalName") or ""),
        str(metadata.get("originalBusinessName") or ""),
        *metric.aliases,
        *[str(alias) for alias in metadata.get("aliases") or []],
    ]
    ranked = sorted({str(label or "").strip() for label in labels if str(label or "").strip()}, key=lambda item: len(normalize_for_match(item)), reverse=True)
    for label in ranked:
        normalized_label = normalize_for_match(label)
        if is_strong_label_text_match(normalized_label, normalized_phrase):
            return label
    return ""


def ambiguous_catalog_metric_identities(selected: List[Tuple[int, str, PlanningAssetEntry]]) -> Set[Tuple[str, str]]:
    by_label: Dict[str, List[PlanningAssetEntry]] = {}
    for _score, matched_label, metric in selected:
        label = normalize_for_match(matched_label)
        if label:
            by_label.setdefault(label, []).append(metric)
    ambiguous: Set[Tuple[str, str]] = set()
    for metrics in by_label.values():
        families = {
            (
                metric.topic,
                metric.table,
                str((metric.metadata or {}).get("canonicalMetricKey") or (metric.metadata or {}).get("aliasOf") or metric.key),
            )
            for metric in metrics
            if metric.table and metric.key
        }
        if len(families) <= 1:
            continue
        ambiguous.update((metric.table, metric.key) for metric in metrics if metric.table and metric.key)
    return ambiguous


def catalog_metric_evidence_payload(metric: PlanningAssetEntry, matched_label: str, score: int, ambiguous: bool = False) -> Dict[str, Any]:
    metadata = metric.metadata or {}
    return {
        "ownerTable": metric.table,
        "metricKey": metric.key,
        "semanticRefId": metric.source_ref_id,
        "docId": metric.source_ref_id,
        "title": metric.title,
        "fusionScore": float(score or 0),
        "sourceType": "SEMANTIC_METRIC",
        "businessName": str(metadata.get("businessName") or metric.title or ""),
        "canonicalMetricKey": str(metadata.get("canonicalMetricKey") or ""),
        "aliasOf": str(metadata.get("aliasOf") or ""),
        "metricLevel": str(metadata.get("metricLevel") or ""),
        "metricGrain": str(metadata.get("metricGrain") or metadata.get("grainHint") or ""),
        "metricIntent": str(metadata.get("metricIntent") or ""),
        "aggregationPolicy": str(metadata.get("aggregationPolicy") or ""),
        "applicableTimeGrain": str(metadata.get("applicableTimeGrain") or ""),
        "timeColumn": str(metadata.get("timeColumn") or metadata.get("time_column") or ""),
        "timeSemantics": metadata.get("timeSemantics") or {},
        "missingValuePolicy": str(metadata.get("missingValuePolicy") or ""),
        "zeroValueMeaning": str(metadata.get("zeroValueMeaning") or ""),
        "formula": str(metadata.get("formula") or metadata.get("metricFormula") or ""),
        "sourceColumns": metadata.get("sourceColumns") or metadata.get("source_columns") or metric.columns,
        "aliases": list(dict.fromkeys([*metric.aliases, *[str(alias) for alias in metadata.get("aliases") or []]])),
        "recallQuery": matched_label,
        "recallQueries": [matched_label] if matched_label else [],
        "matchedMetricLabel": matched_label,
        "metricResolutionType": "semantic_catalog_label",
        "metricResolutionReason": "matched_published_semantic_label:%s" % matched_label,
        "metricResolutionConfidence": 0.0 if ambiguous else 0.97,
        "metricResolutionAmbiguous": bool(ambiguous),
    }


def is_strong_label_text_match(normalized_label: str, normalized_phrase: str) -> bool:
    if not normalized_label or not normalized_phrase:
        return False
    if re.search(r"[a-z0-9]", normalized_label):
        return len(normalized_label) >= 3 and normalized_label in normalized_phrase
    if len(normalized_label) >= 4 and normalized_label in normalized_phrase:
        return True
    return normalized_label == normalized_phrase


def exact_semantic_metric_recall_items(question: str, scored: List[RecallItem]) -> List[RecallItem]:
    """Protect exact governed metric matches from lexical top-k displacement."""
    normalized_question = normalize_for_match(question)
    if not normalized_question:
        return []
    groups: Dict[str, List[RecallItem]] = {}
    for item in scored:
        if str(item.source_type or "").upper() != "SEMANTIC_METRIC":
            continue
        metadata = dict(item.metadata or {})
        labels = [
            str(metadata.get("businessName") or ""),
            str(metadata.get("metricKey") or ""),
            *[str(alias) for alias in metadata.get("aliases") or []],
        ]
        matched = [
            label
            for label in labels
            if is_strong_label_text_match(normalize_for_match(label), normalized_question)
        ]
        if not matched:
            continue
        matched_label = max(matched, key=lambda label: len(normalize_for_match(label)))
        metadata.update(
            {
                "matchedMetricLabel": matched_label,
                "metricResolutionType": "exact_semantic_label",
                "metricResolutionReason": "matched_published_semantic_label:%s" % matched_label,
                "metricResolutionConfidence": 0.97,
                "metricResolutionAmbiguous": False,
            }
        )
        protected = item.model_copy(update={"metadata": metadata})
        groups.setdefault(normalize_for_match(matched_label), []).append(protected)

    selected: List[RecallItem] = []
    for label, group in groups.items():
        if len(group) == 1:
            selected.extend(group)
            continue
        family_keys = {
            (
                str(item.topic or ""),
                str(item.table or ""),
                str((item.metadata or {}).get("canonicalMetricKey") or (item.metadata or {}).get("aliasOf") or (item.metadata or {}).get("metricKey") or ""),
            )
            for item in group
        }
        if len(family_keys) == 1:
            canonical_key = next(iter(family_keys))[2]
            owners = [
                item
                for item in group
                if str((item.metadata or {}).get("metricKey") or "") == canonical_key
                and not str((item.metadata or {}).get("aliasOf") or "")
            ]
            if len(owners) == 1:
                owner = owners[0]
                metadata = dict(owner.metadata or {})
                metadata["metricResolutionReason"] = "%s; canonical_family_owner=%s" % (
                    str(metadata.get("metricResolutionReason") or ""),
                    canonical_key,
                )
                selected.append(owner.model_copy(update={"metadata": metadata}))
                continue
        for item in group:
            metadata = dict(item.metadata or {})
            metadata["metricResolutionAmbiguous"] = True
            metadata["metricResolutionConfidence"] = 0.79
            metadata["metricResolutionReason"] = "%s; ambiguous_label=%s" % (
                str(metadata.get("metricResolutionReason") or ""),
                label,
            )
            selected.append(item.model_copy(update={"metadata": metadata}))
    return sorted(
        selected,
        key=lambda item: (
            bool((item.metadata or {}).get("metricResolutionAmbiguous")) is False,
            float((item.metadata or {}).get("metricResolutionConfidence") or 0.0),
            float(item.fusion_score or 0.0),
        ),
        reverse=True,
    )


def dedupe_recall_items(items: List[RecallItem]) -> List[RecallItem]:
    deduped: List[RecallItem] = []
    seen: Set[str] = set()
    for item in items:
        identity = str(item.doc_id or "")
        if not identity or identity in seen:
            continue
        seen.add(identity)
        deduped.append(item)
    return deduped


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
        topic = str(request.topic or "").strip()
        if not topic:
            return {"success": False, "message": "topic is required", "code": "TOPIC_UNDECLARED"}
        table = request.table_name
        if not table:
            return {"success": False, "message": "tableName is required"}
        pending_dir = self.settings.resolved_topic_path / topic / "pending" / table
        pending_dir.mkdir(parents=True, exist_ok=True)
        existing = self._existing_asset_context(topic, table)
        schema = self._load_schema(table, request)
        physical_metadata = self._load_physical_metadata(table)
        schema = self._enrich_schema_with_physical_metadata(schema, physical_metadata)
        sample_rows = self._load_sample_rows(table, request)
        profile = self._sample_profile(table, schema, sample_rows, request, physical_metadata)
        generated = self._generate_candidate_payload(topic, table, request, schema, sample_rows, profile, existing)
        builder_phases = {
            "schemaDiscovery": {
                "status": "completed",
                "artifact": str(pending_dir / "schema.json"),
                "columnCount": len(schema),
                "primaryKeyColumns": list(physical_metadata.get("primaryKeyColumns") or []),
                "partitionColumns": list(physical_metadata.get("partitionColumns") or []),
                "keyModel": str(physical_metadata.get("keyModel") or ""),
            },
            "sampleProfiling": {
                "status": "completed",
                "artifact": str(pending_dir / "sample_profile.json"),
                "sampleRowCount": len(sample_rows),
            },
            "semanticAnalysis": {
                "status": "completed",
                "mode": str(generated.get("generationMode") or "metadata_only"),
                "artifact": str(pending_dir / "asset.json"),
                "llmUsed": str(generated.get("generationMode") or "").lower() == "llm",
            },
            "tableUsageGovernance": {
                "status": "completed",
                "artifact": str(pending_dir / "asset.json"),
                "reviewRequired": True,
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
            "dataGrain": str(generated.get("dataGrain") or existing.get("dataGrain") or "UNDECLARED"),
            "timeColumn": str(generated.get("timeColumn") or existing.get("timeColumn") or profile.get("timeColumn") or ""),
            "merchantFilterColumn": str(generated.get("merchantFilterColumn") or existing.get("merchantFilterColumn") or ""),
            "entityLookupPolicy": normalize_entity_lookup_policy(
                generated.get("entityLookupPolicy")
                or existing.get("entityLookupPolicy")
                or {},
                inherited_time_column=str(
                    generated.get("timeColumn")
                    or existing.get("timeColumn")
                    or profile.get("timeColumn")
                    or ""
                ),
            ),
            "rowAccessPolicy": normalize_row_access_policy(
                generated.get("rowAccessPolicy")
                or existing.get("rowAccessPolicy")
            ),
            "resultAccessPolicies": normalize_result_access_policies(
                generated.get("resultAccessPolicies")
                or existing.get("resultAccessPolicies")
                or {}
            ),
            "manualNotes": request.manual_notes or str(existing.get("manualNotes") or ""),
            "businessKnowledge": request.business_knowledge or str(existing.get("businessKnowledge") or ""),
            "sampleSqls": request.sample_sqls or list(existing.get("sampleSqls") or []),
            "buildProfile": profile,
            "physicalMetadata": physical_metadata,
            "builderPhases": builder_phases,
            "generationMode": str(generated.get("generationMode") or "metadata_only"),
            "generatedAt": datetime.utcnow().isoformat() + "Z",
            "status": "PENDING_REVIEW",
        }
        if not asset_payload.get("entityLookupPolicy"):
            asset_payload.pop("entityLookupPolicy", None)
        semantic_columns = self._merge_generated_list(
            existing.get("semanticColumns"),
            generated.get("semanticColumns"),
            "semanticColumns",
            schema_columns=schema,
        )
        metrics = self._merge_generated_metrics(existing.get("metrics"), generated.get("metrics"), schema)
        terms = self._merge_generated_list(existing.get("terms"), generated.get("terms"), "terms")
        rules = self._merge_generated_list(existing.get("knowledgeRules"), generated.get("knowledgeRules"), "knowledgeRules")
        metadata_usage = self._metadata_table_usage_profile(semantic_columns, metrics)
        asset_payload["tableUsageProfile"] = normalize_table_usage_profile(
            {
                **metadata_usage,
                **(existing.get("tableUsageProfile") if isinstance(existing.get("tableUsageProfile"), dict) else {}),
                **(generated.get("tableUsageProfile") if isinstance(generated.get("tableUsageProfile"), dict) else {}),
                **(request.table_usage_overrides or {}),
            },
            table,
        )
        asset_payload["schemaColumns"] = schema if isinstance(schema, list) else []
        asset_payload["semanticColumns"] = semantic_columns
        asset_payload["metrics"] = metrics
        asset_payload["terms"] = terms
        asset_payload["knowledgeRules"] = rules
        production_report = topic_asset_production_report(
            topic=topic,
            table=table,
            schema=schema,
            sample_rows=sample_rows,
            profile=profile,
            physical_metadata=physical_metadata,
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
        write_json(pending_dir / "sample_rows.json", sample_rows)
        write_json(pending_dir / "sample_profile.json", profile)
        write_json(pending_dir / "asset.json", asset_payload)
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
        topic = str(request.topic or "").strip()
        if not topic:
            return {"success": False, "message": "topic is required", "code": "TOPIC_UNDECLARED"}
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
        for field, file_name in ACTIVE_SEMANTIC_SIDECAR_FIELDS.items():
            sidecar = read_json(directory / file_name)
            if sidecar:
                payload[field] = sidecar
            else:
                payload.setdefault(field, [])
        return payload

    def _load_physical_metadata(self, table: str) -> Dict[str, Any]:
        providers = ["table_physical_metadata", "show_create_table"]
        for name in providers:
            provider = getattr(self.doris_repository, name, None)
            if not callable(provider):
                continue
            try:
                payload = provider(table)
            except Exception:
                payload = {}
            metadata = normalize_physical_table_metadata(payload)
            if metadata:
                return metadata
        return {}

    def _enrich_schema_with_physical_metadata(
        self,
        schema: List[Dict[str, Any]],
        physical_metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not schema or not physical_metadata:
            return schema
        primary_keys = {str(item).lower() for item in physical_metadata.get("primaryKeyColumns") or []}
        partition_columns = {str(item).lower() for item in physical_metadata.get("partitionColumns") or []}
        bucket_columns = {str(item).lower() for item in physical_metadata.get("bucketColumns") or []}
        enriched: List[Dict[str, Any]] = []
        for column in schema:
            next_column = dict(column)
            name = str(next_column.get("columnName") or "").lower()
            if name in primary_keys:
                next_column["isPrimaryKey"] = True
                next_column["keyType"] = next_column.get("keyType") or str(physical_metadata.get("keyModel") or "KEY")
            if name in partition_columns:
                next_column["isPartitionColumn"] = True
            if name in bucket_columns:
                next_column["isBucketColumn"] = True
            enriched.append(next_column)
        return enriched

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

    def _sample_profile(
        self,
        table: str,
        schema: List[Dict[str, Any]],
        rows: List[Dict[str, Any]],
        request: TopicBuildRequest,
        physical_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        physical_metadata = physical_metadata or {}
        columns = [str(item.get("columnName") or "") for item in schema if str(item.get("columnName") or "")]
        enum_limit = max(1, int(request.enum_value_limit or 20))
        null_rates: Dict[str, float] = {}
        sample_values: Dict[str, List[Any]] = {}
        enum_candidates: Dict[str, List[Any]] = {}
        enum_candidate_profiles: Dict[str, Dict[str, Any]] = {}
        global_enum_profiles = self._load_global_enum_profiles(table, columns, request, enum_limit)
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
            global_profile = global_enum_profiles.get(column) if isinstance(global_enum_profiles, dict) else None
            global_values = []
            if isinstance(global_profile, dict):
                global_values = list(global_profile.get("values") or [])[:enum_limit]
            elif isinstance(global_profile, list):
                global_values = list(global_profile)[:enum_limit]
            candidate_values = global_values or unique_values
            non_null_count = max(0, total - missing)
            if request.enum_discovery_enabled and 0 < len(candidate_values) <= enum_limit and (
                bool(global_values) or self._enum_candidate(column, candidate_values, non_null_count)
            ):
                enum_candidates[column] = candidate_values
                distinct_ratio = round(len(unique_values) / non_null_count, 4) if non_null_count else 0.0
                enum_candidate_profiles[column] = {
                    "discoverySource": "global_profile" if global_values else "sample",
                    "distinctCount": len(candidate_values),
                    "sampleDistinctRatio": distinct_ratio,
                    "sampleRowCount": non_null_count,
                    "exhaustive": bool(isinstance(global_profile, dict) and global_profile.get("exhaustive")),
                    "coverage": float(global_profile.get("coverage") or 0.0) if isinstance(global_profile, dict) else 0.0,
                    "reviewStatus": str(global_profile.get("reviewStatus") or "UNREVIEWED") if isinstance(global_profile, dict) else "UNREVIEWED",
                    "confidence": round(0.9 if global_values else 0.5, 2),
                }
        partition_candidates = dedupe_strings(
            [str(item) for item in physical_metadata.get("partitionColumns") or [] if str(item) in columns]
            + [str(item.get("columnName") or "") for item in schema if bool(item.get("isPartitionColumn"))]
        )
        time_column = partition_candidates[0] if len(partition_candidates) == 1 else ""
        primary_key_columns = dedupe_strings(
            [str(item) for item in physical_metadata.get("primaryKeyColumns") or [] if str(item) in columns]
            + [str(item.get("columnName") or "") for item in schema if bool(item.get("isPrimaryKey"))]
        )
        return {
            "rowCount": len(rows),
            "nullRates": null_rates,
            "sampleValues": sample_values,
            "enumCandidates": enum_candidates,
            "enumCandidateProfiles": enum_candidate_profiles,
            "partitionColumns": partition_candidates,
            "primaryKeyColumns": primary_key_columns,
            "bucketColumns": [str(item) for item in physical_metadata.get("bucketColumns") or [] if str(item) in columns],
            "keyModel": str(physical_metadata.get("keyModel") or ""),
            "timeColumn": time_column,
            "merchantFilterColumn": "",
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
        heuristic["generationMode"] = "metadata_only"
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
            "physicalTableMetadata": {
                "keyModel": profile.get("keyModel") or "",
                "primaryKeyColumns": profile.get("primaryKeyColumns") or [],
                "partitionColumns": profile.get("partitionColumns") or [],
                "bucketColumns": profile.get("bucketColumns") or [],
            },
            "manualNotes": request.manual_notes,
            "businessKnowledge": request.business_knowledge,
            "sampleSqls": request.sample_sqls[:8],
            "existingAsset": {
                "tableComment": existing.get("tableComment"),
                "dataGrain": existing.get("dataGrain"),
                "timeColumn": existing.get("timeColumn"),
                "merchantFilterColumn": existing.get("merchantFilterColumn"),
                "entityLookupPolicy": existing.get("entityLookupPolicy") or {},
                "rowAccessPolicy": existing.get("rowAccessPolicy") or {},
                "resultAccessPolicies": existing.get("resultAccessPolicies") or {},
                "semanticColumns": existing.get("semanticColumns") or [],
                "metrics": existing.get("metrics") or [],
                "terms": existing.get("terms") or [],
                "knowledgeRules": existing.get("knowledgeRules") or [],
                "tableUsageProfile": existing.get("tableUsageProfile") or {},
            },
            "heuristicDraft": heuristic,
        }
        system_prompt = (
            "你是资深数据语义建模助手。请基于表 schema、采样数据、业务备注和历史 SQL，"
            "生成待审核的语义层候选资产。输出必须保守、结构化，不要编造不存在的字段，"
            "不要生成跨表 join 关系。字段角色只允许 KEY、TIME、DIMENSION、ATTRIBUTE。"
            "指标必须引用真实存在的 sourceColumns；派生指标可以引用其他 metricKey。"
            "每个指标必须声明 aggregationPolicy、metricGrain、applicableTimeGrain 和 timeSemantics；"
            "timeSemantics 必须分别声明 asOfPolicy、missingDataPolicy 和 zeroValuePolicy。"
            "每个指标还必须声明 calculationSemantics，说明 nativeTimeGrain/nativeWindowDays、"
            "windowPolicy、timeRollupPolicy、允许或禁止的聚合、所需组件/权重，以及不适用时的"
            "alternativeCapability。比率、平均、快照、去重计数和固定窗口必须根据业务知识声明，"
            "不得从指标名或字段名猜测。可派生度量的字段可声明 allowedAggregations 和 derivableMeasures。"
            "不得把数据缺失默认解释为业务为 0；没有可核验证据时使用 undeclared，不要猜测。"
            "对可作为实体过滤条件的字段，应基于业务证据声明 canonicalEntityRef、comparisonPolicy、"
            "filterOperators、isUniqueEntityKey 与字段级 lookupTimePolicy；唯一实体的无时间查询能力"
            "必须由 lookupTimePolicy 显式声明，不能从字段名、样例值或 KEY 角色推断。"
            "如果字段疑似手机号、邮箱、身份证、地址等敏感信息，请补充 visibilityPolicy 和 maskingPolicy；"
            "如果表有明确租户过滤列，请补充 rowAccessPolicy。"
            "聚合指标结果、时间或其他维度能否返回必须通过 resultAccessPolicies 按语义角色显式声明；"
            "它只约束结果角色，不能替代原始字段访问策略，未声明时保持不可见。"
            "同时生成 tableUsageProfile，说明该表是否允许 Agent 查询、业务分层、权威度、"
            "支持的分析意图/指标/维度，以及适合和不适合回答的问题。"
            "不得根据表名或字段名约定猜测业务分层、租户列、时间列、粒度或可查询性；"
            "没有明确证据时填写 UNDECLARED/false，候选必须经过人工审核。"
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
        # Schema inspection may propose fields and dimensions, but it must not
        # invent business metrics. Keep only previously governed definitions;
        # new metrics must come from business knowledge/LLM draft and review.
        metrics = [dict(item) for item in existing.get("metrics") or [] if isinstance(item, dict)]
        terms = self._heuristic_terms(metrics, semantic_columns, request)
        rules = self._heuristic_rules(topic, profile, request)
        return {
            "tableComment": str(existing.get("tableComment") or request.manual_notes or table),
            "dataGrain": str(existing.get("dataGrain") or "UNDECLARED"),
            "timeColumn": str(existing.get("timeColumn") or profile.get("timeColumn") or ""),
            "merchantFilterColumn": str(existing.get("merchantFilterColumn") or ""),
            "entityLookupPolicy": normalize_entity_lookup_policy(
                existing.get("entityLookupPolicy") or {},
                inherited_time_column=str(
                    existing.get("timeColumn") or profile.get("timeColumn") or ""
                ),
            ),
            "rowAccessPolicy": normalize_row_access_policy(existing.get("rowAccessPolicy") or {}),
            "resultAccessPolicies": normalize_result_access_policies(existing.get("resultAccessPolicies") or {}),
            "semanticColumns": semantic_columns,
            "metrics": metrics,
            "terms": terms,
            "knowledgeRules": rules,
            "tableUsageProfile": self._metadata_table_usage_profile(semantic_columns, metrics),
        }

    def _metadata_table_usage_profile(
        self,
        semantic_columns: List[Dict[str, Any]],
        metrics: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        metric_keys = [str(item.get("metricKey") or "") for item in metrics if isinstance(item, dict) and item.get("metricKey")]
        supported_metrics = dedupe_strings(metric_keys)
        dimensions = [
            str(item.get("columnName") or "")
            for item in semantic_columns
            if isinstance(item, dict)
            and str(item.get("semanticRole") or item.get("role") or "").upper() == "DIMENSION"
            and item.get("columnName")
        ]
        return normalize_table_usage_profile(
            {
                "contractStatus": "UNDECLARED",
                "businessLayer": "UNDECLARED",
                "queryableByAgent": False,
                "authorityLevel": 0,
                "topicRole": "UNDECLARED",
                "supportedMetrics": supported_metrics,
                "supportedDimensions": dimensions,
                "exclusionReason": "TABLE_USAGE_UNDECLARED",
            }
        )

    def _heuristic_semantic_column(self, topic: str, column: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
        name = str(column.get("columnName") or "")
        comment = str(column.get("comment") or column.get("Comment") or name)
        role = str(column.get("semanticRole") or column.get("role") or "UNDECLARED").upper()
        if bool(column.get("isPrimaryKey")):
            role = "KEY"
        elif bool(column.get("isPartitionColumn")):
            role = "TIME"
        aliases = dedupe_strings([name, comment])
        enum_values = [str(item) for item in (profile.get("enumCandidates", {}).get(name) or [])[:20]]
        enum_metadata = dict((profile.get("enumCandidateProfiles", {}) or {}).get(name) or {})
        sample_values = [str(item) for item in (profile.get("sampleValues", {}).get(name) or [])[:8]]
        evidence = ["schema comment=%s" % comment]
        if column.get("isPrimaryKey"):
            evidence.append("physical key column")
        if column.get("isPartitionColumn"):
            evidence.append("physical partition column")
        if column.get("isBucketColumn"):
            evidence.append("physical bucket/distribution column")
        if sample_values:
            evidence.append("samples=[%s]" % ", ".join(sample_values))
        visibility_policy = normalize_visibility_policy(column.get("visibilityPolicy") or {"level": "hidden", "reason": "UNDECLARED_PENDING_REVIEW"})
        masking_policy = normalize_masking_policy(column.get("maskingPolicy") or {"strategy": "full", "reason": "UNDECLARED_PENDING_REVIEW"})
        display_policy = normalize_column_display_policy(column)
        return {
            "columnName": name,
            "businessName": comment,
            "role": role,
            "semanticRole": role,
            "description": comment,
            "aliases": aliases,
            **display_policy,
            "enumValues": enum_values,
            "enumMetadata": enum_metadata,
            "sampleValues": sample_values,
            "confidence": 1.0 if role in {"KEY", "TIME"} else 0.0,
            "evidence": "; ".join(evidence),
            "visibilityPolicy": visibility_policy,
            "maskingPolicy": masking_policy,
        }

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
        primary_key_columns = [str(item) for item in profile.get("primaryKeyColumns") or [] if str(item)]
        partition_columns = [str(item) for item in profile.get("partitionColumns") or [] if str(item)]
        if primary_key_columns:
            rules.append(
                {
                    "ruleId": "primary_key_grain_rule",
                    "title": "主键和数据粒度",
                    "description": "该表物理 Key 候选为 %s；明细去重、记录数和跨表关联前应优先核对这些字段。" % "、".join(primary_key_columns[:8]),
                    "aliases": ["主键", "数据粒度", "去重口径"],
                    "appliesToColumns": primary_key_columns[:12],
                    "cautions": ["Key 字段只表示物理/语义粒度候选，最终指标去重口径仍以 metrics 定义为准"],
                }
            )
        if partition_columns:
            rules.append(
                {
                    "ruleId": "partition_pruning_rule",
                    "title": "分区字段过滤建议",
                    "description": "该表分区/时间候选字段为 %s；有时间窗口的问题应优先下推这些字段以减少扫描范围。" % "、".join(partition_columns[:8]),
                    "aliases": ["分区字段", "分区裁剪", "时间窗口"],
                    "appliesToColumns": partition_columns[:12],
                }
            )
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
            if refs and any((ref not in valid) and not external_metric_dependency(metric, ref) for ref in refs):
                continue
            result.append(metric)
        return dedupe_semantic_items(result, "metrics")

    def _normalize_schema_column(self, item: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(item, dict):
            return {}
        primary_flag = item.get("isPrimaryKey")
        partition_flag = item.get("isPartitionColumn")
        bucket_flag = item.get("isBucketColumn")
        normalized = {
            "columnName": str(item.get("columnName") or item.get("Field") or item.get("name") or ""),
            "dataType": str(item.get("dataType") or item.get("Type") or item.get("type") or ""),
            "comment": str(item.get("comment") or item.get("Comment") or ""),
            "nullable": self._normalize_nullable(item),
            "keyType": str(item.get("keyType") or item.get("Key") or ""),
            "isPrimaryKey": bool(primary_flag) if primary_flag is not None else False,
            "isPartitionColumn": bool(partition_flag) if partition_flag is not None else False,
            "isBucketColumn": bool(bucket_flag) if bucket_flag is not None else False,
        }
        for key in (
            "semanticRole",
            "role",
            "visibilityPolicy",
            "maskingPolicy",
            "defaultVisible",
            "displayPriority",
            "displayScenarios",
            "enumCandidate",
        ):
            if key in item:
                normalized[key] = item[key]
        return normalized

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

    def _load_global_enum_profiles(
        self,
        table: str,
        columns: List[str],
        request: TopicBuildRequest,
        enum_limit: int,
    ) -> Dict[str, Any]:
        provider = getattr(self.doris_repository, "profile_enum_candidates", None)
        if not callable(provider) or not request.enum_discovery_enabled:
            return {}
        candidate_columns = list(columns[:24])
        if not candidate_columns:
            return {}
        try:
            payload = provider(table, request.merchant_id or self.settings.merchant_id, candidate_columns, enum_limit)
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _explicit_enum_column(self, column: str) -> bool:
        del column
        return False

    def _enum_candidate(self, column: str, values: List[Any], observed_count: int = 0) -> bool:
        del column
        if not values or len(values) > 8 or observed_count < 4:
            return False
        return (len(values) / max(observed_count, 1)) <= 0.5

    def _infer_data_grain(self, schema: List[Dict[str, Any]]) -> str:
        del schema
        return "UNDECLARED"

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
    result_access_policy_schema = {
        "type": "object",
        "properties": {
            "visibilityPolicy": visibility_policy_schema,
            "maskingPolicy": masking_policy_schema,
            "defaultVisible": {"type": "boolean"},
            "displayPriority": {"type": "integer"},
            "displayScenarios": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["visibilityPolicy", "maskingPolicy"],
        "additionalProperties": False,
    }
    alternative_capability_schema = {
        "type": "object",
        "properties": {
            "operation": {"type": "string"},
            "entityRole": {"type": "string"},
            "requiredFieldRole": {"type": "string"},
            "requiredTableGrain": {"type": "string"},
            "timeRollupPolicy": {
                "type": "string",
                "enum": sorted(SUPPORTED_CALCULATION_TIME_ROLLUP_POLICIES),
            },
            "windowPolicy": {
                "type": "string",
                "enum": sorted(SUPPORTED_CALCULATION_WINDOW_POLICIES),
            },
            "requiredComponents": {"type": "array", "items": {"type": "string"}},
            "requiredWeightRef": {"type": "string"},
        },
        "additionalProperties": False,
    }
    derivable_measure_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": sorted(SUPPORTED_CALCULATION_AGGREGATIONS),
            },
            "resultSemanticType": {"type": "string"},
            "unit": {"type": "string"},
        },
        "required": ["operation"],
        "additionalProperties": False,
    }
    calculation_semantics_schema = {
        "type": "object",
        "properties": {
            "semanticValueType": {"type": "string"},
            "semanticEntityRole": {"type": "string"},
            "nativeTimeGrain": {"type": "string"},
            "nativeWindowDays": {"type": "integer", "minimum": 0},
            "windowPolicy": {
                "type": "string",
                "enum": sorted(SUPPORTED_CALCULATION_WINDOW_POLICIES),
            },
            "timeRollupPolicy": {
                "type": "string",
                "enum": sorted(SUPPORTED_CALCULATION_TIME_ROLLUP_POLICIES),
            },
            "derivedMeasureTimeRollupPolicy": {
                "type": "string",
                "enum": sorted(SUPPORTED_CALCULATION_TIME_ROLLUP_POLICIES),
            },
            "nativeGrainAnalysisModes": {"type": "array", "items": {"type": "string"}},
            "allowedAggregations": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(SUPPORTED_CALCULATION_AGGREGATIONS)},
            },
            "forbiddenAggregations": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(SUPPORTED_CALCULATION_AGGREGATIONS)},
            },
            "requiredComponents": {"type": "array", "items": {"type": "string"}},
            "requiredWeightRef": {"type": "string"},
            "allowedTableGrains": {"type": "array", "items": {"type": "string"}},
            "enforceAggregationAtNativeGrain": {"type": "boolean"},
            "violationMessage": {"type": "string"},
            "resolution": {
                "type": "string",
                "enum": ["RESELECT_TABLE", "RESELECT_METRIC", "ASK_HUMAN"],
            },
            "alternativeCapability": alternative_capability_schema,
            "derivableMeasures": {"type": "array", "items": derivable_measure_schema},
        },
        "additionalProperties": False,
    }
    entity_lookup_policy_schema = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": sorted(SUPPORTED_ENTITY_LOOKUP_POLICY_MODES),
            },
            "timeColumn": {"type": "string"},
            "timeRequired": {"type": "boolean"},
            "defaultDays": {"type": "integer", "minimum": 1},
        },
        "required": ["mode"],
        "additionalProperties": False,
    }
    semantic_column_schema = {
        "type": "object",
        "properties": {
            "columnName": {"type": "string"},
            "businessName": {"type": "string"},
            "role": {"type": "string", "enum": ["KEY", "TIME", "DIMENSION", "ATTRIBUTE"]},
            "entityRole": {"type": "string"},
            "canonicalEntityRef": {"type": "string"},
            "comparisonPolicy": {
                "type": "string",
                "enum": sorted(SUPPORTED_ENTITY_COMPARISON_POLICIES),
            },
            "isUniqueEntityKey": {"type": "boolean"},
            "filterOperators": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": sorted(SUPPORTED_ENTITY_FILTER_OPERATORS),
                },
            },
            "lookupTimePolicy": entity_lookup_policy_schema,
            "description": {"type": "string"},
            "aliases": {"type": "array", "items": {"type": "string"}},
            "enumValues": {"type": "array", "items": {"type": "string"}},
            "enumMetadata": {
                "type": "object",
                "properties": {
                    "discoverySource": {"type": "string"},
                    "distinctCount": {"type": "integer"},
                    "sampleDistinctRatio": {"type": "number"},
                    "sampleRowCount": {"type": "integer"},
                    "exhaustive": {"type": "boolean"},
                    "coverage": {"type": "number"},
                    "reviewStatus": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "additionalProperties": False,
            },
            "sampleValues": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
            "evidence": {"type": "string"},
            "defaultVisible": {"type": "boolean"},
            "displayPriority": {"type": "integer"},
            "displayScenarios": {"type": "array", "items": {"type": "string"}},
            "visibilityPolicy": visibility_policy_schema,
            "maskingPolicy": masking_policy_schema,
            "calculationSemantics": calculation_semantics_schema,
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
            "currency": {"type": "string"},
            "aggregation": {"type": "string"},
            "aggregationPolicy": {
                "type": "string",
                "enum": sorted(SUPPORTED_METRIC_AGGREGATION_POLICIES),
            },
            "metricGrain": {"type": "string"},
            "applicableTimeGrain": {"type": "string"},
            "timeColumn": {"type": "string"},
            "timeSemantics": {
                "type": "object",
                "properties": {
                    "selectionPolicy": {
                        "type": "string",
                        "enum": sorted(SUPPORTED_METRIC_TIME_SELECTION_POLICIES),
                    },
                    "asOfPolicy": {
                        "type": "string",
                        "enum": sorted(SUPPORTED_METRIC_AS_OF_POLICIES),
                    },
                    "missingDataPolicy": {
                        "type": "string",
                        "enum": sorted(SUPPORTED_METRIC_MISSING_DATA_POLICIES),
                    },
                    "zeroValuePolicy": {
                        "type": "string",
                        "enum": sorted(SUPPORTED_METRIC_ZERO_VALUE_POLICIES),
                    },
                },
                "required": ["selectionPolicy", "asOfPolicy", "missingDataPolicy", "zeroValuePolicy"],
                "additionalProperties": False,
            },
            "calculationSemantics": calculation_semantics_schema,
            "requiredFilters": {"type": "array", "items": {"type": "string"}},
            "conflictsWith": {"type": "array", "items": {"type": "string"}},
            "clarificationQuestion": {"type": "string"},
            "description": {"type": "string"},
            "sourceColumns": {"type": "array", "items": {"type": "string"}},
            "metricDependencies": {"type": "array", "items": {"type": "string"}},
            "requiresMetrics": {"type": "array", "items": {"type": "string"}},
            "requiresTables": {"type": "array", "items": {"type": "string"}},
            "externalMetricRefs": {"type": "array", "items": {"type": "string"}},
            "aliases": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
            "evidence": {"type": "string"},
        },
        "required": [
            "metricKey",
            "businessName",
            "formula",
            "sourceColumns",
            "aggregationPolicy",
            "metricGrain",
            "applicableTimeGrain",
            "timeSemantics",
            "calculationSemantics",
        ],
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
    table_usage_profile_schema = {
        "type": "object",
        "properties": {
            "businessLayer": {"type": "string", "enum": ["ODS", "DWD", "DWM", "ADS", "DIM", "TMP", "STG", "UNKNOWN"]},
            "queryableByAgent": {"type": "boolean"},
            "authorityLevel": {"type": "integer"},
            "topicRole": {"type": "string", "enum": ["ANCHOR", "DETAIL", "DIMENSION", "BRIDGE", "PROFILE", "AUXILIARY", "UNKNOWN"]},
            "defaultForIntents": {"type": "array", "items": {"type": "string"}},
            "supportedIntents": {"type": "array", "items": {"type": "string"}},
            "supportedMetrics": {"type": "array", "items": {"type": "string"}},
            "supportedDimensions": {"type": "array", "items": {"type": "string"}},
            "recommendedFor": {"type": "array", "items": {"type": "string"}},
            "notRecommendedFor": {"type": "array", "items": {"type": "string"}},
            "exclusionReason": {"type": "string"},
        },
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
                "entityLookupPolicy": entity_lookup_policy_schema,
                "rowAccessPolicy": row_access_policy_schema,
                "resultAccessPolicies": {
                    "type": "object",
                    "additionalProperties": result_access_policy_schema,
                },
                "tableUsageProfile": table_usage_profile_schema,
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
    return "hidden"


def normalize_masking_strategy(strategy: str) -> str:
    text = str(strategy or "").strip().lower()
    if text in {"none", "partial", "full", "hash"}:
        return text
    return "full"


def normalize_visibility_policy(policy: Any) -> Dict[str, Any]:
    if not isinstance(policy, dict):
        return {"level": "hidden", "allowedRoles": [], "reason": "UNDECLARED"}
    return {
        "level": normalize_visibility_level(str(policy.get("level") or "")),
        "allowedRoles": dedupe_strings([str(item) for item in policy.get("allowedRoles") or []]),
        "reason": str(policy.get("reason") or ""),
    }


def normalize_masking_policy(policy: Any) -> Dict[str, Any]:
    if not isinstance(policy, dict):
        return {"strategy": "full", "reason": "UNDECLARED"}
    return {
        "strategy": normalize_masking_strategy(str(policy.get("strategy") or "")),
        "reason": str(policy.get("reason") or ""),
    }


def normalize_result_access_policies(policies: Any) -> Dict[str, Dict[str, Any]]:
    """Normalize explicitly governed result policies without inventing roles."""

    if not isinstance(policies, dict):
        return {}
    normalized: Dict[str, Dict[str, Any]] = {}
    for raw_role, raw_policy in policies.items():
        role = str(raw_role or "").strip().upper()
        if not role or not isinstance(raw_policy, dict):
            continue
        normalized[role] = {
            "visibilityPolicy": normalize_visibility_policy(raw_policy.get("visibilityPolicy") or {}),
            "maskingPolicy": normalize_masking_policy(raw_policy.get("maskingPolicy") or {}),
            **normalize_column_display_policy(raw_policy),
        }
    return normalized


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
        "scopeType": str(policy.get("scopeType") or "UNDECLARED"),
        "filterColumn": filter_column,
        "operator": str(policy.get("operator") or "UNDECLARED"),
        "valueSource": str(policy.get("valueSource") or "UNDECLARED"),
        "required": bool(policy.get("required", True)),
        "reason": str(policy.get("reason") or ""),
    }


def default_row_access_policy(filter_column: str) -> Dict[str, Any]:
    del filter_column
    return {}


def sensitive_column_policies(column_name: str, business_name: str = "") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    del column_name, business_name
    return (
        {"level": "hidden", "allowedRoles": [], "reason": "UNDECLARED_PENDING_REVIEW"},
        {"strategy": "full", "reason": "UNDECLARED_PENDING_REVIEW"},
    )


def default_column_display_policy(column_name: str, role: str, visibility_policy: Dict[str, Any]) -> Dict[str, Any]:
    del column_name, role, visibility_policy
    return {"defaultVisible": False, "displayPriority": 1000, "displayScenarios": []}


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
        pending_source_hash = semantic_candidate_source_hash(pending_dir)
        version = version.model_copy(
            update={
                "semantic_version": "semantic-%s" % (pending_source_hash[:12] or "unknown"),
                "source_hash": pending_source_hash,
                "published_at": "",
            }
        )
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
        conflict_detection = combine_semantic_conflict_detection(
            semantic_conflict_detection(asset),
            semantic_catalog_conflict_detection(self.topic_assets, topic, table, asset),
        )
        evaluation_gate = semantic_release_evaluation_gate(asset, validation, drift, conflict_detection)
        release_gate = combine_release_gates(release_gate, evaluation_gate)
        owner = semantic_asset_owner(asset)
        payload = {
            "success": True,
            "publishable": release_gate["publishable"],
            "status": "PREFLIGHT_PASSED" if release_gate["publishable"] else "PREFLIGHT_FAILED",
            "topic": topic,
            "tableName": table,
            "pendingSourceHash": pending_source_hash,
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
        conflict_detection = combine_semantic_conflict_detection(
            semantic_conflict_detection(asset),
            semantic_catalog_conflict_detection(self.topic_assets, topic, table, asset),
        )
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
        self.topic_assets._semantic_source_hash_cache.clear()
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
        for field, file_name in ACTIVE_SEMANTIC_SIDECAR_FIELDS.items():
            sidecar = read_json(directory / file_name)
            if sidecar:
                payload[field] = sidecar
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


def validate_calculation_semantics(
    raw: Any,
    owner_kind: str,
    owner_key: str,
    source_refs: Sequence[str] | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    identity = {"ownerKind": owner_kind, "ownerKey": owner_key}
    if raw is None:
        warnings.append({"code": "CALCULATION_SEMANTICS_UNDECLARED", **identity})
        return {"errors": errors, "warnings": warnings}
    if not isinstance(raw, dict):
        errors.append({"code": "CALCULATION_SEMANTICS_INVALID", **identity})
        return {"errors": errors, "warnings": warnings}

    rollup_policy = str(raw.get("timeRollupPolicy") or "").strip().upper()
    derived_rollup_policy = str(raw.get("derivedMeasureTimeRollupPolicy") or "").strip().upper()
    window_policy = str(raw.get("windowPolicy") or "").strip().upper()
    if rollup_policy and rollup_policy not in SUPPORTED_CALCULATION_TIME_ROLLUP_POLICIES:
        errors.append({"code": "CALCULATION_TIME_ROLLUP_POLICY_INVALID", "value": rollup_policy, **identity})
    if derived_rollup_policy and derived_rollup_policy not in SUPPORTED_CALCULATION_TIME_ROLLUP_POLICIES:
        errors.append({"code": "DERIVED_MEASURE_ROLLUP_POLICY_INVALID", "value": derived_rollup_policy, **identity})
    if window_policy and window_policy not in SUPPORTED_CALCULATION_WINDOW_POLICIES:
        errors.append({"code": "CALCULATION_WINDOW_POLICY_INVALID", "value": window_policy, **identity})

    try:
        native_window_days = int(raw.get("nativeWindowDays") or 0)
    except (TypeError, ValueError):
        native_window_days = -1
    if native_window_days < 0:
        errors.append({"code": "CALCULATION_NATIVE_WINDOW_INVALID", **identity})
    if window_policy in {"EXACT_ONLY", "NATIVE_GRAIN_ONLY"} and native_window_days <= 0:
        errors.append({"code": "CALCULATION_NATIVE_WINDOW_REQUIRED", **identity})
    if rollup_policy == "NOT_COMPOSABLE" and native_window_days <= 0:
        errors.append({"code": "NON_COMPOSABLE_NATIVE_WINDOW_REQUIRED", **identity})
    if rollup_policy == "NOT_COMPOSABLE" and not isinstance(raw.get("alternativeCapability"), dict):
        warnings.append({"code": "NON_COMPOSABLE_ALTERNATIVE_UNDECLARED", **identity})

    allowed = {
        str(item or "").strip().upper()
        for item in raw.get("allowedAggregations") or []
        if str(item or "").strip()
    }
    forbidden = {
        str(item or "").strip().upper()
        for item in raw.get("forbiddenAggregations") or []
        if str(item or "").strip()
    }
    invalid_aggregations = sorted(
        (allowed | forbidden) - set(SUPPORTED_CALCULATION_AGGREGATIONS)
    )
    if invalid_aggregations:
        errors.append(
            {
                "code": "CALCULATION_AGGREGATION_INVALID",
                "aggregations": invalid_aggregations,
                **identity,
            }
        )
    overlap = sorted(allowed & forbidden)
    if overlap:
        errors.append(
            {
                "code": "CALCULATION_AGGREGATION_CONFLICT",
                "aggregations": overlap,
                **identity,
            }
        )

    required_components = [
        str(item or "").strip()
        for item in raw.get("requiredComponents") or []
        if str(item or "").strip()
    ]
    required_weight = str(raw.get("requiredWeightRef") or "").strip()
    if rollup_policy == "WEIGHTED_AVERAGE" and not required_weight:
        errors.append({"code": "WEIGHTED_AVERAGE_WEIGHT_UNDECLARED", **identity})
    if rollup_policy in {"WEIGHTED_AVERAGE", "RATIO_OF_SUMS"} and len(required_components) < 2:
        errors.append({"code": "COMPOSITE_ROLLUP_COMPONENTS_UNDECLARED", **identity})
    available_refs = {str(item) for item in source_refs or [] if str(item)}
    missing_components = sorted(set(required_components) - available_refs)
    if available_refs and missing_components:
        errors.append(
            {
                "code": "CALCULATION_COMPONENT_REF_MISSING",
                "components": missing_components,
                **identity,
            }
        )
    if required_weight and available_refs and required_weight not in available_refs:
        errors.append(
            {
                "code": "CALCULATION_WEIGHT_REF_MISSING",
                "weightRef": required_weight,
                **identity,
            }
        )

    for item in raw.get("derivableMeasures") or []:
        if not isinstance(item, dict):
            errors.append({"code": "DERIVABLE_MEASURE_INVALID", **identity})
            continue
        operation = str(item.get("operation") or "").strip().upper()
        if operation not in SUPPORTED_CALCULATION_AGGREGATIONS:
            errors.append(
                {
                    "code": "DERIVABLE_MEASURE_OPERATION_INVALID",
                    "operation": operation or "UNDECLARED",
                    **identity,
                }
            )
    return {"errors": errors, "warnings": warnings}


def validate_semantic_asset(asset: Dict[str, Any], relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
    table = str(asset.get("tableName") or "")
    schema = schema_columns(asset)
    columns = column_name_set(schema)
    metrics = asset.get("metrics") if isinstance(asset.get("metrics"), list) else []
    semantic_columns = asset.get("semanticColumns") if isinstance(asset.get("semanticColumns"), list) else []
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    row_access_policy = normalize_row_access_policy(asset.get("rowAccessPolicy") or {})
    result_access_policies = normalize_result_access_policies(asset.get("resultAccessPolicies") or {})
    table_usage = normalize_table_usage_profile(asset.get("tableUsageProfile") or {}, table)
    entity_fields = [
        field
        for field in semantic_columns
        if isinstance(field, dict)
        and str(field.get("role") or field.get("semanticRole") or "").strip().upper()
        in ENTITY_SEMANTIC_ROLES
    ]
    table_time_column = str(asset.get("timeColumn") or "").strip()
    raw_entity_lookup_policy = asset.get("entityLookupPolicy")
    if entity_fields and table_time_column and raw_entity_lookup_policy is None:
        errors.append({"code": "ENTITY_LOOKUP_POLICY_UNDECLARED", "column": table_time_column})
    if raw_entity_lookup_policy is not None:
        if not isinstance(raw_entity_lookup_policy, dict):
            errors.append({"code": "ENTITY_LOOKUP_POLICY_INVALID"})
        else:
            lookup_mode = str(raw_entity_lookup_policy.get("mode") or "").strip().lower()
            lookup_time_column = str(raw_entity_lookup_policy.get("timeColumn") or "").strip()
            if lookup_mode not in SUPPORTED_ENTITY_LOOKUP_POLICY_MODES:
                errors.append(
                    {
                        "code": "ENTITY_LOOKUP_POLICY_MODE_INVALID",
                        "mode": lookup_mode or "UNDECLARED",
                    }
                )
            if lookup_time_column and lookup_time_column not in columns:
                errors.append(
                    {
                        "code": "ENTITY_LOOKUP_TIME_COLUMN_MISSING",
                        "column": lookup_time_column,
                    }
                )
            if lookup_mode in {"bounded_default", "default_window"}:
                try:
                    default_days = int(raw_entity_lookup_policy.get("defaultDays"))
                except (TypeError, ValueError):
                    default_days = 0
                if default_days <= 0:
                    errors.append({"code": "ENTITY_LOOKUP_DEFAULT_DAYS_INVALID"})
                if not lookup_time_column:
                    errors.append({"code": "ENTITY_LOOKUP_TIME_COLUMN_UNDECLARED"})
    for entity_field in entity_fields:
        entity_column = str(entity_field.get("columnName") or "").strip()
        canonical_ref = str(
            entity_field.get("canonicalEntityRef")
            or entity_field.get("canonicalEntityType")
            or entity_field.get("entityType")
            or ""
        ).strip()
        comparison_policy = str(entity_field.get("comparisonPolicy") or "").strip().lower()
        if not canonical_ref:
            warnings.append({"code": "CANONICAL_ENTITY_REF_UNDECLARED", "column": entity_column})
        if not comparison_policy:
            warnings.append({"code": "ENTITY_COMPARISON_POLICY_UNDECLARED", "column": entity_column})
        elif comparison_policy not in SUPPORTED_ENTITY_COMPARISON_POLICIES:
            errors.append(
                {
                    "code": "ENTITY_COMPARISON_POLICY_INVALID",
                    "column": entity_column,
                    "comparisonPolicy": comparison_policy,
                }
            )
        raw_unique = entity_field.get(
            "isUniqueEntityKey",
            entity_field.get("isUniqueKey", entity_field.get("is_unique_entity_key")),
        )
        if raw_unique is not None and not isinstance(raw_unique, bool):
            errors.append(
                {
                    "code": "ENTITY_UNIQUE_KEY_DECLARATION_INVALID",
                    "column": entity_column,
                }
            )
        raw_operators = entity_field.get("filterOperators") or entity_field.get(
            "filter_operators"
        )
        if isinstance(raw_operators, str):
            raw_operators = [raw_operators]
        if raw_operators is not None and not isinstance(raw_operators, list):
            errors.append(
                {
                    "code": "ENTITY_FILTER_OPERATORS_INVALID",
                    "column": entity_column,
                }
            )
            raw_operators = []
        operators = {
            normalize_entity_filter_operator(item)
            for item in raw_operators or []
        }
        invalid_operators = sorted(
            operator
            for operator in operators
            if not operator or operator not in SUPPORTED_ENTITY_FILTER_OPERATORS
        )
        if invalid_operators:
            errors.append(
                {
                    "code": "ENTITY_FILTER_OPERATOR_INVALID",
                    "column": entity_column,
                    "operators": invalid_operators,
                }
            )
        if not raw_operators:
            warnings.append(
                {
                    "code": "ENTITY_FILTER_OPERATORS_UNDECLARED",
                    "column": entity_column,
                }
            )

        field_lookup_policy = entity_field.get("lookupTimePolicy") or entity_field.get(
            "lookup_time_policy"
        )
        if field_lookup_policy is not None:
            if not isinstance(field_lookup_policy, dict):
                errors.append(
                    {
                        "code": "FIELD_LOOKUP_TIME_POLICY_INVALID",
                        "column": entity_column,
                    }
                )
            else:
                normalized_field_policy = normalize_entity_lookup_policy(
                    field_lookup_policy,
                    inherited_time_column=table_time_column,
                )
                field_mode = str(normalized_field_policy.get("mode") or "")
                if field_mode not in SUPPORTED_ENTITY_LOOKUP_POLICY_MODES:
                    errors.append(
                        {
                            "code": "FIELD_LOOKUP_TIME_POLICY_MODE_INVALID",
                            "column": entity_column,
                            "mode": field_mode or "UNDECLARED",
                        }
                    )
                field_time_column = str(
                    normalized_field_policy.get("timeColumn") or ""
                )
                if field_time_column and field_time_column not in columns:
                    errors.append(
                        {
                            "code": "FIELD_LOOKUP_TIME_COLUMN_MISSING",
                            "column": entity_column,
                            "timeColumn": field_time_column,
                        }
                    )
                if field_mode in {"bounded_default", "default_window"}:
                    try:
                        field_default_days = int(
                            normalized_field_policy.get("defaultDays") or 0
                        )
                    except (TypeError, ValueError):
                        field_default_days = 0
                    if field_default_days <= 0:
                        errors.append(
                            {
                                "code": "FIELD_LOOKUP_DEFAULT_DAYS_INVALID",
                                "column": entity_column,
                            }
                        )
        elif bool(raw_unique):
            warnings.append(
                {
                    "code": "FIELD_LOOKUP_TIME_POLICY_UNDECLARED",
                    "column": entity_column,
                }
            )
    if row_access_policy and str(row_access_policy.get("filterColumn") or "") not in columns:
        errors.append({"code": "ROW_ACCESS_FILTER_COLUMN_MISSING", "column": row_access_policy.get("filterColumn")})
    if bool(table_usage.get("queryableByAgent")):
        if metrics and "METRIC" not in result_access_policies:
            errors.append({"code": "METRIC_RESULT_ACCESS_POLICY_UNDECLARED"})
        if str(asset.get("timeColumn") or "") and "TIME" not in result_access_policies:
            errors.append({"code": "TIME_RESULT_ACCESS_POLICY_UNDECLARED", "column": asset.get("timeColumn")})
    for role, policy in result_access_policies.items():
        visibility = normalize_visibility_policy(policy.get("visibilityPolicy") or {})
        masking = normalize_masking_policy(policy.get("maskingPolicy") or {})
        if visibility.get("level") == "restricted" and not visibility.get("allowedRoles"):
            errors.append({"code": "RESULT_POLICY_RESTRICTED_WITHOUT_ROLES", "semanticRole": role})
        if visibility.get("level") == "restricted" and masking.get("strategy") == "none":
            warnings.append({"code": "RESULT_POLICY_RESTRICTED_WITHOUT_MASKING", "semanticRole": role})
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
        semantic_role = str(field.get("role") or field.get("semanticRole") or "").strip().upper()
        metric_formula = str(field.get("metricFormula") or "").strip()
        if column_name and column_name not in columns:
            warnings.append({"code": "SEMANTIC_COLUMN_NOT_IN_SCHEMA", "column": column_name})
        if metric_formula and semantic_role not in {"METRIC", "MEASURE"}:
            errors.append(
                {
                    "code": "NON_METRIC_COLUMN_HAS_METRIC_FORMULA",
                    "column": column_name,
                    "semanticRole": semantic_role or "UNDECLARED",
                }
            )
        visibility = normalize_visibility_policy(field.get("visibilityPolicy") or {})
        masking = normalize_masking_policy(field.get("maskingPolicy") or {})
        if visibility.get("level") == "restricted" and not visibility.get("allowedRoles"):
            warnings.append({"code": "RESTRICTED_COLUMN_WITHOUT_ALLOWED_ROLES", "column": column_name})
        if visibility.get("level") == "restricted" and masking.get("strategy") == "none":
            warnings.append({"code": "RESTRICTED_COLUMN_WITHOUT_MASKING", "column": column_name})
        raw_calculation_semantics = field.get("calculationSemantics")
        if raw_calculation_semantics is not None or semantic_role in {
            "KEY",
            "ENTITY",
            "ENTITY_KEY",
            "PRIMARY_KEY",
            "IDENTIFIER",
        }:
            calculation_validation = validate_calculation_semantics(
                raw_calculation_semantics,
                "COLUMN",
                column_name,
            )
            errors.extend(calculation_validation["errors"])
            warnings.extend(calculation_validation["warnings"])
        enum_metadata = field.get("enumMetadata") if isinstance(field.get("enumMetadata"), dict) else {}
        if (
            populated_semantic_enum_fields(field)
            and str(enum_metadata.get("reviewStatus") or "UNREVIEWED").upper() != "APPROVED"
        ) or bool(enum_metadata.get("runtimeSuppressed")):
            warnings.append({"code": "ENUM_VALUES_NOT_BUSINESS_APPROVED", "column": column_name})
        field_aggregation_policy = str(field.get("aggregationPolicy") or "").strip().lower()
        if field_aggregation_policy == "daily_value_only":
            field_time_grain = str(field.get("applicableTimeGrain") or "").strip().lower()
            if not field_time_grain:
                errors.append({"code": "DAILY_VALUE_TIME_GRAIN_UNDECLARED", "column": column_name})
            if re.search(r"\b(?:SUM|AVG)\s*\(", metric_formula, flags=re.IGNORECASE):
                errors.append(
                    {
                        "code": "DAILY_VALUE_CROSS_DAY_AGGREGATION_FORBIDDEN",
                        "column": column_name,
                        "formula": metric_formula,
                    }
                )
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
        table_time_column = str(asset.get("timeColumn") or "").strip()
        time_column = str(metric.get("timeColumn") or table_time_column or "").strip()
        if not time_column:
            errors.append({"code": "METRIC_TIME_COLUMN_UNDECLARED", "metricKey": metric_key})
        elif time_column not in columns:
            errors.append({"code": "METRIC_TIME_COLUMN_MISSING", "metricKey": metric_key, "column": time_column})
        aggregation_policy = str(metric.get("aggregationPolicy") or "").strip().lower()
        formula = str(metric.get("formula") or metric.get("metricFormula") or "").strip()
        metric_type = str(metric.get("metricType") or "").strip().upper()
        ratio_metric = metric_type == "RATIO" or "/" in formula
        metric_grain = str(metric.get("metricGrain") or "").strip().lower()
        applicable_time_grain = str(metric.get("applicableTimeGrain") or "").strip().lower()
        calculation_validation = validate_calculation_semantics(
            metric.get("calculationSemantics"),
            "METRIC",
            metric_key,
            semantic_metric_source_columns(metric),
        )
        errors.extend(calculation_validation["errors"])
        warnings.extend(calculation_validation["warnings"])
        if not metric_grain:
            errors.append({"code": "METRIC_GRAIN_UNDECLARED", "metricKey": metric_key})
        if not applicable_time_grain:
            errors.append({"code": "METRIC_APPLICABLE_TIME_GRAIN_UNDECLARED", "metricKey": metric_key})
        if not aggregation_policy:
            errors.append({"code": "METRIC_AGGREGATION_POLICY_UNDECLARED", "metricKey": metric_key})
        elif aggregation_policy not in SUPPORTED_METRIC_AGGREGATION_POLICIES:
            errors.append(
                {
                    "code": "METRIC_AGGREGATION_POLICY_INVALID",
                    "metricKey": metric_key,
                    "aggregationPolicy": aggregation_policy,
                }
            )
        if aggregation_policy == "period_rollup" and re.search(r"\bAVG\s*\(", formula, flags=re.IGNORECASE):
            errors.append(
                {
                    "code": "PERIOD_ROLLUP_NON_ADDITIVE_FORMULA",
                    "metricKey": metric_key,
                    "formula": formula,
                }
            )
        if ratio_metric and aggregation_policy not in {"ratio_of_sums", "daily_value_only", "period_rollup"}:
            errors.append({"code": "RATIO_AGGREGATION_POLICY_UNDECLARED", "metricKey": metric_key})
        if aggregation_policy == "ratio_of_sums" and (
            "/" not in formula or len(re.findall(r"\bSUM\s*\(", formula, flags=re.IGNORECASE)) < 2
        ):
            errors.append(
                {
                    "code": "RATIO_OF_SUMS_FORMULA_INVALID",
                    "metricKey": metric_key,
                    "formula": formula,
                }
            )
        raw_time_semantics = metric.get("timeSemantics")
        if raw_time_semantics is not None and not isinstance(raw_time_semantics, dict):
            errors.append({"code": "METRIC_TIME_SEMANTICS_INVALID", "metricKey": metric_key})
        elif isinstance(raw_time_semantics, dict):
            selection_policy = str(raw_time_semantics.get("selectionPolicy") or "").strip().lower()
            as_of_policy = str(raw_time_semantics.get("asOfPolicy") or "").strip().lower()
            missing_data_policy = str(raw_time_semantics.get("missingDataPolicy") or "").strip().lower()
            zero_value_policy = str(raw_time_semantics.get("zeroValuePolicy") or "").strip().lower()
            declared_policies = {
                "selectionPolicy": (selection_policy, SUPPORTED_METRIC_TIME_SELECTION_POLICIES),
                "asOfPolicy": (as_of_policy, SUPPORTED_METRIC_AS_OF_POLICIES),
                "missingDataPolicy": (missing_data_policy, SUPPORTED_METRIC_MISSING_DATA_POLICIES),
                "zeroValuePolicy": (zero_value_policy, SUPPORTED_METRIC_ZERO_VALUE_POLICIES),
            }
            for policy_name, (policy_value, supported_values) in declared_policies.items():
                if not policy_value or policy_value == "undeclared":
                    errors.append(
                        {
                            "code": "METRIC_TIME_SEMANTICS_UNDECLARED",
                            "metricKey": metric_key,
                            "policy": policy_name,
                        }
                    )
                elif policy_value not in supported_values:
                    errors.append(
                        {
                            "code": "METRIC_TIME_SEMANTICS_POLICY_INVALID",
                            "metricKey": metric_key,
                            "policy": policy_name,
                            "value": policy_value,
                        }
                    )
            if aggregation_policy == "latest_value_only" and as_of_policy in {"", "not_applicable", "undeclared"}:
                errors.append({"code": "LATEST_VALUE_AS_OF_POLICY_UNDECLARED", "metricKey": metric_key})
            expected_selection_policy = {
                "latest_value_only": "latest_as_of",
                "daily_value_only": "per_time_grain",
                "period_rollup": "period_window",
                "period_recompute": "period_window",
                "ratio_of_sums": "period_window",
            }.get(aggregation_policy, "")
            if (
                expected_selection_policy
                and selection_policy not in {"", "undeclared"}
                and selection_policy != expected_selection_policy
            ):
                errors.append(
                    {
                        "code": "METRIC_TIME_SELECTION_POLICY_CONFLICT",
                        "metricKey": metric_key,
                        "aggregationPolicy": aggregation_policy,
                        "selectionPolicy": selection_policy,
                    }
                )
            if missing_data_policy == "zero_fill" and zero_value_policy == "treat_as_missing":
                errors.append({"code": "METRIC_MISSING_ZERO_POLICY_CONFLICT", "metricKey": metric_key})
            executable_as_of_policies = EXECUTABLE_METRIC_AS_OF_POLICIES_BY_SELECTION.get(
                selection_policy,
                frozenset(),
            )
            if as_of_policy and as_of_policy not in executable_as_of_policies:
                errors.append(
                    {
                        "code": "METRIC_AS_OF_POLICY_NOT_EXECUTABLE",
                        "metricKey": metric_key,
                        "selectionPolicy": selection_policy,
                        "asOfPolicy": as_of_policy,
                    }
                )
            if missing_data_policy and missing_data_policy not in EXECUTABLE_METRIC_MISSING_DATA_POLICIES:
                errors.append(
                    {
                        "code": "METRIC_MISSING_DATA_POLICY_NOT_EXECUTABLE",
                        "metricKey": metric_key,
                        "missingDataPolicy": missing_data_policy,
                    }
                )
            if zero_value_policy and zero_value_policy not in EXECUTABLE_METRIC_ZERO_VALUE_POLICIES:
                errors.append(
                    {
                        "code": "METRIC_ZERO_VALUE_POLICY_NOT_EXECUTABLE",
                        "metricKey": metric_key,
                        "zeroValuePolicy": zero_value_policy,
                    }
                )
        else:
            errors.append({"code": "METRIC_TIME_SEMANTICS_UNDECLARED", "metricKey": metric_key})
        if aggregation_policy == "daily_value_only":
            if not applicable_time_grain:
                errors.append({"code": "DAILY_VALUE_TIME_GRAIN_UNDECLARED", "metricKey": metric_key})
            elif applicable_time_grain != "day":
                errors.append(
                    {
                        "code": "DAILY_VALUE_TIME_GRAIN_INVALID",
                        "metricKey": metric_key,
                        "applicableTimeGrain": applicable_time_grain,
                    }
                )
            if re.search(r"\b(?:SUM|AVG)\s*\(", formula, flags=re.IGNORECASE):
                errors.append(
                    {
                        "code": "DAILY_VALUE_CROSS_DAY_AGGREGATION_FORBIDDEN",
                        "metricKey": metric_key,
                        "formula": formula,
                    }
                )
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
    physical_metadata: Dict[str, Any],
    semantic_columns: List[Dict[str, Any]],
    metrics: List[Dict[str, Any]],
    terms: List[Dict[str, Any]],
    rules: List[Dict[str, Any]],
    builder_phases: Dict[str, Any],
    generation_mode: str,
) -> Dict[str, Any]:
    physical_metadata = physical_metadata or {}
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
            "keyModel": str(physical_metadata.get("keyModel") or profile.get("keyModel") or ""),
            "primaryKeyColumns": [str(item) for item in physical_metadata.get("primaryKeyColumns") or profile.get("primaryKeyColumns") or []],
            "partitionColumns": [str(item) for item in physical_metadata.get("partitionColumns") or profile.get("partitionColumns") or []],
            "bucketColumns": [str(item) for item in physical_metadata.get("bucketColumns") or profile.get("bucketColumns") or []],
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
    schema_fields = asset.get("schemaColumns") if isinstance(asset.get("schemaColumns"), list) else []
    errors = list(validation.get("errors") or []) if isinstance(validation, dict) else []
    blocking_reasons: List[str] = []
    warning_reasons: List[str] = []
    if errors:
        blocking_reasons.append("SEMANTIC_VALIDATION_ERRORS")
    if conflict_detection.get("conflictCount"):
        blocking_reasons.append("SEMANTIC_CONFLICTS")
    if getattr(drift, "missing_live_columns", None):
        blocking_reasons.append("SCHEMA_DRIFT_MISSING_COLUMNS")
    if not fields and not schema_fields:
        blocking_reasons.append("NO_SEMANTIC_FIELDS")
    elif not fields:
        warning_reasons.append("SEMANTIC_FIELD_ANNOTATION_MISSING")
    if not metrics:
        blocking_reasons.append("NO_METRICS")
    return {
        "publishable": not blocking_reasons,
        "severity": "blocking" if blocking_reasons else "passed",
        "blockingReasons": blocking_reasons,
        "warningReasons": warning_reasons,
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


def semantic_catalog_conflict_detection(
    topic_assets: TopicAssetService,
    candidate_topic: str = "",
    candidate_table: str = "",
    candidate_asset: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Detect conflicts across the published catalog, not only inside one asset."""

    entries: List[Tuple[str, str, Dict[str, Any]]] = []
    candidate_seen = False
    for topic in topic_assets.all_topic_names():
        tables_dir = topic_assets.root / topic / "tables"
        if not tables_dir.exists():
            continue
        for table_dir in sorted(path for path in tables_dir.iterdir() if path.is_dir()):
            table = table_dir.name
            if topic == candidate_topic and table == candidate_table and candidate_asset is not None:
                asset = dict(candidate_asset)
                candidate_seen = True
            else:
                asset = topic_assets.load_table_asset(topic, table)
            status = str(asset.get("status") or "PUBLISHED").upper()
            if status in {"RETIRED", "DEPRECATED", "REJECTED"}:
                continue
            entries.append((topic, table, asset))
    if candidate_asset is not None and candidate_topic and candidate_table and not candidate_seen:
        entries.append((candidate_topic, candidate_table, dict(candidate_asset)))

    conflicts: List[Dict[str, Any]] = []
    table_owners: Dict[str, List[str]] = {}
    metric_formulas: Dict[str, List[Dict[str, str]]] = {}
    scoped_aliases: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    globally_sensitive_aliases: Dict[str, List[Dict[str, str]]] = {}
    unresolved_term_aliases: Dict[str, List[Dict[str, str]]] = {}
    enum_labels: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}

    for topic, table, asset in entries:
        table_owners.setdefault(table, []).append(topic)
        for metric in asset.get("metrics") or []:
            if not isinstance(metric, dict):
                continue
            concrete_metric_key = str(metric.get("metricKey") or "").strip()
            metric_key = str(metric.get("canonicalMetricKey") or concrete_metric_key).strip()
            metric_family = str(metric.get("canonicalMetricKey") or metric.get("aliasOf") or concrete_metric_key).strip()
            formula = normalize_semantic_formula(metric.get("formula") or metric.get("metricFormula"))
            if metric_key:
                metric_formulas.setdefault(metric_key, []).append(
                    {"topic": topic, "table": table, "metricKey": concrete_metric_key or metric_key, "formula": formula}
                )
            for alias in metric.get("aliases") or []:
                alias_text = str(alias or "").strip().lower()
                if alias_text:
                    definition = {
                        "topic": topic,
                        "table": table,
                        "metricKey": concrete_metric_key or metric_key,
                        "metricFamily": metric_family,
                        "formula": formula,
                        "metricIntent": str(metric.get("metricIntent") or ""),
                        "metricGrain": str(metric.get("metricGrain") or ""),
                    }
                    scoped_aliases.setdefault((topic, alias_text), []).append(
                        definition
                    )
                    if semantic_metric_alias_requires_global_owner(metric, alias_text):
                        globally_sensitive_aliases.setdefault(alias_text, []).append(definition)
        for term in asset.get("terms") or []:
            if not isinstance(term, dict):
                continue
            term_key = str(term.get("canonicalMetricKey") or term.get("term") or "").strip()
            if not term_key:
                continue
            for alias in term.get("aliases") or []:
                alias_text = str(alias or "").strip().lower()
                if not alias_text:
                    continue
                target = (
                    globally_sensitive_aliases
                    if str(term.get("aliasConflictScope") or "").upper() == "GLOBAL"
                    else unresolved_term_aliases
                )
                target.setdefault(alias_text, []).append(
                    {
                        "topic": topic,
                        "table": table,
                        "metricKey": term_key,
                        "metricFamily": term_key,
                        "formula": "",
                        "metricIntent": "term",
                        "metricGrain": "",
                    }
                )
        for column in asset.get("semanticColumns") or []:
            if not isinstance(column, dict):
                continue
            column_name = str(column.get("columnName") or column.get("name") or "").strip()
            for value, label in semantic_enum_labels(column):
                enum_labels.setdefault((table, column_name, value), []).append({"topic": topic, "label": label})

    for alias, definitions in unresolved_term_aliases.items():
        if alias in globally_sensitive_aliases:
            globally_sensitive_aliases[alias].extend(definitions)
    for table, topics in table_owners.items():
        owners = sorted(set(topics))
        if len(owners) > 1:
            conflicts.append({"type": "duplicate_table_owner", "tableName": table, "topics": owners})
    for metric_key, definitions in metric_formulas.items():
        formulas = sorted({item["formula"] for item in definitions if item["formula"]})
        if len(formulas) > 1:
            conflicts.append(
                {"type": "catalog_metric_formula_conflict", "metricKey": metric_key, "formulas": formulas, "definitions": definitions}
            )
    for (topic, alias), definitions in scoped_aliases.items():
        targets = {(item["metricKey"], item["formula"]) for item in definitions}
        if len(targets) > 1:
            conflicts.append({"type": "topic_metric_alias_conflict", "topic": topic, "alias": alias, "definitions": definitions})
    for alias, definitions in globally_sensitive_aliases.items():
        owners = {(item["topic"], item["table"], item["metricKey"]) for item in definitions}
        families: Set[str] = set()
        for item in definitions:
            family = item["metricFamily"]
            if item["metricIntent"] == "term":
                local_metric_families = {
                    candidate["metricFamily"]
                    for candidate in definitions
                    if candidate["metricIntent"] != "term"
                    and candidate["topic"] == item["topic"]
                    and candidate["table"] == item["table"]
                    and candidate["metricFamily"]
                }
                if len(local_metric_families) == 1:
                    family = next(iter(local_metric_families))
            if family:
                families.add(family)
        if len(owners) > 1 and len(families) > 1:
            conflicts.append(
                {
                    "type": "global_ratio_alias_conflict",
                    "alias": alias,
                    "metricFamilies": sorted(families),
                    "definitions": definitions,
                }
            )
    for (table, column, value), definitions in enum_labels.items():
        labels = sorted({item["label"] for item in definitions if item["label"]})
        if len(labels) > 1:
            conflicts.append(
                {"type": "enum_interpretation_conflict", "tableName": table, "columnName": column, "value": value, "labels": labels}
            )
    return {
        "status": "passed" if not conflicts else "conflict_detected",
        "conflictCount": len(conflicts),
        "conflicts": conflicts[:50],
    }


def combine_semantic_conflict_detection(*reports: Dict[str, Any]) -> Dict[str, Any]:
    conflicts: List[Dict[str, Any]] = []
    seen = set()
    for report in reports:
        for conflict in (report or {}).get("conflicts") or []:
            key = json.dumps(conflict, ensure_ascii=False, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            conflicts.append(conflict)
    return {
        "status": "passed" if not conflicts else "conflict_detected",
        "conflictCount": len(conflicts),
        "conflicts": conflicts[:50],
    }


def normalize_semantic_formula(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).upper()


def semantic_metric_alias_requires_global_owner(metric: Dict[str, Any], alias: str) -> bool:
    del alias
    governance = metric.get("aliasGovernance") if isinstance(metric.get("aliasGovernance"), dict) else {}
    scope = str(metric.get("aliasConflictScope") or governance.get("conflictScope") or "").upper()
    if scope:
        return scope == "GLOBAL"
    metric_type = str(metric.get("metricType") or metric.get("semanticType") or "").upper()
    value_format = str(metric.get("valueFormat") or metric.get("value_format") or "").upper()
    unit = str(metric.get("unit") or "").strip()
    return metric_type in {"RATE", "RATIO", "PERCENTAGE"} or value_format in {
        "PERCENT",
        "PERCENTAGE",
        "RATIO",
    } or unit == "%"


def semantic_ratio_alias_label(alias: str) -> bool:
    del alias
    return False


def semantic_enum_labels(column: Dict[str, Any]) -> List[Tuple[str, str]]:
    if not semantic_enum_business_approved(column):
        return []
    labels: List[Tuple[str, str]] = []
    for key in ("enumMappings", "enumMeanings", "valueLabels"):
        mapping = column.get(key)
        if isinstance(mapping, dict):
            labels.extend((str(value), str(label)) for value, label in mapping.items() if str(value) and str(label))
    for item in column.get("enumValues") or []:
        if not isinstance(item, dict):
            continue
        value = str(item.get("value") or item.get("code") or "")
        label = str(item.get("label") or item.get("name") or item.get("meaning") or "")
        if value and label:
            labels.append((value, label))
    return labels


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
        elif kind == "duplicate_table_owner":
            actions.append(
                {
                    "action": "choose_canonical_table_owner",
                    "tableName": conflict.get("tableName", ""),
                    "topics": conflict.get("topics", []),
                    "requiresOwnerReview": True,
                }
            )
        elif kind in {"catalog_metric_formula_conflict", "topic_metric_alias_conflict", "global_ratio_alias_conflict"}:
            actions.append(
                {
                    "action": "choose_canonical_metric_contract",
                    "metricKey": conflict.get("metricKey", ""),
                    "alias": conflict.get("alias", ""),
                    "definitions": conflict.get("definitions", []),
                    "requiresOwnerReview": True,
                }
            )
        elif kind == "enum_interpretation_conflict":
            actions.append(
                {
                    "action": "choose_canonical_enum_interpretation",
                    "tableName": conflict.get("tableName", ""),
                    "columnName": conflict.get("columnName", ""),
                    "value": conflict.get("value", ""),
                    "labels": conflict.get("labels", []),
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
        "physicalMetadata": {
            "keyModel": str((asset.get("physicalMetadata") or {}).get("keyModel") or ""),
            "primaryKeyColumns": [str(item) for item in ((asset.get("physicalMetadata") or {}).get("primaryKeyColumns") or [])],
            "partitionColumns": [str(item) for item in ((asset.get("physicalMetadata") or {}).get("partitionColumns") or [])],
            "bucketColumns": [str(item) for item in ((asset.get("physicalMetadata") or {}).get("bucketColumns") or [])],
        },
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
    wanted = str(ref or "").strip()
    if not wanted:
        return False
    declared: List[str] = []
    for key in ("metricDependencies", "requiresMetrics", "externalMetricRefs"):
        for item in metric.get(key) or []:
            if isinstance(item, dict):
                declared.append(str(item.get("metricRef") or item.get("metricKey") or "").strip())
            else:
                declared.append(str(item or "").strip())
    for item in metric.get("sourceReferences") or []:
        if not isinstance(item, dict) or str(item.get("referenceType") or "").upper() != "METRIC":
            continue
        declared.append(str(item.get("reference") or item.get("metricRef") or item.get("metricKey") or "").strip())
    return wanted in {item for item in declared if item}


def metric_shaped_reference(ref: str) -> bool:
    del ref
    return False


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


def is_active_semantic_activation_file(path: Path, root: Path) -> bool:
    """Classify files that can change the active runtime semantic view."""

    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return False
    if len(parts) == 2:
        return bool(parts[0]) and parts[1] in ACTIVE_TOPIC_SEMANTIC_FILENAMES
    if len(parts) == 4:
        topic, tables_segment, table, file_name = parts
        return (
            bool(topic)
            and tables_segment == "tables"
            and bool(table)
            and file_name in ACTIVE_TABLE_SEMANTIC_FILENAMES
        )
    return False


def stable_relative_file_digest(root: Path, files: Iterable[Path]) -> str:
    """Hash relative path and bytes in a deterministic, boundary-safe form."""

    try:
        indexed = sorted(
            {
                path.relative_to(root).as_posix(): path
                for path in files
                if path.is_file()
            }.items()
        )
    except (OSError, ValueError):
        return ""
    hasher = hashlib.sha256()
    for relative_text, path in indexed:
        try:
            relative = relative_text.encode("utf-8")
            content = path.read_bytes()
        except OSError:
            return ""
        hasher.update(len(relative).to_bytes(8, "big"))
        hasher.update(relative)
        hasher.update(len(content).to_bytes(8, "big"))
        hasher.update(content)
    return hasher.hexdigest()


def active_semantic_activation_digest(root: Path, files: Iterable[Path]) -> str:
    """Return the identity of the effective, active semantic filesystem view."""

    return stable_relative_file_digest(
        root,
        (path for path in files if is_active_semantic_activation_file(path, root)),
    )


def semantic_activation_signature(
    root: Path,
    files: Iterable[Path],
) -> Optional[SemanticActivationSignature]:
    """Cheap cache signature that also detects out-of-band active file edits."""

    signature: List[Tuple[str, int, int, int, int]] = []
    try:
        indexed = sorted(
            {
                path.relative_to(root).as_posix(): path
                for path in files
                if is_active_semantic_activation_file(path, root)
            }.items()
        )
        for relative, path in indexed:
            stat = path.stat()
            signature.append(
                (
                    relative,
                    int(stat.st_size),
                    int(stat.st_mtime_ns),
                    int(stat.st_ctime_ns),
                    int(stat.st_ino),
                )
            )
    except (OSError, ValueError):
        return None
    return tuple(signature)


def semantic_candidate_source_hash(candidate_dir: Path) -> str:
    """Return a stable hash for the immutable semantic publish candidate."""

    if not candidate_dir.exists() or not candidate_dir.is_dir():
        return ""
    return stable_relative_file_digest(
        candidate_dir,
        (
            path
            for path in candidate_dir.rglob("*")
            if path.is_file() and path.name != "review-result.json"
        ),
    )


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
    return dedupe_strings(
        [str(token) for token in keywords if str(token or "").strip()]
        + question_match_terms(question)
    )[:80]


def targeted_recall_terms(question: str) -> List[str]:
    terms = recall_terms(question, [])
    for term in question_match_terms(question):
        if term and term not in terms:
            terms.append(term)
    return terms


def infer_table_selection_intents(question: str) -> Set[str]:
    del question
    return set()


def table_seed_terms(question: str) -> List[str]:
    return targeted_recall_terms(question)


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
                "topic": str(item.topic or metadata.get("topic") or ""),
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
                "metricGrain": str(metadata.get("metricGrain") or metadata.get("grainHint") or ""),
                "metricIntent": str(metadata.get("metricIntent") or ""),
                "aggregationPolicy": str(metadata.get("aggregationPolicy") or ""),
                "applicableTimeGrain": str(metadata.get("applicableTimeGrain") or ""),
                "timeColumn": str(metadata.get("timeColumn") or metadata.get("time_column") or ""),
                "timeSemantics": metadata.get("timeSemantics") or {},
                "missingValuePolicy": str(metadata.get("missingValuePolicy") or ""),
                "zeroValueMeaning": str(metadata.get("zeroValueMeaning") or ""),
                "selectionGuidance": str(metadata.get("selectionGuidance") or ""),
                "preferredUseCases": metadata.get("preferredUseCases") or [],
                "notPreferredUseCases": metadata.get("notPreferredUseCases") or [],
                "temporalVariants": metadata.get("temporalVariants") or {},
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
                    "metricGrain": str(metadata.get("metricGrain") or current.get("metricGrain") or ""),
                    "metricIntent": str(metadata.get("metricIntent") or current.get("metricIntent") or ""),
                    "aggregationPolicy": str(
                        metadata.get("aggregationPolicy") or current.get("aggregationPolicy") or ""
                    ),
                    "applicableTimeGrain": str(
                        metadata.get("applicableTimeGrain") or current.get("applicableTimeGrain") or ""
                    ),
                    "timeColumn": str(
                        metadata.get("timeColumn")
                        or metadata.get("time_column")
                        or current.get("timeColumn")
                        or ""
                    ),
                    "timeSemantics": metadata.get("timeSemantics") or current.get("timeSemantics") or {},
                    "missingValuePolicy": str(
                        metadata.get("missingValuePolicy") or current.get("missingValuePolicy") or ""
                    ),
                    "zeroValueMeaning": str(
                        metadata.get("zeroValueMeaning") or current.get("zeroValueMeaning") or ""
                    ),
                    "selectionGuidance": str(
                        metadata.get("selectionGuidance") or current.get("selectionGuidance") or ""
                    ),
                    "preferredUseCases": metadata.get("preferredUseCases") or current.get("preferredUseCases") or [],
                    "notPreferredUseCases": (
                        metadata.get("notPreferredUseCases") or current.get("notPreferredUseCases") or []
                    ),
                    "temporalVariants": metadata.get("temporalVariants") or current.get("temporalVariants") or {},
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
