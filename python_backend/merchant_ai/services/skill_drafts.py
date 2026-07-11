from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from merchant_ai.config import Settings
from merchant_ai.models import AgentRunResult, QueryPlan, SkillDraft, SkillDraftReviewRequest, category_display


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", value or "").strip("_").lower()
    return text[:48] or "generated_skill"


class SkillDraftService:
    """Govern free exploration results before they become callable skills."""

    def __init__(self, settings: Settings, skill_root: Optional[Path] = None):
        self.settings = settings
        self.skill_root = skill_root or (settings.resources_root / "runtime" / "agent_skills")
        self.root = settings.resolved_workspace_path / "ops" / "skill_drafts"
        self.index_path = self.root / "skill_drafts.json"
        self.registry_path = settings.resolved_workspace_path / "ops" / "skill_market" / "skill_registry.json"

    def maybe_create_from_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not self._eligible(state):
            return {}
        draft = self._build_draft(state)
        existing = self.list_drafts()
        if any(item.get("sourceRunId") == draft.source_run_id for item in existing):
            return {}
        payload = draft.model_dump(by_alias=True)
        _write_json(self.root / ("%s.json" % draft.draft_id), payload)
        existing.append(payload)
        _write_json(self.index_path, {"items": existing})
        return payload

    def list_drafts(self, status: str = "") -> List[Dict[str, Any]]:
        payload = _read_json(self.index_path, {"items": []})
        items = payload.get("items") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            items = []
        if status:
            return [item for item in items if str(item.get("status") or "") == status]
        return items

    def review_draft(self, draft_id: str, request: SkillDraftReviewRequest) -> Dict[str, Any]:
        items = self.list_drafts()
        target: Dict[str, Any] = {}
        for item in items:
            if str(item.get("draftId") or item.get("draft_id") or "") == draft_id:
                target = item
                break
        if not target:
            return {"success": False, "status": "NOT_FOUND", "draftId": draft_id}
        target["reviewer"] = request.reviewer
        target["reviewNote"] = request.review_note
        target["reviewedAt"] = datetime.utcnow().isoformat() + "Z"
        if request.approved:
            skill_name = self._publish_skill(target)
            target["status"] = "approved"
            target["callable"] = True
            target["publishedSkillName"] = skill_name
            target["skillRegistry"] = self.register_skill(skill_name, target)
        else:
            target["status"] = "rejected"
            target["callable"] = False
        _write_json(self.root / ("%s.json" % draft_id), target)
        _write_json(self.index_path, {"items": items})
        return {"success": True, "status": target["status"], "draft": target}

    def market(self) -> Dict[str, Any]:
        registry = self._load_registry()
        discovered = self._discover_builtin_skills()
        items_by_name = {str(item.get("skillName") or ""): item for item in discovered if item.get("skillName")}
        for item in registry:
            if item.get("skillName"):
                items_by_name[str(item.get("skillName"))] = item
        items = sorted(items_by_name.values(), key=lambda item: (str(item.get("status") or ""), str(item.get("displayName") or item.get("skillName") or "")))
        return {
            "success": True,
            "mode": "skill_market",
            "count": len(items),
            "items": items,
        }

    def register_skill(self, skill_name: str, draft: Dict[str, Any]) -> Dict[str, Any]:
        registry = self._load_registry()
        now = datetime.utcnow().isoformat() + "Z"
        version = "skill-%s" % uuid.uuid4().hex[:10]
        record = {
            "skillName": skill_name,
            "displayName": str(draft.get("title") or skill_name),
            "version": version,
            "status": "active",
            "callable": True,
            "sourceDraftId": str(draft.get("draftId") or draft.get("draft_id") or ""),
            "merchantId": str(draft.get("merchantId") or ""),
            "installScope": {
                "scope": "merchant",
                "merchantIds": [str(draft.get("merchantId") or "")] if draft.get("merchantId") else [],
                "industryTags": [],
            },
            "grayRelease": {
                "enabled": True,
                "stage": "beta",
                "trafficPercent": 10,
                "abortConditions": ["skill_eval_failed", "evidence_gap_rate_above_threshold"],
            },
            "versions": [
                {
                    "version": version,
                    "publishedAt": now,
                    "sourceDraftId": str(draft.get("draftId") or draft.get("draft_id") or ""),
                    "reviewer": str(draft.get("reviewer") or ""),
                }
            ],
            "runtimeStats": {
                "runCount": 0,
                "lastRunAt": "",
                "failureCount": 0,
            },
            "createdAt": now,
            "updatedAt": now,
        }
        next_registry = [item for item in registry if str(item.get("skillName") or "") != skill_name]
        next_registry.append(record)
        self._write_registry(next_registry)
        return record

    def install_skill(
        self,
        skill_name: str,
        scope: str = "merchant",
        merchant_ids: Optional[List[str]] = None,
        industry_tags: Optional[List[str]] = None,
        traffic_percent: int = 100,
    ) -> Dict[str, Any]:
        registry = self._load_registry()
        target = None
        for item in registry:
            if str(item.get("skillName") or "") == skill_name:
                target = item
                break
        if target is None:
            target = self._builtin_market_item(skill_name)
            registry.append(target)
        target["installScope"] = {
            "scope": scope or "merchant",
            "merchantIds": [str(item) for item in (merchant_ids or []) if str(item or "").strip()],
            "industryTags": [str(item) for item in (industry_tags or []) if str(item or "").strip()],
        }
        target["grayRelease"] = {
            "enabled": True,
            "stage": "installed",
            "trafficPercent": max(0, min(100, int(traffic_percent or 0))),
            "abortConditions": ["skill_eval_failed", "manual_disable"],
        }
        target["status"] = "active"
        target["updatedAt"] = datetime.utcnow().isoformat() + "Z"
        self._write_registry(registry)
        return {"success": True, "skill": target}

    def _load_registry(self) -> List[Dict[str, Any]]:
        payload = _read_json(self.registry_path, {"items": []})
        items = payload.get("items") if isinstance(payload, dict) else payload
        return items if isinstance(items, list) else []

    def _write_registry(self, items: List[Dict[str, Any]]) -> None:
        _write_json(self.registry_path, {"items": items[-200:]})

    def _discover_builtin_skills(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if not self.skill_root.exists():
            return items
        for path in sorted(self.skill_root.iterdir()):
            if not path.is_dir() or not (path / "SKILL.md").exists():
                continue
            items.append(self._builtin_market_item(path.name))
        return items

    def _builtin_market_item(self, skill_name: str) -> Dict[str, Any]:
        return {
            "skillName": skill_name,
            "displayName": skill_name.replace("_", " "),
            "version": "builtin",
            "status": "available",
            "callable": True,
            "sourcePath": str(self.skill_root / skill_name / "SKILL.md"),
            "installScope": {"scope": "global_sop", "merchantIds": [], "industryTags": []},
            "grayRelease": {"enabled": False, "stage": "available", "trafficPercent": 100, "abortConditions": []},
            "versions": [{"version": "builtin", "publishedAt": "", "sourceDraftId": ""}],
            "runtimeStats": {"runCount": 0, "lastRunAt": "", "failureCount": 0},
        }

    def _eligible(self, state: Dict[str, Any]) -> bool:
        if not bool(state.get("evidence_graph_verified")):
            return False
        run_result = state.get("agent_run_result") or AgentRunResult()
        if not getattr(getattr(run_result, "verified_evidence", None), "passed", False):
            return False
        plan = state.get("plan") or QueryPlan()
        intent_count = len(getattr(plan, "intents", []) or [])
        fast = state.get("fast_understanding")
        complexity = str(getattr(fast, "complexity", "") or "")
        intent_kind = str(getattr(fast, "intent_kind", "") or "")
        complex_run = (
            intent_count > 1
            or bool(getattr(plan, "dependencies", []) or [])
            or complexity in {"medium", "complex"}
            or intent_kind in {"analysis", "multi_hop", "rule_data_mix"}
            or bool(state.get("analysis_summary"))
            or bool(state.get("analysis_skill_trace"))
        )
        if not complex_run:
            return False
        if getattr(run_result, "evidence_gaps", []) or not getattr(run_result, "task_results", []):
            return False
        return True

    def _build_draft(self, state: Dict[str, Any]) -> SkillDraft:
        plan = state.get("plan") or QueryPlan()
        run_result = state.get("agent_run_result") or AgentRunResult()
        question = str(state.get("question") or "")
        metrics = [str(getattr(intent, "metric_name", "") or "") for intent in plan.intents if getattr(intent, "metric_name", "")]
        categories = [
            category_display(getattr(intent, "category", ""))
            for intent in plan.intents
            if str(getattr(intent, "category", "") or "")
        ]
        metric_label = ", ".join(dict.fromkeys(metrics[:4])) or "verified metrics"
        title = "复用分析流程：%s" % metric_label
        draft_id = "skilldraft_" + uuid.uuid4().hex[:12]
        task_steps = [
            "读取主 Agent 已验证的 QueryGraph 结果和证据报告",
            "按指标、时间窗口和对象维度整理可复用分析输入",
            "只基于 verified evidence 生成分析结论、限制和后续建议",
            "将输出写入独立 Skill artifact，供 AnswerAgent 消费",
        ]
        return SkillDraft(
            draft_id=draft_id,
            status="pending_review",
            callable=False,
            source_thread_id=str(state.get("thread_id") or ""),
            source_run_id=str(state.get("run_id") or ""),
            source_qa_id=str(state.get("qa_id") or ""),
            merchant_id=str(getattr(state.get("merchant"), "merchant_id", "") or ""),
            title=title,
            description="从一次已成功执行的复杂经营分析沉淀出来的候选 Skill，审核前不可被主 Agent 调用。",
            applicability=[
                "问题类型与本轮类似：%s" % (question[:80] or "复杂经营分析"),
                "业务域包含：%s" % (", ".join(dict.fromkeys(categories[:4])) or "已验证业务域"),
                "指标包含：%s" % metric_label,
            ],
            required_inputs=[
                "QueryGraph 已执行完成",
                "EvidenceVerifier 已通过",
                "输入包含已验证 SQL 结果、证据覆盖和必要口径说明",
            ],
            steps=task_steps,
            tools=["semantic_read", "artifact_read", "artifact_grep", "write_checkpoint", "write_skill_output"],
            hard_constraints=[
                "不能新增查询或扩大商家、时间范围、对象范围",
                "不能使用未通过证据校验的数据",
                "不能覆盖语义层正式指标口径",
            ],
            evidence_requirements=[
                "至少一个成功 task_result",
                "verifiedEvidence.passed=true",
                "无 blocking evidence gap",
            ],
            example_questions=[question] if question else [],
            source_artifacts={
                "skillTrace": state.get("analysis_skill_trace") or {},
                "taskCount": len(getattr(run_result, "task_results", []) or []),
                "rowCount": len(getattr(getattr(run_result, "merged_query_bundle", None), "rows", []) or []),
            },
            created_at=datetime.utcnow().isoformat() + "Z",
        )

    def _publish_skill(self, draft: Dict[str, Any]) -> str:
        title = str(draft.get("title") or "generated_skill")
        skill_name = "draft_%s_%s" % (_slug(title), uuid.uuid4().hex[:6])
        target = self.skill_root / skill_name
        target.mkdir(parents=True, exist_ok=True)
        content = self._skill_markdown(skill_name, draft)
        (target / "SKILL.md").write_text(content, encoding="utf-8")
        return skill_name

    def _skill_markdown(self, skill_name: str, draft: Dict[str, Any]) -> str:
        description = str(draft.get("description") or "")
        applicability = "\n".join("- %s" % item for item in draft.get("applicability", []) or [])
        required_inputs = "\n".join("- %s" % item for item in draft.get("requiredInputs", []) or [])
        constraints = "\n".join("- %s" % item for item in draft.get("hardConstraints", []) or [])
        steps = "\n".join("%d. %s" % (index + 1, item) for index, item in enumerate(draft.get("steps", []) or []))
        evidence = "\n".join("- %s" % item for item in draft.get("evidenceRequirements", []) or [])
        return """---
name: %s
description: %s
whenToUse: %s
requiredInputs: %s
constraints: %s
---

# %s

## When To Use

%s

## Required Inputs

%s

## Constraints

%s

## Workflow

%s

## Evidence Requirements

%s
""" % (
            skill_name,
            description.replace("\n", " ")[:500],
            "; ".join(draft.get("applicability", []) or [])[:500],
            "; ".join(draft.get("requiredInputs", []) or [])[:500],
            "; ".join(draft.get("hardConstraints", []) or [])[:500],
            skill_name,
            applicability or "- 与审核通过的候选 Skill 场景一致",
            required_inputs or "- 已验证证据包",
            constraints or "- 只使用已验证证据",
            steps or "1. 读取已验证证据\n2. 输出结构化分析",
            evidence or "- EvidenceVerifier 已通过",
        )
