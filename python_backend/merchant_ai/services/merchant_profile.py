from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from merchant_ai.config import Settings
from merchant_ai.models import FastUnderstandingResult, MerchantInfo, RouteSlots


def _clean_list(values: Any, limit: int = 8) -> List[str]:
    result: List[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _model_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True)
    if hasattr(value, "dict"):
        return value.dict(by_alias=True)
    if isinstance(value, dict):
        return dict(value)
    return {}


class MerchantProfileSummaryService:
    """Build a compact, answer-safe merchant profile summary for BI governance."""

    def summarize(
        self,
        *,
        merchant: MerchantInfo,
        memory_injection: Dict[str, Any],
        memory_constraints: List[Dict[str, Any]],
        route_slots: RouteSlots,
        fast_understanding: FastUnderstandingResult,
    ) -> Dict[str, Any]:
        core_memory = memory_injection.get("coreMemory") if isinstance(memory_injection, dict) else {}
        core_memory = core_memory if isinstance(core_memory, dict) else {}
        recent_focus = core_memory.get("recentFocus") if isinstance(core_memory.get("recentFocus"), dict) else {}
        route_payload = _model_dict(route_slots)
        fast_payload = _model_dict(fast_understanding)
        time_window = route_payload.get("timeWindow") or route_payload.get("time_window") or {}
        if not isinstance(time_window, dict):
            time_window = {}
        required_constraints = [
            item
            for item in memory_constraints or []
            if str(item.get("enforcement") or "") == "required"
        ]
        clarify_constraints = [
            item
            for item in memory_constraints or []
            if str(item.get("enforcement") or "") == "clarify_or_disclose"
        ]
        preferred_metrics = _clean_list(
            route_payload.get("metricFocus")
            or route_payload.get("analysisSignals")
            or route_payload.get("analysis_signals")
            or fast_payload.get("metricFocus")
            or fast_payload.get("metricPhrases")
            or fast_payload.get("metric_phrases")
            or [],
            limit=10,
        )
        if not preferred_metrics:
            preferred_metrics = _clean_list(recent_focus.get("topMetrics"), limit=10)
        topic_candidates = route_payload.get("topicCandidates") or route_payload.get("topic_candidates") or []
        candidate_topics = [
            str((item or {}).get("topic") or "")
            for item in topic_candidates
            if isinstance(item, dict)
        ]
        focus_categories = _clean_list(route_payload.get("categories") or fast_payload.get("businessCategories") or candidate_topics, limit=8)
        if not focus_categories:
            focus_categories = _clean_list(recent_focus.get("topCategories"), limit=8)
        confirmed_rules = []
        for item in required_constraints[:6]:
            confirmed_rules.append(
                {
                    "id": str(item.get("id") or ""),
                    "type": str(item.get("type") or ""),
                    "instruction": str(item.get("instruction") or "")[:240],
                    "targetMetrics": _clean_list(item.get("targetMetrics"), limit=8),
                    "source": str(item.get("source") or ""),
                }
            )
        recent_risks = self._recent_risks(
            recent_focus=recent_focus,
            preferred_metrics=preferred_metrics,
            focus_categories=focus_categories,
            constraints=memory_constraints,
        )
        time_window_days = int(route_payload.get("timeWindowDays") or time_window.get("days") or 0)
        return {
            "merchantId": merchant.merchant_id,
            "merchantName": merchant.merchant_name or merchant.company_name,
            "defaultTimeWindow": time_window_days or 7,
            "defaultTimeWindowDays": time_window_days,
            "preferredMetrics": preferred_metrics,
            "businessFocus": focus_categories,
            "recentRisks": recent_risks,
            "recentFocusPattern": str(recent_focus.get("focusPattern") or ""),
            "confirmedRules": confirmed_rules,
            "confirmedRuleTexts": [item["instruction"] for item in confirmed_rules if item.get("instruction")],
            "disclosureRequiredCount": len(clarify_constraints),
            "source": {
                "merchantProfile": "merchant_service",
                "memory": str(memory_injection.get("source") or "") if isinstance(memory_injection, dict) else "",
                "constraints": len(memory_constraints or []),
            },
        }

    def _recent_risks(
        self,
        *,
        recent_focus: Dict[str, Any],
        preferred_metrics: List[str],
        focus_categories: List[str],
        constraints: List[Dict[str, Any]],
    ) -> List[str]:
        candidates: List[str] = []
        pattern = str(recent_focus.get("focusPattern") or recent_focus.get("focus_pattern") or "")
        for keyword in ["退款率升高", "GMV下滑", "GMV 下降", "订单下降", "工单升高", "客诉升高", "履约异常", "赔付升高"]:
            if keyword.replace(" ", "") in pattern.replace(" ", ""):
                candidates.append(keyword.replace("GMV 下降", "GMV下滑"))
        metric_text = " ".join(preferred_metrics)
        focus_text = " ".join(focus_categories)
        if "退款" in metric_text or "退款" in focus_text:
            candidates.append("退款率升高")
        if "工单" in metric_text or "客诉" in metric_text or "客服" in focus_text:
            candidates.append("客诉/工单升高")
        if "履约" in focus_text or "发货" in focus_text:
            candidates.append("履约异常")
        for item in constraints or []:
            text = str(item.get("instruction") or "")
            if "风险" in text or "异常" in text or "升高" in text or "下降" in text:
                candidates.append(text[:80])
        return _clean_list(candidates, limit=6)


class MerchantProfileStore:
    """Persist merchant preferences and reviewed business profile facts."""

    def __init__(self, settings: Settings, root: Path | None = None):
        self.settings = settings
        self.root = root or (settings.resolved_workspace_path / "ops" / "merchant_profiles")
        self.index_path = self.root / "merchant_profiles.json"

    def get_profile(self, merchant_id: str, include_expired: bool = False) -> Dict[str, Any]:
        profiles = self._load()
        profile = dict(profiles.get(str(merchant_id or "").strip()) or {})
        if not profile:
            profile = self._default_profile(merchant_id)
        if not include_expired:
            profile = self._without_expired(profile)
        return profile

    def upsert_profile(self, merchant_id: str, patch: Dict[str, Any], reviewer: str = "", review_status: str = "reviewed") -> Dict[str, Any]:
        merchant_id = str(merchant_id or "").strip() or self.settings.merchant_id
        profiles = self._load()
        current = dict(profiles.get(merchant_id) or self._default_profile(merchant_id))
        next_profile = self._merge_profile(current, patch or {})
        now = datetime.utcnow().isoformat() + "Z"
        next_profile["merchantId"] = merchant_id
        next_profile["updatedAt"] = now
        next_profile["reviewStatus"] = review_status or next_profile.get("reviewStatus") or "reviewed"
        if reviewer:
            next_profile["reviewer"] = reviewer
            next_profile["reviewedAt"] = now
        history = list(next_profile.get("history") or [])
        history.append(
            {
                "at": now,
                "reviewer": reviewer,
                "reviewStatus": next_profile["reviewStatus"],
                "changedFields": sorted((patch or {}).keys()),
            }
        )
        next_profile["history"] = history[-30:]
        profiles[merchant_id] = next_profile
        self._write(profiles)
        return next_profile

    def merge_runtime_summary(self, merchant_id: str, summary: Dict[str, Any]) -> Dict[str, Any]:
        profile = self.get_profile(merchant_id)
        merged = dict(summary or {})
        merged["profileStore"] = {
            "enabled": True,
            "reviewStatus": profile.get("reviewStatus", ""),
            "updatedAt": profile.get("updatedAt", ""),
            "validUntil": profile.get("validUntil", ""),
        }
        merged["defaultTimeWindow"] = int(profile.get("defaultTimeWindow") or merged.get("defaultTimeWindow") or 7)
        merged["preferredMetrics"] = _clean_list([*(profile.get("preferredMetrics") or []), *(merged.get("preferredMetrics") or [])], limit=12)
        merged["confirmedRules"] = self._merge_rules(profile.get("confirmedRules") or [], merged.get("confirmedRules") or [])
        merged["confirmedRuleTexts"] = _clean_list(
            [*(profile.get("confirmedRuleTexts") or []), *(merged.get("confirmedRuleTexts") or [])],
            limit=12,
        )
        merged["recentRisks"] = _clean_list([*(profile.get("recentRisks") or []), *(merged.get("recentRisks") or [])], limit=10)
        merged["businessFocus"] = _clean_list([*(profile.get("businessFocus") or []), *(merged.get("businessFocus") or [])], limit=10)
        merged["industryTags"] = _clean_list(profile.get("industryTags") or [], limit=8)
        return merged

    def review_profile(self, merchant_id: str, approved: bool, reviewer: str = "", note: str = "") -> Dict[str, Any]:
        status = "approved" if approved else "rejected"
        return self.upsert_profile(
            merchant_id,
            {"reviewNote": note, "validUntil": (datetime.utcnow() + timedelta(days=180)).date().isoformat() if approved else ""},
            reviewer=reviewer,
            review_status=status,
        )

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if not self.index_path.exists():
            return {}
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            items = payload.get("profiles") if isinstance(payload, dict) else payload
            return items if isinstance(items, dict) else {}
        except Exception:
            return {}

    def _write(self, profiles: Dict[str, Dict[str, Any]]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps({"profiles": profiles}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def _default_profile(self, merchant_id: str) -> Dict[str, Any]:
        return {
            "merchantId": str(merchant_id or self.settings.merchant_id),
            "defaultTimeWindow": 7,
            "preferredMetrics": [],
            "confirmedRules": [],
            "confirmedRuleTexts": [],
            "recentRisks": [],
            "businessFocus": [],
            "industryTags": [],
            "reviewStatus": "draft",
            "validUntil": "",
            "createdAt": datetime.utcnow().isoformat() + "Z",
            "updatedAt": "",
            "history": [],
        }

    def _without_expired(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        valid_until = str(profile.get("validUntil") or "")
        if valid_until and valid_until < datetime.utcnow().date().isoformat():
            next_profile = dict(profile)
            next_profile["reviewStatus"] = "expired"
            next_profile["confirmedRules"] = []
            next_profile["confirmedRuleTexts"] = []
            return next_profile
        return profile

    def _merge_profile(self, current: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(current)
        for key in ["defaultTimeWindow", "reviewNote", "validUntil"]:
            if key in patch:
                result[key] = patch.get(key)
        for key in ["preferredMetrics", "confirmedRuleTexts", "recentRisks", "businessFocus", "industryTags"]:
            if key in patch:
                result[key] = _clean_list(patch.get(key), limit=20)
        if "confirmedRules" in patch:
            result["confirmedRules"] = self._merge_rules([], patch.get("confirmedRules") or [])
        return result

    def _merge_rules(self, first: List[Any], second: List[Any]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        seen = set()
        for item in [*(first or []), *(second or [])]:
            if isinstance(item, str):
                rule = {"instruction": item}
            elif isinstance(item, dict):
                rule = dict(item)
            else:
                continue
            key = str(rule.get("id") or rule.get("instruction") or "")[:160]
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(rule)
            if len(result) >= 12:
                break
        return result
