from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from merchant_ai.models import ConversationMessage
from merchant_ai.services.llm import LlmClient


MAX_SHORT_TERM_MESSAGES = 40
MAX_SHORT_TERM_RECENT_MESSAGES = 6
MAX_SHORT_TERM_MESSAGE_CHARS = 1200
MAX_SHORT_TERM_CONTEXT_CHARS = 8000
MAX_SHORT_TERM_SUMMARY_CHARS = 1800


def normalize_message_history(messages: Optional[List[Any]]) -> List[ConversationMessage]:
    normalized: List[ConversationMessage] = []
    for item in list(messages or [])[-MAX_SHORT_TERM_MESSAGES:]:
        try:
            message = item if isinstance(item, ConversationMessage) else ConversationMessage.model_validate(item)
        except Exception:
            continue
        role = str(message.role or "").strip().lower()
        text = str(message.text or "").strip()
        if role not in {"user", "assistant", "system", "tool"} or not text:
            continue
        normalized.append(
            message.model_copy(
                update={
                    "role": role,
                    "text": text[:MAX_SHORT_TERM_MESSAGE_CHARS],
                }
            )
        )
    return normalized


def render_recent_message_history_context(messages: List[ConversationMessage]) -> str:
    if not messages:
        return ""
    lines = [
        "## 当前会话短期记忆",
        "以下是当前上下文窗口内的最近多轮 messages 原文片段，用于指代消解、连续任务和未完成事项承接。",
    ]
    for index, message in enumerate(messages[-MAX_SHORT_TERM_RECENT_MESSAGES:], start=1):
        role = {"user": "用户", "assistant": "助手", "system": "系统", "tool": "工具"}.get(message.role, message.role or "unknown")
        text = re.sub(r"\s+", " ", str(message.text or "")).strip()
        if text:
            lines.append("- %02d %s：%s" % (index, role, text[:MAX_SHORT_TERM_MESSAGE_CHARS]))
    return "\n".join(lines)[-MAX_SHORT_TERM_CONTEXT_CHARS:]


def render_rule_based_message_summary(messages: List[ConversationMessage]) -> str:
    if not messages:
        return ""
    lines = [
        "## 旧会话压缩摘要",
        "模型摘要不可用时使用规则兜底，仅保留较早消息中的关键片段。",
    ]
    for index, message in enumerate(messages[-(MAX_SHORT_TERM_MESSAGES - MAX_SHORT_TERM_RECENT_MESSAGES) :], start=1):
        role = {"user": "用户", "assistant": "助手", "system": "系统", "tool": "工具"}.get(message.role, message.role or "unknown")
        text = re.sub(r"\s+", " ", str(message.text or "")).strip()
        if text:
            lines.append("- %02d %s：%s" % (index, role, text[:500]))
    return "\n".join(lines)[:MAX_SHORT_TERM_SUMMARY_CHARS]


def summarize_message_history_context(
    messages: List[ConversationMessage],
    question: str,
    llm: Optional[LlmClient] = None,
    timeout_seconds: int = 8,
) -> Dict[str, Any]:
    if len(messages) <= MAX_SHORT_TERM_RECENT_MESSAGES:
        return {"summary": "", "usedLlm": False, "sourceMessages": 0, "fallback": False}
    older_messages = messages[:-MAX_SHORT_TERM_RECENT_MESSAGES]
    fallback_summary = render_rule_based_message_summary(older_messages)
    if not llm or not getattr(llm, "configured", False) or not hasattr(llm, "chat"):
        return {"summary": fallback_summary, "usedLlm": False, "sourceMessages": len(older_messages), "fallback": bool(fallback_summary)}

    rows: List[str] = []
    for index, message in enumerate(older_messages[-(MAX_SHORT_TERM_MESSAGES - MAX_SHORT_TERM_RECENT_MESSAGES) :], start=1):
        role = {"user": "用户", "assistant": "助手", "system": "系统", "tool": "工具"}.get(message.role, message.role or "unknown")
        text = re.sub(r"\s+", " ", str(message.text or "")).strip()
        if text:
            rows.append("%02d %s：%s" % (index, role, text[:MAX_SHORT_TERM_MESSAGE_CHARS]))
    if not rows:
        return {"summary": "", "usedLlm": False, "sourceMessages": len(older_messages), "fallback": False}

    system_prompt = (
        "你是商家经营问答系统的会话记忆压缩器。"
        "请只抽取对当前追问仍有用的信息，不补充事实，不推断未出现的数据。"
        "输出中文，控制在 800 字以内。"
    )
    user_prompt = "\n".join(
        [
            "当前用户问题：%s" % (question or ""),
            "",
            "请从以下较早会话消息中提炼短期会话摘要，重点保留：",
            "1. 用户确认过的时间范围、筛选条件、指标口径；",
            "2. 关键对象、实体集合、订单号、商品 ID、退款单号等；",
            "3. 未完成任务、证据缺口、上一轮查询结果中可复用的对象；",
            "4. 用户明确纠正、偏好或冲突信息。",
            "不要复述无关寒暄、长 SQL、完整工具日志。",
            "",
            "较早会话消息：",
            "\n".join(rows),
            "",
            "输出格式：",
            "## 旧会话压缩摘要",
            "- 已确认约束：...",
            "- 关键对象：...",
            "- 未完成任务：...",
            "- 纠正和偏好：...",
        ]
    )
    summary = str(llm.chat(system_prompt, user_prompt, fallback="", timeout_seconds=timeout_seconds) or "").strip()
    if not summary:
        return {"summary": fallback_summary, "usedLlm": False, "sourceMessages": len(older_messages), "fallback": bool(fallback_summary)}
    if "旧会话压缩摘要" not in summary[:80]:
        summary = "## 旧会话压缩摘要\n" + summary
    return {
        "summary": summary[:MAX_SHORT_TERM_SUMMARY_CHARS],
        "usedLlm": True,
        "sourceMessages": len(older_messages),
        "fallback": False,
    }


def render_message_history_context(messages: List[ConversationMessage], question: str = "", llm: Optional[LlmClient] = None) -> Dict[str, Any]:
    summary_trace = summarize_message_history_context(messages, question, llm)
    sections = [summary_trace.get("summary") or "", render_recent_message_history_context(messages)]
    context = "\n\n".join(section.strip() for section in sections if section and section.strip())[-MAX_SHORT_TERM_CONTEXT_CHARS:]
    return {
        "context": context,
        "usedLlm": bool(summary_trace.get("usedLlm")),
        "fallback": bool(summary_trace.get("fallback")),
        "summarySourceMessages": int(summary_trace.get("sourceMessages") or 0),
        "recentMessages": min(len(messages), MAX_SHORT_TERM_RECENT_MESSAGES),
    }


def append_context_section(existing: str, section: str, max_chars: int = MAX_SHORT_TERM_CONTEXT_CHARS) -> str:
    parts = [part for part in [existing.strip(), section.strip()] if part]
    return "\n\n".join(parts)[-max_chars:]


def compact_file_tool_results_for_prompt(results: List[Dict[str, Any]], max_items: int = 6) -> List[Dict[str, Any]]:
    compacted: List[Dict[str, Any]] = []
    for item in results[-max_items:]:
        result = item.get("result") or {}
        payload = {
            "name": item.get("name"),
            "status": item.get("status"),
            "errorType": item.get("errorType"),
            "errorMessage": item.get("errorMessage"),
        }
        if isinstance(result, dict):
            for key in ["relativePath", "merchantUri", "truncated", "estimatedChars", "nextContentOffsetChars"]:
                if key in result:
                    payload[key] = result.get(key)
            if "content" in result:
                payload["content"] = str(result.get("content") or "")[:1800]
            if "items" in result and isinstance(result.get("items"), list):
                payload["items"] = result.get("items")[:8]
            if "hits" in result and isinstance(result.get("hits"), list):
                payload["hits"] = result.get("hits")[:8]
        compacted.append(payload)
    return compacted
