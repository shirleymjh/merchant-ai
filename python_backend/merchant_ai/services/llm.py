from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from merchant_ai.config import Settings


class LlmClient:
    """Small LangChain wrapper with local fallback behavior."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self.last_error = ""
        self.error_events: List[str] = []

    @property
    def configured(self) -> bool:
        return bool(self.settings.openai_api_key)

    def _chat_model(self):
        if self._model is not None:
            return self._model
        if not self.configured:
            return None
        try:
            from langchain_openai import ChatOpenAI

            self._model = ChatOpenAI(
                model=self.settings.openai_model,
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url.rstrip("/"),
                temperature=0,
                timeout=self.settings.llm_request_timeout_seconds,
                max_tokens=self.settings.llm_max_tokens,
            )
        except Exception:
            self.record_error("provider_error: failed to initialize chat model")
            self._model = None
        return self._model

    def record_error(self, error: str) -> None:
        self.last_error = error
        self.error_events.append(error)

    def chat(self, system_prompt: str, user_prompt: str, fallback: str = "", timeout_seconds: Optional[int] = None) -> str:
        model = self._chat_model()
        if model is None:
            return fallback
        try:
            self.last_error = ""
            from langchain_core.messages import HumanMessage, SystemMessage

            result = self._invoke_with_timeout(model, [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)], timeout_seconds)
            if result is None:
                return fallback
            content = getattr(result, "content", "")
            if isinstance(content, list):
                return "\n".join(str(part) for part in content) or fallback
            return str(content or fallback)
        except Exception as exc:
            self.record_error("provider_error: %s" % str(exc)[:300])
            return fallback

    def tool_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: List[Dict[str, Any]],
        fallback: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None,
        tool_choice: Optional[str] = None,
    ) -> Dict[str, Any]:
        model = self._chat_model()
        if model is None:
            return fallback or {"content": "", "toolCalls": []}
        try:
            self.last_error = ""
            from langchain_core.messages import HumanMessage, SystemMessage

            tool_model = self._bind_tools(model, tools, tool_choice)
            result = self._invoke_with_timeout(tool_model, [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)], timeout_seconds)
            if result is None:
                return fallback or {"content": "", "toolCalls": []}
            return {
                "content": self._message_content(result),
                "toolCalls": self._normalize_tool_calls(result),
            }
        except Exception as exc:
            self.record_error("provider_error: %s" % str(exc)[:300])
            return fallback or {"content": "", "toolCalls": []}

    def tool_json_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        tool: Dict[str, Any],
        fallback: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        tool_name = str(((tool or {}).get("function") or {}).get("name") or "")
        result = self.tool_chat(
            system_prompt,
            user_prompt,
            [tool],
            {"content": "", "toolCalls": []},
            timeout_seconds=timeout_seconds,
            tool_choice=tool_name or None,
        )
        for call in result.get("toolCalls") or []:
            if not tool_name or call.get("name") == tool_name:
                args = call.get("args")
                if isinstance(args, dict):
                    return args
        content = str(result.get("content") or "")
        if not content:
            return fallback or {}
        parsed = self._parse_json_text(content)
        return parsed if parsed else (fallback or {})

    def _bind_tools(self, model: Any, tools: List[Dict[str, Any]], tool_choice: Optional[str]) -> Any:
        if not tools or not hasattr(model, "bind_tools"):
            return model
        try:
            if tool_choice:
                return model.bind_tools(tools, tool_choice=tool_choice)
            return model.bind_tools(tools)
        except TypeError:
            return model.bind_tools(tools)
        except Exception as exc:
            self.record_error("provider_error: tool binding failed: %s" % str(exc)[:240])
            return model

    def _message_content(self, result: Any) -> str:
        content = getattr(result, "content", "")
        if isinstance(content, list):
            return "\n".join(str(part) for part in content)
        return str(content or "")

    def _normalize_tool_calls(self, result: Any) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        seen: set[str] = set()
        tool_calls = getattr(result, "tool_calls", None) or []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            args = call.get("args") or {}
            if isinstance(args, str):
                args = self._parse_json_text(args)
            item = {
                "id": str(call.get("id") or ""),
                "name": str(call.get("name") or ""),
                "args": args if isinstance(args, dict) else {},
            }
            fingerprint = tool_call_fingerprint(item)
            if fingerprint not in seen:
                seen.add(fingerprint)
                normalized.append(item)
        raw_calls = (getattr(result, "additional_kwargs", {}) or {}).get("tool_calls") or []
        for call in raw_calls:
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            if not name:
                continue
            args = self._parse_json_text(str(function.get("arguments") or "{}"))
            item = {"id": str(call.get("id") or ""), "name": name, "args": args}
            fingerprint = tool_call_fingerprint(item)
            if fingerprint not in seen:
                seen.add(fingerprint)
                normalized.append(item)
        return normalized

    def _invoke_with_timeout(self, model: Any, messages: List[Any], timeout_seconds: Optional[int] = None) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._ainvoke_with_timeout(model, messages, timeout_seconds))
        self.record_error("provider_error: sync LLM call cannot run inside an active event loop")
        return None

    async def _ainvoke_with_timeout(self, model: Any, messages: List[Any], timeout_seconds: Optional[int] = None) -> Any:
        timeout = max(1, int(timeout_seconds or self.settings.llm_request_timeout_seconds or 1))
        try:
            return await asyncio.wait_for(model.ainvoke(messages), timeout=timeout)
        except asyncio.TimeoutError:
            self.record_error("timeout: provider call exceeded %s seconds" % timeout)
            return None

    def json_chat(self, system_prompt: str, user_prompt: str, fallback: Optional[Dict[str, Any]] = None, timeout_seconds: Optional[int] = None) -> Dict[str, Any]:
        text = self.chat(system_prompt, user_prompt, "", timeout_seconds=timeout_seconds)
        if not text:
            if self.configured and not self.last_error:
                self.record_error("empty_response: provider returned no content")
            return fallback or {}
        parsed = self._parse_json_text(text)
        if parsed:
            return parsed
        return fallback or {}

    def _parse_json_text(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                self.record_error("json_parse_error: %s" % text[:300])
                return {}
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                self.record_error("json_parse_error: %s" % str(exc)[:300])
                return {}


def tool_call_fingerprint(item: Dict[str, Any]) -> str:
    try:
        args = json.dumps(item.get("args") or {}, ensure_ascii=False, sort_keys=True)
    except Exception:
        args = str(item.get("args") or {})
    return "%s:%s:%s" % (item.get("id") or "", item.get("name") or "", args)
