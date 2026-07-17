from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from merchant_ai.config import Settings


SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SemanticRuntimeBindingRegistry:
    """Resolve infrastructure roles declared by published semantic assets."""

    def __init__(self, settings: Settings):
        self.root = settings.resolved_topic_path

    def resolve(self, role: str) -> Dict[str, Any]:
        matches = [item for item in self.bindings() if str(item.get("role") or "") == str(role or "")]
        return matches[0] if len(matches) == 1 else {}

    def bindings(self) -> List[Dict[str, Any]]:
        bindings: List[Dict[str, Any]] = []
        for path in sorted(Path(self.root).glob("*/tables/*/asset.json")):
            try:
                asset = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if str(asset.get("status") or "").upper() != "PUBLISHED":
                continue
            table = str(asset.get("tableName") or "")
            if not safe_identifier(table):
                continue
            for declared in asset.get("runtimeBindings") or []:
                if not isinstance(declared, dict):
                    continue
                binding = dict(declared)
                binding["table"] = table
                binding["sourceRef"] = str(path)
                identifiers = [
                    str(binding.get("lookupColumn") or ""),
                    str(binding.get("idColumn") or ""),
                    str(binding.get("asOfColumn") or ""),
                    *[str(item) for item in binding.get("displayColumns") or []],
                    *[str(item) for item in binding.get("contextColumns") or []],
                ]
                if binding.get("role") and all(safe_identifier(item) for item in identifiers if item):
                    bindings.append(binding)
        return bindings


def safe_identifier(value: str) -> bool:
    return bool(SAFE_IDENTIFIER.fullmatch(str(value or "")))
