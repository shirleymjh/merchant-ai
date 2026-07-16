from __future__ import annotations

from typing import Any, Dict, List, Tuple

from merchant_ai.graph.query_graph_contract import graph_validation_passed
from merchant_ai.graph.state import AgentState
from merchant_ai.models import AgentAction


def action_prerequisite_gaps(
    state: AgentState,
    action: AgentAction,
) -> Tuple[List[str], List[str]]:
    """Return missing declarative prerequisites without choosing another action."""

    missing_keys = [key for key in action.required_state_keys if not state_path_ready(state, key)]
    missing_flags = [flag for flag in action.required_state_flags if not action_state_flag_ready(state, flag)]
    return missing_keys, missing_flags


def action_state_flag_ready(state: AgentState, flag: str) -> bool:
    if flag == "query_graph_validation_passed":
        return graph_validation_passed(state)
    return bool(state.get(flag))


def state_path_value(state: AgentState, path: str) -> Any:
    value: Any = state
    for part in [item for item in str(path or "").split(".") if item]:
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = getattr(value, part, None)
        if value is None:
            break
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True)
    return value


def state_path_ready(state: AgentState, path: str) -> bool:
    value = state_path_value(state, path)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def contract_block_observation(
    action: AgentAction,
    missing_keys: List[str],
    missing_flags: List[str],
    *,
    reason: str = "",
    source: str = "action_contract",
) -> Dict[str, Any]:
    """Build a neutral observation; it deliberately contains no next action."""

    return {
        "status": "pending",
        "source": source,
        "blockedAction": action.id,
        "blockedNode": action.node,
        "missingStateKeys": list(missing_keys),
        "missingStateFlags": list(missing_flags),
        "decisionReason": str(reason or "")[:500],
    }
