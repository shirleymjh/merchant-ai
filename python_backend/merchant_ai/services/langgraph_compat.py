from __future__ import annotations

import warnings
from typing import Any

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
from langchain_core.load.load import Reviver


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"The default value of `allowed_objects` will change.*",
        category=LangChainPendingDeprecationWarning,
    )
    import langgraph.checkpoint.serde.jsonplus as _jsonplus
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph


_jsonplus.LC_REVIVER = Reviver(allowed_objects="core")


def sqlite_saver_class() -> Any:
    from langgraph.checkpoint.sqlite import SqliteSaver

    return SqliteSaver


def postgres_saver_class() -> Any:
    from langgraph.checkpoint.postgres import PostgresSaver

    return PostgresSaver


__all__ = [
    "END",
    "START",
    "MemorySaver",
    "StateGraph",
    "postgres_saver_class",
    "sqlite_saver_class",
]
