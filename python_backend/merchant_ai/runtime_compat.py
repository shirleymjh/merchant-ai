"""Fail-fast checks for the Deep Agents/LangChain runtime contract.

This module intentionally has no LangChain imports.  It can therefore explain
an incompatible environment before the application imports either the legacy
domain graph or the Deep Agent harness.
"""

from __future__ import annotations

import json
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Dict, Iterable, Tuple

from merchant_ai.services.text_parsing import ASCII_DIGITS


MIN_PYTHON = (3, 11)
MAX_PYTHON_MAJOR = 3
PACKAGE_CONTRACT: Dict[str, Tuple[Tuple[int, ...], int]] = {
    "deepagents": ((0, 6, 12), 0),
    "langchain": ((1, 3, 11), 1),
    "langchain-core": ((1, 4, 8), 1),
    "langgraph": ((1, 0, 0), 1),
    "langgraph-checkpoint-sqlite": ((3, 0, 0), 3),
    "langgraph-checkpoint-postgres": ((3, 0, 0), 3),
}


class RuntimeCompatibilityError(RuntimeError):
    """Raised when the process is not running the supported dependency set."""


def _numeric_version(raw: str) -> Tuple[int, ...]:
    text = str(raw or "").strip()
    cursor = 0
    while cursor < len(text) and (text[cursor] in ASCII_DIGITS or text[cursor] == "."):
        cursor += 1
    prefix = text[:cursor].strip(".")
    parts = prefix.split(".") if prefix else []
    if not parts or any(not part or any(character not in ASCII_DIGITS for character in part) for part in parts):
        return ()
    return tuple(int(part) for part in parts)


def _format_version(parts: Iterable[int]) -> str:
    return ".".join(str(part) for part in parts)


def runtime_report() -> Dict[str, object]:
    packages: Dict[str, str] = {}
    errors = []
    python_version = tuple(sys.version_info[:3])
    if python_version < MIN_PYTHON or python_version[0] > MAX_PYTHON_MAJOR:
        errors.append(
            "Python %s is unsupported; use Python >=%s,<4.0"
            % (_format_version(python_version), _format_version(MIN_PYTHON))
        )
    for package, (minimum, supported_major) in PACKAGE_CONTRACT.items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            errors.append("missing package: %s" % package)
            continue
        packages[package] = installed
        parsed = _numeric_version(installed)
        if parsed < minimum:
            errors.append(
                "%s %s is too old; require >=%s"
                % (package, installed, _format_version(minimum))
            )
        elif parsed and parsed[0] != supported_major:
            errors.append(
                "%s %s is outside the tested major version %d"
                % (package, installed, supported_major)
            )
    return {
        "compatible": not errors,
        "python": _format_version(python_version),
        "packages": packages,
        "errors": errors,
    }


def assert_runtime_compatibility() -> Dict[str, object]:
    report = runtime_report()
    if report["errors"]:
        raise RuntimeCompatibilityError("; ".join(report["errors"]))
    return report


def main() -> int:
    report = runtime_report()
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
