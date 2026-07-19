import ast
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BLOCKED_MODULES = {"re", "regex", "re2"}


def _python_sources() -> list[Path]:
    sources: list[Path] = []
    for root in (REPOSITORY_ROOT / "python_backend", REPOSITORY_ROOT / "scripts"):
        for path in root.rglob("*.py"):
            if any(part.startswith(".venv") or part == "__pycache__" for part in path.parts):
                continue
            sources.append(path)
    return sorted(sources)


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _regex_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in BLOCKED_MODULES:
                    violations.append(f"line {node.lineno}: imports {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if str(node.module or "").split(".", 1)[0] in BLOCKED_MODULES:
                violations.append(f"line {node.lineno}: imports from {node.module}")
        elif isinstance(node, ast.Call):
            call_name = _call_name(node)
            if call_name in {"__import__", "import_module"} and node.args:
                imported = node.args[0]
                if (
                    isinstance(imported, ast.Constant)
                    and isinstance(imported.value, str)
                    and imported.value.split(".", 1)[0] in BLOCKED_MODULES
                ):
                    violations.append(f"line {node.lineno}: dynamically imports {imported.value}")
            for keyword in node.keywords:
                if call_name == "raises" and keyword.arg == "match":
                    violations.append(f"line {node.lineno}: pytest.raises(match=...) uses a regular expression")
                if call_name in {"Field", "StringConstraints", "constr"} and keyword.arg in {
                    "pattern",
                    "regex",
                }:
                    violations.append(
                        f"line {node.lineno}: {call_name}({keyword.arg}=...) uses a regular expression"
                    )
    return violations


def test_repository_python_has_no_regular_expression_dependency() -> None:
    failures: list[str] = []
    for path in _python_sources():
        for violation in _regex_violations(path):
            failures.append(f"{path.relative_to(REPOSITORY_ROOT)}: {violation}")

    assert not failures, "Regular expressions are forbidden:\n" + "\n".join(failures)
