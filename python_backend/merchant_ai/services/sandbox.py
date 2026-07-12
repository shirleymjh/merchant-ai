from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

from merchant_ai.config import Settings


@dataclass
class SandboxResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class MerchantAnalysisSandbox:
    """Run reviewed merchant-analysis scripts without exposing a general shell."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.skill_root = (settings.resources_root / "runtime" / "agent_skills").resolve()

    def run_python(self, script: Path, args: List[str], workspace: Path, timeout_seconds: int) -> SandboxResult:
        script_path = script.resolve()
        workspace_path = workspace.resolve()
        if not self._is_within(script_path, self.skill_root):
            return SandboxResult(126, stderr="SANDBOX_SCRIPT_NOT_APPROVED")
        if not script_path.is_file() or script_path.suffix != ".py":
            return SandboxResult(126, stderr="SANDBOX_SCRIPT_INVALID")
        workspace_path.mkdir(parents=True, exist_ok=True)
        for value in args:
            if value.startswith("-"):
                continue
            candidate = Path(value)
            if candidate.is_absolute() and not self._is_within(candidate.resolve(), workspace_path):
                return SandboxResult(126, stderr="SANDBOX_PATH_OUTSIDE_WORKSPACE")
        if str(getattr(self.settings, "sandbox_backend", "local") or "local").lower() in {"container", "docker", "podman"}:
            return self._run_container(script_path, args, workspace_path, timeout_seconds)
        return self._run_local(script_path, args, workspace_path, timeout_seconds)

    def _run_local(self, script_path: Path, args: List[str], workspace_path: Path, timeout_seconds: int) -> SandboxResult:
        command = [self.settings.python_executable, "-I", str(script_path), *args]
        env = {
            "PATH": os.environ.get("PATH", ""),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "PYTHONIOENCODING": "utf-8",
            "MERCHANT_ANALYSIS_SANDBOX": "1",
        }
        try:
            completed = subprocess.run(
                command,
                cwd=str(workspace_path),
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=max(1, int(timeout_seconds or 1)),
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(124, stdout=str(exc.stdout or "")[-4000:], stderr="SANDBOX_TIMEOUT")
        except Exception as exc:
            return SandboxResult(125, stderr="SANDBOX_ERROR: %s" % str(exc)[:500])
        return SandboxResult(
            completed.returncode,
            stdout=str(completed.stdout or "")[-8000:],
            stderr=str(completed.stderr or "")[-8000:],
        )

    def _run_container(self, script_path: Path, args: List[str], workspace_path: Path, timeout_seconds: int) -> SandboxResult:
        runtime = str(getattr(self.settings, "sandbox_container_runtime", "docker") or "docker")
        if not shutil.which(runtime):
            return SandboxResult(125, stderr="SANDBOX_CONTAINER_RUNTIME_UNAVAILABLE")
        relative_script = script_path.relative_to(self.skill_root)
        mapped_args = [self._container_arg(value, workspace_path) for value in args]
        command = [
            runtime,
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--pids-limit",
            "128",
            "--cpus",
            str(max(0.1, float(getattr(self.settings, "sandbox_container_cpus", 1.0) or 1.0))),
            "--memory",
            str(getattr(self.settings, "sandbox_container_memory", "512m") or "512m"),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "-v",
            "%s:/opt/skills:ro" % self.skill_root,
            "-v",
            "%s:/workspace:rw" % workspace_path,
            "-w",
            "/workspace",
            str(getattr(self.settings, "sandbox_container_image", "python:3.11-slim") or "python:3.11-slim"),
            "python",
            "-I",
            "/opt/skills/%s" % relative_script.as_posix(),
            *mapped_args,
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=str(workspace_path),
                env={"PATH": os.environ.get("PATH", "")},
                check=False,
                capture_output=True,
                text=True,
                timeout=max(1, int(timeout_seconds or 1)),
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(124, stdout=str(exc.stdout or "")[-4000:], stderr="SANDBOX_TIMEOUT")
        except Exception as exc:
            return SandboxResult(125, stderr="SANDBOX_CONTAINER_ERROR: %s" % str(exc)[:500])
        return SandboxResult(completed.returncode, str(completed.stdout or "")[-8000:], str(completed.stderr or "")[-8000:])

    def _container_arg(self, value: str, workspace_path: Path) -> str:
        candidate = Path(value)
        if candidate.is_absolute() and self._is_within(candidate.resolve(), workspace_path):
            return "/workspace/%s" % candidate.resolve().relative_to(workspace_path).as_posix()
        return value

    def _is_within(self, path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False
