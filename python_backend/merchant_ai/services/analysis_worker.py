from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from merchant_ai.models import AgentRunResult, MerchantInfo, QueryPlan
from merchant_ai.services.answer import AnswerComposeService, answer_data_package
from merchant_ai.services.llm import LlmClient


@dataclass
class AnalysisWorkerResult:
    answer: str
    trace: Dict[str, Any]


class AnalysisWorkerExecutor:
    """Run long-tail evidence-bound analysis in an isolated worker workspace."""

    def __init__(self, llm: LlmClient):
        self.llm = llm
        self.settings = llm.settings

    def execute(
        self,
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult,
        outputs_path: str = "",
        rule_context: str = "",
        merchant: MerchantInfo | None = None,
        personalization_context: Optional[Dict[str, Any]] = None,
        initial_trace: Optional[Dict[str, Any]] = None,
    ) -> AnalysisWorkerResult:
        started = time.monotonic()
        isolated_run_id = "analysis_%s" % uuid.uuid4().hex[:12]
        workspace = self._workspace(outputs_path, isolated_run_id)
        checkpoint_path = workspace / "analysis_checkpoint.json"
        input_path = workspace / "analysis_input.json"
        output_path = workspace / "analysis_output.json"
        incoming_trace = dict(initial_trace or {})
        trace: Dict[str, Any] = {
            **incoming_trace,
            "workerType": "ANALYSIS_WORKER",
            "subAgentType": "generic_analysis_worker",
            "executionMode": "isolated_analysis_worker",
            "isolatedExecution": True,
            "isolatedRunId": isolated_run_id,
            "workspacePath": str(workspace),
            "inputArtifact": str(input_path),
            "outputArtifact": str(output_path),
            "checkpointPath": str(checkpoint_path),
            "startedAt": datetime.now().isoformat(),
            "_startedMonotonic": started,
            "lifecycleStage": "matched",
            "progress": ["matched"],
        }
        try:
            workspace.mkdir(parents=True, exist_ok=True)
            payload = answer_data_package(
                question,
                plan,
                run_result,
                rule_context,
                merchant=merchant,
                personalization_context=personalization_context,
            )
            payload["questionUnderstanding"] = plan.question_understanding
            payload["executionContract"] = [
                "只基于已验证的 QueryGraph 结果、证据缺口和已召回知识进行分析",
                "不能新增查询、改写 QueryGraph、扩大商家或时间范围",
                "不能把缺失证据当作事实；不确定内容必须作为假设或限制披露",
            ]
            input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            trace["inputRows"] = len(payload.get("dataRows") or [])
            trace["lifecycleStage"] = "isolated_execute"
            trace["progress"] = trace["progress"] + ["isolated_execute"]
            self._write_checkpoint(checkpoint_path, trace, "running")

            answer_service = AnswerComposeService(self.llm)
            answer = answer_service.summarize_analysis(
                question,
                plan,
                run_result,
                outputs_path,
                rule_context,
                merchant=merchant,
                personalization_context=personalization_context,
                allow_skill=False,
            )
            analysis_trace = dict(getattr(answer_service, "last_analysis_skill_trace", {}) or {})
        except Exception as exc:
            return self._fail(
                trace,
                checkpoint_path,
                output_path,
                started,
                "%s: %s" % (type(exc).__name__, str(exc)[:1000]),
            )
        trace.update(
            {
                "analysisTrace": analysis_trace,
                "outputChars": len(answer or ""),
                "completedAt": datetime.now().isoformat(),
                "durationMs": int((time.monotonic() - started) * 1000),
                "lifecycleStage": "completed" if answer else "failed",
                "progress": trace["progress"] + ["progress_synced", "completed" if answer else "failed"],
            }
        )
        trace.pop("_startedMonotonic", None)
        if not answer:
            trace["error"] = "analysis worker produced empty summary"
        output_path.write_text(
            json.dumps({"answerMarkdown": answer, "trace": trace}, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        self._write_checkpoint(checkpoint_path, trace, "completed" if answer else "failed")
        return AnalysisWorkerResult(answer=answer, trace=trace)

    def _workspace(self, outputs_path: str, isolated_run_id: str) -> Path:
        artifact_root = Path(outputs_path) if outputs_path else self.settings.resolved_workspace_path
        return artifact_root / "artifacts" / "analysis_workers" / "general" / "runs" / isolated_run_id

    def _write_checkpoint(self, path: Path, trace: Dict[str, Any], status: str) -> None:
        path.write_text(json.dumps({**trace, "status": status}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def _fail(
        self,
        trace: Dict[str, Any],
        checkpoint_path: Path,
        output_path: Path,
        started: float,
        error: str,
    ) -> AnalysisWorkerResult:
        trace.update(
            {
                "error": error,
                "completedAt": datetime.now().isoformat(),
                "durationMs": int((time.monotonic() - started) * 1000),
                "lifecycleStage": "failed",
                "progress": list(trace.get("progress") or []) + ["failed"],
            }
        )
        trace.pop("_startedMonotonic", None)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps({"answerMarkdown": "", "trace": trace}, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            self._write_checkpoint(checkpoint_path, trace, "failed")
        except Exception as artifact_exc:
            trace["artifactError"] = "%s: %s" % (type(artifact_exc).__name__, str(artifact_exc)[:500])
        return AnalysisWorkerResult(answer="", trace=trace)
