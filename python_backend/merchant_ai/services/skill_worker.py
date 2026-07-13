from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from merchant_ai.models import AgentRunResult, MerchantInfo, QueryPlan
from merchant_ai.services.llm import LlmClient, TaskModelRouter
from merchant_ai.services.sandbox import MerchantAnalysisSandbox


@dataclass
class SkillWorkerResult:
    answer: str
    trace: Dict[str, Any]


class SkillWorkerExecutor:
    """Run complex answer skills in an isolated worker-style workspace."""

    def __init__(self, llm: LlmClient):
        self.llm = llm
        self.settings = llm.settings
        self.model_router = TaskModelRouter(self.settings)
        self.sandbox = MerchantAnalysisSandbox(self.settings)

    def execute_answer_skills(
        self,
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult,
        skill_names: list[str],
        outputs_path: str = "",
        rule_context: str = "",
        merchant: MerchantInfo | None = None,
        personalization_context: Optional[Dict[str, Any]] = None,
        initial_trace: Optional[Dict[str, Any]] = None,
    ) -> list[SkillWorkerResult]:
        names = []
        for name in skill_names or []:
            normalized = str(name or "").strip()
            if normalized and normalized not in names:
                names.append(normalized)
        if not names:
            return []
        max_workers = max(1, min(int(getattr(self.settings, "max_concurrent_skill_workers", 2) or 2), len(names)))
        results: Dict[str, SkillWorkerResult] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for index, skill_name in enumerate(names):
                trace = dict(initial_trace or {})
                trace.update(
                    {
                        "parallelSkillBatch": True,
                        "skillBatchIndex": index,
                        "skillBatchSize": len(names),
                        "maxConcurrency": max_workers,
                    }
                )
                future = executor.submit(
                    self.execute_answer_skill,
                    question,
                    plan,
                    run_result,
                    outputs_path,
                    rule_context,
                    skill_name,
                    merchant,
                    personalization_context,
                    trace,
                )
                futures[future] = skill_name
            for future in as_completed(futures):
                skill_name = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = SkillWorkerResult(
                        answer="",
                        trace={
                            "skillName": skill_name,
                            "workerType": "SKILL_WORKER",
                            "subAgentType": "SKILL_WORKER",
                            "isolatedExecution": True,
                            "parallelSkillBatch": True,
                            "lifecycleStage": "failed",
                            "error": str(exc)[:500],
                            "progress": ["failed:%s" % str(exc)[:80]],
                        },
                    )
                result.trace["parallelSkillBatch"] = True
                result.trace["skillBatchSize"] = len(names)
                result.trace["maxConcurrency"] = max_workers
                results[skill_name] = result
        return [results[name] for name in names if name in results]

    def execute_answer_skill(
        self,
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult,
        outputs_path: str = "",
        rule_context: str = "",
        skill_name: str = "",
        merchant: MerchantInfo | None = None,
        personalization_context: Optional[Dict[str, Any]] = None,
        initial_trace: Optional[Dict[str, Any]] = None,
    ) -> SkillWorkerResult:
        selected_skill = skill_name or "bi_trend_attribution"
        if bool(self.settings.distributed_subagents_enabled):
            return self._execute_distributed_skill(
                question,
                plan,
                run_result,
                outputs_path,
                rule_context,
                selected_skill,
                merchant,
                personalization_context,
                initial_trace,
            )
        started = time.monotonic()
        incoming_trace = dict(initial_trace or {})
        requires_confirmation = bool(
            incoming_trace.get("requiresConfirmation", incoming_trace.get("requires_confirmation", self.settings.skill_confirmation_required))
        )
        confirmed = bool(incoming_trace.get("confirmed", not requires_confirmation))
        isolated_run_id = "skill_%s_%s" % (selected_skill, uuid.uuid4().hex[:10])
        skill_dir = self.settings.resources_root / "runtime" / "agent_skills" / selected_skill
        skill_file = skill_dir / "SKILL.md"
        script = skill_dir / "scripts" / "profile_timeseries.py"
        workspace = self._workspace(outputs_path, selected_skill, isolated_run_id)
        checkpoint_path = workspace / "skill_checkpoint.json"
        input_path = workspace / "skill_input.json"
        output_path = workspace / "skill_output.json"
        context_path = workspace / "skill_context_package.json"
        trace: Dict[str, Any] = {
            "skillName": selected_skill,
            "matchedBy": incoming_trace.get("matchedBy") or "questionUnderstanding+verifiedEvidence",
            "matchTrace": incoming_trace,
            "activated": False,
            "executionMode": "isolated_skill_worker",
            "workerType": "SKILL_WORKER",
            "subAgentType": "SKILL_WORKER",
            "isolatedExecution": True,
            "skillPath": str(skill_file),
            "scriptPath": str(script),
            "lifecycleStage": "matched",
            "requiresConfirmation": requires_confirmation,
            "confirmed": confirmed,
            "isolatedRunId": isolated_run_id,
            "workspacePath": str(workspace),
            "checkpointPath": str(checkpoint_path),
            "contextPackagePath": str(context_path),
            "startedAt": datetime.now().isoformat(),
            "_startedMonotonic": started,
            "progress": ["matched"],
            "reuseCandidate": False,
        }
        if requires_confirmation and not confirmed:
            trace.update(
                {
                    "lifecycleStage": "awaiting_confirmation",
                    "status": "waiting_confirmation",
                    "progress": ["matched", "awaiting_confirmation"],
                    "completedAt": datetime.now().isoformat(),
                    "durationMs": int((time.monotonic() - started) * 1000),
                }
            )
            return SkillWorkerResult(answer="", trace=trace)
        if not skill_file.exists():
            return self._fail(trace, checkpoint_path, "skill package missing")

        from merchant_ai.services.answer import load_skill_frontmatter

        skill_meta = load_skill_frontmatter(skill_file)
        trace["metadata"] = skill_meta
        workspace.mkdir(parents=True, exist_ok=True)
        context_package = self._context_package(
            selected_skill,
            question,
            plan,
            run_result,
            merchant,
            skill_meta,
            input_path,
            output_path,
            context_path,
            bool(rule_context),
            script.exists() and selected_skill == "bi_trend_attribution",
        )
        context_path.write_text(json.dumps(context_package, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        trace["contextPackage"] = self._compact_context_package(context_package)

        payload = self._write_input(
            input_path,
            question,
            plan,
            run_result,
            rule_context,
            merchant,
            personalization_context,
            skill_meta,
            context_package,
        )
        trace.update(
            {
                "activated": True,
                "inputArtifact": str(input_path),
                "outputArtifact": str(output_path),
                "inputRows": len(payload.get("dataRows") or []),
                "lifecycleStage": "isolated_execute",
                "progress": trace["progress"] + ["confirmed" if trace["confirmed"] else "awaiting_confirmation", "isolated_execute"],
            }
        )
        self._write_checkpoint(checkpoint_path, trace, status="running")
        if selected_skill == "bi_trend_attribution":
            return self._execute_script_skill(script, input_path, output_path, checkpoint_path, trace)
        return self._execute_structured_skill(selected_skill, payload, output_path, checkpoint_path, trace)

    def _execute_distributed_skill(
        self,
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult,
        outputs_path: str,
        rule_context: str,
        skill_name: str,
        merchant: MerchantInfo | None,
        personalization_context: Optional[Dict[str, Any]],
        initial_trace: Optional[Dict[str, Any]],
    ) -> SkillWorkerResult:
        from merchant_ai.services.distributed_workers import DistributedSubAgentClient

        trace = dict(initial_trace or {})
        run_id = str(trace.get("parentRunId") or "skill_%s" % uuid.uuid4().hex[:12])
        task_id = "skill_%s_%s" % (skill_name, uuid.uuid4().hex[:10])
        result = DistributedSubAgentClient(self.settings).execute(
            run_id,
            task_id,
            "analysis_skill",
            {
                "question": question,
                "plan": plan.model_dump(by_alias=True),
                "runResult": run_result.model_dump(by_alias=True),
                "outputsPath": outputs_path,
                "ruleContext": rule_context,
                "skillName": skill_name,
                "merchant": merchant.model_dump(by_alias=True) if merchant else {},
                "personalizationContext": dict(personalization_context or {}),
                "initialTrace": {**trace, "distributedWorkerExecution": True},
            },
            timeout_seconds=max(1, int(self.settings.skill_worker_timeout_seconds or 1)),
        )
        if result.status != "completed":
            return SkillWorkerResult(
                answer="",
                trace={
                    **trace,
                    "skillName": skill_name,
                    "workerType": "DISTRIBUTED_SKILL_WORKER",
                    "subAgentType": "SKILL_WORKER",
                    "isolatedExecution": True,
                    "lifecycleStage": result.status,
                    "error": result.error or "distributed skill failed",
                    "distributedTaskId": task_id,
                    "resultArtifactUri": result.artifact_uri,
                },
            )
        payload = result.result
        child_trace = dict(payload.get("trace") or {})
        child_trace.update(
            {
                "workerType": "DISTRIBUTED_SKILL_WORKER",
                "distributedTaskId": task_id,
                "resultArtifactUri": result.artifact_uri,
            }
        )
        return SkillWorkerResult(answer=str(payload.get("answer") or ""), trace=child_trace)

    def _workspace(self, outputs_path: str, skill_name: str, isolated_run_id: str) -> Path:
        artifact_root = Path(outputs_path) if outputs_path else self.settings.resolved_workspace_path / "analysis_skills"
        return artifact_root / "artifacts" / "skill_workers" / skill_name / "runs" / isolated_run_id

    def _write_input(
        self,
        input_path: Path,
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult,
        rule_context: str,
        merchant: MerchantInfo | None,
        personalization_context: Optional[Dict[str, Any]],
        skill_meta: Dict[str, Any],
        context_package: Dict[str, Any],
    ) -> Dict[str, Any]:
        from merchant_ai.services.answer import answer_data_package

        payload = answer_data_package(
            question,
            plan,
            run_result,
            rule_context,
            merchant=merchant,
            personalization_context=personalization_context,
        )
        payload["questionUnderstanding"] = plan.question_understanding
        payload["skillMetadata"] = skill_meta
        payload["skillWorkerContextRef"] = {
            "contextPackagePath": context_package.get("contextPackagePath"),
            "workspacePath": context_package.get("workspacePath"),
            "checkpointPath": context_package.get("checkpointPath"),
        }
        input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return payload

    def _execute_script_skill(
        self,
        script: Path,
        input_path: Path,
        output_path: Path,
        checkpoint_path: Path,
        trace: Dict[str, Any],
    ) -> SkillWorkerResult:
        if not script.exists():
            return self._fail(trace, checkpoint_path, "skill script missing")
        completed = self.sandbox.run_python(
            script,
            ["--input", str(input_path), "--output", str(output_path)],
            output_path.parent,
            max(1, int(self.settings.skill_worker_timeout_seconds or 10)),
        )
        trace["returnCode"] = completed.returncode
        trace["stderr"] = completed.stderr[-1000:]
        if completed.returncode != 0 or not output_path.exists():
            return self._fail(trace, checkpoint_path, completed.stderr[-1000:] or "skill script failed")
        try:
            result = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return self._fail(trace, checkpoint_path, "invalid skill output: %s" % exc)
        return self._complete_from_output(result, checkpoint_path, trace)

    def _execute_structured_skill(
        self,
        skill_name: str,
        payload: Dict[str, Any],
        output_path: Path,
        checkpoint_path: Path,
        trace: Dict[str, Any],
    ) -> SkillWorkerResult:
        from merchant_ai.services.answer import render_structured_skill_answer

        answer = render_structured_skill_answer(skill_name, payload)
        output = {
            "skillName": skill_name,
            "rowCount": len(payload.get("dataRows") or []),
            "answerMarkdown": answer,
            "caveats": [gap.get("code") for gap in payload.get("evidenceGaps") or [] if isinstance(gap, dict)],
        }
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        trace["deterministicRenderer"] = True
        return self._complete_from_output(output, checkpoint_path, trace)

    def _complete_from_output(self, result: Dict[str, Any], checkpoint_path: Path, trace: Dict[str, Any]) -> SkillWorkerResult:
        from merchant_ai.services.answer import answer_skill_reuse_candidate

        trace["outputRows"] = result.get("rowCount", 0)
        trace["findings"] = result.get("findings", [])[:6]
        trace["caveats"] = result.get("caveats", [])[:6]
        trace["lifecycleStage"] = "completed"
        trace["completedAt"] = datetime.now().isoformat()
        trace["durationMs"] = int((time.monotonic() - float(trace.get("_startedMonotonic") or time.monotonic())) * 1000)
        trace.pop("_startedMonotonic", None)
        trace["progress"].extend(["progress_synced", "completed"])
        trace["reuseCandidate"] = bool(
            self.settings.skill_reuse_suggestion_enabled and answer_skill_reuse_candidate(str(trace.get("skillName") or ""), result)
        )
        self._write_checkpoint(checkpoint_path, trace, status="completed")
        answer = str(result.get("answerMarkdown") or "").strip()
        if not answer:
            return self._fail(trace, checkpoint_path, "empty skill answer")
        return SkillWorkerResult(answer=answer, trace=trace)

    def _fail(self, trace: Dict[str, Any], checkpoint_path: Path, message: str) -> SkillWorkerResult:
        trace["error"] = message
        trace["lifecycleStage"] = "failed"
        trace["completedAt"] = datetime.now().isoformat()
        trace["durationMs"] = int((time.monotonic() - float(trace.get("_startedMonotonic") or time.monotonic())) * 1000)
        trace.pop("_startedMonotonic", None)
        trace.setdefault("progress", []).append("failed:%s" % str(message)[:80])
        self._write_checkpoint(checkpoint_path, trace, status="failed")
        return SkillWorkerResult(answer="", trace=trace)

    def _context_package(
        self,
        skill_name: str,
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult,
        merchant: MerchantInfo | None,
        skill_meta: Dict[str, Any],
        input_path: Path,
        output_path: Path,
        context_path: Path,
        has_rule_context: bool,
        can_execute_script: bool,
    ) -> Dict[str, Any]:
        rows = run_result.merged_query_bundle.rows if run_result and run_result.merged_query_bundle else []
        return {
            "packageType": "skill_worker_context",
            "skillName": skill_name,
            "workerType": "SKILL_WORKER",
            "question": question,
            "merchantId": getattr(merchant, "merchant_id", "") if merchant else "",
            "questionUnderstanding": plan.question_understanding,
            "intentCount": len(plan.intents or []),
            "verifiedRowCount": len(rows or []),
            "evidenceGapCount": len(run_result.evidence_gaps or []) if run_result else 0,
            "hasRuleContext": has_rule_context,
            "knowledgeAccess": {
                "required": True,
                "capability": "retrieve_knowledge",
                "availableScopes": ["current_merchant", "current_topic_platform", "skill_private"],
                "merchantIsolation": True,
                "merchantId": getattr(merchant, "merchant_id", "") if merchant else "",
                "preloadedContext": has_rule_context,
                "fallbackTools": ["semantic_grep", "semantic_read"],
            },
            "modelRoute": self.model_router.trace("simple_skill"),
            "skillMetadata": skill_meta,
            "workspacePath": str(context_path.parent),
            "checkpointPath": str(context_path.parent / "skill_checkpoint.json"),
            "contextPackagePath": str(context_path),
            "inputArtifact": str(input_path),
            "outputArtifact": str(output_path),
            "allowedArtifacts": [str(input_path), str(output_path)],
            "fileContextTools": {
                "semantic_ls": "list semantic asset directories and compact manifests",
                "semantic_read": "read approved semantic asset snippets when the skill needs definitions",
                "semantic_grep": "search approved semantic assets without loading full schema",
                "artifact_ls": "list run artifacts available to this worker",
                "artifact_read": "read whitelisted artifacts by path",
                "artifact_grep": "search whitelisted artifacts without loading full files",
            },
            "allowedTools": self._allowed_tools(can_execute_script),
            "executionContract": [
                "只读取 skill_input.json 中的已验证数据和证据缺口",
                "需要语义资产或中间产物细节时，先通过 semantic/artifact 工具按需读取",
                "Skill 不得内置或覆盖平台公共口径；通用定义必须从 knowledge context 或 semantic 工具读取",
                "知识读取只能覆盖当前 merchantId、本 Topic 平台知识和当前 Skill 私有知识",
                "只能写入本次 SkillWorker 工作目录",
                "不能新增查询、改写查询图或扩大商家/时间范围",
                "输出必须写入 skill_output.json 并同步 checkpoint",
            ],
        }

    def _compact_context_package(self, package: Dict[str, Any]) -> Dict[str, Any]:
        keys = [
            "packageType",
            "skillName",
            "workerType",
            "merchantId",
            "intentCount",
            "verifiedRowCount",
            "evidenceGapCount",
            "hasRuleContext",
            "contextPackagePath",
            "inputArtifact",
            "outputArtifact",
            "allowedTools",
            "fileContextTools",
            "knowledgeAccess",
            "modelRoute",
        ]
        return {key: package.get(key) for key in keys if key in package}

    def _allowed_tools(self, can_execute_script: bool) -> Dict[str, Any]:
        tools = {
            "read_skill_input": True,
            "write_skill_output": True,
            "write_checkpoint": True,
            "read_workspace_artifact": True,
            "semantic_ls": True,
            "semantic_read": True,
            "semantic_grep": True,
            "artifact_ls": True,
            "artifact_read": True,
            "artifact_grep": True,
            "retrieve_knowledge": True,
            "execute_script": bool(can_execute_script),
        }
        return tools

    def _write_checkpoint(self, path: Path, trace: Dict[str, Any], status: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "skillName": trace.get("skillName"),
                        "isolatedRunId": trace.get("isolatedRunId"),
                        "workerType": trace.get("workerType"),
                        "subAgentType": trace.get("subAgentType"),
                        "isolatedExecution": trace.get("isolatedExecution"),
                        "status": status,
                        "stage": trace.get("lifecycleStage"),
                        "progress": trace.get("progress") or [],
                        "contextPackage": trace.get("contextPackage") or {},
                        "inputArtifact": trace.get("inputArtifact"),
                        "outputArtifact": trace.get("outputArtifact"),
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        except Exception:
            return
