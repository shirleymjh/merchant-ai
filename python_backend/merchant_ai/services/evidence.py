from __future__ import annotations

from typing import Any, Dict, List, Set

import sqlglot
from sqlglot import exp

from merchant_ai.models import AgentRunResult, AnswerMode, EvidenceGap, QueryPlan, VerifiedEvidence
from merchant_ai.services.entity_contracts import (
    entity_filter_contract_hash,
    entity_filter_sql_hash,
)
from merchant_ai.services.memory_constraints import memory_constraint_evidence_gaps
from merchant_ai.services.query import (
    semantic_filter_contract_hash,
    semantic_filter_execution_hash,
    semantic_filter_value_hash,
)
from merchant_ai.services.text_parsing import (
    is_ascii_identifier,
    leading_iso_date_parts,
    literal_spans,
)


class EvidenceVerifier:
    def verify(
        self,
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult,
        memory_constraints: List[Dict[str, Any]] | None = None,
        allowed_knowledge_refs: Set[str] | None = None,
    ) -> VerifiedEvidence:
        # Evidence gaps are monotonic across the execution pipeline.  Runtime,
        # graph validation and node workers may already have produced typed
        # failures that this verifier cannot reconstruct from result rows.  A
        # later verification pass may enrich those gaps, but must never erase
        # them by starting from an empty list.
        gaps: List[EvidenceGap] = [
            gap.model_copy(deep=True)
            for gap in run_result.evidence_gaps or []
        ]
        gaps.extend(execution_operational_evidence_gaps(run_result))
        gaps.extend(entity_filter_evidence_gaps(run_result))
        gaps.extend(semantic_filter_evidence_gaps(run_result))
        covered = self._covered_keys(run_result)
        derived_evidence = self._derived_evidence(plan, run_result)
        gaps.extend(self._metric_spec_gaps(plan, derived_evidence))
        gaps.extend(snapshot_alignment_evidence_gaps(run_result))
        gaps.extend(planner_degraded_evidence_gaps(run_result))
        zero_filled_component_tasks, zero_filled_edges = self._zero_filled_derived_components(plan, run_result)
        table_names = set(run_result.merged_query_bundle.tables)
        if plan.evidence_contracts:
            gaps.extend(self._contract_gaps(plan.evidence_contracts, run_result, allowed_knowledge_refs, zero_filled_component_tasks))
            covered.update(self._covered_contract_labels(plan.evidence_contracts, run_result, allowed_knowledge_refs))
        else:
            for evidence in plan.final_required_evidence:
                if not evidence:
                    continue
                if self._natural_evidence_covered(evidence, covered, table_names):
                    continue
                gaps.append(EvidenceGap(code="MISSING_REQUIRED_EVIDENCE", evidence=evidence, reason="finalRequiredEvidence 未被结果字段或使用表覆盖"))
        required_disclosures = metric_resolution_disclosures(plan)
        for disclosure in snapshot_alignment_disclosures(run_result):
            if disclosure not in required_disclosures:
                required_disclosures.append(disclosure)
        for disclosure in planner_degraded_disclosures(run_result):
            if disclosure not in required_disclosures:
                required_disclosures.append(disclosure)
        gaps.extend(metric_resolution_warning_gaps(plan))
        gaps.extend(memory_constraint_evidence_gaps(question, plan, memory_constraints or []))
        for gap in run_result.evidence_check.gaps:
            if any(edge and edge in str(gap) for edge in zero_filled_edges):
                continue
            gaps.append(EvidenceGap(code="DEPENDENCY_GAP", evidence=gap, reason=gap))
        succeeded_tasks = [item for item in run_result.task_results if not item.query_bundle.failed]
        failed_tasks = [item for item in run_result.task_results if item.query_bundle.failed]
        consumed_entity_set_task_ids = {
            str(dependency.anchor_task_id or "")
            for dependency in plan.dependencies
            if str(dependency.anchor_task_id or "")
            and str(dependency.dependent_task_id or "")
        }
        consumed_entity_set_task_ids.update(
            str(task_id or "")
            for intent in plan.intents
            for task_id in intent.depends_on_task_ids
            if str(task_id or "")
        )
        if succeeded_tasks and failed_tasks:
            gaps.append(
                EvidenceGap(
                    code="PARTIAL_EVIDENCE",
                    reason="部分节点执行成功，部分节点失败或被依赖跳过；回答只能基于已验证证据",
                )
            )
        for task_result in run_result.task_results:
            if task_result.query_bundle.failed:
                code = classify_task_failure(task_result.query_bundle.error or task_result.summary)
                gaps.append(
                    EvidenceGap(
                        code=code,
                        task_id=task_result.task_id,
                        evidence=task_result.query_bundle.sql[:240],
                        reason=task_result.query_bundle.error or task_result.summary,
                    )
                )
            for repair in task_result.sql_repairs:
                if repair.success and repair.error_code in {"MEM_ALLOC_FAILED", "TIMEOUT"} and "resource-safe" in repair.error_message:
                    gaps.append(
                        EvidenceGap(
                            code="RESOURCE_DEGRADED_QUERY",
                            task_id=task_result.task_id,
                            evidence=task_result.query_bundle.sql[:240],
                            reason="Doris %s 后使用资源保护 SQL 降级执行，结果可能只覆盖收敛后的行数或实体范围" % repair.error_code,
                            severity="warning",
                            disclosure_required=True,
                            source="task",
                            answer_instruction="说明该节点因 Doris 资源限制使用了降级 SQL，结论应按部分覆盖理解。",
                        )
                    )
            if (
                task_result.entity_set
                and task_result.entity_set.truncated
                and task_result.task_id in consumed_entity_set_task_ids
            ):
                gaps.append(
                    EvidenceGap(
                        code="ENTITY_SET_TRUNCATED",
                        task_id=task_result.task_id,
                        evidence=task_result.entity_set.join_key,
                        reason="上游实体超过本轮最大传递数量，dependent 结果可能只覆盖部分实体",
                    )
                )
        gaps = dedupe_evidence_gaps([classify_evidence_gap(gap) for gap in gaps])
        blocking_gaps = [gap for gap in gaps if gap.severity == "blocking"]
        warning_gaps = [gap for gap in gaps if gap.severity == "warning"]
        partial_reason = "；".join(gap.reason for gap in blocking_gaps[:3])
        return VerifiedEvidence(
            passed=not blocking_gaps,
            covered_evidence=sorted(covered),
            derived_evidence=derived_evidence,
            gaps=gaps,
            blocking_gaps=blocking_gaps,
            warning_gaps=warning_gaps,
            answer_guard_required=bool(blocking_gaps or warning_gaps or required_disclosures),
            required_disclosures=required_disclosures,
            partial_answer_reason=partial_reason,
        )

    def _covered_keys(self, run_result: AgentRunResult) -> Set[str]:
        keys: Set[str] = set()
        for row in run_result.merged_query_bundle.rows:
            keys.update(str(key) for key in row.keys())
        for task_result in run_result.task_results:
            for row in task_result.query_bundle.rows[:20]:
                keys.update(str(key) for key in row.keys())
        return keys

    def _derived_evidence(self, plan: QueryPlan, run_result: AgentRunResult) -> List[Dict[str, Any]]:
        derived: List[Dict[str, Any]] = []
        results_by_task = {item.task_id: item for item in run_result.task_results}
        for intent in plan.intents:
            task_result = results_by_task.get(intent.plan_task_id)
            rows = task_result.query_bundle.rows if task_result else []
            row_columns = self._row_columns(rows)
            observed_roles = sorted(
                {
                    str(row.get("__timeWindowRole") or "").strip()
                    for row in rows
                    if str(row.get("__timeWindowRole") or "").strip()
                }
            )
            expected_role = str(
                getattr(intent.time_range, "window_role", "")
                or (intent.metric_resolution or {}).get("timeWindowRole")
                or ""
            ).strip()
            raw_specs = [item for item in intent.metric_specs if isinstance(item, dict)]
            specs = raw_specs or (
                [
                    {
                        "metricName": intent.metric_name or intent.metric_column,
                        "metricColumn": intent.metric_column,
                        "metricFormula": intent.metric_formula,
                        "sourceColumns": formula_columns(intent.metric_formula),
                        "sourceTaskId": intent.plan_task_id,
                    }
                ]
                if intent.metric_formula or intent.metric_name or intent.metric_column
                else []
            )
            for spec in specs:
                metric_alias = str(
                    spec.get("metricName")
                    or spec.get("metric_name")
                    or spec.get("metricColumn")
                    or spec.get("metric_column")
                    or ""
                ).strip()
                formula = str(spec.get("metricFormula") or spec.get("metric_formula") or "").strip()
                source_columns = [
                    str(item)
                    for item in spec.get("sourceColumns")
                    or spec.get("source_columns")
                    or formula_columns(formula)
                    if str(item or "").strip()
                ]
                alias_covered = bool(metric_alias and metric_alias in row_columns)
                value_covered = bool(
                    alias_covered
                    and any(row.get(metric_alias) not in (None, "") for row in rows)
                )
                source_covered = self._metric_spec_source_covered(intent, spec, task_result)
                role_covered = not expected_role or expected_role in observed_roles
                task_succeeded = bool(task_result and not task_result.query_bundle.failed)
                has_rows = bool(rows)
                derived.append(
                    {
                        "taskId": intent.plan_task_id,
                        "sourceTaskId": str(
                            spec.get("sourceTaskId") or spec.get("source_task_id") or intent.plan_task_id or ""
                        ),
                        "table": intent.preferred_table,
                        "metric": metric_alias,
                        "formula": formula,
                        "sourceColumns": source_columns,
                        "semanticRefId": str(
                            spec.get("semanticRefId")
                            or spec.get("semantic_ref_id")
                            or (intent.metric_resolution or {}).get("semanticRefId")
                            or ""
                        ),
                        "expectedTimeWindowRole": expected_role,
                        "observedTimeWindowRoles": observed_roles,
                        "timeWindowRole": expected_role if role_covered else "",
                        "aliasCovered": alias_covered,
                        "valueCovered": value_covered,
                        "sourceCovered": source_covered,
                        "timeWindowRoleCovered": role_covered,
                        "taskSucceeded": task_succeeded,
                        "hasRows": has_rows,
                        "metricSpec": bool(raw_specs),
                        "covered": bool(
                            task_succeeded
                            and has_rows
                            and alias_covered
                            and value_covered
                            and source_covered
                            and role_covered
                        ),
                    }
                )
        return derived

    def _metric_spec_source_covered(self, intent: Any, spec: Dict[str, Any], task_result: Any) -> bool:
        if not task_result:
            return False
        contract = task_result.node_plan_contract
        if intent.preferred_table and str(contract.preferred_table or "") != str(intent.preferred_table):
            return False
        expected_alias = str(
            spec.get("metricName") or spec.get("metric_name") or spec.get("metricColumn") or spec.get("metric_column") or ""
        ).strip()
        expected_formula = canonical_formula(spec.get("metricFormula") or spec.get("metric_formula") or "")
        expected_columns = {
            str(item)
            for item in spec.get("sourceColumns")
            or spec.get("source_columns")
            or formula_columns(str(spec.get("metricFormula") or spec.get("metric_formula") or ""))
            if str(item or "").strip()
        }
        expected_source_task = str(spec.get("sourceTaskId") or spec.get("source_task_id") or "").strip()
        for contract_spec in contract.metric_specs or []:
            if not isinstance(contract_spec, dict):
                continue
            actual_alias = str(
                contract_spec.get("metricName")
                or contract_spec.get("metric_name")
                or contract_spec.get("metricColumn")
                or contract_spec.get("metric_column")
                or ""
            ).strip()
            if not expected_alias or actual_alias != expected_alias:
                continue
            actual_formula = canonical_formula(
                contract_spec.get("metricFormula") or contract_spec.get("metric_formula") or ""
            )
            if expected_formula and actual_formula != expected_formula:
                continue
            actual_columns = {
                str(item)
                for item in contract_spec.get("sourceColumns")
                or contract_spec.get("source_columns")
                or formula_columns(str(contract_spec.get("metricFormula") or contract_spec.get("metric_formula") or ""))
                if str(item or "").strip()
            }
            if expected_columns and actual_columns != expected_columns:
                continue
            actual_source_task = str(
                contract_spec.get("sourceTaskId") or contract_spec.get("source_task_id") or ""
            ).strip()
            if expected_source_task and actual_source_task != expected_source_task:
                continue
            return True
        return False

    def _metric_spec_gaps(
        self,
        plan: QueryPlan,
        derived_evidence: List[Dict[str, Any]],
    ) -> List[EvidenceGap]:
        gaps: List[EvidenceGap] = []
        for item in derived_evidence:
            if not item.get("metricSpec") or not item.get("taskSucceeded") or not item.get("hasRows"):
                continue
            metric = str(item.get("metric") or "")
            task_id = str(item.get("taskId") or "")
            source_task_id = str(item.get("sourceTaskId") or "")
            details = {
                "metric": metric,
                "sourceTaskId": source_task_id,
                "expectedTimeWindowRole": item.get("expectedTimeWindowRole") or "",
                "observedTimeWindowRoles": item.get("observedTimeWindowRoles") or [],
            }
            if not item.get("aliasCovered"):
                gaps.append(
                    EvidenceGap(
                        code="MISSING_METRIC_ALIAS",
                        task_id=task_id,
                        evidence=metric,
                        reason="metricSpecs 声明的结果别名未由执行结果精确产出",
                        details=details,
                    )
                )
            elif not item.get("valueCovered"):
                gaps.append(
                    EvidenceGap(
                        code="MISSING_METRIC_VALUE",
                        task_id=task_id,
                        evidence=metric,
                        reason="metricSpecs 声明的结果别名没有可验证值",
                        details=details,
                    )
                )
            if not item.get("sourceCovered"):
                gaps.append(
                    EvidenceGap(
                        code="METRIC_SOURCE_CONTRACT_MISMATCH",
                        task_id=task_id,
                        evidence=metric,
                        reason="结果指标的别名、公式、sourceColumns 或 sourceTaskId 与 NodePlanContract 不一致",
                        details=details,
                    )
                )
            if not item.get("timeWindowRoleCovered"):
                gaps.append(
                    EvidenceGap(
                        code="TIME_WINDOW_ROLE_MISMATCH",
                        task_id=task_id,
                        evidence=metric,
                        reason="结果行的时间窗口角色与计划契约不一致",
                        details=details,
                    )
                )
        time_contract = (plan.question_understanding or {}).get("timeWindowContract") or {}
        if not isinstance(time_contract, dict) or not (
            time_contract.get("requiresMultipleWindows") or time_contract.get("requiresComparison")
        ):
            return gaps
        metric_identities = {
            str(item.get("semanticRefId") or item.get("metric") or "")
            for item in derived_evidence
            if item.get("metricSpec") and str(item.get("semanticRefId") or item.get("metric") or "")
        }
        covered_roles = {
            (
                str(item.get("semanticRefId") or item.get("metric") or ""),
                str(item.get("timeWindowRole") or ""),
            )
            for item in derived_evidence
            if item.get("covered")
        }
        expected_roles = [
            str(item.get("windowRole") or "")
            for item in time_contract.get("windows") or []
            if isinstance(item, dict) and str(item.get("windowRole") or "")
        ]
        if not expected_roles:
            expected_roles = ["primary", "comparison"]
        for metric_identity in sorted(metric_identities):
            for role in expected_roles:
                if (metric_identity, role) in covered_roles:
                    continue
                gaps.append(
                    EvidenceGap(
                        code="MISSING_TIME_WINDOW_EVIDENCE",
                        evidence=metric_identity,
                        reason="多窗口契约缺少指标在 %s 窗口的完整证据" % role,
                        missing_metric=metric_identity,
                        missing_time_range=role,
                        details={"metricIdentity": metric_identity, "timeWindowRole": role},
                    )
                )
        return gaps

    def _contract_gaps(
        self,
        contracts: List[Dict[str, Any]],
        run_result: AgentRunResult,
        allowed_knowledge_refs: Set[str] | None = None,
        zero_filled_component_tasks: Set[str] | None = None,
    ) -> List[EvidenceGap]:
        gaps: List[EvidenceGap] = []
        zero_filled_component_tasks = zero_filled_component_tasks or set()
        for contract in contracts:
            level = str(contract.get("requiredLevel") or contract.get("required_level") or "required").lower()
            if level in {"optional", "info", "warning"}:
                continue
            evidence_source = str(contract.get("evidenceSource") or contract.get("evidence_source") or "").lower()
            if evidence_source in {"knowledge_ref", "knowledge", "rule"}:
                if self._knowledge_contract_covered(contract, allowed_knowledge_refs):
                    continue
                semantic_label = str(contract.get("semanticLabel") or contract.get("semantic_label") or "knowledge evidence")
                gaps.append(
                    EvidenceGap(
                        code="MISSING_REQUIRED_EVIDENCE",
                        task_id=str(contract.get("taskId") or contract.get("task_id") or ""),
                        evidence=semantic_label,
                        reason="%s 缺少召回知识引用" % semantic_label,
                    )
                )
                continue
            task_id = str(contract.get("taskId") or contract.get("task_id") or "")
            table = str(contract.get("table") or "")
            columns = [str(item) for item in contract.get("columns", []) if item]
            columns_any_of = normalize_any_of(contract.get("columnsAnyOf") or contract.get("columns_any_of") or [])
            semantic_aliases = normalize_semantic_aliases(contract.get("semanticAliases") or contract.get("semantic_aliases") or {})
            semantic_label = str(contract.get("semanticLabel") or contract.get("semantic_label") or table or task_id)
            task_result = self._matching_task_result(task_id, table, run_result)
            if not task_result:
                gaps.append(
                    EvidenceGap(
                        code="MISSING_REQUIRED_EVIDENCE",
                        task_id=task_id,
                        evidence=semantic_label,
                        reason="%s 未找到对应 task/table 执行结果" % semantic_label,
                    )
                )
                continue
            if task_result.query_bundle.failed:
                code = classify_task_failure(task_result.query_bundle.error or task_result.summary)
                gaps.append(
                    EvidenceGap(
                        code=code,
                        task_id=task_result.task_id,
                        evidence=semantic_label,
                        reason=task_result.query_bundle.error or task_result.summary,
                    )
                )
                continue
            if not task_result.query_bundle.rows:
                if task_result.task_id in zero_filled_component_tasks:
                    continue
                gaps.append(
                    EvidenceGap(
                        code="ZERO_ROWS",
                        task_id=task_result.task_id,
                        evidence=semantic_label,
                        reason="%s 执行成功但返回 0 行" % (task_result.task_id or semantic_label),
                    )
                )
                continue
            row_columns = self._row_columns(task_result.query_bundle.rows)
            missing_columns = [column for column in columns if not self._column_covered(column, row_columns, semantic_aliases)]
            if missing_columns:
                gaps.append(
                    EvidenceGap(
                        code=missing_gap_code(contract),
                        task_id=task_result.task_id,
                        evidence=",".join(missing_columns),
                        reason="%s 缺少结构化证据字段: %s" % (semantic_label, ",".join(missing_columns)),
                    )
                )
            for group in columns_any_of:
                if any(self._column_covered(column, row_columns, semantic_aliases) for column in group):
                    continue
                gaps.append(
                    EvidenceGap(
                        code="MISSING_ENTITY_KEY",
                        task_id=task_result.task_id,
                        evidence="/".join(group),
                        reason="%s 缺少任一可用实体键: %s" % (semantic_label, "/".join(group)),
                    )
                )
        return gaps

    def _zero_filled_derived_components(self, plan: QueryPlan, run_result: AgentRunResult) -> tuple[Set[str], Set[str]]:
        results_by_task = {item.task_id: item for item in run_result.task_results}
        zero_row_tasks = {
            item.task_id
            for item in run_result.task_results
            if not item.query_bundle.failed and not item.query_bundle.rows and int(item.query_bundle.effective_row_count() or 0) == 0
        }
        tasks: Set[str] = set()
        edges: Set[str] = set()
        for intent in plan.intents:
            if intent.answer_mode != AnswerMode.DERIVED:
                continue
            derived_result = results_by_task.get(intent.plan_task_id)
            if not derived_result or derived_result.query_bundle.failed:
                continue
            if not (derived_result.query_bundle.rows or int(derived_result.query_bundle.effective_row_count() or 0) > 0):
                continue
            resolution = intent.metric_resolution or {}
            formula = str(intent.metric_formula or resolution.get("formula") or "")
            components = [item for item in resolution.get("componentMetrics") or [] if isinstance(item, dict)]
            if "/" not in formula or len(components) < 2:
                continue
            numerator_task = str(components[0].get("taskId") or "")
            denominator_task = str(components[1].get("taskId") or "")
            denominator_result = results_by_task.get(denominator_task)
            if not numerator_task or numerator_task not in zero_row_tasks:
                continue
            if not denominator_result or denominator_result.query_bundle.failed:
                continue
            if not (denominator_result.query_bundle.rows or int(denominator_result.query_bundle.effective_row_count() or 0) > 0):
                continue
            tasks.add(numerator_task)
            edges.add("%s->%s" % (numerator_task, intent.plan_task_id))
        return tasks, edges

    def _covered_contract_labels(
        self,
        contracts: List[Dict[str, Any]],
        run_result: AgentRunResult,
        allowed_knowledge_refs: Set[str] | None = None,
    ) -> Set[str]:
        labels: Set[str] = set()
        for contract in contracts:
            evidence_source = str(contract.get("evidenceSource") or contract.get("evidence_source") or "").lower()
            if evidence_source in {"knowledge_ref", "knowledge", "rule"}:
                if self._knowledge_contract_covered(contract, allowed_knowledge_refs):
                    label = str(contract.get("semanticLabel") or contract.get("semantic_label") or "")
                    if label:
                        labels.add(label)
                continue
            task_result = self._matching_task_result(str(contract.get("taskId") or contract.get("task_id") or ""), str(contract.get("table") or ""), run_result)
            if not task_result or task_result.query_bundle.failed or not task_result.query_bundle.rows:
                continue
            columns = [str(item) for item in contract.get("columns", []) if item]
            columns_any_of = normalize_any_of(contract.get("columnsAnyOf") or contract.get("columns_any_of") or [])
            semantic_aliases = normalize_semantic_aliases(contract.get("semanticAliases") or contract.get("semantic_aliases") or {})
            row_columns = self._row_columns(task_result.query_bundle.rows)
            if columns and any(not self._column_covered(column, row_columns, semantic_aliases) for column in columns):
                continue
            if columns_any_of and any(
                not any(self._column_covered(column, row_columns, semantic_aliases) for column in group)
                for group in columns_any_of
            ):
                continue
            label = str(contract.get("semanticLabel") or contract.get("semantic_label") or "")
            if label:
                labels.add(label)
        return labels

    def _knowledge_contract_covered(self, contract: Dict[str, Any], allowed_knowledge_refs: Set[str] | None = None) -> bool:
        refs = contract.get("knowledgeRefs") or contract.get("knowledge_refs") or []
        if not isinstance(refs, list):
            refs = [refs] if refs else []
        normalized = {str(ref or "").strip() for ref in refs if str(ref or "").strip()}
        if not normalized:
            return False
        if allowed_knowledge_refs is None:
            return True
        return normalized.issubset({str(ref or "").strip() for ref in allowed_knowledge_refs if str(ref or "").strip()})

    def _matching_task_result(self, task_id: str, table: str, run_result: AgentRunResult):
        if task_id:
            return next(
                (task_result for task_result in run_result.task_results if task_result.task_id == task_id),
                None,
            )
        if not table:
            return None
        table_matches = [
            task_result
            for task_result in run_result.task_results
            if table in task_result.query_bundle.tables
        ]
        if len(table_matches) == 1:
            return table_matches[0]
        return None

    def _row_columns(self, rows: List[Dict[str, Any]]) -> Set[str]:
        columns: Set[str] = set()
        for row in rows[:20]:
            columns.update(str(key) for key in row.keys())
        return columns

    def _column_covered(self, column: str, row_columns: Set[str], semantic_aliases: Dict[str, Set[str]] | None = None) -> bool:
        if column in row_columns:
            return True
        aliases = (semantic_aliases or {}).get(column, set())
        return bool(aliases & row_columns)

    def _natural_evidence_covered(self, evidence: str, covered: Set[str], table_names: Set[str]) -> bool:
        evidence_text = str(evidence or "").strip().casefold()
        if not evidence_text:
            return False
        return any(
            explicit_evidence_reference(evidence_text, reference)
            for reference in {*covered, *table_names}
        )


def explicit_evidence_reference(evidence_text: str, reference: Any) -> bool:
    """Match a result/table reference without allowing identifier substrings.

    ``id`` is not evidence for ``refund_order_id``.  Natural evidence labels may
    still embed an exact governed identifier (for example ``orders.order_id``),
    but both sides must be identifier boundaries.  This keeps the fallback
    generic while preventing a short, unrelated column from satisfying a more
    specific evidence obligation.
    """

    normalized_reference = str(reference or "").strip().strip("`").casefold()
    if not normalized_reference:
        return False
    if evidence_text == normalized_reference:
        return True
    if (
        is_ascii_identifier(normalized_reference)
        and normalized_reference == normalized_reference.lower()
    ):
        return bool(
            literal_spans(
                evidence_text,
                (normalized_reference,),
                ascii_word_boundary=True,
            )
        )
    # Non-identifier labels are deliberately exact.  Their aliases belong in
    # the typed evidence contract rather than being inferred by substring.
    return False


def normalize_any_of(value: Any) -> List[List[str]]:
    groups: List[List[str]] = []
    if not isinstance(value, list):
        return groups
    for group in value:
        if isinstance(group, list):
            columns = [str(item) for item in group if item]
        elif group:
            columns = [str(group)]
        else:
            columns = []
        if columns:
            groups.append(columns)
    return groups


def formula_columns(formula: str) -> List[str]:
    keywords = {
        "SUM",
        "COUNT",
        "AVG",
        "MIN",
        "MAX",
        "DISTINCT",
        "CASE",
        "WHEN",
        "THEN",
        "ELSE",
        "END",
        "NULLIF",
        "COALESCE",
        "IFNULL",
        "CAST",
        "AS",
        "DECIMAL",
        "DOUBLE",
        "AND",
        "OR",
        "NOT",
        "IN",
        "IS",
        "NULL",
    }
    try:
        parsed = sqlglot.parse_one(str(formula or ""), read="doris")
    except Exception:
        return []
    return list(
        dict.fromkeys(
            column.name
            for column in parsed.find_all(exp.Column)
            if column.name and column.name.upper() not in keywords
        )
    )


def canonical_formula(value: Any) -> str:
    """Normalize formatting only; keep the published calculation semantics."""

    return "".join(
        str(value or "").replace("`", "").split()
    ).lower()


def normalize_semantic_aliases(value: Any) -> Dict[str, Set[str]]:
    aliases: Dict[str, Set[str]] = {}
    if not isinstance(value, dict):
        return aliases
    for key, raw_values in value.items():
        alias_key = str(key or "")
        if not alias_key:
            continue
        if isinstance(raw_values, list):
            aliases[alias_key] = {str(item) for item in raw_values if item}
        elif raw_values:
            aliases[alias_key] = {str(raw_values)}
    return aliases


def missing_gap_code(contract: Dict[str, Any]) -> str:
    evidence_role = str(
        contract.get("evidenceRole")
        or contract.get("evidence_role")
        or contract.get("semanticRole")
        or contract.get("semantic_role")
        or ""
    ).lower()
    if evidence_role in {"entity", "entity_key", "dimension"}:
        return "MISSING_ENTITY_KEY"
    if evidence_role in {"metric", "measure", "ratio", "derived_metric"}:
        return "MISSING_METRIC_EVIDENCE"
    return "MISSING_REQUIRED_COLUMNS"


def snapshot_alignment_applicable(run_result: AgentRunResult | None) -> bool:
    if not run_result:
        return False
    alignment = getattr(run_result, "snapshot_alignment", None)
    if alignment is None:
        return False
    status = str(getattr(alignment, "status", "") or "").strip().upper()
    return bool(
        getattr(alignment, "sources", None)
        or getattr(alignment, "strategy", "")
        or getattr(alignment, "common_anchor_time_value", "")
        or getattr(alignment, "disclosure_required", False)
        or status not in {"", "NOT_APPLICABLE"}
    )


def planner_degraded_reasons(run_result: AgentRunResult | None) -> List[Dict[str, Any]]:
    """Return active planner failures recorded by the runtime.

    Planner availability is operational evidence, not business evidence.  Keep
    the structured failure attached to the run so a validated deterministic
    fallback can be disclosed and an unsafe/no fallback can fail closed.
    """

    if not run_result:
        return []
    reasons: List[Dict[str, Any]] = []
    seen: Set[tuple[str, str]] = set()
    for raw in run_result.degraded_reasons or []:
        if not isinstance(raw, dict):
            continue
        stage = str(raw.get("stage") or "").strip().lower()
        code = str(raw.get("code") or "").strip()
        if stage != "planner" or raw.get("active") is False:
            continue
        identity = (code, str(raw.get("reason") or ""))
        if identity in seen:
            continue
        seen.add(identity)
        reasons.append(dict(raw))
    return reasons


def execution_operational_evidence_gaps(run_result: AgentRunResult | None) -> List[EvidenceGap]:
    """Fail closed when execution ended before a task-level result existed."""

    if not run_result or not run_result.merged_query_bundle.failed:
        return []
    # Normal task failures are classified with their task id later in verify().
    # This guard covers worker/bootstrap/fingerprint failures that have no task
    # record from which the verifier could otherwise reconstruct a gap.
    if any(item.query_bundle.failed for item in run_result.task_results or []):
        return []
    error = str(
        run_result.merged_query_bundle.error
        or run_result.merged_query_bundle.summary
        or "query execution failed before producing task evidence"
    )
    return [
        EvidenceGap(
            code="EXECUTION_OPERATIONAL_FAILURE",
            evidence=str(run_result.executed_query_graph_fingerprint or "execution"),
            reason=error[:500],
            severity="blocking",
            disclosure_required=True,
            source="execution",
            answer_instruction="说明本轮执行链路失败且没有形成可验证任务结果，不能输出业务结论或把失败解释为 0。",
            suggested_action="retry_execution_or_answer_with_operational_gap",
        )
    ]


def planner_degraded_evidence_gaps(run_result: AgentRunResult | None) -> List[EvidenceGap]:
    gaps: List[EvidenceGap] = []
    for degraded in planner_degraded_reasons(run_result):
        fallback_used = bool(degraded.get("fallbackUsed") or degraded.get("fallback_used"))
        coverage_passed = bool(
            degraded.get("fallbackCoveragePassed") or degraded.get("fallback_coverage_passed")
        )
        safe_fallback = fallback_used and coverage_passed
        planner_code = str(degraded.get("code") or "PLANNER_OPERATIONAL_FAILURE")
        reason = str(degraded.get("reason") or planner_code)
        gaps.append(
            EvidenceGap(
                code="PLANNER_DEGRADED_FALLBACK" if safe_fallback else "PLANNER_OPERATIONAL_FAILURE",
                evidence=planner_code,
                reason=(
                    "Planner 服务异常，本轮使用了通过问题覆盖校验的确定性降级规划；业务结果仍需披露该恢复来源"
                    if safe_fallback
                    else "Planner 服务异常且没有通过问题覆盖校验的安全规划，不能输出业务结论"
                ),
                severity="warning" if safe_fallback else "blocking",
                disclosure_required=True,
                source="planner",
                answer_instruction=(
                    "说明 Planner 服务异常，本轮由通过覆盖校验的确定性降级规划完成取数；不要表述为正常规划链路。"
                    if safe_fallback
                    else "说明 Planner 服务异常且未形成安全可执行规划，不能据此输出业务结论。"
                ),
                suggested_action=(
                    "answer_with_planner_fallback_disclosure"
                    if safe_fallback
                    else "retry_planner_or_answer_with_operational_gap"
                ),
                details={
                    "plannerCode": planner_code,
                    "plannerReason": reason[:500],
                    "fallbackUsed": fallback_used,
                    "fallbackCoveragePassed": coverage_passed,
                    "plannerTrace": list(degraded.get("trace") or [])[:12],
                },
            )
        )
    return gaps


def planner_degraded_disclosures(run_result: AgentRunResult | None) -> List[str]:
    disclosures: List[str] = []
    for degraded in planner_degraded_reasons(run_result):
        fallback_used = bool(degraded.get("fallbackUsed") or degraded.get("fallback_used"))
        coverage_passed = bool(
            degraded.get("fallbackCoveragePassed") or degraded.get("fallback_coverage_passed")
        )
        text = (
            "本轮 Planner 服务异常，取数计划来自通过问题覆盖校验的确定性降级规划。"
            if fallback_used and coverage_passed
            else "本轮 Planner 服务异常，未形成通过问题覆盖校验的安全取数计划。"
        )
        if text not in disclosures:
            disclosures.append(text)
    return disclosures


def snapshot_alignment_incomplete(run_result: AgentRunResult | None) -> bool:
    if not snapshot_alignment_applicable(run_result):
        return False
    alignment = run_result.snapshot_alignment
    return not bool(alignment.aligned and alignment.complete)


def snapshot_alignment_evidence_gaps(run_result: AgentRunResult | None) -> List[EvidenceGap]:
    """Turn runtime snapshot coverage into answer-blocking evidence contracts.

    A missing component row is not evidence of a business zero.  The executor's
    typed snapshot contract is the authority for whether sources cover the same
    physical window; the verifier must therefore fail closed whenever that
    contract is incomplete.
    """

    if not snapshot_alignment_incomplete(run_result):
        return []
    alignment = run_result.snapshot_alignment
    gaps: List[EvidenceGap] = []
    common_details = {
        "snapshotStatus": str(alignment.status or ""),
        "strategy": str(alignment.strategy or ""),
        "commonAnchorTimeValue": str(alignment.common_anchor_time_value or ""),
        "aligned": bool(alignment.aligned),
        "complete": bool(alignment.complete),
    }
    uncovered_sources = [
        source
        for source in alignment.sources
        if not bool(source.compatible) or not bool(source.coverage_complete)
    ]
    for source in uncovered_sources:
        compatible = bool(source.compatible)
        code = "SNAPSHOT_SOURCE_COVERAGE_INCOMPLETE" if compatible else "SNAPSHOT_SOURCE_UNAVAILABLE"
        reason = (
            "数据来源未完整覆盖本轮统一时间窗口，相关结果不可用于完整结论，且不能把缺失解释为 0"
            if compatible
            else "数据来源未能绑定到本轮统一时间窗口，相关结果不可用，且不能把缺失解释为 0"
        )
        expected_range = snapshot_source_expected_range(source)
        details = {
            **common_details,
            "taskId": str(source.task_id or ""),
            "table": str(source.table or ""),
            "sourceStatus": str(source.status or ""),
            "sourceMinTimeValue": str(source.source_min_time_value or ""),
            "sourceMaxTimeValue": str(source.source_max_time_value or ""),
            "effectiveStartTimeValue": str(source.effective_start_time_value or ""),
            "effectiveEndTimeValue": str(source.effective_end_time_value or ""),
            "executionStartValue": str(source.execution_start_value or ""),
            "executionEndValue": str(source.execution_end_value or ""),
            "compatible": compatible,
            "coverageComplete": bool(source.coverage_complete),
            "sourceReason": str(source.reason or ""),
        }
        gaps.append(
            EvidenceGap(
                code=code,
                task_id=str(source.task_id or ""),
                evidence=str(source.table or source.task_id or "snapshot_source"),
                reason=reason,
                severity="blocking",
                disclosure_required=True,
                source="freshness",
                answer_instruction="说明该数据来源未完整覆盖统一时间窗口，相关结果不可用，不能把缺失解释为 0。",
                suggested_action="align_source_windows_or_answer_with_gap",
                missing_time_range=expected_range,
                details={key: value for key, value in details.items() if value not in ("", None, [], {})},
            )
        )
    if not alignment.aligned or not gaps:
        gaps.insert(
            0,
            EvidenceGap(
                code="SNAPSHOT_ALIGNMENT_INCOMPLETE",
                evidence="snapshot_alignment",
                reason="本轮所需数据来源未完成统一时间对齐，不能进行完整比较，且不能把未覆盖区间解释为 0",
                severity="blocking",
                disclosure_required=True,
                source="freshness",
                answer_instruction="说明本轮数据来源未完成统一时间对齐，未对齐结果不可用，不能把缺失解释为 0。",
                suggested_action="align_source_windows_or_answer_with_gap",
                missing_time_range=snapshot_time_display(alignment.common_anchor_time_value),
                details={key: value for key, value in common_details.items() if value not in ("", None, [], {})},
            ),
        )
    return gaps


def snapshot_alignment_disclosures(run_result: AgentRunResult | None) -> List[str]:
    if not snapshot_alignment_applicable(run_result):
        return []
    alignment = run_result.snapshot_alignment
    anchor = snapshot_time_display(alignment.common_anchor_time_value)
    disclosures: List[str] = []
    if anchor:
        disclosures.append("本轮可比数据已统一到同一时间口径，数据截至 %s。" % anchor)
    if snapshot_alignment_incomplete(run_result):
        unavailable_count = sum(
            1
            for source in alignment.sources
            if not bool(source.compatible) or not bool(source.coverage_complete)
        )
        if unavailable_count:
            disclosures.append(
                "有 %d 个数据来源未完整覆盖统一时间窗口，相关结果不可用，不能把缺失解释为 0。"
                % unavailable_count
            )
        else:
            disclosures.append("本轮数据来源未完成统一时间对齐，未对齐结果不可用，不能把缺失解释为 0。")
    return disclosures


def snapshot_source_expected_range(source: Any) -> str:
    start = str(source.effective_start_time_value or source.execution_start_value or "").strip()
    end = str(source.effective_end_time_value or source.execution_end_value or "").strip()
    if start and end:
        return "%s..%s" % (start, end)
    return start or end


def snapshot_time_display(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) == 8 and text.isascii() and text.isdigit():
        return "%s-%s-%s" % (text[:4], text[4:6], text[6:8])
    date_parts = leading_iso_date_parts(text)
    if date_parts is not None:
        year, month, day = date_parts
        return "%s-%s-%s" % (year, month, day)
    return text[:40]


def entity_filter_evidence_gaps(run_result: AgentRunResult) -> List[EvidenceGap]:
    """Consume the pre-mask identity proof; never re-check display rows."""

    gaps: List[EvidenceGap] = []
    for task in run_result.task_results:
        contract = task.node_plan_contract
        obligations = [
            item
            for item in contract.entity_filter_obligations
            if item.required and item.status == "bound"
        ]
        if not obligations or task.query_bundle.failed:
            continue
        column = str(contract.filter_column or "")
        proof = task.entity_filter_verification
        expected_contract_hash = entity_filter_contract_hash(contract)
        expected_sql_hash = entity_filter_sql_hash(task.query_bundle.sql)
        if (
            proof.status == "not_required"
            or proof.contract_hash != expected_contract_hash
            or proof.sql_hash != expected_sql_hash
        ):
            gaps.append(
                EvidenceGap(
                    code="ENTITY_FILTER_VERIFICATION_MISSING",
                    task_id=task.task_id,
                    evidence=column,
                    reason="实体过滤执行结果缺少与当前 SQL/契约绑定的内部验证证明",
                    severity="blocking",
                    source="entity_filter",
                    answer_instruction="不能把这些结果表述为用户指定 ID 的明细。",
                )
            )
            continue
        if not proof.verified:
            gaps.append(
                EvidenceGap(
                    code=proof.code or "ENTITY_FILTER_RESULT_UNVERIFIABLE",
                    task_id=task.task_id,
                    evidence=column,
                    reason=proof.reason or "实体过滤内部验证未通过",
                    severity="blocking",
                    source="entity_filter",
                    answer_instruction="不能使用对象身份与请求不一致的明细结果。",
                    details={
                        "proofStatus": proof.status,
                        "unexpectedValueCount": proof.unexpected_value_count,
                        "coverageComplete": proof.coverage_complete,
                    },
                )
            )
            continue
        missing = list(proof.missing_values or [])
        if missing:
            gaps.append(
                EvidenceGap(
                    code="ENTITY_FILTER_VALUE_NOT_FOUND",
                    task_id=task.task_id,
                    evidence=column,
                    reason="部分请求 ID 在已完成查询中没有匹配记录",
                    severity="warning",
                    disclosure_required=True,
                    source="entity_filter",
                    answer_instruction="逐项说明哪些请求 ID 未返回记录，不要把其他 ID 的结果替代它们。",
                    details={"missingValues": missing[:20]},
                )
            )
    return gaps


def semantic_filter_evidence_gaps(run_result: AgentRunResult) -> List[EvidenceGap]:
    """Require an SQL-bound proof for every immutable user-filter obligation."""

    gaps: List[EvidenceGap] = []
    for task in run_result.task_results:
        if task.query_bundle.failed:
            continue
        contract = task.node_plan_contract
        nodes = {
            item.node_id: item
            for item in contract.semantic_query.filter_nodes
            if item.node_id and item.node_type == "predicate"
        }
        obligations = [item for item in contract.semantic_filter_obligations if item.required]
        if nodes and not obligations:
            gaps.append(
                EvidenceGap(
                    code="SEMANTIC_FILTER_OBLIGATION_MISSING",
                    task_id=task.task_id,
                    evidence=contract.semantic_query.root_filter_node_id,
                    reason="执行结果包含语义过滤图，但缺少规划阶段冻结的用户条件义务",
                    severity="blocking",
                    source="semantic_filter",
                    answer_instruction="不能使用缺少用户条件义务账本的查询结果回答问题。",
                )
            )
            continue
        if not obligations:
            continue
        proof = task.semantic_filter_verification
        expected_contract_hash = semantic_filter_contract_hash(contract)
        expected_sql_hash = semantic_filter_execution_hash(
            task.query_bundle.sql,
            task.query_bundle.params,
        )
        if (
            proof.status == "not_required"
            or proof.contract_hash != expected_contract_hash
            or proof.sql_hash != expected_sql_hash
        ):
            gaps.append(
                EvidenceGap(
                    code="SEMANTIC_FILTER_VERIFICATION_MISSING",
                    task_id=task.task_id,
                    evidence=contract.semantic_query.root_filter_node_id,
                    reason="语义过滤执行结果缺少与当前 SQL 和条件契约绑定的验证证明",
                    severity="blocking",
                    source="semantic_filter",
                    answer_instruction="不能声称结果满足了用户给出的全部过滤条件。",
                    details={
                        "requiredCount": len(obligations),
                        "proofStatus": proof.status,
                    },
                )
            )
            continue
        obligation_by_id = {item.obligation_id: item for item in obligations}
        proof_by_id = {
            item.obligation_id: item
            for item in proof.predicate_proofs
            if item.obligation_id
        }
        ledger_mismatches: List[str] = []
        if len(obligation_by_id) != len(obligations) or len(proof_by_id) != len(proof.predicate_proofs):
            ledger_mismatches.append("duplicate_obligation_or_proof_id")
        for obligation_id, obligation in obligation_by_id.items():
            item = proof_by_id.get(obligation_id)
            node_id = str(obligation.node_id or obligation.predicate_id or "")
            node = nodes.get(node_id)
            if item is None or node is None:
                ledger_mismatches.append(obligation_id)
                continue
            if (
                item.node_id != node_id
                or item.semantic_ref_id != obligation.semantic_ref_id
                or item.bound_table != obligation.bound_table
                or item.bound_field != obligation.bound_field
                or str(item.member_kind or "").lower() != str(obligation.member_kind or "").lower()
                or str(item.operator or "").lower() != str(obligation.operator or "").lower()
                or sorted(item.resolved_value_hashes)
                != sorted(semantic_filter_value_hash(value) for value in obligation.resolved_values)
                or not item.verified
            ):
                ledger_mismatches.append(obligation_id)
        if set(proof_by_id) != set(obligation_by_id):
            ledger_mismatches.append("proof_obligation_set_mismatch")
        if ledger_mismatches:
            gaps.append(
                EvidenceGap(
                    code="SEMANTIC_FILTER_PROOF_LEDGER_MISMATCH",
                    task_id=task.task_id,
                    evidence=contract.semantic_query.root_filter_node_id,
                    reason="语义过滤证明与规划义务账本不一致",
                    severity="blocking",
                    source="semantic_filter",
                    answer_instruction="不能使用条件证明被替换、遗漏或重复的查询结果。",
                    details={"mismatchIds": sorted(set(ledger_mismatches))[:20]},
                )
            )
            continue
        if (
            not proof.verified
            or not proof.coverage_complete
            or proof.required_count != len(obligations)
            or proof.verified_count != len(obligations)
        ):
            gaps.append(
                EvidenceGap(
                    code=proof.code or "SEMANTIC_FILTER_VERIFICATION_FAILED",
                    task_id=task.task_id,
                    evidence=contract.semantic_query.root_filter_node_id,
                    reason=proof.reason or "部分用户过滤条件没有进入最终 SQL，或布尔结构发生变化",
                    severity="blocking",
                    source="semantic_filter",
                    answer_instruction="不能声称结果满足了用户给出的全部过滤条件。",
                    details={
                        "requiredCount": len(obligations),
                        "verifiedCount": proof.verified_count,
                        "coverageComplete": proof.coverage_complete,
                    },
                )
            )
    return gaps


def classify_evidence_gap(gap: EvidenceGap) -> EvidenceGap:
    warning_codes = {
        "FIELD_AMBIGUOUS",
        "ENTITY_SET_TRUNCATED",
        "OPTIONAL_EVIDENCE_MISSING",
        "RESOURCE_DEGRADED_QUERY",
        "MEMORY_METRIC_DISPUTE_REQUIRES_CLARIFICATION",
        "ENTITY_FILTER_VALUE_NOT_FOUND",
    }
    info_codes: Set[str] = set()
    default_severity = "warning" if gap.code in warning_codes else "info" if gap.code in info_codes else "blocking"
    severity = normalize_evidence_severity(gap.severity, default_severity)
    instruction = gap.answer_instruction or answer_instruction_for_gap(gap)
    details = dict(gap.details or {})
    details.setdefault("gapCode", gap.gap_code or gap.code)
    details.setdefault("sourceNodeId", gap.source_node_id or gap.task_id)
    details.setdefault("evidence", gap.evidence)
    return gap.model_copy(
        update={
            "gap_code": gap.gap_code or gap.code,
            "source_node_id": gap.source_node_id or gap.task_id,
            "severity": severity,
            "disclosure_required": gap.disclosure_required or severity in {"blocking", "warning"},
            "source": gap.source or evidence_gap_source(gap.code),
            "answer_instruction": instruction,
            "suggested_action": gap.suggested_action or suggested_action_for_gap(gap),
            "missing_metric": gap.missing_metric or missing_metric_for_gap(gap),
            "missing_dimension": gap.missing_dimension or missing_dimension_for_gap(gap),
            "missing_time_range": gap.missing_time_range or missing_time_range_for_gap(gap),
            "details": {key: value for key, value in details.items() if value not in ("", None, [], {})},
        }
    )


def normalize_evidence_severity(value: Any, default: str = "blocking") -> str:
    """Map producer vocabularies onto the verifier's closed severity set."""

    normalized = str(value or "").strip().lower()
    if not normalized:
        return default
    if normalized in {"fatal", "critical", "error", "block", "blocked", "blocking"}:
        return "blocking"
    if normalized in {"warn", "warning", "degraded", "partial"}:
        return "warning"
    if normalized in {"info", "informational"}:
        return "info"
    # An unknown producer severity must not create a new fail-open state.
    return "blocking"


def dedupe_evidence_gaps(gaps: List[EvidenceGap]) -> List[EvidenceGap]:
    """Deduplicate retries without weakening the strongest gap contract."""

    deduped: List[EvidenceGap] = []
    positions: Dict[tuple[str, str, str, str], int] = {}
    severity_rank = {"": 0, "info": 1, "warning": 2, "blocking": 3}
    for gap in gaps:
        identity = (
            str(gap.gap_code or gap.code or ""),
            str(gap.source_node_id or gap.task_id or ""),
            str(gap.evidence or ""),
            str(gap.reason or ""),
        )
        position = positions.get(identity)
        if position is None:
            positions[identity] = len(deduped)
            deduped.append(gap)
            continue
        current = deduped[position]
        if severity_rank.get(str(gap.severity or ""), 0) > severity_rank.get(str(current.severity or ""), 0):
            deduped[position] = gap
    return deduped


def evidence_gap_source(code: str) -> str:
    if code.startswith("SNAPSHOT_"):
        return "freshness"
    if code.startswith("MISSING") or code in {"ZERO_ROWS"}:
        return "contract"
    if code.startswith("UPSTREAM") or code in {"JOIN_KEY_NOT_PRODUCED", "DEPENDENCY_GAP"}:
        return "dependency"
    if code.startswith("DERIVED_METRIC"):
        return "calculation"
    if code.startswith("MEMORY_"):
        return "memory"
    if code.startswith("ENTITY_FILTER_"):
        return "entity_filter"
    if code in {"FIELD_AMBIGUOUS"}:
        return "field_policy"
    return "task"


def answer_instruction_for_gap(gap: EvidenceGap) -> str:
    if gap.code.startswith("SNAPSHOT_"):
        return "说明本轮数据来源未完整覆盖统一时间窗口，相关结果不可用，不能把缺失解释为 0。"
    if gap.code == "FIELD_AMBIGUOUS":
        return "说明该字段或指标口径未完全确认，不能把候选值表述为已确认事实。"
    if gap.code == "ZERO_ROWS":
        return "说明 SQL 成功但返回 0 行，不能解释为业务指标为 0。"
    if gap.code.startswith("UPSTREAM") or gap.code == "JOIN_KEY_NOT_PRODUCED":
        return "说明下游节点因上游实体缺失或依赖键缺失未完整执行。"
    if gap.code.startswith("DERIVED_METRIC"):
        return "说明派生指标缺少可计算的分子/分母或组件证据，不能把缺失解释为 0。"
    if gap.code.startswith("MEMORY_CONSTRAINT"):
        return "说明长期记忆约束未被本轮 QueryGraph/证据覆盖，不要声称已应用该历史偏好或纠错。"
    if gap.code.startswith("MEMORY_METRIC_DISPUTE"):
        return "说明长期记忆存在口径争议，当前仍以语义层/指标中心定义为准。"
    if gap.code in {"SQL_EXECUTION_FAILED", "UNKNOWN_COLUMN", "MEM_ALLOC_FAILED", "TIMEOUT"}:
        return "说明该节点 SQL/执行失败，不能基于该节点输出业务结论。"
    if gap.code == "RESOURCE_DEGRADED_QUERY":
        return "说明该节点因 Doris 资源限制使用降级 SQL，结论只能按部分覆盖理解。"
    return "说明该证据缺口对回答范围的影响。"


def suggested_action_for_gap(gap: EvidenceGap) -> str:
    if gap.code.startswith("SNAPSHOT_"):
        return "align_source_windows_or_answer_with_gap"
    if gap.code in {"ZERO_ROWS"}:
        return "answer_with_zero_rows_disclosure"
    if gap.code in {"SQL_EXECUTION_FAILED", "UNKNOWN_COLUMN", "MEM_ALLOC_FAILED", "TIMEOUT"}:
        return "retry_repair_or_answer_with_gap"
    if gap.code.startswith("MISSING") or gap.code.startswith("DERIVED_METRIC"):
        return "supplement_evidence_or_answer_with_gap"
    if gap.code.startswith("UPSTREAM") or gap.code == "JOIN_KEY_NOT_PRODUCED":
        return "repair_dependency_or_answer_with_gap"
    if gap.code.startswith("MEMORY_"):
        return "disclose_memory_constraint_gap"
    return "answer_with_gap"


def missing_metric_for_gap(gap: EvidenceGap) -> str:
    text = " ".join([gap.evidence or "", gap.reason or ""])
    return _value_after_markers(text, ("metric", "指标"))


def missing_dimension_for_gap(gap: EvidenceGap) -> str:
    text = " ".join([gap.evidence or "", gap.reason or ""])
    return _value_after_markers(text, ("dimension", "维度", "字段"))


def _value_after_markers(text: str, markers: tuple[str, ...]) -> str:
    spans = literal_spans(
        text,
        markers,
        case_sensitive=False,
    )
    for _, end, _ in spans:
        cursor = end
        separator_seen = False
        while cursor < len(text) and (
            text[cursor].isspace() or text[cursor] in {":", "：", "="}
        ):
            separator_seen = True
            cursor += 1
        if not separator_seen:
            continue
        start = cursor
        while cursor < len(text):
            character = text[cursor]
            allowed = (
                character.isascii()
                and (
                    character.isalnum()
                    or character in {"_", ".", "-"}
                )
            ) or "\u4e00" <= character <= "\u9fff"
            if not allowed:
                break
            cursor += 1
        if cursor > start:
            return text[start:cursor]
    return ""


def missing_time_range_for_gap(gap: EvidenceGap) -> str:
    text = " ".join([gap.evidence or "", gap.reason or ""])
    if "time" in text.lower() or "时间" in text or "日期" in text:
        return gap.evidence or gap.reason
    return ""


def metric_resolution_disclosures(plan: QueryPlan) -> List[str]:
    disclosures: List[str] = []
    for resolution in metric_resolutions(plan):
        warning = str(resolution.get("fieldWarning") or resolution.get("field_warning") or "")
        confidence = float(resolution.get("confidence") or 0)
        if warning and confidence >= 0.7 and warning not in disclosures:
            disclosures.append(warning)
    return disclosures


def metric_resolution_warning_gaps(plan: QueryPlan) -> List[EvidenceGap]:
    gaps: List[EvidenceGap] = []
    for resolution in metric_resolutions(plan):
        confidence = float(resolution.get("confidence") or 0)
        if not resolution.get("metricKey") and resolution.get("requestedMetricRef"):
            gaps.append(
                EvidenceGap(
                    code="FIELD_AMBIGUOUS",
                    evidence=str(resolution.get("requestedMetricRef") or ""),
                    reason="语义层没有找到可确认的指标口径",
                    severity="warning",
                    disclosure_required=True,
                    source="field_policy",
                )
            )
        elif 0 < confidence < 0.7:
            gaps.append(
                EvidenceGap(
                    code="FIELD_AMBIGUOUS",
                    evidence=str(resolution.get("requestedMetricRef") or resolution.get("metricKey") or ""),
                    reason=str(resolution.get("fieldWarning") or "指标由弱匹配得到，口径需要人工确认"),
                    severity="warning",
                    disclosure_required=True,
                    source="field_policy",
                )
            )
    return gaps


def metric_resolution_confirmed(plan: QueryPlan) -> bool:
    for resolution in metric_resolutions(plan):
        if float(resolution.get("confidence") or 0) >= 0.7 and resolution.get("metricKey"):
            return True
    return False


def metric_resolutions(plan: QueryPlan) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for intent in plan.intents:
        if intent.metric_resolution:
            items.append(intent.metric_resolution)
    for contract in plan.evidence_contracts:
        resolution = contract.get("metricResolution") or contract.get("metric_resolution")
        if isinstance(resolution, dict):
            items.append(resolution)
    deduped: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for item in items:
        key = "%s:%s:%s" % (item.get("requestedMetricRef"), item.get("ownerTable"), item.get("metricKey"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def classify_task_failure(message: str) -> str:
    text = message or ""
    lower = text.lower()
    if "mem_alloc_failed" in lower or "memory limit" in lower or "low water mark" in lower:
        return "MEM_ALLOC_FAILED"
    if "unknown column" in lower or ("unknown" in lower and "column" in lower):
        return "UNKNOWN_COLUMN"
    if "timeout" in lower or "timed out" in lower:
        return "TIMEOUT"
    for code in [
        "DERIVED_METRIC_COMPONENTS_MISSING",
        "DERIVED_METRIC_GROUP_KEY_MISSING",
        "DERIVED_METRIC_NO_JOINED_COMPONENT_ROWS",
        "DERIVED_METRIC_ZERO_ROWS",
        "DERIVED_METRIC_FAILED",
        "JOIN_KEY_NOT_PRODUCED",
        "JOIN_KEY_VALUES_EMPTY",
        "UPSTREAM_SQL_FAILED",
        "UPSTREAM_ZERO_ROWS",
        "DEPENDENCY_KEY_NOT_IN_SCHEMA",
    ]:
        if code in text:
            return code
    if "上游实体集缺失" in text:
        return "UPSTREAM_ENTITY_MISSING"
    return "SQL_EXECUTION_FAILED"
