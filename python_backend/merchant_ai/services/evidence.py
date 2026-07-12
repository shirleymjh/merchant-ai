from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from merchant_ai.models import AgentRunResult, EvidenceGap, QueryPlan, VerifiedEvidence
from merchant_ai.services.memory_constraints import memory_constraint_evidence_gaps


class EvidenceVerifier:
    def verify(
        self,
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult,
        memory_constraints: List[Dict[str, Any]] | None = None,
        allowed_knowledge_refs: Set[str] | None = None,
    ) -> VerifiedEvidence:
        gaps: List[EvidenceGap] = []
        covered = self._covered_keys(run_result)
        derived_evidence = self._derived_evidence(plan, run_result)
        table_names = set(run_result.merged_query_bundle.tables)
        if plan.evidence_contracts:
            gaps.extend(self._contract_gaps(plan.evidence_contracts, run_result, allowed_knowledge_refs))
            covered.update(self._covered_contract_labels(plan.evidence_contracts, run_result, allowed_knowledge_refs))
        else:
            for evidence in plan.final_required_evidence:
                if not evidence:
                    continue
                if self._natural_evidence_covered(evidence, covered, table_names):
                    continue
                gaps.append(EvidenceGap(code="MISSING_REQUIRED_EVIDENCE", evidence=evidence, reason="finalRequiredEvidence 未被结果字段或使用表覆盖"))
        required_disclosures = metric_resolution_disclosures(plan)
        gaps.extend(metric_resolution_warning_gaps(plan))
        gaps.extend(memory_constraint_evidence_gaps(question, plan, memory_constraints or []))
        for gap in run_result.evidence_check.gaps:
            gaps.append(EvidenceGap(code="DEPENDENCY_GAP", evidence=gap, reason=gap))
        succeeded_tasks = [item for item in run_result.task_results if not item.query_bundle.failed]
        failed_tasks = [item for item in run_result.task_results if item.query_bundle.failed]
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
            if task_result.entity_set and task_result.entity_set.truncated:
                gaps.append(
                    EvidenceGap(
                        code="ENTITY_SET_TRUNCATED",
                        task_id=task_result.task_id,
                        evidence=task_result.entity_set.join_key,
                        reason="上游实体超过本轮最大传递数量，dependent 结果可能只覆盖部分实体",
                    )
                )
        gaps = [classify_evidence_gap(gap) for gap in gaps]
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
            if not intent.metric_formula:
                continue
            task_result = results_by_task.get(intent.plan_task_id)
            row_columns = self._row_columns(task_result.query_bundle.rows) if task_result else set()
            metric_alias = intent.metric_name or intent.metric_column
            covered = bool(metric_alias and metric_alias in row_columns)
            if not covered and metric_alias:
                covered = self._column_covered(metric_alias, row_columns)
            derived.append(
                {
                    "taskId": intent.plan_task_id,
                    "table": intent.preferred_table,
                    "metric": metric_alias,
                    "formula": intent.metric_formula,
                    "sourceColumns": formula_columns(intent.metric_formula),
                    "covered": covered,
                }
            )
        return derived

    def _contract_gaps(
        self,
        contracts: List[Dict[str, Any]],
        run_result: AgentRunResult,
        allowed_knowledge_refs: Set[str] | None = None,
    ) -> List[EvidenceGap]:
        gaps: List[EvidenceGap] = []
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
                        code=missing_gap_code(missing_columns),
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
        for task_result in run_result.task_results:
            if task_id and task_result.task_id == task_id:
                return task_result
        for task_result in run_result.task_results:
            if table and table in task_result.query_bundle.tables:
                return task_result
        return None

    def _row_columns(self, rows: List[Dict[str, Any]]) -> Set[str]:
        columns: Set[str] = set()
        for row in rows[:20]:
            columns.update(str(key) for key in row.keys())
        return columns

    def _column_covered(self, column: str, row_columns: Set[str], semantic_aliases: Dict[str, Set[str]] | None = None) -> bool:
        if column in row_columns:
            return True
        aliases = {
            "refund_related_pay_amt": {"refund_related_pay_amt", "refund_related_pay_amt_raw", "pay_amt", "sum_pay_amt"},
            "order_cnt": {"order_cnt", "cnt", "count", "sub_order_cnt"},
            "refund_cnt": {"refund_cnt", "cnt", "count", "refund_bill_cnt"},
            "ticket_cnt": {"ticket_cnt", "cnt", "count", "ticket_bill_cnt"},
            "repay_cnt": {"repay_cnt", "cnt", "count", "repay_bill_cnt"},
            "coupon_cnt": {"coupon_cnt", "cnt", "count"},
            "scm_cnt": {"scm_cnt", "cnt", "count"},
            "goods_cnt": {"goods_cnt", "cnt", "count"},
            "repay_amt": {"repay_amt", "sum_repay_amt"},
            "order_pay_amt": {"order_pay_amt", "pay_amt", "sum_pay_amt"},
        }
        if semantic_aliases:
            for key, values in semantic_aliases.items():
                aliases.setdefault(key, set()).update(values)
        return bool(aliases.get(column, set()) & row_columns)

    def _natural_evidence_covered(self, evidence: str, covered: Set[str], table_names: Set[str]) -> bool:
        if evidence in covered or evidence in table_names:
            return True
        evidence_text = evidence.lower()
        if any(table and table.lower() in evidence_text for table in table_names):
            return True
        return any(key and key.lower() in evidence_text for key in covered)


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
    columns: List[str] = []
    for token in re.findall(r"`?([A-Za-z_][A-Za-z0-9_]*)`?", formula or ""):
        if token.upper() in keywords:
            continue
        if token not in columns:
            columns.append(token)
    return columns


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


def missing_gap_code(columns: List[str]) -> str:
    if all(is_entity_key(column) for column in columns):
        return "MISSING_ENTITY_KEY"
    if any(is_metric_key(column) for column in columns):
        return "MISSING_METRIC_EVIDENCE"
    return "MISSING_REQUIRED_COLUMNS"


def classify_evidence_gap(gap: EvidenceGap) -> EvidenceGap:
    warning_codes = {
        "FIELD_AMBIGUOUS",
        "ENTITY_SET_TRUNCATED",
        "OPTIONAL_EVIDENCE_MISSING",
        "RESOURCE_DEGRADED_QUERY",
        "MEMORY_METRIC_DISPUTE_REQUIRES_CLARIFICATION",
    }
    info_codes: Set[str] = set()
    severity = "warning" if gap.code in warning_codes else "info" if gap.code in info_codes else "blocking"
    instruction = gap.answer_instruction or answer_instruction_for_gap(gap)
    details = dict(gap.details or {})
    details.setdefault("gapCode", gap.gap_code or gap.code)
    details.setdefault("sourceNodeId", gap.source_node_id or gap.task_id)
    details.setdefault("evidence", gap.evidence)
    return gap.model_copy(
        update={
            "gap_code": gap.gap_code or gap.code,
            "source_node_id": gap.source_node_id or gap.task_id,
            "severity": gap.severity or severity,
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


def evidence_gap_source(code: str) -> str:
    if code.startswith("MISSING") or code in {"ZERO_ROWS"}:
        return "contract"
    if code.startswith("UPSTREAM") or code in {"JOIN_KEY_NOT_PRODUCED", "DEPENDENCY_GAP"}:
        return "dependency"
    if code.startswith("DERIVED_METRIC"):
        return "calculation"
    if code.startswith("MEMORY_"):
        return "memory"
    if code in {"FIELD_AMBIGUOUS"}:
        return "field_policy"
    return "task"


def answer_instruction_for_gap(gap: EvidenceGap) -> str:
    if gap.code == "FIELD_AMBIGUOUS":
        return "说明该指标口径未完全确认，不能表述为已确认的独立退款金额。"
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
    match = re.search(r"(metric|指标)[:=： ]+([A-Za-z0-9_\\.\\-\\u4e00-\\u9fff]+)", text)
    return match.group(2) if match else ""


def missing_dimension_for_gap(gap: EvidenceGap) -> str:
    text = " ".join([gap.evidence or "", gap.reason or ""])
    match = re.search(r"(dimension|维度|字段)[:=： ]+([A-Za-z0-9_\\.\\-\\u4e00-\\u9fff]+)", text)
    return match.group(2) if match else ""


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


def is_entity_key(column: str) -> bool:
    return column in {"order_id", "sub_order_id", "spu_id", "spu_name", "refund_id", "ticket_id", "bill_id", "coupon_id", "discount_rel_id", "pt"}


def is_metric_key(column: str) -> bool:
    text = column.lower()
    return any(token in text for token in ["amt", "cnt", "count", "rate", "gmv", "pay", "repay"])
