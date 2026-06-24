from __future__ import annotations

from typing import Dict, List, Optional

from merchant_ai.config import Settings
from merchant_ai.graph.state import AgentState
from merchant_ai.models import AgentAction, AgentDecision, AnswerMode, IntentType, PlannerReflectionResult, QuestionRoute


MAX_MAIN_AGENT_ACTIONS = 18
MAX_RETRIEVE_ACTIONS = 3
MAX_PLAN_ACTIONS = 1
MAX_GRAPH_REPAIR_ACTIONS = 2


class AgentActionRegistry:
    """Registry of Lead Agent actions and their LangGraph node bindings."""

    def __init__(self):
        actions = [
            AgentAction(id="route_topic", node="route_topic", agent="LeadAgent", description="route BI topic"),
            AgentAction(id="retrieve_knowledge", node="retrieve_knowledge", agent="KnowledgeAgent", description="retrieve semantic rules, terms and thin skill policies"),
            AgentAction(id="compact_assets", node="compact_assets", agent="KnowledgeAgent", description="build semantic asset pack"),
            AgentAction(id="plan_graph", node="plan_query_graph", agent="PlannerAgent", description="build QueryGraph"),
            AgentAction(id="reflect_plan", node="reflect_query_graph", agent="PlannerCriticAgent", description="review QueryGraph before validation"),
            AgentAction(id="validate_graph", node="validate_query_graph", agent="PlannerCriticAgent", description="validate QueryGraph against asset pack"),
            AgentAction(id="repair_graph", node="repair_query_graph", agent="PlannerAgent", description="repair QueryGraph"),
            AgentAction(id="execute_graph", node="execute_query_graph", agent="NodeAgent", description="execute QueryGraph nodes"),
            AgentAction(id="repair_sql", node="repair_sql", agent="NodeAgent", description="review node-level SQL repair status"),
            AgentAction(id="verify_evidence", node="verify_evidence_graph", agent="EvidenceVerifierAgent", description="verify evidence before answer"),
            AgentAction(id="answer_rule", node="answer_rule", agent="RuleAnswerAgent", description="answer platform rule questions from recalled knowledge"),
            AgentAction(id="answer_data", node="answer_analysis", agent="AnswerAgent", description="compose BI answer from verified evidence"),
            AgentAction(id="answer", node="answer_analysis", agent="AnswerAgent", description="legacy alias for answer_data"),
            AgentAction(id="ask_human", node="human_in_loop", agent="LeadAgent", description="request clarification"),
            AgentAction(id="cache_answer", node="cache_answer", agent="LeadAgent", description="cache final answer"),
        ]
        self._by_id: Dict[str, AgentAction] = {action.id: action for action in actions}
        self._by_node: Dict[str, AgentAction] = {action.node: action for action in actions}

    def get(self, action_id: str) -> AgentAction:
        return self._by_id.get(action_id) or AgentAction(id=action_id, node=action_id, agent="LeadAgent")

    def by_node(self, node: str) -> AgentAction:
        return self._by_node.get(node) or AgentAction(id=node, node=node, agent="LeadAgent")

    def actions(self, action_ids: List[str]) -> List[AgentAction]:
        return [self.get(action_id) for action_id in action_ids]

    def public_action_ids(self) -> List[str]:
        return list(self._by_id.keys())


class V2AgentPolicy:
    """Dynamic Lead Agent policy backed by an action registry."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings
        self.registry = AgentActionRegistry()

    @property
    def max_main_actions(self) -> int:
        return self.settings.agent_main_rounds if self.settings else MAX_MAIN_AGENT_ACTIONS

    @property
    def max_retrieve_actions(self) -> int:
        return self.settings.agent_retrieve_rounds if self.settings else MAX_RETRIEVE_ACTIONS

    @property
    def max_plan_actions(self) -> int:
        return self.settings.agent_plan_rounds if self.settings else MAX_PLAN_ACTIONS

    @property
    def max_graph_repair_actions(self) -> int:
        return self.settings.agent_graph_repair_rounds if self.settings else MAX_GRAPH_REPAIR_ACTIONS

    def decide(self, state: AgentState) -> AgentDecision:
        action_ids, reason, budget_exhausted = self._candidate_action_ids(state)
        selected_action = action_ids[0] if action_ids else "answer_data"
        action = self.registry.get(selected_action)
        return AgentDecision(
            selected_action=action.id,
            selected_node=action.node,
            available_actions=action_ids,
            reason=reason,
            budget_exhausted=budget_exhausted,
        )

    def next_action(self, state: AgentState) -> str:
        return self.decide(state).selected_node

    def available_actions(self, state: AgentState) -> List[AgentAction]:
        action_ids, _, _ = self._candidate_action_ids(state)
        return self.registry.actions(action_ids)

    def _candidate_action_ids(self, state: AgentState) -> tuple[List[str], str, bool]:
        route = state.get("routing_decision")
        if route and route.route == QuestionRoute.GREETING:
            return ["answer_data"], "greeting route uses lightweight answer", False
        if state.get("human_clarification_required"):
            return ["ask_human"], "human clarification is required", False
        if route and route.route == QuestionRoute.INVALID:
            return ["ask_human"], "question is outside supported business scope", False
        if int(state.get("react_round") or 0) >= self.max_main_actions:
            if state.get("agent_run_result") and not state.get("evidence_graph_verified"):
                return ["verify_evidence"], "evidence must be verified after execution before answer", True
            return ["answer_data"], "main agent action budget exhausted", True
        if not state.get("topic_routed"):
            return ["route_topic"], "topic has not been routed", False
        if not state.get("data_discovered"):
            return ["retrieve_knowledge"], "semantic knowledge has not been retrieved", False
        if self.has_rule_recall_ready(state):
            return ["answer_rule"], "platform rule knowledge is ready; answer without BI QueryGraph", False
        if self.has_rule_answer_plan(state):
            return ["answer_rule"], "platform rule answer plan is ready", False
        if not state.get("planning_assets_compacted"):
            return ["compact_assets"], "planning asset pack has not been compacted", False
        plan = state.get("plan")
        if (
            (not plan or not plan.intents)
            and state.get("pending_knowledge_requests")
            and int(state.get("query_graph_retrieve_count") or 0) < self.max_retrieve_actions
        ):
            return ["retrieve_knowledge", "compact_assets", "answer_data"], "planner requested more semantic knowledge", False
        if (not plan or not plan.intents) and state.get("planner_provider_error"):
            if not state.get("query_graph_validated"):
                return ["validate_graph"], "Planner provider failed; validate empty graph as structured gap", True
            return ["answer_data"], "Planner provider failed; answer with structured planning gap", True
        if (not plan or not plan.intents) and int(state.get("query_graph_plan_attempts") or 0) < self.max_plan_actions:
            return ["plan_graph", "retrieve_knowledge"], "QueryGraph has not been planned", False
        if (
            plan
            and plan.intents
            and state.get("pending_knowledge_requests")
            and int(state.get("query_graph_retrieve_count") or 0) < self.max_retrieve_actions
        ):
            return ["retrieve_knowledge", "compact_assets", "plan_graph"], "Resolver requested scoped semantic metric evidence", False
        if plan and plan.intents and not state.get("query_graph_reflected"):
            return ["reflect_plan", "validate_graph"], "QueryGraph needs planner reflection before validation", False
        reflection = normalize_reflection(state.get("planner_reflection"))
        if reflection and not reflection.passed:
            if (
                reflection.repair_reason == "ANCHOR_MISMATCH"
                and int(state.get("query_graph_plan_attempts") or 0) < self.max_plan_actions
            ):
                return ["plan_graph", "repair_graph", "answer_data"], "planner reflection found anchor mismatch; rerun LLM understanding", False
            if (
                reflection.repair_reason == "ANALYSIS_CONTRACT_MISSING"
                and int(state.get("query_graph_plan_attempts") or 0) < self.max_plan_actions
            ):
                return ["plan_graph", "repair_graph", "answer_data"], "planner reflection found missing analysis evidence contract; rerun LLM understanding", False
            if (
                reflection.repair_reason == "MISSING_REQUIRED_EVIDENCE"
                and not reflection.suggested_knowledge_requests
                and int(state.get("query_graph_plan_attempts") or 0) < self.max_plan_actions
            ):
                return ["plan_graph", "repair_graph", "answer_data"], "planner reflection found uncovered analysis evidence; rerun LLM understanding", False
            has_blocking_issue = any(str(issue.get("severity") or "") == "error" for issue in reflection.issues)
            if (
                reflection.suggested_knowledge_requests
                and (reflection.repair_reason or has_blocking_issue)
                and int(state.get("query_graph_retrieve_count") or 0) < self.max_retrieve_actions
            ):
                return ["retrieve_knowledge", "repair_graph", "answer_data"], "planner reflection requested more knowledge", False
            if int(state.get("query_graph_repair_attempts") or 0) < self.max_graph_repair_actions:
                return ["repair_graph", "answer_data"], "planner reflection found repairable graph issues", False
            return ["answer_data"], "planner reflection failed and repair budget is exhausted", True
        if not state.get("query_graph_validated"):
            return ["validate_graph"], "QueryGraph has not been validated", False
        validation = state.get("query_graph_validation_result")
        if validation and not validation.valid:
            if self.validation_requires_reunderstand(validation) and int(state.get("query_graph_plan_attempts") or 0) < self.max_plan_actions:
                return ["plan_graph", "answer_data"], "validator found contract mismatch; rerun LLM understanding", False
            if validation.recommended_knowledge_requests and int(state.get("query_graph_retrieve_count") or 0) < self.max_retrieve_actions:
                return ["retrieve_knowledge", "repair_graph", "answer_data"], "validator requested more semantic knowledge", False
            if validation.repairable and int(state.get("query_graph_repair_attempts") or 0) < self.max_graph_repair_actions:
                return ["repair_graph", "answer_data"], "validator found repairable graph gaps", False
            return ["answer_data"], "QueryGraph validation failed and cannot be repaired in budget", True
        if self.has_executable_plan(state) and not state.get("sql_generated"):
            return ["execute_graph"], "validated QueryGraph is ready for NodeAgent execution", False
        if self.has_graph_repairable_execution_gap(state):
            if int(state.get("query_graph_repair_attempts") or 0) < self.max_graph_repair_actions:
                return ["repair_graph", "answer_data"], "NodeAgent found graph-repairable plan contract gaps", False
            return ["answer_data"], "graph-repairable execution gaps remain but repair budget is exhausted", True
        if not state.get("sql_repair_reviewed") and self.has_sql_failure(state):
            return ["repair_sql", "verify_evidence"], "one or more node SQL executions failed", False
        if state.get("agent_run_result") and not state.get("evidence_graph_verified"):
            return ["verify_evidence"], "evidence has not been verified", False
        if self.has_graph_repairable_execution_gap(state):
            if int(state.get("query_graph_repair_attempts") or 0) < self.max_graph_repair_actions:
                return ["repair_graph", "answer_data"], "evidence verifier found graph-repairable dependency gaps", False
            return ["answer_data"], "graph-repairable evidence gaps remain but repair budget is exhausted", True
        if not state.get("chat_bi_completed"):
            return ["answer_data"], "ready to compose BI answer", False
        return ["cache_answer"], "answer is complete and can be cached", False

    def has_executable_plan(self, state: AgentState) -> bool:
        plan = state.get("plan")
        if not plan:
            return False
        for intent in plan.intents:
            if intent.intent_type == IntentType.VALID and intent.answer_mode != AnswerMode.RULE:
                return True
        return False

    def has_rule_answer_plan(self, state: AgentState) -> bool:
        if state.get("chat_bi_completed"):
            return False
        plan = state.get("plan")
        if not plan or not plan.intents:
            return False
        return all(intent.intent_type == IntentType.VALID and intent.answer_mode == AnswerMode.RULE for intent in plan.intents)

    def has_rule_recall_ready(self, state: AgentState) -> bool:
        return bool(state.get("rule_recall_ready")) and not state.get("chat_bi_completed")

    def has_sql_failure(self, state: AgentState) -> bool:
        run_result = state.get("agent_run_result")
        if not run_result:
            return False
        return any(result.query_bundle.failed for result in run_result.task_results)

    def has_graph_repairable_execution_gap(self, state: AgentState) -> bool:
        run_result = state.get("agent_run_result")
        if not run_result:
            return False
        repairable_codes = {
            "JOIN_KEY_NOT_PRODUCED",
            "DEPENDENCY_KEY_NOT_IN_SCHEMA",
            "DEPENDENCY_KEY_NOT_PRODUCED",
            "PLAN_CONTRACT_MISMATCH",
            "MISSING_METRIC_COLUMN",
            "MISSING_GROUP_BY_COLUMN",
            "MISSING_OUTPUT_KEY",
            "MISSING_UPSTREAM_ENTITY",
            "CONTRACT_REQUIRED_EVIDENCE_GAP",
        }
        result_by_task = {result.task_id: result for result in run_result.task_results}
        for gap in run_result.evidence_gaps:
            if gap.code not in repairable_codes:
                continue
            if gap.code == "MISSING_UPSTREAM_ENTITY" and upstream_missing_is_execution_result(result_by_task.get(gap.task_id)):
                continue
            return True
        return False

    def validation_requires_reunderstand(self, validation: object) -> bool:
        codes = {str(gap.code) for gap in getattr(validation, "gaps", [])}
        return bool(codes & {"SCOPE_NOT_NARROWING", "OBJECTIVE_NOT_COMPILED"})


def normalize_reflection(value: object) -> Optional[PlannerReflectionResult]:
    if isinstance(value, PlannerReflectionResult):
        return value
    if isinstance(value, dict):
        try:
            return PlannerReflectionResult.model_validate(value)
        except Exception:
            return None
    return None


def upstream_missing_is_execution_result(task_result: object) -> bool:
    contract = getattr(task_result, "node_plan_contract", None)
    if not contract:
        return False
    for entity in getattr(contract, "upstream_entity_sets", []) or []:
        reason = ""
        if isinstance(entity, dict):
            reason = str(entity.get("missingReason") or entity.get("missing_reason") or "")
        if reason in {"UPSTREAM_SQL_FAILED", "UPSTREAM_ZERO_ROWS"}:
            return True
    return False
