from __future__ import annotations

from typing import Dict, List, Optional

from merchant_ai.config import Settings
from merchant_ai.graph.state import AgentState
from merchant_ai.graph.query_graph_contract import graph_validation_attempted, graph_validation_passed
from merchant_ai.models import AgentAction, AgentDecision, AnswerMode, IntentType, PlannerReflectionResult, QuestionRoute, SkillMatchState
from merchant_ai.services.answer import analysis_summary_required, answer_skill_required
from merchant_ai.services.capabilities import CapabilityRegistry, features_from_fast_understanding
from merchant_ai.services.quick_metrics import is_metric_definition_question


MAX_MAIN_AGENT_ACTIONS = 16
MAX_RETRIEVE_ACTIONS = 3
MAX_PLAN_ACTIONS = 1
MAX_GRAPH_REPAIR_ACTIONS = 2
REPAIRABLE_QUERY_GRAPH_GAP_CODES = {
    "JOIN_KEY_NOT_PRODUCED",
    "JOIN_KEY_VALUES_EMPTY",
    "DEPENDENCY_KEY_NOT_IN_SCHEMA",
    "DEPENDENCY_KEY_NOT_PRODUCED",
    "MISSING_DEPENDENCY_KEY",
    "PLAN_CONTRACT_MISMATCH",
    "MISSING_METRIC_COLUMN",
    "MISSING_GROUP_BY_COLUMN",
    "MISSING_OUTPUT_KEY",
    "MISSING_UPSTREAM_ENTITY",
    "CONTRACT_REQUIRED_EVIDENCE_GAP",
    "MEMORY_CONSTRAINT_UNAPPLIED",
}


class AgentActionRegistry:
    """Registry of Lead Agent actions and their LangGraph node bindings."""

    def __init__(self):
        actions = [
            AgentAction(
                id="route_topic",
                node="route_topic",
                agent="LeadAgent",
                description="route BI topic",
                expected_state_flags=["topic_routed"],
            ),
            AgentAction(
                id="fast_understand",
                node="fast_understand",
                agent="LeadAgent",
                description="cheap complexity and intent sketch before heavy planning",
                required_state_flags=["topic_routed"],
                expected_state_flags=["fast_understood"],
                fallback_action="route_topic",
            ),
            AgentAction(
                id="recall_memory",
                node="recall_memory",
                agent="LeadAgent",
                description="recall governed merchant memory after topic routing",
                required_state_flags=["topic_routed"],
                expected_state_flags=["memory_recalled"],
                fallback_action="route_topic",
            ),
            AgentAction(
                id="try_fast_metric",
                node="try_fast_metric",
                agent="LeadAgent",
                description="legacy compatibility path for one uniquely matched published semantic metric",
                required_state_flags=["fast_understood"],
                expected_state_flags=["fast_metric_attempted"],
                fallback_action="retrieve_knowledge",
            ),
            AgentAction(
                id="retrieve_knowledge",
                node="retrieve_knowledge",
                agent="KnowledgeAgent",
                description="retrieve semantic rules, terms and thin skill policies",
                required_state_flags=["topic_routed"],
                expected_state_flags=["data_discovered"],
                fallback_action="route_topic",
            ),
            AgentAction(
                id="compact_assets",
                node="compact_assets",
                agent="KnowledgeAgent",
                description="build semantic asset pack",
                required_state_flags=["data_discovered"],
                expected_state_flags=["planning_assets_compacted"],
                fallback_action="retrieve_knowledge",
            ),
            AgentAction(
                id="query_metric",
                node="query_metric",
                agent="MetricTool",
                description="resolve, validate, execute and verify one governed semantic metric inside the current Topic workspace",
                required_state_flags=["planning_assets_compacted"],
                expected_state_flags=["query_metric_attempted"],
                fallback_action="plan_graph",
            ),
            AgentAction(
                id="plan_graph",
                node="plan_query_graph",
                agent="PlannerAgent",
                description="build QueryGraph",
                required_state_flags=["planning_assets_compacted"],
                expected_state_keys=["plan.intents"],
                fallback_action="compact_assets",
            ),
            AgentAction(
                id="reflect_plan",
                node="reflect_query_graph",
                agent="PlannerCriticAgent",
                description="review QueryGraph before validation",
                required_state_keys=["plan.intents"],
                expected_state_flags=["query_graph_reflected"],
                fallback_action="plan_graph",
            ),
            AgentAction(
                id="validate_graph",
                node="validate_query_graph",
                agent="PlannerCriticAgent",
                description="validate QueryGraph against asset pack",
                required_state_flags=["planning_assets_compacted"],
                expected_state_keys=["query_graph_validation_result"],
                expected_state_flags=["query_graph_validation_attempted"],
                fallback_action="compact_assets",
            ),
            AgentAction(
                id="repair_graph",
                node="repair_query_graph",
                agent="PlannerAgent",
                description="repair QueryGraph",
                required_state_keys=["plan.intents"],
                required_state_flags=["planning_assets_compacted"],
                expected_state_flags=["query_graph_repair_attempted"],
                fallback_action="plan_graph",
            ),
            AgentAction(
                id="execute_graph",
                node="execute_query_graph",
                agent="NodeAgent",
                description="execute QueryGraph nodes",
                required_state_keys=["plan.intents", "query_graph_validation_result"],
                required_state_flags=["query_graph_validation_passed"],
                expected_state_flags=["sql_generated", "sql_repair_reviewed"],
                fallback_action="validate_graph",
            ),
            AgentAction(
                id="execute_graph_direct",
                node="execute_query_graph_direct",
                agent="NodeWorker",
                description="execute the validated QueryGraph with the lightweight direct worker path",
                required_state_keys=["plan.intents", "query_graph_validation_result"],
                required_state_flags=["query_graph_validation_passed"],
                expected_state_flags=["sql_generated", "sql_repair_reviewed"],
                fallback_action="execute_graph_agent",
            ),
            AgentAction(
                id="execute_graph_agent",
                node="execute_query_graph_agent",
                agent="NodeAgent",
                description="execute the validated QueryGraph with autonomous bounded Sub-Agent loops",
                required_state_keys=["plan.intents", "query_graph_validation_result"],
                required_state_flags=["query_graph_validation_passed"],
                expected_state_flags=["sql_generated", "sql_repair_reviewed"],
                fallback_action="execute_graph_direct",
            ),
            AgentAction(
                id="repair_sql",
                node="repair_sql",
                agent="NodeAgent",
                description="review node-level SQL repair status",
                required_state_keys=["agent_run_result.task_results"],
                required_state_flags=["sql_generated"],
                expected_state_flags=["sql_repair_reviewed"],
                fallback_action="execute_graph",
            ),
            AgentAction(
                id="verify_evidence",
                node="verify_evidence_graph",
                agent="EvidenceVerifierAgent",
                description="verify evidence before answer",
                required_state_keys=["agent_run_result.task_results"],
                required_state_flags=["sql_generated"],
                expected_state_flags=["evidence_graph_verified"],
                fallback_action="execute_graph",
            ),
            AgentAction(
                id="run_analysis_skill",
                node="run_analysis_skill",
                agent="SkillWorker",
                description="dynamically dispatch an isolated analysis skill worker before final answer",
                required_state_keys=["agent_run_result.task_results"],
                required_state_flags=["sql_generated", "evidence_graph_verified"],
                expected_state_keys=["analysis_summary"],
                fallback_action="verify_evidence",
            ),
            AgentAction(
                id="run_analysis_worker",
                node="run_analysis_worker",
                agent="AnalysisWorker",
                description="optionally dispatch a generic isolated analysis worker for long-tail analysis before final answer",
                required_state_keys=["agent_run_result.task_results"],
                required_state_flags=["sql_generated", "evidence_graph_verified"],
                expected_state_keys=["analysis_summary"],
                fallback_action="answer_data",
            ),
            AgentAction(
                id="delegate_subagent",
                node="delegate_subagent",
                agent="LeadAgent",
                description="delegate bounded document, Python, hypothesis, Skill or query work to isolated Sub-Agents",
                expected_state_flags=["subagent_delegation_completed"],
                fallback_action="answer_data",
            ),
            AgentAction(
                id="explore_hypotheses",
                node="explore_hypotheses",
                agent="LeadAgent",
                description="generate, validate and execute independent QueryGraphs for competing merchant hypotheses, then prune or expand them",
                expected_state_flags=["hypothesis_exploration_completed"],
                fallback_action="answer_data",
            ),
            AgentAction(
                id="answer_rule",
                node="answer_rule",
                agent="RuleAnswerAgent",
                description="answer platform rule questions from recalled knowledge",
                expected_state_keys=["answer"],
            ),
            AgentAction(
                id="answer_data",
                node="answer_analysis",
                agent="AnswerAgent",
                description="compose BI answer from verified evidence",
                expected_state_keys=["answer"],
            ),
            AgentAction(
                id="ask_human",
                node="human_in_loop",
                agent="LeadAgent",
                description="request clarification",
                expected_state_keys=["answer"],
            ),
            AgentAction(
                id="cache_answer",
                node="cache_answer",
                agent="LeadAgent",
                description="cache final answer",
                expected_state_keys=["answer", "response_context"],
            ),
            AgentAction(
                id="terminal_end",
                node="terminal_end",
                agent="Runtime",
                description="end a terminal run without more tools",
                expected_state_keys=["answer"],
                expected_state_flags=["chat_bi_completed"],
            ),
        ]
        self._aliases = {"answer": "answer_data"}
        self._by_id: Dict[str, AgentAction] = {action.id: action for action in actions}
        self._by_node: Dict[str, AgentAction] = {action.node: action for action in actions}
        self._validate_contract(actions)

    def get(self, action_id: str) -> AgentAction:
        canonical_id = self._aliases.get(action_id, action_id)
        if canonical_id not in self._by_id:
            raise KeyError("unregistered agent action: %s" % action_id)
        return self._by_id[canonical_id]

    def by_node(self, node: str) -> AgentAction:
        if node not in self._by_node:
            raise KeyError("unregistered agent action node: %s" % node)
        return self._by_node[node]

    def actions(self, action_ids: List[str]) -> List[AgentAction]:
        return [self.get(action_id) for action_id in action_ids]

    def public_action_ids(self) -> List[str]:
        return list(self._by_id.keys())

    def node_for(self, action_id: str) -> str:
        return self.get(action_id).node

    def policy_routing_map(self) -> Dict[str, str]:
        """Return the one authoritative LangGraph branch map for Lead actions."""

        return {node: node for node in self._by_node}

    def _validate_contract(self, actions: List[AgentAction]) -> None:
        action_ids = [action.id for action in actions]
        action_nodes = [action.node for action in actions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("agent action ids must be unique")
        if len(action_nodes) != len(set(action_nodes)):
            raise ValueError("agent action nodes must be unique; use an explicit action alias")
        if any(not action.id or not action.node for action in actions):
            raise ValueError("agent actions require non-empty ids and nodes")
        for alias, canonical_id in self._aliases.items():
            if alias in self._by_id or canonical_id not in self._by_id:
                raise ValueError("invalid agent action alias: %s" % alias)
        for action in actions:
            if action.fallback_action and action.fallback_action not in self._by_id:
                raise ValueError("fallback action is not registered: %s" % action.fallback_action)
            if not action.expected_state_keys and not action.expected_state_flags:
                raise ValueError("agent action has no declared postcondition: %s" % action.id)


class V2AgentPolicy:
    """Dynamic Lead Agent policy backed by an action registry."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings
        self.registry = AgentActionRegistry()
        self.capabilities = CapabilityRegistry.from_settings(settings)

    @property
    def max_main_actions(self) -> int:
        return self.settings.agent_main_rounds if self.settings else MAX_MAIN_AGENT_ACTIONS

    @property
    def max_retrieve_actions(self) -> int:
        return self.settings.agent_retrieve_rounds if self.settings else MAX_RETRIEVE_ACTIONS

    def supplemental_retrieve_count(self, state: AgentState) -> int:
        return int(state.get("query_graph_supplemental_retrieve_count") or 0)

    def can_retrieve_supplemental(self, state: AgentState) -> bool:
        return self.supplemental_retrieve_count(state) < self.max_retrieve_actions

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
        if state.get("run_canceled"):
            return ["terminal_end"], "run was canceled by user; end without persistence or post-processing", True
        terminal = state.get("terminal_status") or {}
        if terminal.get("active"):
            return ["terminal_end"], "terminal runtime guard stops all further orchestration", True
        if state.get("run_budget_exhausted"):
            if self.has_task_results(state) and not state.get("evidence_graph_verified"):
                return ["verify_evidence"], "run budget exhausted after SQL execution; evidence verification is still mandatory", True
            return ["answer_data"], "run budget exhausted; answer with collected results", True
        plan = state.get("plan")
        if state.get("human_clarification_required") or bool(
            plan and getattr(plan, "clarification_needs", None)
        ):
            return ["ask_human"], "planner requires human clarification before QueryGraph validation", False
        if state.get("middleware_loop_blocked"):
            if self.has_task_results(state) and not self.evidence_verification_attempted(state):
                return ["verify_evidence"], "loop guard stopped orchestration after SQL execution; evidence verification is still mandatory", True
            return ["answer_data"], "middleware loop guard blocked repeated action pattern", True
        route = state.get("routing_decision")
        if route and route.route == QuestionRoute.GREETING:
            return ["answer_data"], "greeting route uses lightweight answer", False
        if route and route.route == QuestionRoute.INVALID:
            return ["ask_human"], "question is outside supported business scope", False
        if int(state.get("react_round") or 0) >= self.max_main_actions:
            if self.has_task_results(state) and not state.get("evidence_graph_verified"):
                return ["verify_evidence"], "evidence must be verified after execution before answer", True
            if self.has_executable_plan(state) and not state.get("sql_generated"):
                if self.has_blocking_reflection_issues(state):
                    return ["answer_data"], "main agent budget exhausted with unresolved PlannerCritic errors; answer with structured planning gap", True
                if self.has_validated_executable_plan(state):
                    return ["execute_graph_direct"], "main agent budget exhausted; finish the validated QueryGraph through the lightweight direct worker", True
                if not self.graph_validation_attempted(state):
                    return ["validate_graph"], "main agent budget exhausted; validate QueryGraph before any NodeAgent execution", True
                return ["answer_data"], "main agent budget exhausted with invalid QueryGraph; answer with structured planning gap", True
            if self.has_unresolved_planning_work(state) and not self.graph_validation_attempted(state):
                return ["validate_graph"], "main agent budget exhausted with unresolved planning work; produce structured gap before answer", True
            return ["answer_data"], "main agent action budget exhausted", True
        if not state.get("topic_routed"):
            return ["route_topic"], "topic has not been routed", False
        if self.memory_recall_required(state):
            return ["recall_memory"], "topic workspace is ready; recall governed merchant memory with topic-aware context", False
        if bool(getattr(self.settings, "lead_agent_autonomous_enabled", False)):
            actions = self.autonomous_candidate_action_ids(state)
            if actions:
                reflection = normalize_reflection(state.get("planner_reflection"))
                if reflection and not reflection.passed:
                    reason = self.repair_decision_reason(reflection, actions[0])
                elif state.get("planner_provider_error"):
                    reason = "Planner provider failed; autonomous Lead Agent must validate and close with an explicit gap"
                else:
                    reason = "autonomous Lead Agent tool catalog filtered only by runtime safety contracts"
                return actions, reason, bool(state.get("planner_provider_error") and not (state.get("plan") and state["plan"].intents))
        if not state.get("data_discovered"):
            if not state.get("fast_understood") and state.get("route_slots"):
                return ["fast_understand"], "fast understanding has not classified complexity and intent kind", False
            fast = state.get("fast_understanding")
            if fast and getattr(fast, "intent_kind", "") == "rule_only":
                return ["retrieve_knowledge", "answer_rule"], "fast understanding classified rule-only; retrieve rule knowledge first", False
            if state.get("fast_metric_completed"):
                return ["cache_answer"], "Lead Agent accepted a verified single semantic metric result", False
            if self.general_delegation_needed(state):
                return ["delegate_subagent", "retrieve_knowledge"], "request contains bounded document or Python work suitable for an isolated Sub-Agent", False
            return ["retrieve_knowledge"], self.fast_understanding_reason(state, "semantic knowledge refresh is required before metric/query tools"), False
        if self.has_rule_recall_ready(state):
            return ["answer_rule"], "platform rule knowledge is ready; answer without BI QueryGraph", False
        if self.has_rule_answer_plan(state):
            return ["answer_rule"], "platform rule answer plan is ready", False
        if not state.get("planning_assets_compacted"):
            return ["compact_assets"], "planning asset pack has not been compacted", False
        plan = state.get("plan")
        if self.has_pending_knowledge_requests(state):
            if self.knowledge_recall_stalled(state):
                return self.stalled_knowledge_actions(state), "KnowledgeRequest recall produced no new refs; continue with current assets or close with gap", False
            if self.can_retrieve_supplemental(state):
                return ["retrieve_knowledge", "compact_assets", "plan_graph", "repair_graph", "answer_data"], self.pending_knowledge_reason(state), False
            if not self.graph_validation_attempted(state):
                return ["validate_graph"], "knowledge request budget exhausted; validate current graph as structured gap", True
        if state.get("query_metric_completed") and self.evidence_verification_passed(state):
            return ["answer_data"], "query_metric executed and verified one governed semantic metric", False
        if not state.get("query_metric_attempted") and self.query_metric_candidate(state):
            return ["query_metric", "plan_graph"], "single governed metric can be resolved through query_metric after Topic workspace recall", False
        if (
            (not plan or not plan.intents)
            and state.get("planner_provider_error")
            and not self.planner_degraded_fail_fast(state)
            and self.hypothesis_recovery_needed(state)
        ):
            return ["explore_hypotheses", "validate_graph", "answer_data"], "Planner provider failed; independently seed and execute bounded hypothesis QueryGraphs from semantic assets", False
        if (not plan or not plan.intents) and state.get("planner_provider_error"):
            if not self.graph_validation_attempted(state):
                return ["validate_graph"], "Planner provider failed; validate empty graph as structured gap", True
            return ["answer_data"], "Planner provider failed; answer with structured planning gap", True
        if (not plan or not plan.intents) and int(state.get("query_graph_plan_attempts") or 0) < self.max_plan_actions:
            return ["plan_graph", "retrieve_knowledge"], "QueryGraph has not been planned", False
        if plan and plan.intents and not state.get("query_graph_reflected"):
            if self.fast_path_bypasses_reflection(state):
                if not self.graph_validation_attempted(state):
                    return ["validate_graph"], "fast candidate skips Planner Reflection but must pass deterministic validation", False
            else:
                return ["reflect_plan", "validate_graph"], "QueryGraph needs planner reflection before validation", False
        reflection = normalize_reflection(state.get("planner_reflection"))
        if reflection and not reflection.passed:
            repair_actions = self.repair_request_actions(reflection)
            if "semantic_read" in repair_actions and self.can_retrieve_supplemental(state) and not self.knowledge_recall_stalled(state):
                return ["retrieve_knowledge", "compact_assets", "repair_graph", "plan_graph", "answer_data"], self.repair_decision_reason(reflection, "retrieve_knowledge"), False
            if "semantic_read" in repair_actions and self.knowledge_recall_stalled(state):
                return self.stalled_knowledge_actions(state), "PlannerCritic requested semantic knowledge but recall has no new refs; continue without another retrieve", False
            if self.has_structural_anchor_repair_issue(reflection) and int(state.get("query_graph_repair_attempts") or 0) < self.max_graph_repair_actions:
                return ["repair_graph", "answer_data"], self.repair_decision_reason(reflection, "repair_graph"), False
            if "re_understand" in repair_actions and self.can_reunderstand(state):
                return ["plan_graph", "repair_graph", "answer_data"], self.repair_decision_reason(reflection, "plan_graph"), False
            if "graph_repair" in repair_actions and int(state.get("query_graph_repair_attempts") or 0) < self.max_graph_repair_actions:
                return ["repair_graph", "retrieve_knowledge", "answer_data"], self.repair_decision_reason(reflection, "repair_graph"), False
            if "answer_with_gap" in repair_actions:
                if not self.graph_validation_attempted(state):
                    return ["validate_graph"], "PlannerCritic requested answer_with_gap; validate QueryGraph before answering", True
                return ["answer_data"], self.repair_decision_reason(reflection, "answer_data"), True
            has_blocking_issue = any(str(issue.get("severity") or "") == "error" for issue in reflection.issues)
            if (
                reflection.suggested_knowledge_requests
                and (reflection.repair_reason or has_blocking_issue)
                and self.can_retrieve_supplemental(state)
            ):
                return ["retrieve_knowledge", "repair_graph", "answer_data"], "planner reflection requested more knowledge", False
            if int(state.get("query_graph_repair_attempts") or 0) < self.max_graph_repair_actions:
                return ["repair_graph", "answer_data"], "planner reflection found repairable graph issues", False
            if not self.graph_validation_attempted(state):
                return ["validate_graph"], "planner reflection repair budget exhausted; validate QueryGraph before answer", True
            return ["answer_data"], "planner reflection failed and repair budget is exhausted", True
        if not self.graph_validation_attempted(state):
            return ["validate_graph"], "QueryGraph has not been validated", False
        validation = state.get("query_graph_validation_result")
        if validation and not validation.valid:
            if self.validation_requires_knowledge(validation) and self.can_retrieve_supplemental(state) and not self.knowledge_recall_stalled(state):
                return ["retrieve_knowledge", "compact_assets", "repair_graph", "answer_data"], "validator requested missing semantic knowledge before graph repair", False
            if self.validation_requires_knowledge(validation) and self.knowledge_recall_stalled(state):
                if self.validation_requires_reunderstand(validation) and self.can_reunderstand(state):
                    return ["plan_graph", "repair_graph", "answer_data"], "validator requested knowledge but recall stalled; rerun understanding or repair current graph", False
                if validation.repairable and int(state.get("query_graph_repair_attempts") or 0) < self.max_graph_repair_actions:
                    return ["repair_graph", "answer_data"], "validator requested knowledge but recall stalled; repair current graph", False
            if self.validation_requires_reunderstand(validation) and self.can_reunderstand(state):
                return ["plan_graph", "answer_data"], "validator found contract mismatch; rerun LLM understanding", False
            if validation.repairable and int(state.get("query_graph_repair_attempts") or 0) < self.max_graph_repair_actions:
                return ["repair_graph", "answer_data"], "validator found repairable graph gaps", False
            return ["answer_data"], "QueryGraph validation failed and cannot be repaired in budget", True
        if self.has_blocking_reflection_issues(state):
            return ["answer_data"], "PlannerCritic still has blocking errors; do not execute even if structural validation passed", True
        if self.has_validated_executable_plan(state) and not state.get("sql_generated"):
            actions = self.execution_action_ids(state)
            return actions, "validated QueryGraph is ready; LeadAgent selects direct worker or bounded Sub-Agent from runtime cost signals", False
        if self.has_graph_repairable_execution_gap(state):
            if int(state.get("query_graph_repair_attempts") or 0) < self.max_graph_repair_actions:
                return ["repair_graph", "answer_data"], "NodeAgent found graph-repairable plan contract gaps", False
            return ["answer_data"], "graph-repairable execution gaps remain but repair budget is exhausted", True
        if not state.get("sql_repair_reviewed") and self.has_sql_failure(state):
            return ["repair_sql", "verify_evidence"], "one or more node SQL executions failed", False
        if self.has_task_results(state) and not self.evidence_verification_attempted(state):
            return ["verify_evidence"], "evidence has not been verified", False
        if self.has_graph_repairable_execution_gap(state):
            if int(state.get("query_graph_repair_attempts") or 0) < self.max_graph_repair_actions:
                return ["repair_graph", "answer_data"], "evidence verifier found graph-repairable dependency gaps", False
            return ["answer_data"], "graph-repairable evidence gaps remain but repair budget is exhausted", True
        if self.hypothesis_exploration_needed(state):
            actions = ["explore_hypotheses"]
            if self.analysis_skill_needed(state):
                actions.append("run_analysis_skill")
            if self.analysis_worker_needed(state):
                actions.append("run_analysis_worker")
            actions.append("answer_data")
            if not state.get("subagent_delegation_attempted"):
                actions.insert(1, "delegate_subagent")
            return actions, "verified baseline evidence is ready; LeadAgent may independently delegate or test competing hypotheses", False
        if self.analysis_worker_needed(state) or self.analysis_skill_needed(state):
            actions = []
            if self.analysis_skill_needed(state):
                actions.append("run_analysis_skill")
            if self.analysis_worker_needed(state):
                actions.append("run_analysis_worker")
            actions.append("answer_data")
            if not state.get("subagent_delegation_attempted"):
                actions.insert(1, "delegate_subagent")
            return actions, "verified evidence is ready; LeadAgent may choose direct answer, generic AnalysisWorker, or matched SkillWorker", False
        if not state.get("chat_bi_completed"):
            return ["answer_data"], "ready to compose BI answer", False
        return ["cache_answer"], "answer is complete and can be cached", False

    def autonomous_candidate_action_ids(self, state: AgentState) -> List[str]:
        """Expose every currently safe tool instead of prescribing a workflow stage."""
        plan = state.get("plan")
        validation = state.get("query_graph_validation_result")
        run_result = state.get("agent_run_result")
        reflection = normalize_reflection(state.get("planner_reflection"))
        has_plan = bool(plan and getattr(plan, "intents", None))
        has_tasks = bool(getattr(run_result, "task_results", None))
        if state.get("fast_metric_completed"):
            return ["cache_answer"]
        early_understanding = not state.get("data_discovered") and not has_plan and not has_tasks
        if early_understanding and not state.get("fast_understood"):
            return ["fast_understand"]
        if self.has_pending_knowledge_requests(state):
            if self.can_retrieve_supplemental(state) and not self.knowledge_recall_stalled(state):
                return ["retrieve_knowledge"]
            return self.stalled_knowledge_actions(state)
        if not state.get("data_discovered"):
            if self.general_delegation_needed(state):
                return ["delegate_subagent", "retrieve_knowledge"]
            return ["retrieve_knowledge"]
        if self.has_rule_recall_ready(state) or self.has_rule_answer_plan(state):
            return ["answer_rule"]
        if state.get("data_discovered") and not state.get("planning_assets_compacted"):
            return ["compact_assets"]
        if state.get("query_metric_completed") and self.evidence_verification_passed(state):
            return ["answer_data"]
        if not state.get("query_metric_attempted") and self.query_metric_candidate(state):
            return ["query_metric", "plan_graph"]
        if not has_plan:
            if self.graph_validation_attempted(state):
                return ["answer_data"]
            if state.get("planner_provider_error"):
                if not self.planner_degraded_fail_fast(state) and self.hypothesis_recovery_needed(state):
                    return ["explore_hypotheses", "validate_graph", "answer_data"]
                return ["validate_graph"]
            if int(state.get("query_graph_plan_attempts") or 0) < self.max_plan_actions:
                return ["plan_graph", "retrieve_knowledge"]
            return ["validate_graph", "answer_data"]
        repair_actions = self.repair_request_actions(reflection) if reflection and not reflection.passed else []
        if reflection and not reflection.passed:
            if "semantic_read" in repair_actions and self.can_retrieve_supplemental(state) and not self.knowledge_recall_stalled(state):
                return ["retrieve_knowledge", "repair_graph", "answer_data"]
            if "semantic_read" in repair_actions and self.knowledge_recall_stalled(state):
                return self.stalled_knowledge_actions(state)
            if self.has_structural_anchor_repair_issue(reflection) and int(state.get("query_graph_repair_attempts") or 0) < self.max_graph_repair_actions:
                return ["repair_graph", "answer_data"]
            if "re_understand" in repair_actions and self.can_reunderstand(state):
                return ["plan_graph", "repair_graph", "answer_data"]
            if "graph_repair" in repair_actions and int(state.get("query_graph_repair_attempts") or 0) < self.max_graph_repair_actions:
                return ["repair_graph", "retrieve_knowledge", "answer_data"]
        if has_plan and not state.get("query_graph_reflected"):
            if self.fast_path_bypasses_reflection(state):
                if not self.graph_validation_attempted(state):
                    return ["validate_graph"]
            else:
                return ["reflect_plan", "validate_graph"]
        if has_plan and not self.graph_validation_attempted(state):
            if reflection and not reflection.passed and int(state.get("query_graph_repair_attempts") or 0) < self.max_graph_repair_actions:
                return ["repair_graph", "validate_graph", "answer_data"]
            return ["validate_graph"]
        if validation and not validation.valid:
            if self.validation_requires_knowledge(validation) and self.can_retrieve_supplemental(state) and not self.knowledge_recall_stalled(state):
                return ["retrieve_knowledge", "compact_assets", "repair_graph", "answer_data"]
            if self.validation_requires_knowledge(validation) and self.knowledge_recall_stalled(state):
                if self.validation_requires_reunderstand(validation) and self.can_reunderstand(state):
                    return ["plan_graph", "repair_graph", "answer_data"]
                if validation.repairable and int(state.get("query_graph_repair_attempts") or 0) < self.max_graph_repair_actions:
                    return ["repair_graph", "answer_data"]
            if self.validation_requires_reunderstand(validation) and self.can_reunderstand(state):
                return ["plan_graph", "answer_data"]
            if validation.repairable and int(state.get("query_graph_repair_attempts") or 0) < self.max_graph_repair_actions:
                return ["repair_graph", "answer_data"]
            return ["answer_data"]
        if self.has_blocking_reflection_issues(state):
            return ["answer_data"]
        if self.has_validated_executable_plan(state) and not state.get("sql_generated"):
            return self.execution_action_ids(state)
        if self.has_graph_repairable_execution_gap(state):
            if int(state.get("query_graph_repair_attempts") or 0) < self.max_graph_repair_actions:
                return ["repair_graph", "answer_data"]
            return ["answer_data"]
        if state.get("sql_generated") and self.has_sql_failure(state) and not state.get("sql_repair_reviewed"):
            return ["repair_sql", "verify_evidence"]
        if has_tasks and not self.evidence_verification_attempted(state):
            return ["verify_evidence"]
        if self.has_graph_repairable_execution_gap(state):
            if int(state.get("query_graph_repair_attempts") or 0) < self.max_graph_repair_actions:
                return ["repair_graph", "answer_data"]
            return ["answer_data"]
        if self.evidence_verification_passed(state) and self.hypothesis_exploration_needed(state):
            actions = ["explore_hypotheses"]
            if self.analysis_skill_needed(state):
                actions.append("run_analysis_skill")
            if self.analysis_worker_needed(state):
                actions.append("run_analysis_worker")
            actions.append("answer_data")
            return actions if state.get("subagent_delegation_attempted") else [actions[0], "delegate_subagent"] + actions[1:]
        if self.evidence_verification_passed(state) and (self.analysis_worker_needed(state) or self.analysis_skill_needed(state)):
            actions = []
            if self.analysis_skill_needed(state):
                actions.append("run_analysis_skill")
            if self.analysis_worker_needed(state):
                actions.append("run_analysis_worker")
            actions.append("answer_data")
            return actions if state.get("subagent_delegation_attempted") else [actions[0], "delegate_subagent"] + actions[1:]
        if self.evidence_verification_attempted(state) or state.get("planner_provider_error") or (has_plan and validation and not validation.valid):
            return ["answer_data"]
        if state.get("chat_bi_completed"):
            return ["cache_answer"]
        return []

    def memory_recall_required(self, state: AgentState) -> bool:
        route = state.get("routing_decision")
        return bool(
            state.get("memory_recalled") is False
            and route
            and route.route == QuestionRoute.BUSINESS
            and not state.get("fast_understood")
            and not state.get("data_discovered")
            and not state.get("planning_assets_compacted")
            and not self.has_executable_plan(state)
            and not self.has_task_results(state)
            and not state.get("confirmation_evidence_reused")
            and not state.get("human_clarification_required")
        )

    def general_delegation_needed(self, state: AgentState) -> bool:
        if state.get("subagent_delegation_attempted"):
            return False
        context = state.get("request_context")
        files = list(getattr(context, "offloaded_files", None) or [])
        question = str(state.get("question") or "").lower()
        document_signal = bool(files) or "[用户附件上下文]" in question
        python_signal = any(str(path).lower().endswith(".py") for path in files) and any(
            token in question for token in ("python", "批量分析", "批处理", "模拟计算", "运行脚本")
        )
        return document_signal or python_signal

    def fast_metric_candidate(self, state: AgentState) -> bool:
        if self.fast_metric_decision(state.get("fast_understanding")).eligible:
            return True
        fast = state.get("fast_understanding")
        metric_phrases = {
            str(item).strip().lower()
            for item in (getattr(fast, "metric_phrases", None) or [])
            if str(item or "").strip()
        }
        return bool(
            is_metric_definition_question(str(state.get("question") or ""))
            and len(metric_phrases) == 1
        )

    def query_metric_candidate(self, state: AgentState) -> bool:
        if not state.get("planning_assets_compacted"):
            return False
        if self.has_pending_knowledge_requests(state):
            return False
        if state.get("plan") and getattr(state["plan"], "intents", None):
            return False
        if state.get("query_metric_completed"):
            return False
        fast = state.get("fast_understanding")
        if not fast:
            return False
        decision = self.fast_metric_decision(fast)
        return bool(decision.eligible)

    def fast_metric_decision(self, fast: object):
        features = features_from_fast_understanding(fast)
        analysis_intent = str(getattr(fast, "analysis_intent", "") or "").strip().lower()
        object_refs = getattr(fast, "object_refs", {}) or {}
        if (
            analysis_intent == "trend"
            and features.metric_count == 1
            and not object_refs
        ):
            # Fast understanding treats any trend signal as complex analysis,
            # while the metric executor supports a bounded single-metric daily
            # series. Project that typed shape onto the low-risk capability.
            features = features.model_copy(
                update={
                    "intent_kind": "metric_query",
                    "complexity": "simple",
                    "analysis_intent": "metric",
                    "domain_count": 1,
                    "requires_explanation": False,
                    "needs_planner": False,
                }
            )
        elif (
            features.intent_kind == "metric_query"
            and features.complexity == "simple"
            and features.metric_count == 1
            and not features.requires_explanation
            and not object_refs
        ):
            # Semantic recall can include adjacent domains. The governed fast
            # executor still resolves exactly one published metric contract, so
            # let that contract arbitrate support instead of treating recall
            # breadth as query complexity.
            features = features.model_copy(update={"domain_count": 1, "needs_planner": False})
        return self.capabilities.evaluate("metric_fast_entry", features)

    def has_unresolved_planning_work(self, state: AgentState) -> bool:
        plan = state.get("plan")
        validation = state.get("query_graph_validation_result")
        return bool(
            self.has_pending_knowledge_requests(state)
            or state.get("planner_repair_requests")
            or (plan and getattr(plan, "knowledge_requests", None))
            or not (plan and plan.intents)
            or (validation and getattr(validation, "gaps", None) and not validation.valid)
        )

    def has_pending_knowledge_requests(self, state: AgentState) -> bool:
        return bool(state.get("pending_knowledge_requests"))

    def pending_knowledge_reason(self, state: AgentState) -> str:
        requests = state.get("pending_knowledge_requests") or []
        first = requests[0] if requests else None
        reason = str(getattr(first, "reason", "") or getattr(first, "type", "") or "")
        query = str(getattr(first, "query", "") or "")
        suffix = (": %s" % query[:120]) if query else ""
        return "LeadAgent observed pending KnowledgeRequest%s; retrieve before planning/repair (%s)" % (suffix, reason)

    def knowledge_recall_stalled(self, state: AgentState) -> bool:
        context = state.get("lead_decision_context") or {}
        progress = context.get("progress") or {}
        return bool(progress.get("knowledgeRecallStalled"))

    def stalled_knowledge_actions(self, state: AgentState) -> List[str]:
        plan = state.get("plan")
        actions: List[str] = []
        if state.get("data_discovered") and not state.get("planning_assets_compacted"):
            actions.append("compact_assets")
        if state.get("planning_assets_compacted") and (not plan or not getattr(plan, "intents", None)) and self.can_reunderstand(state):
            actions.append("plan_graph")
        if plan and getattr(plan, "intents", None) and int(state.get("query_graph_repair_attempts") or 0) < self.max_graph_repair_actions:
            actions.append("repair_graph")
        if plan and getattr(plan, "intents", None) and not self.graph_validation_attempted(state):
            actions.append("validate_graph")
        actions.append("answer_data")
        return dedupe_action_ids(actions)

    def fast_understanding_reason(self, state: AgentState, default: str) -> str:
        fast = state.get("fast_understanding")
        if not fast:
            return default
        return "%s; fastUnderstanding intent=%s complexity=%s needsPlanner=%s" % (
            default,
            getattr(fast, "intent_kind", "unknown"),
            getattr(fast, "complexity", "unknown"),
            getattr(fast, "needs_planner", True),
        )

    def repair_request_actions(self, reflection: PlannerReflectionResult) -> set[str]:
        actions = {str(item.action or "") for item in reflection.repair_requests if getattr(item, "action", "")}
        if self.has_structural_anchor_repair_issue(reflection):
            actions.add("graph_repair")
        if reflection.repair_reason and not actions:
            fallback = {
                "ANCHOR_MISMATCH": "re_understand",
                "METRIC_RESOLUTION_LOW_CONFIDENCE": "re_understand",
                "ANALYSIS_EVIDENCE_NOT_COVERED": "re_understand",
                "MISSING_DOMAIN": "semantic_read",
                "METRIC_RESOLUTION_NEEDED": "semantic_read",
                "FRESHNESS_GAP": "semantic_read",
                "MISSING_EDGE": "graph_repair",
                "SCHEMA_DRIFT": "answer_with_gap",
            }
            action = fallback.get(reflection.repair_reason)
            if action:
                actions.add(action)
        return actions

    def has_structural_anchor_repair_issue(self, reflection: PlannerReflectionResult) -> bool:
        issue_codes = {
            str(issue.get("code") or "")
            for issue in reflection.issues
            if isinstance(issue, dict) and issue.get("code")
        }
        return bool(issue_codes & STRUCTURAL_ANCHOR_REPAIR_CODES)

    def can_reunderstand(self, state: AgentState) -> bool:
        attempts = int(state.get("query_graph_plan_attempts") or 0)
        # First planning pass is the normal understanding attempt. Critic-driven
        # re-understand is a repair action and gets its own bounded allowance.
        return attempts < (self.max_plan_actions + self.max_graph_repair_actions)

    def repair_decision_reason(self, reflection: PlannerReflectionResult, selected: str) -> str:
        actions = sorted(self.repair_request_actions(reflection))
        issue_codes = [
            str(issue.get("code") or "")
            for issue in reflection.issues[:6]
            if isinstance(issue, dict) and issue.get("code")
        ]
        return "PlannerCritic produced repairRequests actions=%s issues=%s; LeadAgent selects %s" % (
            actions,
            issue_codes,
            selected,
        )

    def has_executable_plan(self, state: AgentState) -> bool:
        plan = state.get("plan")
        if not plan:
            return False
        for intent in plan.intents:
            if intent.intent_type == IntentType.VALID and intent.answer_mode != AnswerMode.RULE:
                return True
        return False

    def execution_action_ids(self, state: AgentState) -> List[str]:
        policy = state.get("execution_tier_policy") or {}
        modes = list(policy.get("allowedModes") or [])
        if not modes:
            modes = [str(policy.get("defaultMode") or "direct")]
        actions = []
        for mode in modes:
            action = "execute_graph_agent" if str(mode) == "subagent" else "execute_graph_direct"
            if action not in actions:
                actions.append(action)
        return actions or ["execute_graph_direct"]

    def has_validated_executable_plan(self, state: AgentState) -> bool:
        if not self.has_executable_plan(state) or not graph_validation_passed(state):
            return False
        validation = state.get("query_graph_validation_result")
        return bool(validation and getattr(validation, "valid", False) and not self.has_blocking_reflection_issues(state))

    def graph_validation_attempted(self, state: AgentState) -> bool:
        return graph_validation_attempted(state)

    def has_blocking_reflection_issues(self, state: AgentState) -> bool:
        reflection = normalize_reflection(state.get("planner_reflection"))
        if not reflection or reflection.passed:
            return False
        return any(str(issue.get("severity") or "") == "error" for issue in reflection.issues)

    def has_task_results(self, state: AgentState) -> bool:
        run_result = state.get("agent_run_result")
        return bool(run_result and getattr(run_result, "task_results", None))

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

    def analysis_skill_needed(self, state: AgentState) -> bool:
        if self.planner_degraded_fail_fast(state):
            return False
        if state.get("analysis_skill_bypassed"):
            return False
        match = state.get("skill_match")
        if isinstance(match, SkillMatchState) and match.status == "no_match":
            return False
        if self.fast_path_verified_graph(state):
            return False
        if state.get("analysis_summary") or state.get("analysis_skill_trace"):
            return False
        plan = state.get("plan")
        run_result = state.get("agent_run_result")
        if not plan or not run_result or not getattr(run_result, "task_results", None):
            return False
        if not self.evidence_verification_passed(state):
            return False
        return bool(answer_skill_required(plan, run_result, bool(state.get("rule_recall_context", ""))))

    def analysis_worker_needed(self, state: AgentState) -> bool:
        if self.planner_degraded_fail_fast(state):
            return False
        if state.get("analysis_worker_completed"):
            return False
        if state.get("analysis_summary") or state.get("analysis_worker_trace"):
            return False
        if self.fast_path_verified_graph(state):
            return False
        plan = state.get("plan")
        run_result = state.get("agent_run_result")
        if not plan or not run_result or not getattr(run_result, "task_results", None):
            return False
        if not self.evidence_verification_passed(state):
            return False
        return bool(analysis_summary_required(plan))

    def hypothesis_exploration_needed(self, state: AgentState) -> bool:
        if self.planner_degraded_fail_fast(state):
            return False
        if self.fast_path_verified_graph(state):
            return False
        if not bool(getattr(self.settings, "hypothesis_query_exploration_enabled", False)):
            return False
        if state.get("hypothesis_exploration_completed") or not self.evidence_verification_passed(state):
            return False
        if not self.evidence_verification_passed(state):
            return False
        hypotheses = list((state.get("hypothesis_exploration") or {}).get("hypotheses") or [])
        if len(hypotheses) < 2:
            return False
        signals = (state.get("hypothesis_exploration") or {}).get("questionSignals") or {}
        fast = state.get("fast_understanding")
        complex_signal = bool(signals.get("mentionsAttribution") or signals.get("mentionsDrop"))
        complex_signal = complex_signal or bool(
            getattr(fast, "requires_explanation", False)
            or str(getattr(fast, "complexity", "") or "") == "complex"
        )
        if not complex_signal:
            return False
        plan = state.get("plan")
        understanding = getattr(plan, "question_understanding", {}) or {} if plan else {}
        if str(understanding.get("source") or "") == "canonical_recalled_metric_diagnostic_fallback":
            driver_contracts = understanding.get("diagnosticDriverContracts") or understanding.get("diagnostic_driver_contracts") or []
            if not driver_contracts:
                # A verified result-metric trend is enough to validate the
                # user's premise, but not enough to launch unrelated causal
                # probes. Only governed driver contracts may widen this path.
                return False
        return bool(plan and getattr(plan, "intents", None))

    def hypothesis_recovery_needed(self, state: AgentState) -> bool:
        if self.planner_degraded_fail_fast(state):
            return False
        if not state.get("planner_provider_error"):
            return False
        if not bool(getattr(self.settings, "hypothesis_query_exploration_enabled", False)):
            return False
        if state.get("hypothesis_exploration_completed") or not state.get("planning_assets_compacted"):
            return False
        hypotheses = list((state.get("hypothesis_exploration") or {}).get("hypotheses") or [])
        pack = state.get("planning_asset_pack")
        return len(hypotheses) >= 2 and bool(pack and getattr(pack, "metrics", None))

    def planner_degraded_fail_fast(self, state: AgentState) -> bool:
        degraded = state.get("planner_degraded") or {}
        return bool(degraded.get("active") and degraded.get("stopExpensivePostProcessing", True))

    def evidence_verification_passed(self, state: AgentState) -> bool:
        run_result = state.get("agent_run_result")
        verified = getattr(run_result, "verified_evidence", None) if run_result else None
        accepted = state.get("evidence_accepted")
        if accepted is None:
            accepted = state.get("evidence_graph_verified")
        elif accepted is False and state.get("evidence_graph_verified") and str(state.get("verification_status") or "") != "failed":
            accepted = True
        return bool(accepted and verified and getattr(verified, "passed", False))

    def evidence_verification_attempted(self, state: AgentState) -> bool:
        status = str(state.get("verification_status") or "")
        return status in {"passed", "failed"} or bool(state.get("evidence_graph_verified"))

    def fast_path_verified_graph(self, state: AgentState) -> bool:
        latency = state.get("latency_optimization") or {}
        return bool(latency.get("eligible")) and (
            str(latency.get("state") or "") == "fast_verified"
            or str(latency.get("mode") or "") == "fast_path_verified_graph"
        )

    def fast_path_bypasses_reflection(self, state: AgentState) -> bool:
        latency = state.get("latency_optimization") or {}
        return bool(latency.get("eligible")) and (
            str(latency.get("state") or "") in {"fast_candidate", "fast_verified"}
            or str(latency.get("mode") or "") in {"fast_path", "fast_path_candidate_graph", "fast_path_verified_graph"}
        )

    def has_graph_repairable_execution_gap(self, state: AgentState) -> bool:
        run_result = state.get("agent_run_result")
        if not run_result:
            return False
        result_by_task = {result.task_id: result for result in run_result.task_results}
        for gap in run_result.evidence_gaps:
            if gap.code not in REPAIRABLE_QUERY_GRAPH_GAP_CODES:
                continue
            if gap.code == "MISSING_UPSTREAM_ENTITY" and upstream_missing_is_execution_result(result_by_task.get(gap.task_id)):
                continue
            return True
        return False

    def validation_requires_reunderstand(self, validation: object) -> bool:
        codes = {str(gap.code) for gap in getattr(validation, "gaps", [])}
        return bool(
            codes
            & {
                "SCOPE_NOT_NARROWING",
                "OBJECTIVE_NOT_COMPILED",
                "OBJECT_REF_FILTER_MISSING",
                "CALCULATION_NUMERATOR_SAME_AS_DENOMINATOR",
                "METRIC_RESOLUTION_LOW_CONFIDENCE",
            }
        )

    def validation_requires_knowledge(self, validation: object) -> bool:
        codes = {str(gap.code) for gap in getattr(validation, "gaps", [])}
        if codes and codes <= {"CALCULATION_NUMERATOR_SAME_AS_DENOMINATOR"}:
            return False
        if getattr(validation, "recommended_knowledge_requests", None):
            return True
        return bool(
            codes
            & {
                "PENDING_KNOWLEDGE_REQUEST",
                "MISSING_FIELD",
                "MISSING_TABLE",
                "MISSING_RELATIONSHIP",
                "MISSING_METRIC_DEPENDENCY",
                "REQUESTED_MEASURE_NOT_PLANNED",
                "METRIC_RESOLUTION_NEEDED",
                "METRIC_RESOLUTION_LOW_CONFIDENCE",
                "CALCULATION_NUMERATOR_NOT_EVENT_METRIC",
            }
        )


def normalize_reflection(value: object) -> Optional[PlannerReflectionResult]:
    if isinstance(value, PlannerReflectionResult):
        return value
    if isinstance(value, dict):
        try:
            return PlannerReflectionResult.model_validate(value)
        except Exception:
            return None
    return None


STRUCTURAL_ANCHOR_REPAIR_CODES = {
    "ROOT_METRIC_NOT_ROOT",
    "ROOT_METRIC_NOT_MOST_SPECIFIC",
    "SIBLING_METRIC_WRONGLY_DEPENDENT",
    "FAKE_DEPENDENCY",
    "SCOPE_NOT_NARROWING",
    "OBJECTIVE_NOT_COMPILED",
}


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


def dedupe_action_ids(action_ids: List[str]) -> List[str]:
    seen: set[str] = set()
    deduped: List[str] = []
    for action_id in action_ids:
        value = str(action_id or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
