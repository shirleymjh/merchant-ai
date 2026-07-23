from __future__ import annotations

from typing import Dict, List, Optional

from merchant_ai.config import Settings
from merchant_ai.graph.action_contract import action_prerequisite_gaps, contract_block_observation
from merchant_ai.graph.evidence_verification_contract import (
    evidence_verification_attempted,
    evidence_verification_passed,
)
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
    "AGGREGATE_OUTPUT_CONTRACT_MISMATCH",
    "GROUP_BY_CONTRACT_MISMATCH",
}
RECOVERABLE_EXECUTION_EVIDENCE_GAP_CODES = {
    "EXECUTION_OPERATIONAL_FAILURE",
    "MEM_ALLOC_FAILED",
    "SQL_EXECUTION_FAILED",
    "TIMEOUT",
    "UNKNOWN_COLUMN",
    "UPSTREAM_SQL_FAILED",
    "UPSTREAM_ZERO_ROWS",
    "ZERO_ROWS",
}
RECOVERABLE_EVIDENCE_ACTION_TOKENS = {
    "align_",
    "repair_",
    "restore_",
    "retry_",
    "rerun_",
    "supplement_",
}
RECOVERABLE_FRESHNESS_STATUSES = {
    "CHECK_FAILED",
    "NO_TIME_COLUMN",
    "STALE_REQUIRES_GRAPH_REPREPARATION",
    "ZERO_ROWS",
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
                id="observe_contract_block",
                node="observe_contract_block",
                agent="Runtime",
                description="record a blocked action contract and return its observation to LeadAgent",
                expected_state_flags=["contract_block_observed"],
            ),
            AgentAction(
                id="lead_arbitrate",
                node="lead_arbitrate",
                agent="LeadAgent",
                description="internal placeholder requiring the Lead model to select one safe catalog action",
                expected_state_flags=["lead_arbitration_observed"],
            ),
            AgentAction(
                id="fast_understand",
                node="fast_understand",
                agent="LeadAgent",
                description="cheap complexity and intent sketch before heavy planning",
                required_state_flags=["topic_routed"],
                expected_state_flags=["fast_understood"],
            ),
            AgentAction(
                id="recall_memory",
                node="recall_memory",
                agent="LeadAgent",
                description="recall governed merchant memory after topic routing",
                required_state_flags=["topic_routed"],
                expected_state_flags=["memory_recalled"],
            ),
            AgentAction(
                id="try_fast_metric",
                node="try_fast_metric",
                agent="LeadAgent",
                description="legacy compatibility path for one uniquely matched published semantic metric",
                required_state_flags=["fast_understood"],
                expected_state_flags=["fast_metric_attempted"],
            ),
            AgentAction(
                id="retrieve_knowledge",
                node="retrieve_knowledge",
                agent="KnowledgeAgent",
                description="retrieve semantic rules, terms and thin skill policies",
                required_state_flags=["topic_routed"],
                expected_state_flags=["data_discovered"],
            ),
            AgentAction(
                id="compact_assets",
                node="compact_assets",
                agent="KnowledgeAgent",
                description="build semantic asset pack",
                required_state_flags=["data_discovered"],
                expected_state_flags=["planning_assets_compacted"],
            ),
            AgentAction(
                id="query_metric",
                node="query_metric",
                agent="MetricTool",
                description="resolve and validate one governed semantic metric graph, then return the observation to LeadAgent",
                required_state_flags=["planning_assets_compacted"],
                expected_state_flags=["query_metric_attempted"],
            ),
            AgentAction(
                id="plan_graph",
                node="plan_query_graph",
                agent="PlannerAgent",
                description=(
                    "compile QueryGraph from exact semantic evidence already read by the Core Agent; "
                    "Planner has no hidden filesystem loop"
                ),
                required_state_flags=["planning_assets_compacted"],
                expected_state_keys=["plan.intents"],
            ),
            AgentAction(
                id="reflect_plan",
                node="reflect_query_graph",
                agent="PlannerCriticAgent",
                description="review QueryGraph before validation",
                required_state_keys=["plan.intents"],
                expected_state_flags=["query_graph_reflected"],
            ),
            AgentAction(
                id="validate_graph",
                node="validate_query_graph",
                agent="PlannerCriticAgent",
                description="validate QueryGraph against asset pack",
                required_state_flags=["planning_assets_compacted"],
                expected_state_keys=["query_graph_validation_result"],
                expected_state_flags=["query_graph_validation_attempted"],
            ),
            AgentAction(
                id="repair_graph",
                node="repair_query_graph",
                agent="PlannerAgent",
                description="repair QueryGraph",
                required_state_keys=["plan.intents"],
                required_state_flags=["planning_assets_compacted"],
                expected_state_flags=["query_graph_repair_progressed"],
            ),
            AgentAction(
                id="execute_graph",
                node="execute_query_graph",
                agent="NodeAgent",
                description="execute QueryGraph nodes",
                required_state_keys=["plan.intents", "query_graph_validation_result"],
                required_state_flags=["query_graph_validation_passed"],
                expected_state_flags=["sql_generated"],
            ),
            AgentAction(
                id="execute_graph_direct",
                node="execute_query_graph_direct",
                agent="NodeWorker",
                description="execute the validated QueryGraph with the lightweight direct worker path",
                required_state_keys=["plan.intents", "query_graph_validation_result"],
                required_state_flags=["query_graph_validation_passed"],
                expected_state_flags=["sql_generated"],
            ),
            AgentAction(
                id="execute_graph_agent",
                node="execute_query_graph_agent",
                agent="NodeAgent",
                description="execute the validated QueryGraph with autonomous bounded Sub-Agent loops",
                required_state_keys=["plan.intents", "query_graph_validation_result"],
                required_state_flags=["query_graph_validation_passed"],
                expected_state_flags=["sql_generated"],
            ),
            AgentAction(
                id="repair_sql",
                node="repair_sql",
                agent="NodeAgent",
                description="review node-level SQL repair status",
                required_state_keys=["agent_run_result.task_results"],
                required_state_flags=["sql_generated"],
                expected_state_flags=["sql_repair_reviewed"],
            ),
            AgentAction(
                id="verify_evidence",
                node="verify_evidence_graph",
                agent="EvidenceVerifierAgent",
                description="verify evidence before answer",
                required_state_keys=["agent_run_result.task_results"],
                required_state_flags=["sql_generated"],
                expected_state_flags=["evidence_graph_verified"],
            ),
            AgentAction(
                id="run_analysis_skill",
                node="run_analysis_skill",
                agent="SkillWorker",
                description="dynamically dispatch an isolated analysis skill worker before final answer",
                required_state_keys=["agent_run_result.task_results"],
                required_state_flags=["sql_generated", "evidence_graph_verified"],
                expected_state_keys=["analysis_summary"],
            ),
            AgentAction(
                id="run_analysis_worker",
                node="run_analysis_worker",
                agent="AnalysisWorker",
                description="optionally dispatch a generic isolated analysis worker for long-tail analysis before final answer",
                required_state_keys=["agent_run_result.task_results"],
                required_state_flags=["sql_generated", "evidence_graph_verified"],
                expected_state_keys=["analysis_summary"],
            ),
            AgentAction(
                id="delegate_subagent",
                node="delegate_subagent",
                agent="LeadAgent",
                description=(
                    "dispatch a runtime-governed document/Python/query/Skill worker; use DeepAgent task, not this "
                    "action, for ordinary read-only context isolation"
                ),
                expected_state_flags=["subagent_delegation_completed"],
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
        self._internal_ids = {"observe_contract_block", "lead_arbitrate"}
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
        return [action_id for action_id in self._by_id if action_id not in self._internal_ids]

    def routing_action_ids(self) -> List[str]:
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
        action_ids, blocked = self.contract_safe_action_ids(state, action_ids)
        state["action_catalog_contract_blocks"] = blocked
        if not action_ids and blocked:
            if len(blocked) == 1:
                (blocked_contract,) = blocked
                blocked_action = self.registry.get(str(blocked_contract["action"]))
                observation = contract_block_observation(
                    blocked_action,
                    list(blocked_contract["missingStateKeys"]),
                    list(blocked_contract["missingStateFlags"]),
                    reason=reason,
                    source="policy_catalog_filter",
                )
                observation["blockedCatalog"] = blocked
            else:
                blocked_catalog = sorted(blocked, key=lambda item: str(item["action"]))
                observation = {
                    "status": "pending",
                    "source": "policy_catalog_filter",
                    "blockedAction": "",
                    "blockedNode": "",
                    "blockedActions": [str(item["action"]) for item in blocked_catalog],
                    "blockedNodes": sorted({str(item["node"]) for item in blocked_catalog}),
                    "missingStateKeys": sorted(
                        {
                            str(key)
                            for item in blocked_catalog
                            for key in item["missingStateKeys"]
                        }
                    ),
                    "missingStateFlags": sorted(
                        {
                            str(flag)
                            for item in blocked_catalog
                            for flag in item["missingStateFlags"]
                        }
                    ),
                    "decisionReason": str(reason or "")[:500],
                    "blockedCatalog": blocked_catalog,
                }
            state["contract_block_observation"] = observation
            state["contract_block_observed"] = False
            action_ids = ["observe_contract_block"]
            reason = "all proposed actions were removed by declarative prerequisite contracts"
        elif blocked:
            reason = "%s; contract filter removed unsafe actions=%s" % (
                reason,
                [item["action"] for item in blocked],
            )
        if len(action_ids) > 1:
            selected_action = "lead_arbitrate"
            decision_source = "lead_arbitration_pending"
        else:
            selected_action = next(iter(action_ids), "ask_human")
            decision_source = "runtime_single_action"
        action = self.registry.get(selected_action)
        return AgentDecision(
            selected_action=action.id,
            selected_node=action.node,
            available_actions=action_ids,
            reason=reason,
            budget_exhausted=budget_exhausted,
            source=decision_source,
        )

    def contract_safe_action_ids(
        self,
        state: AgentState,
        action_ids: List[str],
    ) -> tuple[List[str], List[Dict[str, object]]]:
        safe: List[str] = []
        blocked: List[Dict[str, object]] = []
        for action_id in action_ids:
            action = self.registry.get(action_id)
            missing_keys, missing_flags = action_prerequisite_gaps(state, action)
            if missing_keys or missing_flags:
                blocked.append(
                    {
                        "action": action.id,
                        "node": action.node,
                        "missingStateKeys": missing_keys,
                        "missingStateFlags": missing_flags,
                    }
                )
                continue
            safe.append(action.id)
        return safe, blocked

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
        if state.get("chat_bi_completed"):
            return ["cache_answer"], "answer action completed; close the run through the terminal cache contract", False
        if state.get("run_budget_exhausted"):
            if self.has_task_results(state) and not self.evidence_verification_attempted(state):
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
            if self.has_task_results(state) and not self.evidence_verification_attempted(state):
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
        actions = self.autonomous_candidate_action_ids(state)
        if actions:
            reflection = normalize_reflection(state.get("planner_reflection"))
            if reflection and not reflection.passed:
                reason = "Lead Agent received a runtime-safe repair catalog from PlannerCritic observations"
            elif state.get("planner_provider_error"):
                reason = "Planner provider failed; Lead Agent must select a safe recovery or close with an explicit gap"
            else:
                reason = "Lead Agent tool catalog filtered only by declarative prerequisites and runtime safety contracts"
            return actions, reason, bool(
                state.get("planner_provider_error")
                and not (state.get("plan") and state["plan"].intents)
            )
        return ["ask_human"], "no runtime-safe Lead action is currently eligible", True

    def autonomous_candidate_action_ids(self, state: AgentState) -> List[str]:
        """Build an unordered ReAct tool catalog from state contracts.

        This method only decides whether an action is currently safe and useful.
        It never ranks business actions: when several actions remain, the Lead
        model must select one; model failure is handled by the fail-closed path.
        """

        eligible: set[str] = set()
        plan = state.get("plan")
        validation = state.get("query_graph_validation_result")
        run_result = state.get("agent_run_result")
        reflection = normalize_reflection(state.get("planner_reflection"))
        has_plan = bool(plan and getattr(plan, "intents", None))
        has_tasks = bool(getattr(run_result, "task_results", None))

        if state.get("fast_metric_completed"):
            return ["cache_answer"]

        if self.memory_recall_required(state):
            eligible.add("recall_memory")
        if (
            not state.get("fast_understood")
            and not has_plan
            and not has_tasks
            and not state.get("planner_provider_error")
            and not self.graph_validation_attempted(state)
        ):
            eligible.add("fast_understand")
        if self.general_delegation_needed(state):
            eligible.add("delegate_subagent")

        pending_knowledge = self.has_pending_knowledge_requests(state)
        knowledge_can_progress = self.can_retrieve_supplemental(state) and not self.knowledge_recall_stalled(state)
        if not state.get("data_discovered") or (pending_knowledge and knowledge_can_progress):
            eligible.add("retrieve_knowledge")

        if self.has_rule_recall_ready(state) or self.has_rule_answer_plan(state):
            eligible.add("answer_rule")

        # DeepAgent's grounded path never runs the legacy compact_assets node.
        # Its minimal execution view becomes ready only after the explicit
        # GroundedQueryContract has been deterministically compiled.
        assets_ready = bool(
            state.get("planning_assets_compacted")
            or state.get("grounded_query_compiled")
        )
        if state.get("data_discovered") and not assets_ready:
            eligible.add("compact_assets")

        if assets_ready:
            if not has_plan:
                if state.get("planner_provider_error"):
                    if not self.graph_validation_attempted(state):
                        eligible.add("validate_graph")
                    else:
                        eligible.add("answer_data")
                elif (
                    int(state.get("query_graph_plan_attempts") or 0) < self.max_plan_actions
                    or (
                        bool(state.get("core_managed_filesystem"))
                        and self.can_reunderstand(state)
                    )
                ):
                    eligible.add("plan_graph")
                    if self.can_retrieve_supplemental(state):
                        eligible.add("retrieve_knowledge")
            else:
                repair_actions = self.repair_request_actions(reflection) if reflection and not reflection.passed else []
                repair_budget = self.graph_repair_attempt_count(state) < self.max_graph_repair_actions
                validation_attempted = self.graph_validation_attempted(state)
                reflection_pending = bool(
                    not state.get("query_graph_reflected")
                    and not self.fast_path_bypasses_reflection(state)
                )

                # QueryGraph lifecycle is tri-state and phase ordered.  A default
                # GraphValidationResult(valid=False) means "not run", never
                # "failed".  Reflection, validation and repair therefore cannot
                # all become eligible from the same unvalidated state.
                if not validation_attempted:
                    if reflection_pending:
                        eligible.add("reflect_plan")
                    elif reflection and not reflection.passed:
                        if "semantic_read" in repair_actions and knowledge_can_progress:
                            eligible.add("retrieve_knowledge")
                        elif "re_understand" in repair_actions and self.can_reunderstand(state):
                            eligible.add("plan_graph")
                        elif repair_budget and (
                            "graph_repair" in repair_actions
                            or self.has_structural_anchor_repair_issue(reflection)
                            or bool(reflection.issues)
                        ):
                            eligible.add("repair_graph")
                        if not eligible.intersection({"retrieve_knowledge", "plan_graph", "repair_graph"}):
                            eligible.add("answer_data")
                    else:
                        eligible.add("validate_graph")
                elif not graph_validation_passed(state):
                    if self.validation_requires_knowledge(validation) and knowledge_can_progress:
                        eligible.add("retrieve_knowledge")
                    elif self.validation_requires_reunderstand(validation) and self.can_reunderstand(state):
                        eligible.add("plan_graph")
                    elif validation and validation.repairable and repair_budget:
                        eligible.add("repair_graph")
                    if not eligible.intersection({"retrieve_knowledge", "plan_graph", "repair_graph"}):
                        eligible.add("answer_data")
                elif self.has_blocking_reflection_issues(state):
                    eligible.add("answer_data")
                elif self.has_validated_executable_plan(state) and not state.get("sql_generated"):
                    eligible.update(self.execution_action_ids(state))

        if state.get("sql_generated") and self.has_sql_failure(state) and not state.get("sql_repair_reviewed"):
            eligible.add("repair_sql")
        if has_tasks and not self.evidence_verification_attempted(state):
            eligible.add("verify_evidence")

        # Execution observations may reveal that the current QueryGraph chose
        # the wrong source/window even when SQL itself completed. Re-open the
        # governed Planner (and optional semantic retrieval) instead of forcing
        # the Lead to turn a recoverable zero-row/freshness/SQL gap directly
        # into an answer. The Planner remains bounded by can_reunderstand(), and
        # every replacement graph must still pass reflection/validation before
        # another SQL execution action becomes eligible.
        if assets_ready and self.execution_recovery_needed(state):
            if self.can_reunderstand(state) and not state.get("planner_provider_error"):
                eligible.add("plan_graph")
            if knowledge_can_progress:
                eligible.add("retrieve_knowledge")

        if self.has_graph_repairable_execution_gap(state):
            if self.graph_repair_attempt_count(state) < self.max_graph_repair_actions:
                eligible.add("repair_graph")
            else:
                eligible.add("answer_data")

        if self.evidence_verification_passed(state):
            if self.analysis_skill_needed(state):
                eligible.add("run_analysis_skill")
            if self.analysis_worker_needed(state):
                eligible.add("run_analysis_worker")
            if not state.get("subagent_delegation_attempted"):
                eligible.add("delegate_subagent")
            eligible.add("answer_data")
        elif self.evidence_verification_attempted(state):
            eligible.add("answer_data")

        if state.get("query_metric_completed") and self.evidence_verification_passed(state):
            eligible.add("answer_data")
        if state.get("planner_provider_error") and self.graph_validation_attempted(state):
            eligible.add("answer_data")

        return [
            action_id
            for action_id in self.registry.public_action_ids()
            if action_id in eligible
        ]

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
        return bool(files)

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
        if plan and getattr(plan, "intents", None) and self.graph_repair_attempt_count(state) < self.max_graph_repair_actions:
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

    def graph_repair_attempt_count(self, state: AgentState) -> int:
        """Return committed attempts for the active graph-and-issue scope.

        A repair that only suspends for supplemental knowledge remains visible
        in repair history, but the workflow does not commit it to this budget.
        """

        scope_key = str(state.get("query_graph_repair_scope_key") or "")
        scope_attempts = state.get("query_graph_repair_scope_attempts") or {}
        if scope_key and isinstance(scope_attempts, dict):
            return int(scope_attempts.get(scope_key, 0) or 0)
        if "query_graph_repair_scope_attempt_count" in state:
            return int(state.get("query_graph_repair_scope_attempt_count") or 0)
        return int(state.get("query_graph_repair_attempts") or 0)

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

    def has_successful_zero_row(self, state: AgentState) -> bool:
        run_result = state.get("agent_run_result")
        if not run_result:
            return False
        return any(
            bool(result.success)
            and not bool(result.query_bundle.failed)
            and int(result.query_bundle.effective_row_count()) == 0
            for result in run_result.task_results
        )

    def has_incomplete_freshness_or_snapshot(self, state: AgentState) -> bool:
        run_result = state.get("agent_run_result")
        reports = list(getattr(run_result, "freshness_reports", None) or state.get("freshness_reports") or [])
        for report in reports:
            status = str(getattr(report, "status", "") or "").upper()
            if status in RECOVERABLE_FRESHNESS_STATUSES:
                return True
            if status.startswith("STALE_") and status != "STALE_USE_REALTIME_FALLBACK":
                return True
            if getattr(report, "coverage_complete", True) is False:
                return True
        alignment = getattr(run_result, "snapshot_alignment", None) if run_result else None
        alignment_status = str(getattr(alignment, "status", "") or "").upper()
        if alignment_status and alignment_status != "NOT_APPLICABLE":
            return not bool(getattr(alignment, "aligned", False) and getattr(alignment, "complete", False))
        return False

    def has_recoverable_evidence_gap(self, state: AgentState) -> bool:
        run_result = state.get("agent_run_result")
        if not run_result:
            return False
        gaps = list(getattr(run_result, "evidence_gaps", None) or [])
        verified = getattr(run_result, "verified_evidence", None)
        gaps.extend(getattr(verified, "gaps", None) or [])
        for gap in gaps:
            code = str(getattr(gap, "code", "") or getattr(gap, "gap_code", "") or "").upper()
            # Dependency/graph-contract gaps already have the narrower
            # repair_graph action and its own budget. Do not widen those into
            # a second competing recovery path.
            if code in REPAIRABLE_QUERY_GRAPH_GAP_CODES:
                continue
            if code in RECOVERABLE_EXECUTION_EVIDENCE_GAP_CODES or code.startswith("SNAPSHOT_"):
                return True
            suggested_action = str(getattr(gap, "suggested_action", "") or "").lower()
            if any(token in suggested_action for token in RECOVERABLE_EVIDENCE_ACTION_TOKENS):
                return True
        return False

    def execution_recovery_needed(self, state: AgentState) -> bool:
        return bool(
            self.has_sql_failure(state)
            or self.has_successful_zero_row(state)
            or self.has_incomplete_freshness_or_snapshot(state)
            or self.has_recoverable_evidence_gap(state)
        )

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
        hypotheses = list((state.get("hypothesis_exploration") or {}).get("hypotheses") or [])
        if len(hypotheses) < 2:
            return False
        plan = state.get("plan")
        understanding = getattr(plan, "question_understanding", {}) or {} if plan else {}
        requires_explanation = str(
            understanding.get("requiresExplanation", understanding.get("requires_explanation", False))
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        required_evidence = understanding.get("requiredEvidenceIntents") or understanding.get("required_evidence_intents") or []
        if not requires_explanation and not required_evidence:
            return False
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
        return evidence_verification_passed(state)

    def evidence_verification_attempted(self, state: AgentState) -> bool:
        return evidence_verification_attempted(state)

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
