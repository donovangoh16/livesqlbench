"""Forward-only coordinator for variants 4 and 5.

The coordinator performs no SQL reasoning. It invokes phase owners, transfers
stored artifacts, and checks only whether the required phase artifact exists.
Backward routing is intentionally deferred to the next milestone.
"""

import json
import logging
import sys
import hashlib
from copy import deepcopy
from typing import Any

from multi_agent.state import PHASES, initial_multi_agent_state


logger = logging.getLogger(__name__)

PHASE_AGENT_NAMES = {
    "preprocessing": "sql_preprocessing_agent",
    "planning": "sql_planning_agent",
    "sql_generation": "sql_generation_agent",
    "post_processing": "sql_post_processing_agent",
}


PHASE_COMPLETION_KEYS = {
    "preprocessing": "preprocessing_completed",
    "planning": "query_plan_validated",
    "sql_generation": "sql_generation_completed",
    "post_processing": "task_done",
}

MAX_COORDINATOR_TRANSITIONS = 12

TOKEN_KEYS = (
    "input_tokens", "cached_input_tokens", "uncached_input_tokens",
    "output_tokens", "thought_tokens", "tool_prompt_tokens", "total_tokens",
)

PHASE_ALIASES = {
    "preprocessing": "preprocessing",
    "pre_processing": "preprocessing",
    "query_planning": "planning",
    "planning": "planning",
    "sql_generation": "sql_generation",
    "generation": "sql_generation",
    "postprocessing": "post_processing",
    "post_processing": "post_processing",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def phase_message(
    phase: str, state: dict, initial_message: str = "", recovery: dict | None = None,
) -> str:
    """Create a compact artifact handoff without semantic transformation."""
    question = state.get("user_query", "")
    common = f"User Query:\n{question}\n\nShared steps remaining: {state.get('steps_remaining', 0)}."
    recovery_text = ""
    if recovery:
        recovery_text = (
            "\n\nRECOVERY DIRECTIVE FROM VALIDATOR:\n"
            + _json({
                "return_to_phase": recovery.get("return_to_phase"),
                "required_action": recovery.get("required_action"),
                "retryable": recovery.get("retryable"),
            })
        )
    if phase == "preprocessing":
        base = common if recovery else (initial_message or common)
        if recovery:
            base += "\n\nCURRENT PREPROCESSING_CONTEXT:\n" + _json(
                state.get("preprocessing_context", {})
            )
        return base + recovery_text
    if phase == "planning":
        return common + "\n\nPREPROCESSING_CONTEXT:\n" + _json(
            state.get("preprocessing_context", {})
        ) + recovery_text
    if phase == "sql_generation":
        return common + "\n\nVALIDATED QUERY_PLAN:\n" + _json(
            state.get("query_plan", {})
        ) + recovery_text
    if phase == "post_processing":
        return (
            common
            + "\n\nVALIDATED QUERY_PLAN:\n" + _json(state.get("query_plan", {}))
            + "\n\nVALIDATED DRAFT_SQL:\n" + str(state.get("draft_sql", ""))
            + recovery_text
        )
    raise ValueError(f"Unsupported multi-agent phase: {phase}")


def recovery_directive(state: dict, failed_phase: str) -> dict | None:
    """Read existing validator routing fields without interpreting semantics."""
    candidates = []
    if failed_phase == "planning":
        candidates.append(state.get("query_plan_validation"))
    if failed_phase in {"sql_generation", "post_processing"}:
        history = state.get("sql_validation_history", []) or []
        if history:
            candidates.append(history[-1])
    if failed_phase == "post_processing":
        candidates.append(state.get("last_execution_error_diagnosis"))

    for value in reversed(candidates):
        if not isinstance(value, dict) or not value.get("return_to_phase"):
            continue
        target = PHASE_ALIASES.get(str(value["return_to_phase"]).lower())
        if target is None:
            continue
        return {
            "return_to_phase": target,
            "required_action": value.get("required_action")
            or value.get("recommended_action")
            or "revise_the_rejected_artifact",
            "retryable": value.get("retryable", value.get("retry_allowed", True)),
        }
    return None


def invalidate_for_return(state: dict, target_phase: str) -> None:
    """Invalidate the target artifact and every downstream artifact."""
    target_index = PHASES.index(target_phase)
    if target_index <= PHASES.index("preprocessing"):
        state["preprocessing_completed"] = False
    if target_index <= PHASES.index("planning"):
        for key in ("query_plan", "query_plan_version"):
            state.pop(key, None)
        state["query_plan_validated"] = False
        state["query_plan_completed"] = False
    if target_index <= PHASES.index("sql_generation"):
        for key in (
            "draft_sql", "draft_sql_version", "draft_sql_plan_version",
            "last_execution_sql", "last_execution_result", "last_execution_error",
        ):
            state.pop(key, None)
        state["sql_generation_completed"] = False
        state["last_execution_succeeded"] = None
    if target_index <= PHASES.index("post_processing"):
        state["task_done"] = False


def _counter_snapshot(state: dict) -> dict:
    usage = state.get("token_usage", {}) or {}
    return {
        "model_calls": int(state.get("model_turns", 0) or 0),
        "tool_calls": len(state.get("tool_trajectory", []) or []),
        **{key: int(usage.get(key, 0) or 0) for key in TOKEN_KEYS},
    }


def _record_agent_delta(state: dict, phase: str, before: dict) -> None:
    after = _counter_snapshot(state)
    metrics = state.setdefault("agent_metrics", {}).setdefault(phase, {})
    for key in ("model_calls", "tool_calls", *TOKEN_KEYS):
        metrics[key] = int(metrics.get(key, 0) or 0) + max(
            0, int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0)
        )


def _evidence_snapshot(state: dict) -> dict:
    """Compactly identify whether recovery acquired new upstream evidence."""
    context = state.get("preprocessing_context", {}) or {}
    prepared_kb = state.get("prepared_knowledge_context", {}) or {}
    inspections = state.get("database_inspections", {}) or {}
    payload = {
        "tables": context.get("selected_tables", []),
        "columns": context.get("selected_columns", []),
        "joins": context.get("selected_join_edges", []),
        "knowledge": prepared_kb.get("knowledge", []),
        "inspections": inspections,
    }
    return {
        "fingerprint": hashlib.sha1(_json(payload).encode()).hexdigest()[:12],
        "table_count": len(payload["tables"]),
        "column_count": len(payload["columns"]),
        "join_count": len(payload["joins"]),
        "knowledge_count": len(payload["knowledge"]),
        "inspection_count": len(payload["inspections"]),
    }


class MultiAgentCoordinator:
    """Run the four specialists sequentially over one shared task state."""

    def __init__(self, runtime) -> None:
        self.runtime = runtime
        self._task_states: dict[str, dict] = {}

    def has_task(self, task_id: str) -> bool:
        return task_id in self._task_states

    async def init_task(self, task_id: str, state: dict, reset: bool = True) -> dict:
        if reset or task_id not in self._task_states:
            shared = deepcopy(state)
            shared.update(initial_multi_agent_state())
            self._task_states[task_id] = shared
        return {"task_id": task_id, "multi_agent": True, "initialized": True}

    async def run(self, task_id: str, initial_message: str) -> dict:
        if task_id not in self._task_states:
            raise KeyError(f"Multi-agent task is not initialized: {task_id}")

        state = self._task_states[task_id]
        final_response = ""
        phase_index = 0
        transition_count = 0
        pending_recovery = None
        while phase_index < len(PHASES):
            if transition_count >= MAX_COORDINATOR_TRANSITIONS:
                state["multi_agent_stop_reason"] = "coordinator_transition_limit"
                break
            phase = PHASES[phase_index]
            phase_number = phase_index + 1
            state["current_phase"] = phase
            state["active_phase_agent"] = phase
            sequence = list(state.get("phase_agent_sequence", []))
            sequence.append(phase)
            state["phase_agent_sequence"] = sequence
            phase_metrics = state.setdefault("agent_metrics", {}).setdefault(phase, {})
            phase_metrics["activations"] = int(
                phase_metrics.get("activations", 0) or 0
            ) + 1
            counters_before = _counter_snapshot(state)

            logger.info(
                "[%s] MULTI-AGENT phase %d/%d: %s (%s), shared steps remaining=%s",
                task_id,
                phase_number,
                len(PHASES),
                PHASE_AGENT_NAMES[phase],
                phase,
                state.get("steps_remaining", 0),
            )
            print(
                f"[{task_id}] MULTI-AGENT phase {phase_number}/{len(PHASES)}: "
                f"{PHASE_AGENT_NAMES[phase]} ({phase}), shared steps "
                f"remaining={state.get('steps_remaining', 0)}",
                file=sys.stderr,
                flush=True,
            )

            runtime_profile = f"multi_{phase}"
            state["runtime_agent_profile"] = runtime_profile
            active_recovery = pending_recovery
            if active_recovery:
                state["multi_agent_recovery_directive"] = active_recovery
            else:
                state.pop("multi_agent_recovery_directive", None)
            phase_task_id = f"{task_id}__{phase}"
            await self.runtime.init_session(
                task_id=phase_task_id, state=state, reset=True,
            )
            result = await self.runtime.run_turn(
                task_id=phase_task_id,
                message=phase_message(
                    phase, state, initial_message, recovery=pending_recovery,
                ),
            )
            pending_recovery = None
            state = result.get("state", state)
            state.pop("multi_agent_recovery_directive", None)
            if active_recovery:
                events = state.get("routing_events", []) or []
                if events:
                    after = _evidence_snapshot(state)
                    before = events[-1].get("evidence_before", {})
                    events[-1]["evidence_after"] = after
                    events[-1]["new_evidence_acquired"] = (
                        before.get("fingerprint") != after.get("fingerprint")
                    )
                    state["routing_events"] = events
            _record_agent_delta(state, phase, counters_before)
            final_response = result.get("response", final_response)
            self._task_states[task_id] = state

            completed = bool(state.get(PHASE_COMPLETION_KEYS[phase], False))
            state.setdefault("phase_completion", {})[phase] = completed

            if not completed:
                directive = recovery_directive(state, phase)
                if directive:
                    target = directive["return_to_phase"]
                    target_index = PHASES.index(target)
                    if target_index <= phase_index:
                        invalidate_for_return(state, target)
                        transitions = list(state.get("phase_transitions", []))
                        transitions.append(f"{phase}->{target}")
                        state["phase_transitions"] = transitions
                        if target_index < phase_index:
                            state["backward_transitions"] = int(
                                state.get("backward_transitions", 0) or 0
                            ) + 1
                        else:
                            state["phase_retries"] = int(
                                state.get("phase_retries", 0) or 0
                            ) + 1
                        state["last_recovery_directive"] = directive
                        routing_events = list(state.get("routing_events", []))
                        routing_events.append({
                            "from": phase,
                            "to": target,
                            "required_action": directive.get("required_action"),
                            "retryable": directive.get("retryable"),
                            "evidence_before": _evidence_snapshot(state),
                        })
                        state["routing_events"] = routing_events
                        pending_recovery = directive
                        phase_index = target_index
                        transition_count += 1
                        continue
                state["multi_agent_stop_reason"] = f"{phase}_incomplete"
                logger.warning(
                    "[%s] MULTI-AGENT stopped: %s did not complete its artifact",
                    task_id, PHASE_AGENT_NAMES[phase],
                )
                break

            if phase != PHASES[-1]:
                next_phase = PHASES[phase_index + 1]
                transitions = list(state.get("phase_transitions", []))
                transitions.append(f"{phase}->{next_phase}")
                state["phase_transitions"] = transitions
            phase_index += 1
            transition_count += 1

        state["active_phase_agent"] = None
        state["submission_attempted"] = bool(state.get("task_done"))
        self._task_states[task_id] = state
        return {
            "task_id": task_id,
            "response": final_response,
            "state": state,
            "multi_agent": True,
        }
