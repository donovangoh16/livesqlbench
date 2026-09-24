"""Shared state contract for the four phase agents."""

from typing import Any


PHASES = (
    "preprocessing",
    "planning",
    "sql_generation",
    "post_processing",
)

ARTIFACT_KEYS = {
    "preprocessing": ("preprocessing_context", "preprocessing_completed"),
    "planning": ("query_plan", "query_plan_version", "query_plan_validated"),
    "sql_generation": (
        "draft_sql", "draft_sql_version", "draft_sql_plan_version",
        "sql_generation_completed",
    ),
    "post_processing": (
        "last_execution_result", "last_execution_error", "final_sql",
    ),
}


def initial_multi_agent_state() -> dict[str, Any]:
    """Return only multi-agent additions; task-level state remains shared."""
    return {
        "current_phase": "preprocessing",
        "active_phase_agent": None,
        "phase_agent_sequence": [],
        "phase_transitions": [],
        "backward_transitions": 0,
        "phase_retries": 0,
        "last_recovery_directive": None,
        "routing_events": [],
        "phase_completion": {phase: False for phase in PHASES},
        "agent_metrics": {
            phase: {
                "activations": 0,
                "model_calls": 0,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "uncached_input_tokens": 0,
                "output_tokens": 0,
                "thought_tokens": 0,
                "tool_prompt_tokens": 0,
                "tool_calls": 0,
                "total_tokens": 0,
            }
            for phase in PHASES
        },
    }
