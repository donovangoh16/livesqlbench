"""Capability-adaptive harness for experiment variants 2 and 3.

This module classifies agent-specific tools into shared events and phases.  The
It records shared events and enforces only correctness-related workflow gates.
Retry budgets and context compaction intentionally belong to later milestones.
"""

import hashlib
import json
import re
from typing import Any


BASELINE_EVENTS = {
    "get_schema": ("schema_read", "preprocessing"),
    "get_all_column_meanings": ("schema_read", "preprocessing"),
    "get_column_meaning": ("schema_read", "preprocessing"),
    "get_all_external_knowledge_names": ("kb_read", "preprocessing"),
    "get_knowledge_definition": ("kb_read", "preprocessing"),
    "get_all_knowledge_definitions": ("kb_read", "preprocessing"),
    "execute_sql": ("sql_execution", "generation"),
    "submit_sql": ("submission", "submission"),
}

IMPROVED_EVENTS = {
    "prepare_schema_context": ("schema_read", "preprocessing"),
    "prepare_knowledge_context": ("kb_read", "preprocessing"),
    "inspect_database": ("data_inspection", "preprocessing"),
    "finalize_preprocessing_context": ("preprocessing_finalization", "preprocessing"),
    "generate_and_validate_query_plan": ("plan_attempt", "planning"),
    "validate_sql_to_plan": ("sql_validation", "generation"),
    "execute_validated_sql": ("sql_execution", "generation"),
    "diagnose_last_execution_error": ("diagnosis", "post_processing"),
    "submit_validated_sql": ("submission", "submission"),
}

RETRY_EVENTS = {
    "preprocessing_finalization", "plan_attempt", "sql_validation", "sql_execution",
}


def _new_metrics(adapter: str) -> dict:
    return {
        "mode": "workflow_enforcement",
        "adapter": adapter,
        "current_phase": "preprocessing",
        "event_counts": {},
        "phase_counts": {},
        "phase_transitions": [],
        "duplicate_calls_detected": 0,
        "no_progress_events_detected": 0,
        "no_progress_streak": 0,
        "max_no_progress_streak": 0,
        "retry_counts": {},
        "blocked_calls": 0,
        "block_reasons": {},
        "termination_reason": None,
        "calls_observed": 0,
    }


def _metrics(state: dict) -> tuple[dict, str]:
    adapter = "improved" if state.get("active_agent_profile") == "improved" else "baseline"
    return state.setdefault("harness_metrics", _new_metrics(adapter)), adapter


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except TypeError:
        return str(value)


def _parse_response(response: Any) -> Any:
    if isinstance(response, (dict, list)):
        return response
    text = str(response).strip()
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def _response_signature(event: str, response: Any) -> str:
    parsed = _parse_response(response)
    if isinstance(parsed, dict) and event in {
        "preprocessing_finalization", "plan_attempt", "sql_validation", "sql_execution",
    }:
        errors = parsed.get("errors") or parsed.get("diagnoses") or []
        compact_errors = []
        for error in errors if isinstance(errors, list) else [errors]:
            if isinstance(error, dict):
                compact_errors.append((
                    error.get("code") or error.get("type"),
                    error.get("path") or error.get("message"),
                ))
            else:
                compact_errors.append(str(error))
        material = {
            "event": event,
            "valid": parsed.get("valid"),
            "success": parsed.get("success"),
            "complete": parsed.get("complete"),
            "errors": compact_errors,
        }
    else:
        material = {"event": event, "response": parsed}
    return hashlib.sha256(_canonical(material).encode()).hexdigest()[:16]


def _response_status(response: Any) -> tuple[bool | None, bool]:
    parsed = _parse_response(response)
    if isinstance(parsed, dict):
        success = parsed.get("valid", parsed.get("success"))
        error = bool(parsed.get("error")) or success is False
        return success if isinstance(success, bool) else None, error
    text = str(parsed).lower()
    if "correct!" in text:
        return True, False
    return None, bool(re.search(r"\b(?:error|incorrect)\b", text))


def _sql_fingerprint(sql: Any) -> str:
    raw = str(sql or "").strip()
    if not raw:
        return ""
    try:
        from sqlglot import parse
        trees = [tree for tree in parse(raw, dialect="postgres") if tree is not None]
        if not trees:
            raise ValueError("no SQL statements parsed")
        canonical = ";".join(
            tree.sql(
                dialect="postgres", pretty=False, normalize=True, comments=False,
            )
            for tree in trees
        )
    except Exception:
        # Preserve the gate for SQL unsupported by SQLGlot while avoiding
        # formatting-only differences where basic normalization is sufficient.
        canonical = re.sub(r"/\*.*?\*/|--[^\r\n]*", " ", raw, flags=re.S)
        canonical = re.sub(r"\s+", " ", canonical).strip().rstrip(";").lower()
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _is_query_sql(sql: Any) -> bool:
    try:
        from sqlglot import parse_one
        tree = parse_one(str(sql), dialect="postgres")
        return tree is not None and type(tree).__name__.lower() in {
            "select", "union", "intersect", "except",
        }
    except Exception:
        return bool(re.match(r"^\s*(?:with\b.*?\bselect\b|select\b)", str(sql), re.I | re.S))


def _block(state: dict, reason: str, required_action: str) -> dict:
    metrics, _ = _metrics(state)
    metrics["blocked_calls"] += 1
    reasons = metrics["block_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "allowed": False,
        "harness_blocked": True,
        "reason": reason,
        "required_action": required_action,
    }


def authorize_tool_call(state: dict, tool_name: str, args: dict) -> dict | None:
    """Enforce Milestone 2 ordering gates for variants 2 and 3 only."""
    if state.get("active_harness_profile") != "improved":
        return None
    _, adapter = _metrics(state)
    if state.get("_harness_submitted"):
        return _block(state, "submission_already_attempted", "stop")

    if adapter == "baseline":
        if tool_name == "execute_sql":
            if not state.get("_harness_schema_grounded"):
                return _block(state, "schema_required_before_execution", "call_schema_tool")
            sql_hash = _sql_fingerprint(args.get("sql"))
            if sql_hash and sql_hash == state.get("_harness_failed_sql_hash"):
                return _block(state, "failed_sql_unchanged", "change_sql_before_retry")
        elif tool_name == "submit_sql":
            sql = args.get("sql", "")
            is_query = _is_query_sql(sql)
            if not state.get("_harness_schema_grounded"):
                return _block(state, "schema_required_before_submission", "call_schema_tool")
            if is_query and _sql_fingerprint(sql) != state.get("_harness_successful_sql_hash"):
                return _block(
                    state, "submitted_sql_not_successfully_executed",
                    "execute_the_exact_current_sql_successfully",
                )
    else:
        if tool_name == "generate_and_validate_query_plan":
            if not state.get("preprocessing_completed", False):
                return _block(
                    state, "preprocessing_not_finalized", "finalize_preprocessing_context",
                )
        elif tool_name == "validate_sql_to_plan":
            if not state.get("query_plan_validated", False):
                return _block(state, "validated_plan_required", "generate_and_validate_query_plan")
        elif tool_name == "execute_validated_sql":
            if not state.get("sql_generation_completed", False) or not state.get("draft_sql"):
                return _block(state, "validated_sql_required", "validate_sql_to_plan")
        elif tool_name == "diagnose_last_execution_error":
            if state.get("last_execution_succeeded") is not False:
                return _block(state, "execution_error_required", "execute_validated_sql")
        elif tool_name == "submit_validated_sql":
            if not state.get("sql_generation_completed", False) or not state.get("draft_sql"):
                return _block(state, "validated_sql_required", "validate_sql_to_plan")
            category = str((state.get("query_plan") or {}).get("category", "Query"))
            if category == "Query" and (
                state.get("last_execution_succeeded") is not True
                or _sql_fingerprint(state.get("last_execution_sql"))
                != _sql_fingerprint(state.get("draft_sql"))
            ):
                return _block(
                    state, "current_query_sql_not_successfully_executed",
                    "execute_validated_sql",
                )
    return None


def _transition(metrics: dict, next_phase: str) -> None:
    current = metrics.get("current_phase", "preprocessing")
    if current != next_phase:
        metrics.setdefault("phase_transitions", []).append(f"{current}->{next_phase}")
        metrics["current_phase"] = next_phase


def observe_tool_call(
    state: dict, tool_name: str, args: dict, tool_response: Any,
) -> None:
    """Record one tool call when the improved harness is active."""
    if state.get("active_harness_profile") != "improved":
        return

    metrics, adapter = _metrics(state)
    mapping = IMPROVED_EVENTS if adapter == "improved" else BASELINE_EVENTS
    event, call_phase = mapping.get(tool_name, ("other", "unknown"))
    metrics["calls_observed"] += 1
    metrics["event_counts"][event] = metrics["event_counts"].get(event, 0) + 1
    metrics["phase_counts"][call_phase] = metrics["phase_counts"].get(call_phase, 0) + 1
    if event in RETRY_EVENTS:
        metrics["retry_counts"][event] = max(0, metrics["event_counts"][event] - 1)

    fingerprint = hashlib.sha256(
        f"{tool_name}:{_canonical(args)}".encode()
    ).hexdigest()[:16]
    seen = state.setdefault("_harness_call_fingerprints", {})
    if seen.get(fingerprint, 0):
        metrics["duplicate_calls_detected"] += 1
    seen[fingerprint] = seen.get(fingerprint, 0) + 1

    signature = _response_signature(event, tool_response)
    if signature == state.get("_harness_last_progress_signature"):
        metrics["no_progress_streak"] += 1
        metrics["no_progress_events_detected"] += 1
    else:
        metrics["no_progress_streak"] = 0
    metrics["max_no_progress_streak"] = max(
        metrics["max_no_progress_streak"], metrics["no_progress_streak"],
    )
    state["_harness_last_progress_signature"] = signature

    success, error = _response_status(tool_response)
    parsed_response = _parse_response(tool_response)
    blocked = isinstance(parsed_response, dict) and parsed_response.get("harness_blocked")
    if not blocked and event == "schema_read" and not error:
        state["_harness_schema_grounded"] = True
    if not blocked and event == "sql_execution":
        sql = args.get("sql") if adapter == "baseline" else state.get("draft_sql")
        sql_hash = _sql_fingerprint(sql)
        if error:
            state["_harness_failed_sql_hash"] = sql_hash
        else:
            state["_harness_successful_sql_hash"] = sql_hash
            # google.adk State exposes mapping reads/writes but has no pop().
            state["_harness_failed_sql_hash"] = None
    if not blocked and event == "submission":
        state["_harness_submitted"] = True
        metrics["termination_reason"] = "submission"
    target_phase = metrics.get("current_phase", call_phase) if blocked else call_phase
    if not blocked and adapter == "improved":
        if event == "preprocessing_finalization" and success:
            target_phase = "planning"
        elif event == "plan_attempt" and success:
            target_phase = "generation"
        elif event == "sql_execution" and error:
            target_phase = "post_processing"
        elif event == "diagnosis":
            target_phase = "post_processing"
        elif event == "submission":
            target_phase = "completed"
    elif not blocked:
        if event == "sql_execution":
            target_phase = "post_processing" if error else "generation"
        elif event == "submission":
            target_phase = "completed"
    if target_phase != "unknown":
        _transition(metrics, target_phase)


def finalize_harness_metrics(state: dict) -> dict | None:
    """Fill terminal observation fields after the agent turn finishes."""
    if state.get("active_harness_profile") != "improved":
        return None
    metrics, _ = _metrics(state)
    if metrics.get("termination_reason") is None:
        if state.get("steps_remaining", 1) <= 0:
            metrics["termination_reason"] = "step_budget_exhausted"
        elif state.get("model_turns", 0) >= 60:
            metrics["termination_reason"] = "model_turn_budget_exhausted"
        else:
            metrics["termination_reason"] = "agent_stopped"
    return metrics
