"""Capability-adaptive harness for experiment variants 2 and 3.

This module classifies agent-specific tools into shared events and phases. It
records shared events and enforces correctness-related workflow gates plus
Milestone 3A duplicate/no-progress controls, 3A.2 semantic recovery gates,
and calibrated Milestone 3B event budgets. Context compaction belongs to a
later milestone.
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

NO_PROGRESS_EVENTS = RETRY_EVENTS | {"submission"}

RECOVERY_ACTIONS = {
    "preprocessing_finalization": "change_preprocessing_evidence_before_retry",
    "plan_attempt": "change_the_plan_or_return_to_preprocessing",
    "sql_validation": "change_the_sql_or_revise_the_plan",
    "sql_execution": "diagnose_the_error_or_change_the_sql",
    "submission": "satisfy_the_submission_gate_or_stop",
}

# Limits are deliberately above the largest successful matched-run paths:
# Query needed 4 preprocessing finalizations; Management needed 10 plan and 5
# SQL-validation attempts. These are runaway guards, not target call counts.
IMPROVED_EVENT_BUDGETS = {
    "Query": {
        "schema_read": 4,
        "kb_read": 4,
        "data_inspection": 5,
        "preprocessing_finalization": 5,
        "plan_attempt": 5,
        "sql_validation": 4,
        "sql_execution": 3,
        "diagnosis": 2,
    },
    "Management": {
        "schema_read": 4,
        "kb_read": 4,
        "data_inspection": 4,
        "preprocessing_finalization": 6,
        "plan_attempt": 12,
        "sql_validation": 7,
        "sql_execution": 1,
        "diagnosis": 1,
    },
}

BASELINE_EVENT_BUDGETS = {
    "schema_read": 10,
    "kb_read": 10,
    "sql_execution": 8,
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
        "duplicate_calls_blocked": 0,
        "no_progress_events_detected": 0,
        "no_progress_calls_blocked": 0,
        "execution_diagnoses_required": 0,
        "execution_diagnoses_completed": 0,
        "semantic_reviews_required": 0,
        "semantic_reviews_completed": 0,
        "empty_result_advisories": 0,
        "management_no_progress_advisories": 0,
        "budget_warnings": 0,
        "budget_blocks": 0,
        "budget_warning_events": {},
        "budget_limits": {},
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
            "error": parsed.get("error"),
            "errors": compact_errors,
        }
        if event == "sql_execution" and parsed.get("success") is True:
            material.update({
                "columns": parsed.get("columns"),
                "sample_row_count": parsed.get("sample_row_count"),
                "empty": parsed.get("empty"),
                "numeric_ranges": parsed.get("numeric_ranges"),
                "alerts": parsed.get("alerts"),
            })
    else:
        material = {"event": event, "response": parsed}
    return hashlib.sha256(_canonical(material).encode()).hexdigest()[:16]


def _progress_signature(event: str, args: dict, response: Any) -> str:
    """Represent outcome plus evidence that is meaningful for retry progress."""
    response_signature = _response_signature(event, response)
    if event not in {
        "preprocessing_finalization", "plan_attempt", "sql_validation",
    }:
        return response_signature
    if event == "preprocessing_finalization":
        evidence = {
            key: args.get(key)
            for key in (
                "selected_tables", "selected_columns", "selected_join_edges",
                "required_knowledge_phrases", "selected_knowledge", "unresolved_items",
            )
        }
    else:
        # A materially revised plan or SQL draft is progress even when a
        # validator reports the same broad error category.
        evidence = args
    material = {"response": response_signature, "evidence": evidence}
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


def _semantic_warning_reasons(response: Any) -> list[str]:
    parsed = _parse_response(response)
    if not isinstance(parsed, dict) or parsed.get("success") is not True:
        return []
    reasons = []
    if parsed.get("alerts"):
        reasons.append("execution_alert")
    return reasons


def _block(state: dict, reason: str, required_action: str) -> dict:
    metrics, _ = _metrics(state)
    metrics["blocked_calls"] += 1
    reasons = metrics["block_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1
    if reason == "duplicate_call_without_state_change":
        metrics["duplicate_calls_blocked"] += 1
    elif reason == "no_progress_without_state_change":
        metrics["no_progress_calls_blocked"] += 1
    elif reason == "event_budget_exhausted":
        metrics["budget_blocks"] += 1
    return {
        "allowed": False,
        "harness_blocked": True,
        "reason": reason,
        "required_action": required_action,
    }


def _event_for(adapter: str, tool_name: str) -> tuple[str, str]:
    mapping = IMPROVED_EVENTS if adapter == "improved" else BASELINE_EVENTS
    return mapping.get(tool_name, ("other", "unknown"))


def _is_management_context(state: dict, tool_name: str, args: dict) -> bool:
    category = args.get("category") or state.get("task_category")
    if not category:
        category = (state.get("query_plan") or {}).get("category")
    if category:
        return str(category).lower() == "management"
    if tool_name in {"execute_sql", "submit_sql"} and args.get("sql"):
        return not _is_query_sql(args.get("sql"))
    return False


def _event_budget(
    state: dict, adapter: str, tool_name: str, args: dict,
) -> tuple[str, int | None]:
    event, _ = _event_for(adapter, tool_name)
    if event == "submission":
        return event, None
    if adapter == "baseline":
        return event, BASELINE_EVENT_BUDGETS.get(event)
    category = "Management" if _is_management_context(state, tool_name, args) else "Query"
    return event, IMPROVED_EVENT_BUDGETS[category].get(event)


def _budget_gate(
    state: dict, adapter: str, tool_name: str, args: dict,
) -> dict | None:
    event, limit = _event_budget(state, adapter, tool_name, args)
    if limit is None:
        return None
    metrics, _ = _metrics(state)
    metrics["budget_limits"][event] = limit
    used = int(metrics.get("event_counts", {}).get(event, 0) or 0)
    if used >= limit:
        return _block(
            state, "event_budget_exhausted",
            f"{event}_limit_{limit}_reached_move_to_next_phase_or_stop",
        )
    warning_at = max(1, limit - 1)
    warned = state.setdefault("_harness_budget_events_warned", {})
    if used + 1 >= warning_at and not warned.get(event):
        warned[event] = True
        metrics["budget_warnings"] += 1
        warning_events = metrics["budget_warning_events"]
        warning_events[event] = warning_events.get(event, 0) + 1
        state["_harness_pending_budget_warning"] = (
            f"{event} attempt {used + 1}/{limit}; converge or change phase"
        )
    return None


def take_budget_warning(state: dict) -> str | None:
    """Return and clear a concise warning queued for a harness-enabled call."""
    if state.get("active_harness_profile") != "improved":
        return None
    warning = state.get("_harness_pending_budget_warning")
    state["_harness_pending_budget_warning"] = None
    return str(warning) if warning else None


def _call_key(state: dict, adapter: str, tool_name: str, args: dict) -> str:
    """Identify an exact call within the current meaningful artifact state."""
    context: dict[str, Any] = {
        "revision": int(state.get("_harness_progress_revision", 0) or 0),
    }
    if adapter == "improved":
        if tool_name == "generate_and_validate_query_plan":
            context["preprocessing_revision"] = int(
                state.get("_harness_preprocessing_revision", 0) or 0
            )
        elif tool_name == "execute_validated_sql":
            context["sql"] = _sql_fingerprint(state.get("draft_sql"))
            context["plan_version"] = state.get("query_plan_version")
        elif tool_name == "diagnose_last_execution_error":
            context["sql"] = _sql_fingerprint(state.get("last_execution_sql"))
            context["execution_succeeded"] = state.get("last_execution_succeeded")
    material = {"tool": tool_name, "args": args, "context": context}
    return hashlib.sha256(_canonical(material).encode()).hexdigest()[:20]


def _repetition_gate(state: dict, adapter: str, tool_name: str, args: dict) -> dict | None:
    event, _ = _event_for(adapter, tool_name)
    if event == "submission":
        return None
    # Let the more informative semantic-recovery gates explain what must
    # change after a failed or suspicious execution.
    if adapter == "improved" and (
        state.get("_harness_execution_diagnosis_required")
        or state.get("_harness_semantic_review_required")
    ) and tool_name in {"validate_sql_to_plan", "execute_validated_sql"}:
        return None
    revision = int(state.get("_harness_progress_revision", 0) or 0)
    no_progress_blocks = state.get("_harness_no_progress_blocks", {}) or {}
    if (
        event in NO_PROGRESS_EVENTS
        and not _is_management_context(state, tool_name, args)
        and no_progress_blocks.get(event) == revision
    ):
        return _block(
            state, "no_progress_without_state_change",
            RECOVERY_ACTIONS.get(event, "change_arguments_or_relevant_state"),
        )
    key = _call_key(state, adapter, tool_name, args)
    if (state.get("_harness_completed_call_keys", {}) or {}).get(key):
        return _block(
            state, "duplicate_call_without_state_change",
            RECOVERY_ACTIONS.get(event, "change_arguments_or_relevant_state"),
        )
    return None


def authorize_tool_call(state: dict, tool_name: str, args: dict) -> dict | None:
    """Enforce Milestones 2, 3A, and 3A.2 for variants 2 and 3 only."""
    if state.get("active_harness_profile") != "improved":
        return None
    _, adapter = _metrics(state)
    if state.get("_harness_submitted"):
        return _block(state, "submission_already_attempted", "stop")
    budget_block = _budget_gate(state, adapter, tool_name, args)
    if budget_block:
        return budget_block
    repetition_block = _repetition_gate(state, adapter, tool_name, args)
    if repetition_block:
        return repetition_block

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
            failed_sql = state.get("_harness_execution_diagnosis_required")
            if failed_sql:
                if state.get("_harness_diagnosed_failed_sql_hash") != failed_sql:
                    return _block(
                        state, "execution_diagnosis_required_before_correction",
                        "diagnose_last_execution_error",
                    )
                proposed_sql = _sql_fingerprint(args.get("sql"))
                proposed_plan = args.get("plan_version")
                if (
                    proposed_sql == failed_sql
                    and proposed_plan == state.get("_harness_failed_plan_version")
                ):
                    return _block(
                        state, "correction_required_after_diagnosis",
                        "change_the_sql_or_revise_the_query_plan",
                    )
            semantic_review = state.get("_harness_semantic_review_required") or {}
            if semantic_review:
                proposed_sql = _sql_fingerprint(args.get("sql"))
                proposed_plan = args.get("plan_version")
                if (
                    proposed_sql == semantic_review.get("sql_hash")
                    and proposed_plan == semantic_review.get("plan_version")
                ):
                    return _block(
                        state, "semantic_revision_required",
                        "revise_the_sql_or_query_plan_then_validate_again",
                    )
        elif tool_name == "execute_validated_sql":
            failed_sql = state.get("_harness_execution_diagnosis_required")
            if failed_sql:
                if state.get("_harness_diagnosed_failed_sql_hash") != failed_sql:
                    return _block(
                        state, "execution_diagnosis_required_before_retry",
                        "diagnose_last_execution_error",
                    )
                if (
                    _sql_fingerprint(state.get("draft_sql")) == failed_sql
                    and state.get("draft_sql_plan_version")
                    == state.get("_harness_failed_plan_version")
                ):
                    return _block(
                        state, "correction_required_after_diagnosis",
                        "change_and_validate_the_sql_or_revise_the_query_plan",
                    )
            if not state.get("sql_generation_completed", False) or not state.get("draft_sql"):
                return _block(state, "validated_sql_required", "validate_sql_to_plan")
        elif tool_name == "diagnose_last_execution_error":
            if state.get("last_execution_succeeded") is not False:
                return _block(state, "execution_error_required", "execute_validated_sql")
        elif tool_name == "submit_validated_sql":
            if not state.get("sql_generation_completed", False) or not state.get("draft_sql"):
                return _block(state, "validated_sql_required", "validate_sql_to_plan")
            category = str((state.get("query_plan") or {}).get("category", "Query"))
            if category == "Query" and state.get("_harness_execution_diagnosis_required"):
                return _block(
                    state, "unresolved_execution_error", "diagnose_then_correct_validate_execute",
                )
            if category == "Query" and state.get("_harness_semantic_review_required"):
                return _block(
                    state, "semantic_review_required_before_submission",
                    "revise_plan_or_sql_then_validate_and_execute",
                )
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
    event, call_phase = _event_for(adapter, tool_name)
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

    success, error = _response_status(tool_response)
    parsed_response = _parse_response(tool_response)
    blocked = isinstance(parsed_response, dict) and parsed_response.get("harness_blocked")
    if not blocked:
        signature = _progress_signature(event, args, tool_response)
        event_signatures = state.setdefault("_harness_event_signatures", {})
        event_streaks = state.setdefault("_harness_event_no_progress_streaks", {})
        previous_signature = event_signatures.get(event)
        if event in NO_PROGRESS_EVENTS and previous_signature == signature:
            streak = int(event_streaks.get(event, 0) or 0) + 1
            event_streaks[event] = streak
            metrics["no_progress_streak"] = streak
            metrics["no_progress_events_detected"] += 1
            metrics["max_no_progress_streak"] = max(
                metrics["max_no_progress_streak"], streak,
            )
            # The repeated result is the second equivalent attempt. The next
            # attempt must first change relevant state or take recovery action.
            if _is_management_context(state, tool_name, args):
                metrics["management_no_progress_advisories"] += 1
            else:
                blocks = state.setdefault("_harness_no_progress_blocks", {})
                blocks[event] = int(state.get("_harness_progress_revision", 0) or 0)
        else:
            event_streaks[event] = 0
            metrics["no_progress_streak"] = 0
            if previous_signature != signature:
                state["_harness_progress_revision"] = int(
                    state.get("_harness_progress_revision", 0) or 0
                ) + 1
        event_signatures[event] = signature

        if event == "preprocessing_finalization" and success:
            state["_harness_preprocessing_revision"] = int(
                state.get("_harness_preprocessing_revision", 0) or 0
            ) + 1

        completed = state.setdefault("_harness_completed_call_keys", {})
        completed[_call_key(state, adapter, tool_name, args)] = True

    if not blocked and event == "schema_read" and not error:
        state["_harness_schema_grounded"] = True
    if not blocked and event == "sql_execution":
        sql = args.get("sql") if adapter == "baseline" else state.get("draft_sql")
        sql_hash = _sql_fingerprint(sql)
        if error:
            state["_harness_failed_sql_hash"] = sql_hash
            if adapter == "improved":
                state["_harness_execution_diagnosis_required"] = sql_hash
                state["_harness_failed_plan_version"] = state.get(
                    "draft_sql_plan_version"
                )
                state["_harness_diagnosed_failed_sql_hash"] = None
                metrics["execution_diagnoses_required"] += 1
        else:
            state["_harness_successful_sql_hash"] = sql_hash
            # google.adk State exposes mapping reads/writes but has no pop().
            state["_harness_failed_sql_hash"] = None
            if adapter == "improved":
                state["_harness_execution_diagnosis_required"] = None
                state["_harness_failed_plan_version"] = None
                state["_harness_diagnosed_failed_sql_hash"] = None
                warning_reasons = _semantic_warning_reasons(tool_response)
                parsed_execution = _parse_response(tool_response)
                if isinstance(parsed_execution, dict) and (
                    parsed_execution.get("empty") is True
                    or parsed_execution.get("has_rows") is False
                ):
                    metrics["empty_result_advisories"] += 1
                previous_review = state.get("_harness_semantic_review_required")
                if warning_reasons:
                    state["_harness_semantic_review_required"] = {
                        "sql_hash": sql_hash,
                        "plan_version": state.get("draft_sql_plan_version"),
                        "reasons": warning_reasons,
                    }
                    metrics["semantic_reviews_required"] += 1
                else:
                    if previous_review:
                        metrics["semantic_reviews_completed"] += 1
                    state["_harness_semantic_review_required"] = None
    if not blocked and adapter == "improved" and event == "diagnosis" and not error:
        required_hash = state.get("_harness_execution_diagnosis_required")
        if required_hash:
            state["_harness_diagnosed_failed_sql_hash"] = required_hash
            metrics["execution_diagnoses_completed"] += 1
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
