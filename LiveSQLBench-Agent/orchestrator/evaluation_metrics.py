"""Uniform evaluation-only metrics for experiment variants 0--5.

This module never changes agent or harness state.  It translates recorded task
artifacts into common, phase, harness, and multi-agent metric groups and marks
architecturally unavailable groups as non-applicable.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from shared.config import settings
from schema_linking_metrics import (
    final_attempted_sql, selected_kb_ids, sql_columns, sql_schema_elements,
)
from trajectory_phase_metrics import (
    EXECUTION_TOOLS, POSTPROCESSING_TOOLS, SUBMISSION_TOOLS,
    _execution_succeeded, _state_backed_sql, _structural_features,
    _validation_succeeded,
)


IMPROVED_VARIANTS = {1, 3, 4, 5}
HARNESS_VARIANTS = {2, 3, 5}
MULTI_VARIANTS = {4, 5}
PLAN_TOOLS = {"generate_query_plan", "generate_and_validate_query_plan"}
WORKFLOW_BLOCK_REASONS = {
    "schema_required_before_execution", "schema_required_before_submission",
    "failed_sql_unchanged", "submitted_sql_not_successfully_executed",
    "preprocessing_not_finalized", "new_preprocessing_evidence_required",
    "validated_plan_required", "semantic_plan_contract_required",
    "execution_diagnosis_required_before_correction",
    "execution_diagnosis_required_before_retry", "correction_required_after_diagnosis",
    "semantic_revision_required", "validated_sql_required", "execution_error_required",
    "latest_sql_contract_required", "unresolved_execution_error",
    "semantic_review_required_before_submission",
    "current_query_sql_not_successfully_executed",
}


def _payload(value: Any) -> dict:
    if isinstance(value, dict):
        return value.get("result", value) if isinstance(value.get("result"), dict) else value
    try:
        parsed = json.loads(str(value or ""))
        return parsed.get("result", parsed) if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        # Trajectories deliberately store compact previews.  Preserve the
        # leading control fields when a long JSON response ends in
        # ``...<truncated>`` and therefore cannot be decoded as a whole.
        text = str(value or "")
        fallback = {}
        for key in ("valid", "harness_blocked", "semantic_valid"):
            match = re.search(rf'"{key}"\s*:\s*(true|false)', text, re.I)
            if match:
                fallback[key] = match.group(1).lower() == "true"
        for key in ("reason", "termination_reason"):
            match = re.search(rf'"{key}"\s*:\s*"([^"]+)"', text)
            if match:
                fallback[key] = match.group(1)
        return fallback


def _set_metric(predicted: Iterable, expected: Iterable) -> dict[str, Any]:
    predicted, expected = set(predicted), set(expected)
    correct = predicted & expected
    precision = len(correct) / len(predicted) if predicted else (1.0 if not expected else 0.0)
    recall = len(correct) / len(expected) if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "selected_count": len(predicted), "expected_count": len(expected),
        "correct_count": len(correct), "precision": precision, "recall": recall,
        "f1": f1, "exact_match": predicted == expected,
    }


def _schema_catalog(database: str) -> tuple[set[str], set[str]]:
    path = Path(settings.data_dir) / database / f"{database}_schema.txt"
    text = path.read_text() if path.exists() else ""
    tables, columns = set(), set()
    for table, body in re.findall(
        r'CREATE\s+TABLE\s+"?([A-Za-z_]\w*)"?\s*\((.*?)\);', text, re.I | re.S,
    ):
        table = table.lower()
        tables.add(table)
        for line in body.splitlines():
            line = line.strip().rstrip(",")
            if not line or re.match(
                r"(?:PRIMARY|FOREIGN|UNIQUE|CHECK|CONSTRAINT)\b", line, re.I
            ):
                continue
            match = re.match(r'"?([A-Za-z_]\w*)"?\s+', line)
            if match:
                columns.add(f"{table}.{match.group(1).lower()}")
    return tables, columns


def _kb_catalog(database: str) -> set[str]:
    path = Path(settings.data_dir) / database / f"{database}_kb.jsonl"
    available = set()
    if not path.exists():
        return available
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        for value in (item.get("id"), item.get("knowledge"), item.get("name")):
            if value is not None:
                available.add(str(value).strip().lower())
    return available


def _all_kb_references(calls: list[dict]) -> set[str]:
    references = {str(value).lower() for value in selected_kb_ids(calls)}
    for call in calls:
        args = call.get("args") or {}
        for key in ("knowledge_id", "kb_id"):
            if args.get(key) is not None:
                references.add(str(args[key]).strip().lower())
        if args.get("knowledge_name") is not None:
            response_id = None
            payload = _payload(call.get("result"))
            if payload.get("id") is not None:
                response_id = payload["id"]
            if response_id is None:
                references.add(str(args["knowledge_name"]).strip().lower())
        if call.get("tool") in PLAN_TOOLS:
            def walk(value):
                if isinstance(value, dict):
                    for key, nested in value.items():
                        if key in {"knowledge_id", "kb_id"} and nested is not None:
                            references.add(str(nested).strip().lower())
                        else:
                            walk(nested)
                elif isinstance(value, list):
                    for nested in value:
                        walk(nested)
            walk(args.get("plan") or {})
    return references


def _declared_sql_aliases(sql: str | None) -> set[str]:
    """Return aliases created by SQL expressions, CTEs, and derived tables."""
    text = str(sql or "")
    aliases = {
        match.lower() for match in re.findall(
            r'\bAS\s+"?([A-Za-z_]\w*)"?', text, re.I,
        )
    }
    aliases.update(
        match.lower() for match in re.findall(
            r'(?:\bWITH|,)\s*"?([A-Za-z_]\w*)"?\s+AS\s*\(', text, re.I,
        )
    )
    return aliases


def _normalize_schema_columns(columns: set[str], database: str) -> set[str]:
    """Resolve unqualified source columns and discard derived output aliases."""
    _, valid_columns = _schema_catalog(database)
    by_name: dict[str, set[str]] = {}
    for qualified in valid_columns:
        by_name.setdefault(qualified.split(".", 1)[1], set()).add(qualified)
    normalized = set()
    for value in columns:
        value = str(value).lower()
        if value in valid_columns:
            normalized.add(value)
            continue
        name = value.split(".", 1)[-1]
        matches = by_name.get(name, set())
        if len(matches) == 1:
            normalized.update(matches)
    return normalized


def _invalid_reference_metrics(
    tables: set[str], columns: set[str], kb: set[str], database: str,
    sql: str | None = None,
) -> dict[str, Any]:
    valid_tables, valid_columns = _schema_catalog(database)
    valid_column_names = {value.split(".", 1)[1] for value in valid_columns}
    invalid_tables = {value for value in tables if value not in valid_tables}
    invalid_columns = set()
    aliases = _declared_sql_aliases(sql)
    schema_columns = {
        value for value in columns
        if not (
            value.split(".", 1)[-1] in aliases
            and value not in valid_columns
            and value.split(".", 1)[-1] not in valid_column_names
        )
    }
    for value in schema_columns:
        if value.startswith("*."):
            if value[2:] not in valid_column_names and value[2:] not in aliases:
                invalid_columns.add(value)
        elif value not in valid_columns and value.split(".", 1)[-1] not in aliases:
            invalid_columns.add(value)
    valid_kb = _kb_catalog(database)
    invalid_kb = {value for value in kb if value not in valid_kb}
    total = len(tables) + len(schema_columns) + len(kb)
    invalid_total = len(invalid_tables) + len(invalid_columns) + len(invalid_kb)
    def rate(invalid, referenced):
        return len(invalid) / len(referenced) if referenced else None
    return {
        "table_referenced_count": len(tables),
        "invalid_table_count": len(invalid_tables),
        "invalid_table_rate": rate(invalid_tables, tables),
        "invalid_tables": sorted(invalid_tables),
        "column_referenced_count": len(schema_columns),
        "invalid_column_count": len(invalid_columns),
        "invalid_column_rate": rate(invalid_columns, schema_columns),
        "invalid_columns": sorted(invalid_columns),
        "kb_referenced_count": len(kb),
        "invalid_kb_count": len(invalid_kb),
        "invalid_kb_rate": rate(invalid_kb, kb),
        "invalid_kb_references": sorted(invalid_kb),
        "referenced_count": total, "invalid_count": invalid_total,
        "overall_invalid_reference_rate": invalid_total / total if total else None,
        "task_has_invalid_reference": invalid_total > 0,
    }


def _last_finalized_context(calls: list[dict]) -> dict | None:
    for call in reversed(calls):
        if call.get("tool") != "finalize_preprocessing_context":
            continue
        result = _payload(call.get("result"))
        if result.get("valid") is True:
            return call
    return None


def _final_sql(calls: list[dict]) -> str | None:
    submissions = [call for call in calls if call.get("tool") in SUBMISSION_TOOLS]
    if submissions:
        return _state_backed_sql(submissions[-1], calls)
    validations = [call for call in calls if call.get("tool") == "validate_sql_to_plan"]
    if validations:
        return _state_backed_sql(validations[-1], calls)
    return final_attempted_sql(calls)


def _preprocessing_references(call: dict | None) -> tuple[set, set, set, set]:
    if not call:
        return set(), set(), set(), set()
    args = call.get("args") or {}
    tables = {str(value).lower() for value in args.get("selected_tables", [])}
    columns = set()
    for group in args.get("selected_columns", []) or []:
        if not isinstance(group, dict):
            continue
        table = str(group.get("table", "")).lower()
        values = group.get("columns")
        if values is None and group.get("column", group.get("name")):
            values = [group.get("column", group.get("name"))]
        for value in values or []:
            name = value.get("name") if isinstance(value, dict) else value
            if table and name:
                columns.add(f"{table}.{str(name).lower()}")
    joins = set()
    for edge in args.get("selected_join_edges", []) or []:
        if isinstance(edge, dict) and edge.get("left") and edge.get("right"):
            joins.add(tuple(sorted((str(edge["left"]).lower(), str(edge["right"]).lower()))))
    kb = set()
    for item in args.get("selected_knowledge", []) or []:
        value = item.get("id", item.get("knowledge_id", item.get("kb_id"))) if isinstance(item, dict) else item
        if value is not None:
            kb.add(str(value))
    return tables, columns, joins, kb


def _phase_metrics(task: dict, calls: list[dict], result: dict) -> dict:
    database = str(task.get("selected_database", ""))
    expected_tables, expected_joins = sql_schema_elements(task.get("sol_sql"))
    expected_columns = _normalize_schema_columns(sql_columns(task.get("sol_sql")), database)
    expected_kb = {str(value) for value in task.get("external_knowledge", []) or []}
    finalized = _last_finalized_context(calls)
    pre_tables, pre_columns, pre_joins, pre_kb = _preprocessing_references(finalized)
    pre_columns = _normalize_schema_columns(pre_columns, database)
    pre_scores = {
        "completion": finalized is not None,
        "table": _set_metric(pre_tables, expected_tables),
        "column": _set_metric(pre_columns, expected_columns),
        "join_path": _set_metric(pre_joins, expected_joins),
        "kb": _set_metric(pre_kb, expected_kb),
    }
    pre_scores["complete_grounding"] = all(
        expected.issubset(selected) for selected, expected in (
            (pre_tables, expected_tables), (pre_columns, expected_columns),
            (pre_joins, expected_joins), (pre_kb, expected_kb),
        )
    ) and finalized is not None

    plans = [call for call in calls if call.get("tool") in PLAN_TOOLS]
    plan_valid = [_payload(call.get("result")).get("valid") is True for call in plans]
    latest_plan = (plans[-1].get("args") or {}).get("plan", {}) if plans else {}
    plan_features = set()
    if latest_plan.get("joins"):
        plan_features.add("join")
    for field, feature in (
        ("requires_cte", "cte"), ("requires_subquery", "subquery"),
        ("requires_set_operation", "union"), ("grouping", "group_by"),
        ("ordering", "order_by"), ("limit", "limit"), ("distinct", "distinct"),
    ):
        if latest_plan.get(field):
            plan_features.add(feature)
    if latest_plan.get("conditions") or latest_plan.get("population_constraints") or latest_plan.get("global_filters"):
        plan_features.add("where")
    plan_text = json.dumps(latest_plan, default=str).lower()
    if re.search(r"\b(?:count|sum|avg|min|max)\s*\(", plan_text):
        plan_features.add("aggregate")
    if re.search(r"\bover\s*\(", plan_text):
        plan_features.add("window")
    expected_features = _structural_features("\n".join(map(str, task.get("sol_sql", []) or [])))
    query_structure = _set_metric(plan_features, expected_features)
    last_plan_payload = _payload(plans[-1].get("result")) if plans else {}
    semantic_checks = (last_plan_payload.get("semantic_contract") or {}).get("checks", {})
    validations = [call for call in calls if call.get("tool") == "validate_sql_to_plan"]
    rejection_reasons = []
    for call in validations:
        payload = _payload(call.get("result"))
        if payload.get("valid") is not True:
            rejection_reasons.extend(
                str(item.get("code", item.get("type", "unknown")))
                for item in payload.get("diagnoses", []) if isinstance(item, dict)
            )

    category = str(task.get("category", "Query"))
    executions = [call for call in calls if call.get("tool") in EXECUTION_TOOLS]
    first_failure = bool(executions) and not _execution_succeeded(executions[0])
    if category == "Management":
        eligible = bool(validations) and not _validation_succeeded(validations[0])
        attempts = max(0, len(validations) - 1) if eligible else 0
        resolved = eligible and any(_validation_succeeded(call) for call in validations[1:])
    else:
        eligible = first_failure
        attempts = max(0, len(executions) - 1) if eligible else 0
        resolved = eligible and any(_execution_succeeded(call) for call in executions[1:])

    formula = result.get("kb_formula_preservation") or {}
    return {
        "applicable": True,
        "preprocessing": pre_scores,
        "query_planning": {
            "attempts": len(plans), "completion": any(plan_valid),
            "first_attempt_accepted": bool(plan_valid and plan_valid[0]),
            "query_structural_agreement": query_structure if category == "Query" else None,
            "management_contract_complete": (
                bool(semantic_checks.get("management_contract_complete"))
                if category == "Management" and semantic_checks else None
            ),
            "kb_formula_preservation": formula or None,
        },
        "sql_generation": {
            "validation_attempts": len(validations),
            "first_validation_passed": bool(validations and _validation_succeeded(validations[0])),
            "rejection_reasons": sorted(set(rejection_reasons)),
        },
        "postprocessing": {
            "eligible": eligible, "correction_attempted": attempts > 0,
            "correction_attempts": attempts, "resolved": bool(resolved),
            "recovered_task": bool(resolved and result.get("phase1_passed")),
        },
    }


def _harness_metrics(calls: list[dict], raw: dict) -> dict:
    guarded = 0
    workflow_blocks = []
    submission_attempts = 0
    compliant_submissions = 0
    for index, call in enumerate(calls):
        payload = _payload(call.get("result"))
        reason = str(payload.get("reason", ""))
        if call.get("tool") in {
            "execute_sql", "submit_sql", "generate_and_validate_query_plan",
            "validate_sql_to_plan", "execute_validated_sql",
            "diagnose_last_execution_error", "submit_validated_sql",
        }:
            guarded += 1
        if reason in WORKFLOW_BLOCK_REASONS:
            workflow_blocks.append((index, reason))
        if call.get("tool") in SUBMISSION_TOOLS:
            submission_attempts += 1
            if not payload.get("harness_blocked"):
                compliant_submissions += 1

    recovery_tools = {
        "schema_required_before_execution": {"get_schema", "get_schema_summary", "prepare_schema_context"},
        "schema_required_before_submission": {"get_schema", "get_schema_summary", "prepare_schema_context"},
        "preprocessing_not_finalized": {"finalize_preprocessing_context"},
        "new_preprocessing_evidence_required": {"prepare_schema_context", "prepare_knowledge_context", "inspect_database", "finalize_preprocessing_context"},
        "validated_plan_required": PLAN_TOOLS,
        "validated_sql_required": {"validate_sql_to_plan"},
        "execution_diagnosis_required_before_correction": POSTPROCESSING_TOOLS,
        "execution_diagnosis_required_before_retry": POSTPROCESSING_TOOLS,
        "current_query_sql_not_successfully_executed": EXECUTION_TOOLS,
    }
    recoverable = [(index, reason) for index, reason in workflow_blocks if reason in recovery_tools]
    recovered = sum(
        any(call.get("tool") in recovery_tools[reason] for call in calls[index + 1:])
        for index, reason in recoverable
    )
    return {
        "applicable": True,
        "guarded_attempts": guarded,
        "hard_constraint_violations": len(workflow_blocks),
        "hard_constraint_violation_attempt_rate": len(workflow_blocks) / guarded if guarded else None,
        "recoverable_blocks": len(recoverable), "recovered_blocks": recovered,
        "post_block_recovery_rate": recovered / len(recoverable) if recoverable else None,
        "submission_attempts": submission_attempts,
        "compliant_submissions": compliant_submissions,
        "submission_gate_compliance_rate": compliant_submissions / submission_attempts if submission_attempts else None,
        "violation_reasons": dict(raw.get("block_reasons", {})),
        "termination_reason": raw.get("termination_reason"),
        "raw_blocked_call_count": int(raw.get("blocked_calls", 0) or 0),
    }


def _multi_metrics(raw: dict) -> dict:
    completion = raw.get("phase_completion", {}) or {}
    routing = raw.get("routing_events", []) or []
    backward = [event for event in routing if event.get("from") != event.get("to")]
    recovered = sum(bool(completion.get(event.get("from"))) for event in backward)
    agents = raw.get("agent_metrics", {}) or {}
    total_tokens = sum(int(values.get("total_tokens", 0) or 0) for values in agents.values())
    token_share = {
        phase: int(values.get("total_tokens", 0) or 0) / total_tokens
        if total_tokens else None
        for phase, values in agents.items()
    }
    return {
        "applicable": True,
        "phase_completion": completion,
        "end_to_end_phase_completion": bool(completion) and all(completion.values()),
        "backward_transitions": int(raw.get("backward_transitions", 0) or 0),
        "recovered_handoffs": recovered,
        "handoff_recovery_rate": recovered / len(backward) if backward else None,
        "new_evidence_recovery_rate": (
            sum(bool(event.get("new_evidence_acquired")) for event in backward) / len(backward)
            if backward else None
        ),
        "per_agent_token_share": token_share,
        "agent_metrics": agents,
        "phase_retries": int(raw.get("phase_retries", 0) or 0),
        "routing_events": routing,
        "coordinator_termination_reason": raw.get("multi_agent_stop_reason"),
    }


def calculate_task_evaluation_metrics(task: dict, result: dict) -> dict[str, Any]:
    calls = result.get("tool_trajectory") if isinstance(result.get("tool_trajectory"), list) else []
    experiment = result.get("experiment") or {}
    variant = int(experiment.get("number", -1))
    final_sql = _final_sql(calls)
    predicted_tables, predicted_joins = sql_schema_elements(final_sql)
    database = str(task.get("selected_database", ""))
    predicted_columns = _normalize_schema_columns(sql_columns(final_sql), database)
    expected_tables, expected_joins = sql_schema_elements(task.get("sol_sql"))
    expected_columns = _normalize_schema_columns(sql_columns(task.get("sol_sql")), database)
    predicted_kb = {str(value) for value in selected_kb_ids(calls)}
    expected_kb = {str(value) for value in task.get("external_knowledge", []) or []}
    kb_refs = _all_kb_references(calls)
    final_features = _structural_features(final_sql)
    expected_features = _structural_features("\n".join(map(str, task.get("sol_sql", []) or [])))
    submissions = [call for call in calls if call.get("tool") in SUBMISSION_TOOLS]

    common = {
        "task_success": bool(result.get("phase1_passed")),
        "submission_present": bool(submissions),
        "table": _set_metric(predicted_tables, expected_tables),
        "column": _set_metric(predicted_columns, expected_columns),
        "join_path": _set_metric(predicted_joins, expected_joins),
        "kb": _set_metric(predicted_kb, expected_kb),
        "invalid_references": _invalid_reference_metrics(
            predicted_tables, sql_columns(final_sql), kb_refs, database, final_sql,
        ),
        "final_sql_structural_agreement": _set_metric(final_features, expected_features),
        "steps_used": int(result.get("steps_used", 0) or 0),
        "total_tokens": int(result.get("total_tokens", 0) or 0),
        "elapsed_seconds": float(result.get("elapsed_seconds", 0) or 0),
    }
    return {
        "common": common,
        "phase_specific": (
            _phase_metrics(task, calls, result) if variant in IMPROVED_VARIANTS
            else {"applicable": False, "reason": "structured_phase_artifacts_unavailable"}
        ),
        "harness_specific": (
            _harness_metrics(calls, result.get("harness_metrics") or {})
            if variant in HARNESS_VARIANTS
            else {"applicable": False, "reason": "improved_harness_inactive"}
        ),
        "multi_agent_specific": (
            _multi_metrics(result.get("multi_agent_metrics") or {})
            if variant in MULTI_VARIANTS
            else {"applicable": False, "reason": "single_agent_orchestration"}
        ),
    }


def aggregate_evaluation_metrics(results: list[dict]) -> dict[str, Any]:
    evaluated = [result.get("evaluation_metrics") for result in results]
    evaluated = [value for value in evaluated if isinstance(value, dict)]
    if not evaluated:
        return {}
    common = [value["common"] for value in evaluated]
    def mean(path, applicable_only=False):
        values = []
        for item in evaluated if applicable_only else common:
            value = item
            for key in path:
                if not isinstance(value, dict) or key not in value:
                    value = None
                    break
                value = value[key]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
            elif isinstance(value, bool):
                values.append(float(value))
        return sum(values) / len(values) if values else None
    def micro(component):
        selected = sum(item[component]["selected_count"] for item in common)
        expected = sum(item[component]["expected_count"] for item in common)
        correct = sum(item[component]["correct_count"] for item in common)
        precision = correct / selected if selected else (1.0 if not expected else 0.0)
        recall = correct / expected if expected else 1.0
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    invalid = [item["invalid_references"] for item in common]
    invalid_count = sum(item["invalid_count"] for item in invalid)
    referenced_count = sum(item["referenced_count"] for item in invalid)
    output = {
        "tasks": len(common), "task_success_rate": mean(("task_success",)),
        "submission_rate": mean(("submission_present",)),
        "table_f1": micro("table"), "column_f1": micro("column"),
        "join_path_f1": micro("join_path"), "kb_f1": micro("kb"),
        "overall_invalid_reference_rate": invalid_count / referenced_count if referenced_count else None,
        "invalid_table_rate": (
            sum(item["invalid_table_count"] for item in invalid)
            / sum(item["table_referenced_count"] for item in invalid)
            if sum(item["table_referenced_count"] for item in invalid) else None
        ),
        "invalid_column_rate": (
            sum(item["invalid_column_count"] for item in invalid)
            / sum(item["column_referenced_count"] for item in invalid)
            if sum(item["column_referenced_count"] for item in invalid) else None
        ),
        "invalid_kb_rate": (
            sum(item["invalid_kb_count"] for item in invalid)
            / sum(item["kb_referenced_count"] for item in invalid)
            if sum(item["kb_referenced_count"] for item in invalid) else None
        ),
        "task_invalid_reference_rate": sum(item["task_has_invalid_reference"] for item in invalid) / len(invalid),
        "final_sql_structural_f1": micro("final_sql_structural_agreement"),
        "average_steps_used": mean(("steps_used",)),
        "average_total_tokens": mean(("total_tokens",)),
        "average_latency_seconds": mean(("elapsed_seconds",)),
    }
    phases = [value["phase_specific"] for value in evaluated if value["phase_specific"].get("applicable")]
    if phases:
        def phase_micro(name):
            selected = sum(item["preprocessing"][name]["selected_count"] for item in phases)
            expected = sum(item["preprocessing"][name]["expected_count"] for item in phases)
            correct = sum(item["preprocessing"][name]["correct_count"] for item in phases)
            precision = correct / selected if selected else (1.0 if not expected else 0.0)
            recall = correct / expected if expected else 1.0
            return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        query_structures = [
            item["query_planning"]["query_structural_agreement"] for item in phases
            if isinstance(item["query_planning"].get("query_structural_agreement"), dict)
        ]
        management_contracts = [
            item["query_planning"]["management_contract_complete"] for item in phases
            if item["query_planning"].get("management_contract_complete") is not None
        ]
        formula_checks = [
            item["query_planning"]["kb_formula_preservation"] for item in phases
            if isinstance(item["query_planning"].get("kb_formula_preservation"), dict)
        ]
        formula_expected = sum(
            int(item.get("expected_count", item.get("checked_knowledge_count", 0)) or 0)
            for item in formula_checks
        )
        formula_preserved = sum(
            int(item.get("preserved_count", item.get("preserved_knowledge_count", 0)) or 0)
            for item in formula_checks
        )
        rejection_reasons = Counter(
            reason for item in phases
            for reason in item["sql_generation"].get("rejection_reasons", [])
        )
        eligible_post = [item["postprocessing"] for item in phases if item["postprocessing"]["eligible"]]
        output["phase_specific"] = {
            "applicable_tasks": len(phases),
            "preprocessing_completion_rate": sum(item["preprocessing"]["completion"] for item in phases) / len(phases),
            "preprocessing_table_f1": phase_micro("table"),
            "preprocessing_column_f1": phase_micro("column"),
            "preprocessing_join_path_f1": phase_micro("join_path"),
            "preprocessing_kb_f1": phase_micro("kb"),
            "complete_grounding_rate": sum(item["preprocessing"]["complete_grounding"] for item in phases) / len(phases),
            "plan_completion_rate": sum(item["query_planning"]["completion"] for item in phases) / len(phases),
            "first_plan_acceptance_rate": sum(item["query_planning"]["first_attempt_accepted"] for item in phases) / len(phases),
            "average_plan_attempts": sum(item["query_planning"]["attempts"] for item in phases) / len(phases),
            "average_query_structural_f1": sum(item["f1"] for item in query_structures) / len(query_structures) if query_structures else None,
            "management_contract_completeness_rate": sum(management_contracts) / len(management_contracts) if management_contracts else None,
            "kb_formula_preservation_rate": formula_preserved / formula_expected if formula_expected else None,
            "first_validation_pass_rate": sum(item["sql_generation"]["first_validation_passed"] for item in phases) / len(phases),
            "average_validation_attempts": sum(item["sql_generation"]["validation_attempts"] for item in phases) / len(phases),
            "validation_rejection_reasons": dict(sorted(rejection_reasons.items())),
            "postprocessing_eligible_tasks": len(eligible_post),
            "correction_attempt_rate": sum(item["correction_attempted"] for item in eligible_post) / len(eligible_post) if eligible_post else None,
            "average_correction_attempts": sum(item["correction_attempts"] for item in eligible_post) / len(eligible_post) if eligible_post else None,
            "resolution_rate": sum(item["resolved"] for item in eligible_post) / len(eligible_post) if eligible_post else None,
            "recovered_task_rate": sum(item["postprocessing"]["recovered_task"] for item in phases) / len(phases),
        }
    else:
        output["phase_specific"] = {"applicable_tasks": 0, "applicable": False}

    harness = [value["harness_specific"] for value in evaluated if value["harness_specific"].get("applicable")]
    if harness:
        guarded = sum(item["guarded_attempts"] for item in harness)
        violations = sum(item["hard_constraint_violations"] for item in harness)
        recoverable = sum(item["recoverable_blocks"] for item in harness)
        recovered = sum(item["recovered_blocks"] for item in harness)
        submissions_total = sum(item["submission_attempts"] for item in harness)
        compliant = sum(item["compliant_submissions"] for item in harness)
        violation_reasons = Counter()
        termination_reasons = Counter()
        for item in harness:
            violation_reasons.update(item.get("violation_reasons") or {})
            if item.get("termination_reason"):
                termination_reasons[str(item["termination_reason"])] += 1
        output["harness_specific"] = {
            "applicable_tasks": len(harness),
            "hard_constraint_violation_attempt_rate": violations / guarded if guarded else None,
            "post_block_recovery_rate": recovered / recoverable if recoverable else None,
            "submission_gate_compliance_rate": compliant / submissions_total if submissions_total else None,
            "raw_blocked_call_count": sum(item["raw_blocked_call_count"] for item in harness),
            "constraint_violations_by_reason": dict(sorted(violation_reasons.items())),
            "termination_reasons": dict(sorted(termination_reasons.items())),
        }
    else:
        output["harness_specific"] = {"applicable_tasks": 0, "applicable": False}

    multi = [value["multi_agent_specific"] for value in evaluated if value["multi_agent_specific"].get("applicable")]
    if multi:
        backwards = sum(item["backward_transitions"] for item in multi)
        recovered = sum(item["recovered_handoffs"] for item in multi)
        token_phases = set().union(*(item["per_agent_token_share"] for item in multi))
        completion_phases = set().union(*(item["phase_completion"] for item in multi))
        stop_reasons = Counter(
            str(item["coordinator_termination_reason"]) for item in multi
            if item.get("coordinator_termination_reason")
        )
        evidence_rates = [
            item["new_evidence_recovery_rate"] for item in multi
            if item.get("new_evidence_recovery_rate") is not None
        ]
        output["multi_agent_specific"] = {
            "applicable_tasks": len(multi),
            "end_to_end_phase_completion_rate": sum(item["end_to_end_phase_completion"] for item in multi) / len(multi),
            "average_backward_transitions": backwards / len(multi),
            "handoff_recovery_rate": recovered / backwards if backwards else None,
            "phase_completion_rate": {
                phase: sum(bool(item["phase_completion"].get(phase)) for item in multi) / len(multi)
                for phase in sorted(completion_phases)
            },
            "average_phase_retries": sum(item["phase_retries"] for item in multi) / len(multi),
            "new_evidence_recovery_rate": sum(evidence_rates) / len(evidence_rates) if evidence_rates else None,
            "coordinator_termination_reasons": dict(sorted(stop_reasons.items())),
            "average_per_agent_token_share": {
                phase: sum((item["per_agent_token_share"].get(phase) or 0) for item in multi) / len(multi)
                for phase in sorted(token_phases)
            },
        }
    else:
        output["multi_agent_specific"] = {"applicable_tasks": 0, "applicable": False}
    return output
