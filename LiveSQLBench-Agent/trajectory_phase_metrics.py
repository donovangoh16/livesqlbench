"""Phase grouping and observable metrics for LiveSQLBench trajectories."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from schema_linking_metrics import (
    _set_scores,
    final_attempted_sql,
    selected_kb_ids,
    sql_schema_elements,
)


PREPROCESSING_TOOLS = {
    "get_schema",
    "get_schema_summary",
    "get_selected_schema",
    "get_all_column_meanings",
    "get_column_meaning",
    "get_all_external_knowledge_names",
    "get_knowledge_definition",
    "get_all_knowledge_definitions",
    "rank_relevant_tables",
    "rank_relevant_columns",
    "find_join_paths",
    "identify_knowledge_requirements",
    "rank_relevant_knowledge",
    "check_kb_completeness",
    "finalize_preprocessing_context",
    "prepare_schema_context",
    "prepare_knowledge_context",
    "inspect_database",
}

QUERY_PLANNING_TOOLS = {
    "generate_query_plan", "validate_query_plan", "generate_and_validate_query_plan",
}
SQL_GENERATION_TOOLS = {"validate_sql_to_plan"}
POSTPROCESSING_TOOLS = {"diagnose_execution_error", "diagnose_last_execution_error"}
EXECUTION_TOOLS = {"execute_sql", "execute_validated_sql"}
SUBMISSION_TOOLS = {"submit_sql", "submit_validated_sql"}


def group_trajectory(trajectory: Any) -> dict[str, list[dict]]:
    """Assign tool calls to the four observable pipeline phases."""
    grouped = {
        "preprocessing": [],
        "query_planning": [],
        "sql_generation": [],
        "postprocessing": [],
    }
    first_execution_seen = False
    first_execution_failed = False
    for call in trajectory if isinstance(trajectory, list) else []:
        tool = call.get("tool")
        if tool in PREPROCESSING_TOOLS:
            grouped["preprocessing"].append(call)
        elif tool in QUERY_PLANNING_TOOLS:
            grouped["query_planning"].append(call)
        elif tool == "validate_sql_to_plan" and first_execution_failed:
            grouped["postprocessing"].append(call)
        elif tool in SQL_GENERATION_TOOLS:
            grouped["sql_generation"].append(call)
        elif tool in POSTPROCESSING_TOOLS:
            grouped["postprocessing"].append(call)
        elif tool in EXECUTION_TOOLS and not first_execution_seen:
            # One observable artifact represents both the implicit plan and its
            # initial SQL translation, so retain it under both phase views.
            grouped["sql_generation"].append(call)
            first_execution_seen = True
            first_execution_failed = not _execution_succeeded(call)
        elif tool in EXECUTION_TOOLS and first_execution_failed:
            grouped["postprocessing"].append(call)
    return grouped


def _sql(call: dict | None) -> str | None:
    return str((call.get("args") or {}).get("sql")) if call and (call.get("args") or {}).get("sql") else None


def _execution_succeeded(call: dict | None) -> bool:
    if not call:
        return False
    result = str(call.get("result") or "").lower()
    error_markers = ("sql error:", "error calling db environment", "execution timed out")
    return not any(marker in result for marker in error_markers)


def _parse_succeeded(sql: str | None) -> bool:
    if not sql:
        return False
    from schema_linking_metrics import _parse_all

    return bool(_parse_all(sql))


def _structural_features(sql: str | None) -> set[str]:
    if not sql:
        return set()
    from schema_linking_metrics import _parse_all
    from sqlglot import exp

    trees = _parse_all(sql)
    features = set()
    mappings = {
        "join": exp.Join,
        "cte": exp.CTE,
        "subquery": exp.Subquery,
        "aggregate": exp.AggFunc,
        "window": exp.Window,
        "union": exp.Union,
        "intersect": exp.Intersect,
        "except": exp.Except,
        "group_by": exp.Group,
        "having": exp.Having,
        "order_by": exp.Order,
        "limit": exp.Limit,
        "distinct": exp.Distinct,
        "where": exp.Where,
        "update": exp.Update,
        "delete": exp.Delete,
        "insert": exp.Insert,
        "create": exp.Create,
        "alter": exp.Alter,
    }
    for name, expression_type in mappings.items():
        if any(tree.find(expression_type) is not None for tree in trees):
            features.add(name)
    return features


def task_phase_metrics(row: pd.Series) -> pd.Series:
    grouped = group_trajectory(row.get("tool_trajectory"))
    preprocessing = grouped["preprocessing"]
    first_call = grouped["sql_generation"][0] if grouped["sql_generation"] else None
    first_sql = _sql(first_call)
    postprocessing_calls = grouped["postprocessing"]
    later_executes = [
        call for call in postprocessing_calls if call.get("tool") in EXECUTION_TOOLS
    ]
    error_diagnoses = [
        call for call in postprocessing_calls
        if call.get("tool") == "diagnose_execution_error"
    ]
    all_calls = row.get("tool_trajectory") if isinstance(row.get("tool_trajectory"), list) else []
    submissions = [c for c in all_calls if c.get("tool") in SUBMISSION_TOOLS]
    final_sql = _sql(submissions[-1]) if submissions else final_attempted_sql(row.get("tool_trajectory"))

    ground_truth_tables, ground_truth_joins = sql_schema_elements(row.get("sol_sql"))
    initial_tables, initial_joins = sql_schema_elements(first_sql)
    final_tables, final_joins = sql_schema_elements(final_sql)
    expected_kb = {str(value) for value in (row.get("external_knowledge") or [])}
    retrieved_kb = {str(value) for value in selected_kb_ids(preprocessing)}
    plan_kb = set()
    for call in all_calls:
        if call.get("tool") in {"generate_query_plan", "generate_and_validate_query_plan"}:
            plan = (call.get("args") or {}).get("plan") or {}
            for field in ("calculations", "conditions", "filters"):
                for item in plan.get(field, []) or []:
                    if isinstance(item, dict):
                        kid = item.get("knowledge_id", item.get("kb_id"))
                        if kid is not None:
                            plan_kb.add(str(kid))
    sql_validation_passed = any(
        call.get("tool") == "validate_sql_to_plan"
        and '"valid": true' in str(call.get("result", "")).lower()
        for call in all_calls
    )
    sql_kb = plan_kb if sql_validation_passed else set()

    initial_features = _structural_features(first_sql)
    ground_truth_features = _structural_features("\n".join(str(x) for x in (row.get("sol_sql") or [])))
    final_features = _structural_features(final_sql)

    counts = Counter(call.get("tool") for call in preprocessing)
    token_usage = row.get("token_usage") if isinstance(row.get("token_usage"), dict) else {}

    def token_value(name: str) -> int:
        value = row.get(name)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            value = token_usage.get(name, 0)
        return int(value or 0)

    input_tokens = token_value("input_tokens")
    output_tokens = token_value("output_tokens")
    total_tokens = token_value("total_tokens")
    cached_input_tokens = token_value("cached_input_tokens")
    steps_used = int(row.get("steps_used", 0) or 0)
    experiment = row.get("experiment") if isinstance(row.get("experiment"), dict) else {}
    improved_profile = experiment.get("active_agent_profile") == "improved"
    planning_attempts = sum(
        call.get("tool") in {"generate_query_plan", "generate_and_validate_query_plan"}
        for call in all_calls
    )
    metrics: dict[str, Any] = {
        "phase_groups": grouped,
        "preprocessing_call_count": len(preprocessing),
        "query_planning_call_count": len(grouped["query_planning"]),
        "sql_generation_call_count": len(grouped["sql_generation"]),
        "postprocessing_call_count": len(grouped["postprocessing"]),
        "tool_response_character_count": sum(
            len(str(call.get("result") or "")) for call in all_calls
        ),
        "exposed_tool_count": 9 if improved_profile else 8,
        "schema_retrieved": (
            counts["get_schema"] + counts["get_schema_summary"]
            + counts["get_selected_schema"] > 0
        ),
        "column_metadata_retrieved": (
            counts["get_all_column_meanings"] + counts["get_column_meaning"] > 0
        ),
        "knowledge_catalogue_retrieved": counts["get_all_external_knowledge_names"] > 0,
        "initial_sql": first_sql,
        "initial_sql_parse_success": _parse_succeeded(first_sql),
        "initial_sql_execution_success": _execution_succeeded(first_call),
        "first_execution_failed": bool(first_call) and not _execution_succeeded(first_call),
        "postprocessing_eligible": bool(first_call) and not _execution_succeeded(first_call),
        "postprocessing_attempt_count": len(later_executes),
        "postprocessing_diagnosis_count": len(error_diagnoses),
        "submission_present": bool(submissions),
        "final_sql": final_sql,
        "final_passed": bool(row.get("phase1_passed")),
        "postprocessing_attempted": bool(later_executes),
        "initial_execution_error_resolved": (
            bool(first_call)
            and not _execution_succeeded(first_call)
            and any(_execution_succeeded(call) for call in later_executes)
        ),
        "planning_features_initial": sorted(initial_features),
        "planning_features_ground_truth": sorted(ground_truth_features),
        "planning_features_final": sorted(final_features),
        "metric_input_tokens": input_tokens,
        "metric_output_tokens": output_tokens,
        "metric_total_tokens": total_tokens,
        "metric_cached_input_tokens": cached_input_tokens,
        "metric_uncached_input_tokens": max(0, input_tokens - cached_input_tokens),
        "prompt_cache_hit_rate": (
            cached_input_tokens / input_tokens if input_tokens else 0.0
        ),
        "tokens_per_tool_step": total_tokens / steps_used if steps_used else None,
        "tokens_per_success": total_tokens if bool(row.get("phase1_passed")) else None,
        "token_usage_available": bool(token_usage.get("usage_event_count", 0)),
        "model_call_count": int(token_usage.get("usage_event_count", 0) or 0),
        "planning_retry_count": max(0, planning_attempts - 1),
        "sql_validation_retry_count": max(
            0,
            sum(call.get("tool") == "validate_sql_to_plan" for call in all_calls) - 1,
        ),
    }
    metrics.update(_set_scores(retrieved_kb, expected_kb, "phase_kb"))
    metrics.update(_set_scores(plan_kb, expected_kb, "kb_plan_usage"))
    metrics.update(_set_scores(sql_kb, expected_kb, "kb_sql_usage"))
    metrics["kb_retrieval_complete"] = retrieved_kb == expected_kb
    metrics["kb_plan_usage_complete"] = plan_kb == expected_kb
    metrics["kb_sql_usage_complete"] = sql_kb == expected_kb
    metrics["all_required_kb_used"] = (
        retrieved_kb == expected_kb == plan_kb == sql_kb
    )
    metrics.update(_set_scores(initial_tables, ground_truth_tables, "initial_table"))
    metrics.update(_set_scores(initial_joins, ground_truth_joins, "initial_join_path"))
    metrics.update(_set_scores(initial_features, ground_truth_features, "initial_plan_structure"))
    metrics.update(_set_scores(final_tables, ground_truth_tables, "final_table"))
    metrics.update(_set_scores(final_joins, ground_truth_joins, "final_join_path"))
    metrics["plan_structure_exact_match"] = initial_features == ground_truth_features
    metrics["final_structure_exact_match"] = final_features == ground_truth_features
    metrics["structure_changed_during_postprocessing"] = initial_features != final_features
    return pd.Series(metrics)


def add_phase_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([frame.copy(), frame.apply(task_phase_metrics, axis=1)], axis=1)
