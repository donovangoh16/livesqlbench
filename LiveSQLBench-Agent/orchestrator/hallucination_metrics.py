"""Evaluation-only hallucination metrics for pipeline artifacts.

These metrics compare references in each observable artifact with the reference
SQL/KB annotations.  Consequently, ``hallucination_rate`` means reference
divergence (false discoveries), not necessarily a reference to a nonexistent
database object.  Explicit unsupported-reference errors are reported
separately.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from schema_linking_metrics import sql_columns, sql_schema_elements


PLAN_TOOLS = {"generate_query_plan", "generate_and_validate_query_plan"}
VALIDATION_TOOLS = {"validate_sql_to_plan"}
SUBMISSION_TOOLS = {"submit_sql", "submit_validated_sql"}
UNSUPPORTED_MARKERS = (
    "does not exist",
    "unknown table",
    "unknown column",
    "unsupported reference",
    "selected table is not",
    "selected column is not",
    "selected knowledge entry does not exist",
)


def _ids(values: Any) -> set[str]:
    if not isinstance(values, (list, tuple, set)):
        return set()
    return {str(value) for value in values if value is not None}


def _score(predicted: Iterable, expected: Iterable) -> dict[str, Any]:
    predicted, expected = set(predicted), set(expected)
    false_positive = predicted - expected
    return {
        "referenced_count": len(predicted),
        "supported_count": len(predicted & expected),
        "hallucinated_count": len(false_positive),
        "hallucination_rate": (
            len(false_positive) / len(predicted) if predicted else 0.0
        ),
        "hallucinated_references": sorted(false_positive, key=str),
    }


def _call_sql(call: dict | None, calls: list[dict]) -> str | None:
    if not call:
        return None
    sql = (call.get("args") or {}).get("sql")
    if sql:
        return str(sql)
    index = calls.index(call)
    for prior in reversed(calls[:index]):
        if prior.get("tool") in VALIDATION_TOOLS:
            sql = (prior.get("args") or {}).get("sql")
            if sql:
                return str(sql)
    return None


def _last_valid_preprocessing(calls: list[dict]) -> dict:
    candidates = [
        call for call in calls if call.get("tool") == "finalize_preprocessing_context"
    ]
    for call in reversed(candidates):
        result = call.get("result")
        try:
            payload = result if isinstance(result, dict) else json.loads(str(result))
        except (TypeError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict) and payload.get("valid") is True:
            return call
    return candidates[-1] if candidates else {}


def _preprocessing_refs(call: dict) -> tuple[set[str], set[str], set[str]]:
    args = call.get("args") or {}
    tables = {str(value).lower() for value in args.get("selected_tables", [])}
    columns: set[str] = set()
    for group in args.get("selected_columns", []) or []:
        if not isinstance(group, dict):
            continue
        table = str(group.get("table", "")).lower()
        # The preprocessing tool accepts both grouped and normalized-flat
        # column artifacts. Evaluation must recognize both representations.
        values = group.get("columns")
        if values is None and group.get("column", group.get("name")):
            values = [group.get("column", group.get("name"))]
        for value in values or []:
            name = value.get("name") if isinstance(value, dict) else value
            if table and name:
                columns.add(f"{table}.{str(name).lower()}")
    knowledge = set()
    for value in args.get("selected_knowledge", []) or []:
        if isinstance(value, dict):
            value = value.get("id", value.get("knowledge_id", value.get("kb_id")))
        if value is not None:
            knowledge.add(str(value))
    return tables, columns, knowledge


def _walk(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _plan_refs(call: dict) -> tuple[set[str], set[str], set[str]]:
    plan = (call.get("args") or {}).get("plan") or {}
    tables = {str(value).lower() for value in plan.get("source_tables", []) or []}
    columns, knowledge = set(), set()
    for key, value in _walk(plan):
        if key in {"knowledge_id", "kb_id"} and value is not None:
            knowledge.add(str(value))
        if isinstance(value, str):
            columns.update(
                f"{table.lower()}.{column.lower()}"
                for table, column in re.findall(
                    r"\b([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)\b", value
                )
            )
    return tables, columns, knowledge


def calculate_hallucination_metrics(
    task_data: dict, trajectory: Any
) -> dict[str, Any]:
    """Return phase-local false-discovery metrics for one completed task."""
    calls = trajectory if isinstance(trajectory, list) else []
    expected_tables, _ = sql_schema_elements(task_data.get("sol_sql"))
    expected_columns = sql_columns(task_data.get("sol_sql"))
    expected_kb = _ids(task_data.get("external_knowledge"))

    pre_tables, pre_columns, pre_kb = _preprocessing_refs(
        _last_valid_preprocessing(calls)
    )
    plan_call = next((c for c in reversed(calls) if c.get("tool") in PLAN_TOOLS), {})
    plan_tables, plan_columns, plan_kb = _plan_refs(plan_call)
    validation_calls = [c for c in calls if c.get("tool") in VALIDATION_TOOLS]
    initial_sql = _call_sql(validation_calls[0], calls) if validation_calls else None
    submissions = [c for c in calls if c.get("tool") in SUBMISSION_TOOLS]
    final_call = submissions[-1] if submissions else (validation_calls[-1] if validation_calls else None)
    final_sql = _call_sql(final_call, calls)
    initial_tables, _ = sql_schema_elements(initial_sql)
    initial_columns = sql_columns(initial_sql)
    final_tables, _ = sql_schema_elements(final_sql)
    final_columns = sql_columns(final_sql)

    explicit_events = []
    for call in calls:
        result = str(call.get("result") or "").lower()
        if any(marker in result for marker in UNSUPPORTED_MARKERS):
            explicit_events.append(call.get("tool", "unknown"))

    phases = {
        "preprocessing": {
            "tables": _score(pre_tables, expected_tables),
            "columns": _score(pre_columns, expected_columns),
            "kb": _score(pre_kb, expected_kb),
        },
        "query_planning": {
            "tables": _score(plan_tables, expected_tables),
            "columns": _score(plan_columns, expected_columns),
            "kb": _score(plan_kb, expected_kb),
        },
        "sql_generation": {
            "tables": _score(initial_tables, expected_tables),
            "columns": _score(initial_columns, expected_columns),
        },
        "postprocessing": {
            "tables": _score(final_tables, expected_tables),
            "columns": _score(final_columns, expected_columns),
            "applicable": len(validation_calls) > 1,
        },
    }
    return {
        "definition": "false_positive_references_divided_by_all_references",
        "interpretation": "reference_divergence_proxy_not_database_nonexistence",
        "phases": phases,
        "explicit_unsupported_reference_detected": bool(explicit_events),
        "explicit_unsupported_reference_events": explicit_events,
    }
