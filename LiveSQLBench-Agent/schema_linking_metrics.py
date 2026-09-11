"""Schema-linking metrics derived from agent trajectories and reference SQL."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

import pandas as pd
import sqlglot
from sqlglot import exp


def _items(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if item is not None and str(item).strip()]


def final_attempted_sql(trajectory: Any) -> str | None:
    """Use final submitted SQL, or the last executed SQL when no submission exists."""
    calls = trajectory if isinstance(trajectory, list) else []
    for tool_name in (
        "submit_validated_sql", "submit_sql",
        "execute_validated_sql", "execute_sql",
    ):
        for call in reversed(calls):
            if call.get("tool") == tool_name:
                sql = (call.get("args") or {}).get("sql")
                if sql:
                    return str(sql)
    return None


def _parse_all(value: Any) -> list[exp.Expression]:
    trees = []
    for sql in _items(value):
        try:
            trees.extend(
                tree for tree in sqlglot.parse(sql, dialect="postgres") if tree is not None
            )
        except Exception:
            continue
    return trees


def sql_schema_elements(value: Any) -> tuple[set[str], set[tuple[str, str]]]:
    """Return physical tables and qualified equi-join column edges.

    CTE aliases are excluded from the physical table set. Join edges are
    canonical unordered pairs such as
    ("orders.customer_id", "customers.customer_id").
    """
    trees = _parse_all(value)
    cte_names = {
        cte.alias_or_name.lower()
        for tree in trees
        for cte in tree.find_all(exp.CTE)
        if cte.alias_or_name
    }
    alias_map: dict[str, str] = {}
    tables: set[str] = set()
    for tree in trees:
        for table in tree.find_all(exp.Table):
            name = table.name.lower() if table.name else ""
            if not name or name in cte_names:
                continue
            tables.add(name)
            alias_map[name] = name
            if table.alias_or_name:
                alias_map[table.alias_or_name.lower()] = name

    join_edges: set[tuple[str, str]] = set()
    for tree in trees:
        for join in tree.find_all(exp.Join):
            on_expression = join.args.get("on")
            if on_expression is None:
                continue
            for equality in on_expression.find_all(exp.EQ):
                left, right = equality.left, equality.right
                if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                    continue
                if not left.table or not right.table:
                    continue
                left_table = alias_map.get(left.table.lower(), left.table.lower())
                right_table = alias_map.get(right.table.lower(), right.table.lower())
                if left_table in cte_names or right_table in cte_names:
                    continue
                endpoints = sorted(
                    [f"{left_table}.{left.name.lower()}", f"{right_table}.{right.name.lower()}"]
                )
                join_edges.add((endpoints[0], endpoints[1]))
    return tables, join_edges


def sql_columns(value: Any) -> set[str]:
    """Return canonical column references from SQL.

    Qualified columns are resolved from aliases to physical table names. CTE
    output references are excluded because their underlying physical columns
    are collected inside the CTE. Unqualified references use ``*.column`` so
    equivalent unqualified references remain comparable.
    """
    columns: set[str] = set()
    for tree in _parse_all(value):
        cte_names = {
            cte.alias_or_name.lower()
            for cte in tree.find_all(exp.CTE)
            if cte.alias_or_name
        }
        alias_map: dict[str, str] = {}
        for table in tree.find_all(exp.Table):
            name = table.name.lower() if table.name else ""
            if not name or name in cte_names:
                continue
            alias_map[name] = name
            if table.alias_or_name:
                alias_map[table.alias_or_name.lower()] = name
        for column in tree.find_all(exp.Column):
            if not column.name:
                continue
            if not column.table:
                columns.add(f"*.{column.name.lower()}")
                continue
            qualifier = column.table.lower()
            if qualifier in cte_names:
                continue
            columns.add(f"{alias_map.get(qualifier, qualifier)}.{column.name.lower()}")
    return columns


def _response_id(response: Any) -> int | str | None:
    if isinstance(response, dict):
        payload = response
    else:
        text = str(response or "").strip()
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            match = re.search(r'"?id"?\s*:\s*"?(\d+)', text, re.IGNORECASE)
            return int(match.group(1)) if match else None
    if isinstance(payload, dict) and "id" in payload:
        try:
            return int(payload["id"])
        except (TypeError, ValueError):
            return str(payload["id"])
    if isinstance(payload, dict) and "result" in payload:
        return _response_id(payload["result"])
    return None


def selected_kb_ids(trajectory: Any) -> set[int | str]:
    selected = set()
    for call in trajectory if isinstance(trajectory, list) else []:
        if call.get("tool") == "get_knowledge_definition":
            knowledge_id = _response_id(call.get("result"))
            if knowledge_id is not None:
                selected.add(knowledge_id)
        elif call.get("tool") == "prepare_knowledge_context":
            # Variants 1 and 3 return the retrieved definitions together in one
            # compact response instead of calling get_knowledge_definition once
            # per entry. Count only definitions actually returned under
            # ``knowledge``; suggested or merely required IDs are not retrievals.
            payload = call.get("result")
            if not isinstance(payload, dict):
                try:
                    payload = json.loads(str(payload or ""))
                except (json.JSONDecodeError, TypeError):
                    payload = {}
            if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
                payload = payload["result"]
            for item in payload.get("knowledge", []) if isinstance(payload, dict) else []:
                knowledge_id = _response_id(item)
                if knowledge_id is not None:
                    selected.add(knowledge_id)
    return selected


def _normalise_ids(value: Any) -> set[int | str]:
    values: Iterable[Any] = value if isinstance(value, (list, tuple, set)) else []
    result = set()
    for item in values:
        try:
            result.add(int(item))
        except (TypeError, ValueError):
            result.add(str(item))
    return result


def _set_scores(predicted: set, expected: set, prefix: str) -> dict[str, Any]:
    correct = predicted & expected
    precision = len(correct) / len(predicted) if predicted else float("nan")
    recall = len(correct) / len(expected) if expected else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if predicted and expected and precision + recall
        else float("nan")
    )
    return {
        f"{prefix}_selected_count": len(predicted),
        f"{prefix}_ground_truth_count": len(expected),
        f"{prefix}_correct_count": len(correct),
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        f"{prefix}_exact_match": predicted == expected,
        f"{prefix}_false_positive": sorted(predicted - expected),
        f"{prefix}_false_negative": sorted(expected - predicted),
    }


def task_schema_linking_metrics(row: pd.Series) -> pd.Series:
    predicted_sql = final_attempted_sql(row.get("tool_trajectory"))
    predicted_tables, predicted_joins = sql_schema_elements(predicted_sql)
    expected_tables, expected_joins = sql_schema_elements(row.get("sol_sql"))
    predicted_kb = selected_kb_ids(row.get("tool_trajectory"))
    expected_kb = _normalise_ids(row.get("external_knowledge"))

    metrics = {
        "schema_linking_sql": predicted_sql,
        "selected_tables": sorted(predicted_tables),
        "ground_truth_tables": sorted(expected_tables),
        "selected_join_edges": sorted(predicted_joins),
        "ground_truth_join_edges": sorted(expected_joins),
        "selected_kb_ids_unique": sorted(predicted_kb, key=str),
        "ground_truth_kb_ids": sorted(expected_kb, key=str),
    }
    metrics.update(_set_scores(predicted_tables, expected_tables, "table"))
    metrics.update(_set_scores(predicted_joins, expected_joins, "join_path"))
    metrics.update(_set_scores(predicted_kb, expected_kb, "kb"))
    metrics["join_path_evaluable"] = bool(expected_joins)
    metrics["join_path_correct"] = predicted_joins == expected_joins
    return pd.Series(metrics)


def add_schema_linking_metrics(analysis_df: pd.DataFrame) -> pd.DataFrame:
    metrics = analysis_df.apply(task_schema_linking_metrics, axis=1)
    return pd.concat([analysis_df.copy(), metrics], axis=1)


def micro_scores(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    selected = int(frame[f"{prefix}_selected_count"].sum())
    expected = int(frame[f"{prefix}_ground_truth_count"].sum())
    correct = int(frame[f"{prefix}_correct_count"].sum())
    precision = correct / selected if selected else float("nan")
    recall = correct / expected if expected else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if selected and expected and precision + recall
        else float("nan")
    )
    return {"precision": precision, "recall": recall, "f1": f1}
