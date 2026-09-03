"""Explainable SQL-structure difficulty classification for EDA."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


DIFFICULTY_ORDER = ["easy", "non_nested_complex", "nested_complex"]


def _sql_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if item is not None and str(item).strip()]


def management_sql_features(value: Any) -> dict[str, Any]:
    """Extract PostgreSQL management features not reliably exposed by SQLGlot."""
    sql_text = "\n".join(_sql_items(value))
    uncommented = re.sub(r"/\*.*?\*/|--[^\n]*", " ", sql_text, flags=re.DOTALL)
    upper = uncommented.upper()
    procedural_patterns = [
        r"\bCREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\b",
        r"\bCREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\b",
        r"\bCREATE\s+(?:OR\s+REPLACE\s+)?TRIGGER\b",
        r"\bDO\s+\$\$",
        r"\bLANGUAGE\s+PLPGSQL\b",
        r"\bDECLARE\b",
        r"\bBEGIN\b.*\bEND\b",
    ]
    target_pattern = re.compile(
        r"\b(?:ALTER\s+TABLE|UPDATE|INSERT\s+INTO|DELETE\s+FROM|"
        r"TRUNCATE(?:\s+TABLE)?|CREATE\s+(?:OR\s+REPLACE\s+)?"
        r"(?:TABLE|VIEW|MATERIALIZED\s+VIEW|FUNCTION|PROCEDURE|TRIGGER)|"
        r"DROP\s+(?:TABLE|VIEW|MATERIALIZED\s+VIEW|FUNCTION|PROCEDURE|TRIGGER))"
        r"\s+(?:IF\s+(?:NOT\s+)?EXISTS\s+)?([\w.\"]+)",
        re.IGNORECASE,
    )
    targets = sorted({match.strip('"').lower() for match in target_pattern.findall(sql_text)})
    semantic_statements = [
        statement.strip()
        for statement in uncommented.split(";")
        if statement.strip()
        and not re.match(r"^\s*(COMMENT|GRANT)\b", statement, re.IGNORECASE)
    ]
    return {
        "management_has_procedural_sql": any(
            re.search(pattern, upper, re.DOTALL) for pattern in procedural_patterns
        ),
        "management_modified_objects": targets,
        "management_modified_object_count": len(targets),
        "management_semantic_statement_count": len(semantic_statements),
    }


def _classify_row(row: pd.Series) -> tuple[str | None, str]:
    nested_query = bool(row["has_cte"] or row["has_subquery"] or row["has_set_operation"])

    if row["category"] == "Query":
        if nested_query:
            return "nested_complex", "query contains CTE, subquery, or set operation"
        if row["has_join"] or row["unique_table_count"] > 1:
            return "non_nested_complex", "query joins or references multiple tables without nesting"
        return "easy", "single-table query without join or nesting"

    if row["category"] == "Management":
        multi_object_dependency = (
            row["management_modified_object_count"] > 1
            and row["management_semantic_statement_count"] > 1
        )
        if row["management_has_procedural_sql"] or multi_object_dependency:
            return "nested_complex", "procedural or dependent multi-object management"
        relational_dependency = bool(
            nested_query
            or row["has_join"]
            or row["has_aggregate"]
            or row["has_window"]
            or row["unique_table_count"] > 1
        )
        if relational_dependency:
            return "non_nested_complex", "management operation has a relational dependency"
        return "easy", "single-target management without relational or procedural dependency"

    return None, "unsupported task category"


def add_structural_difficulty(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with management features and auditable difficulty labels."""
    classified = analysis_df.copy()
    management_features = classified["sol_sql"].apply(management_sql_features).apply(pd.Series)
    for column in management_features:
        classified[column] = management_features[column]

    labels = classified.apply(_classify_row, axis=1, result_type="expand")
    labels.columns = ["structural_difficulty", "difficulty_reason"]
    classified[labels.columns] = labels
    classified["structural_difficulty"] = pd.Categorical(
        classified["structural_difficulty"], categories=DIFFICULTY_ORDER, ordered=True
    )
    return classified


def success_summaries(analysis_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate success by task type and by task type plus difficulty."""
    by_type = (
        analysis_df.groupby("category", observed=True)["passed_int"]
        .agg(tasks="size", passed="sum", success_rate="mean")
        .reset_index()
    )
    by_difficulty = (
        analysis_df.groupby(
            ["category", "structural_difficulty"], observed=False
        )["passed_int"]
        .agg(tasks="size", passed="sum", success_rate="mean")
        .reset_index()
    )
    for summary in (by_type, by_difficulty):
        summary["passed"] = summary["passed"].astype(int)
        summary["failed"] = summary["tasks"] - summary["passed"]
    return by_type, by_difficulty
