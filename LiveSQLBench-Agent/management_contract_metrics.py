"""Best-effort SQL contract metrics for Management tasks."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from sqlglot import exp

from schema_linking_metrics import _parse_all, _set_scores


def _items(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if item is not None]


def sql_management_contract(value: Any) -> dict[str, set[str]]:
    """Extract operation, target, predicate, and mutation contract features."""
    operations: set[str] = set()
    targets: set[str] = set()
    predicates: set[str] = set()
    mutations: set[str] = set()

    raw = "\n".join(_items(value))
    for keyword in re.findall(
        r"\b(UPDATE|INSERT|DELETE|MERGE|CREATE|ALTER|DROP|TRUNCATE)\b",
        raw,
        flags=re.IGNORECASE,
    ):
        operations.add(keyword.lower())

    comparison_types = (
        exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE,
        exp.In, exp.Like, exp.ILike, exp.Between, exp.Is,
    )
    for tree in _parse_all(value):
        for node in tree.walk():
            if isinstance(node, exp.Update):
                operations.add("update")
                if isinstance(node.this, exp.Table) and node.this.name:
                    targets.add(node.this.name.lower())
                for assignment in node.expressions:
                    if isinstance(assignment, exp.EQ) and isinstance(assignment.left, exp.Column):
                        mutations.add(f"set:{assignment.left.name.lower()}")
            elif isinstance(node, exp.Delete):
                operations.add("delete")
                table = node.this if isinstance(node.this, exp.Table) else None
                if table is not None and table.name:
                    targets.add(table.name.lower())
            elif isinstance(node, exp.Insert):
                operations.add("insert")
                table = node.this
                if isinstance(table, exp.Schema):
                    if isinstance(table.this, exp.Table) and table.this.name:
                        targets.add(table.this.name.lower())
                    for column in table.expressions:
                        if isinstance(column, exp.Identifier):
                            mutations.add(f"insert:{column.name.lower()}")
                elif isinstance(table, exp.Table) and table.name:
                    targets.add(table.name.lower())
            elif isinstance(node, (exp.Create, exp.Alter, exp.Drop)):
                operation = type(node).__name__.lower()
                operations.add(operation)
                target = node.this
                if isinstance(target, exp.Table) and target.name:
                    targets.add(target.name.lower())
                elif getattr(target, "name", None):
                    targets.add(str(target.name).lower())
            elif isinstance(node, exp.ColumnDef) and node.name:
                mutations.add(f"define:{node.name.lower()}")

        for where in tree.find_all(exp.Where):
            for column in where.find_all(exp.Column):
                predicates.add(f"column:{column.name.lower()}")
            for comparison in where.walk():
                if isinstance(comparison, comparison_types):
                    predicates.add(f"operator:{type(comparison).__name__.lower()}")

    # SQLGlot treats some procedural statements as Command. Preserve their
    # principal target objects using conservative keyword patterns.
    for match in re.finditer(
        r"\b(?:UPDATE|INSERT\s+INTO|DELETE\s+FROM|ALTER\s+TABLE|DROP\s+TABLE|"
        r"CREATE\s+TABLE|TRUNCATE(?:\s+TABLE)?)\s+(?:IF\s+(?:NOT\s+)?EXISTS\s+)?"
        r"(?:ONLY\s+)?(?:[\w\"]+\.)?([\w\"]+)",
        raw,
        flags=re.IGNORECASE,
    ):
        targets.add(match.group(1).strip('"').lower())

    return {
        "operation": operations,
        "target": targets,
        "predicate": predicates,
        "mutation": mutations,
    }


def add_management_contract_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    def one(row: pd.Series) -> pd.Series:
        predicted = sql_management_contract(row.get("initial_sql"))
        expected = sql_management_contract(row.get("sol_sql"))
        metrics: dict[str, Any] = {}
        for component in ("operation", "target", "predicate", "mutation"):
            metrics.update(_set_scores(
                predicted[component], expected[component], f"management_{component}"
            ))
        metrics["management_operation_exact"] = (
            predicted["operation"] == expected["operation"]
        )
        return pd.Series(metrics)

    return pd.concat([frame.copy(), frame.apply(one, axis=1)], axis=1)
