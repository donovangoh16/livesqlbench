"""Reusable loader for the task-aware Variant 0/1/3 EDA."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from management_contract_metrics import add_management_contract_metrics
from schema_linking_metrics import micro_scores
from trajectory_phase_metrics import add_phase_metrics
from schema_linking_metrics import _parse_all
from sql_difficulty import management_sql_features
from sqlglot import exp


RUN_FILES = {
    "Variant 0": Path("results/eval_7db_train.json"),
    "Variant 1": Path("results/eval_variant_1_milestone_3b_agent_full.json"),
    "Variant 3": Path("results/eval_variant_3_train_full_after_grounding_fix.json"),
}
DATA_FILE = Path("splits/lite_7db_train.jsonl")


def _safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else np.nan


def _difficulty(row: pd.Series) -> str:
    trees = _parse_all(row.get("sol_sql"))
    has_nested = any(
        any(tree.find(kind) is not None for kind in (exp.CTE, exp.Subquery, exp.Union, exp.Intersect, exp.Except))
        for tree in trees
    )
    has_join = any(tree.find(exp.Join) is not None for tree in trees)
    has_relational = has_join or has_nested or any(
        tree.find(exp.AggFunc) is not None or tree.find(exp.Window) is not None
        for tree in trees
    )
    tables = {table.name.lower() for tree in trees for table in tree.find_all(exp.Table) if table.name}
    if row.get("category") == "Query":
        if has_nested:
            return "nested_complex"
        return "non_nested_complex" if has_join or len(tables) > 1 else "easy"
    management = management_sql_features(row.get("sol_sql"))
    if management["management_has_procedural_sql"] or (
        management["management_modified_object_count"] > 1
        and management["management_semantic_statement_count"] > 1
    ):
        return "nested_complex"
    return "non_nested_complex" if has_relational or len(tables) > 1 else "easy"


def load_variant_phase_data() -> pd.DataFrame:
    metadata = pd.read_json(DATA_FILE, lines=True)[[
        "instance_id", "selected_database", "category", "difficulty_tier",
        "sol_sql", "external_knowledge",
    ]]
    frames = []
    for label, path in RUN_FILES.items():
        run = pd.DataFrame(json.loads(path.read_text())["results"])
        run = run.drop(columns=[
            column for column in metadata.columns
            if column != "instance_id" and column in run.columns
        ])
        run = run.merge(metadata, on="instance_id", how="left", validate="one_to_one")
        run["variant"] = label
        run["phase1_passed"] = run["phase1_passed"].astype(bool)
        frames.append(add_phase_metrics(run))
    result = pd.concat(frames, ignore_index=True)
    result["structural_difficulty"] = result.apply(_difficulty, axis=1)
    counts = result.groupby("variant")["instance_id"].nunique()
    if not counts.eq(75).all():
        raise ValueError(f"Expected 75 matched tasks per variant, got {counts.to_dict()}")
    return result


def summarise_task_aware(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, subset in frame.groupby("variant", sort=False):
        query = subset[subset["category"].eq("Query")]
        management = subset[subset["category"].eq("Management")]
        query_errors = query[query["postprocessing_eligible"]]
        management_validated = management[management["validation_present"]]
        management_rejected = management_validated[
            management_validated["initial_validation_rejected"]
        ]
        join_subset = subset[subset["initial_join_path_ground_truth_count"].gt(0)]
        rows.append({
            "variant": variant,
            "overall_accuracy": subset["final_passed"].mean(),
            "pre_table_f1": micro_scores(subset, "initial_table")["f1"],
            "pre_column_f1": micro_scores(subset, "initial_column")["f1"],
            "pre_join_f1": micro_scores(join_subset, "initial_join_path")["f1"],
            "pre_kb_f1": micro_scores(subset, "phase_kb")["f1"],
            "query_plan_structure_f1": micro_scores(query, "initial_plan_structure")["f1"],
            "query_plan_structure_exact": query["plan_structure_exact_match"].mean(),
            "management_operation_exact": management["management_operation_exact"].mean(),
            "management_target_f1": micro_scores(management, "management_target")["f1"],
            "management_predicate_f1": micro_scores(management, "management_predicate")["f1"],
            "management_mutation_f1": micro_scores(management, "management_mutation")["f1"],
            "query_initial_execution_success": query["initial_sql_execution_success"].mean(),
            "query_retry_rate_after_error": _safe_mean(query_errors["postprocessing_attempted"]),
            "query_error_resolution_rate": _safe_mean(query_errors["initial_execution_error_resolved"]),
            "management_validation_rejection_rate": _safe_mean(
                management_validated["initial_validation_rejected"]
            ),
            "management_revision_rate_after_rejection": _safe_mean(
                management_rejected["validation_revision_attempted"]
            ),
            "management_validation_resolution_rate": _safe_mean(
                management_rejected["validation_rejection_resolved"]
            ),
            "query_initial_errors": len(query_errors),
            "management_initial_rejections": len(management_rejected),
        })
    return pd.DataFrame(rows)


def build_task_aware_evaluation() -> tuple[pd.DataFrame, pd.DataFrame]:
    phase_data = load_variant_phase_data()
    task_data = add_management_contract_metrics(phase_data)
    return task_data, summarise_task_aware(task_data)
