"""Parallel evaluation runner for single-turn text-to-SQL."""

import asyncio
import argparse
import json
import logging
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any, Callable, Awaitable, Dict, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import settings
from experiment.variants import get_variant

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def run_parallel_evaluation(
    tasks: List[dict],
    run_single_task: Callable[[dict], Awaitable[Dict[str, Any]]],
    output_path: str,
    concurrency: int = 5,
    experiment: dict = None,
):
    semaphore = asyncio.Semaphore(concurrency)
    results: List[Dict[str, Any]] = []
    results_lock = asyncio.Lock()
    total_reward = 0.0
    p1_count = 0
    completed = 0
    token_totals = {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "cached_input_tokens": 0,
    }
    harness_totals = {
        "tasks_observed": 0,
        "calls_observed": 0,
        "duplicate_calls_detected": 0,
        "no_progress_events_detected": 0,
        "blocked_calls": 0,
        "block_reasons": {},
        "max_no_progress_streak": 0,
        "event_counts": {},
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    async def _save():
        n = len(results)
        if n == 0:
            return
        output = {
            "mode": "single-turn",
            "experiment": experiment or {},
            "metrics": {
                "total_tasks": n,
                "total_reward": total_reward,
                "average_reward": total_reward / n,
                "phase1_rate": p1_count / n,
                "phase1_count": p1_count,
                **token_totals,
                "average_tokens_per_task": token_totals["total_tokens"] / n,
                "tokens_per_success": (
                    token_totals["total_tokens"] / p1_count if p1_count else None
                ),
                "prompt_cache_hit_rate": (
                    token_totals["cached_input_tokens"] / token_totals["input_tokens"]
                    if token_totals["input_tokens"] else 0.0
                ),
                **(
                    {"harness_observation": harness_totals}
                    if harness_totals["tasks_observed"] else {}
                ),
            },
            "results": results,
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

    async def _run_one(i: int, td: dict):
        nonlocal total_reward, p1_count, completed
        instance_id = td["instance_id"]
        async with semaphore:
            logger.info("=== Task %d/%d: %s ===", i + 1, len(tasks), instance_id)
            try:
                r = await run_single_task(td)
            except Exception as e:
                logger.error("Error: %s: %s", instance_id, e)
                traceback.print_exc()
                r = {"task_id": instance_id, "error": str(e), "total_reward": 0}

        async with results_lock:
            results.append(r)
            total_reward += r.get("total_reward", 0)
            if r.get("phase1_passed"):
                p1_count += 1
            for key in token_totals:
                token_totals[key] += int(r.get(key, 0) or 0)
            harness = r.get("harness_metrics")
            if isinstance(harness, dict):
                harness_totals["tasks_observed"] += 1
                for key in (
                    "calls_observed", "duplicate_calls_detected",
                    "no_progress_events_detected", "blocked_calls",
                ):
                    harness_totals[key] += int(harness.get(key, 0) or 0)
                harness_totals["max_no_progress_streak"] = max(
                    harness_totals["max_no_progress_streak"],
                    int(harness.get("max_no_progress_streak", 0) or 0),
                )
                for event, count in harness.get("event_counts", {}).items():
                    harness_totals["event_counts"][event] = (
                        harness_totals["event_counts"].get(event, 0) + int(count or 0)
                    )
                for reason, count in harness.get("block_reasons", {}).items():
                    harness_totals["block_reasons"][reason] = (
                        harness_totals["block_reasons"].get(reason, 0) + int(count or 0)
                    )
            completed += 1
            if completed % 5 == 0 or completed == len(tasks):
                await _save()

    await asyncio.gather(*[_run_one(i, td) for i, td in enumerate(tasks)])
    await _save()

    n = len(tasks)
    if n:
        logger.info(
            "\nDone! Tasks: %d, Avg Reward: %.4f, Pass: %d/%d (%.1f%%)",
            n, total_reward / n, p1_count, n, p1_count / n * 100,
        )


def load_tasks(
    data_path: str,
    limit: int = None,
    category: str = None,
    database: str = None,
) -> List[dict]:
    tasks = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))

    # Ground-truth/KG files are evaluation overlays: they may contain only the
    # instance id, solution SQL, KB ids, and tests. Enrich them from the base
    # task record while preserving the overlay's evaluation fields.
    if any("selected_database" not in task or "query" not in task for task in tasks):
        project_root = Path(__file__).resolve().parent.parent
        candidates = [
            Path(settings.data_path),
            project_root / "livesqlbench-base-lite" / "livesqlbench_data.jsonl",
            project_root / "livesqlbench-base-full" / "livesqlbench_data.jsonl",
        ]
        needed = {task.get("instance_id") for task in tasks}
        base_records = {}
        for candidate in dict.fromkeys(candidates):
            if not candidate.exists() or candidate.resolve() == Path(data_path).resolve():
                continue
            with candidate.open() as base_file:
                for line in base_file:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    instance_id = record.get("instance_id")
                    if instance_id in needed and instance_id not in base_records:
                        base_records[instance_id] = record
        tasks = [
            {**base_records.get(task.get("instance_id"), {}), **task}
            for task in tasks
        ]

        missing = [
            task.get("instance_id") for task in tasks
            if "selected_database" not in task or "query" not in task
        ]
        if missing:
            raise ValueError(
                "Could not enrich ground-truth records from a base dataset: "
                + ", ".join(str(value) for value in missing[:10])
            )
    if category:
        tasks = [
            task for task in tasks
            if str(task.get("category", "")).lower() == category.lower()
        ]
    if database:
        tasks = [
            task for task in tasks
            if str(task.get("selected_database", "")).lower() == database.lower()
        ]
    if limit:
        tasks = tasks[:limit]
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Single-turn text-to-SQL evaluation")
    parser.add_argument("--data", default=settings.data_path)
    parser.add_argument("--output", default="results/eval_single_turn.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument(
        "--category", choices=("Query", "Management"), default=None,
        help="Run only Query or Management tasks; applied before --limit",
    )
    parser.add_argument(
        "--database", default=None,
        help="Run only tasks from this database; applied before --limit",
    )
    parser.add_argument(
        "--variant", type=int, choices=range(4),
        default=settings.experiment_variant,
        help="Experiment variant: 0=baseline, 1=agent, 2=harness, 3=combined",
    )
    args = parser.parse_args()

    from orchestrator.single_turn import run_single_task

    variant = get_variant(args.variant)
    tasks = load_tasks(args.data, args.limit, args.category, args.database)
    logger.info(
        "Single-turn variant %d: requested=(%s, %s), active=(%s, %s)",
        variant.number,
        variant.requested_agent_profile,
        variant.requested_harness_profile,
        variant.active_agent_profile,
        variant.active_harness_profile,
    )
    logger.info("Evaluating %d tasks with concurrency=%d", len(tasks), args.concurrency)

    asyncio.run(run_parallel_evaluation(
        tasks=tasks,
        run_single_task=partial(run_single_task, variant_number=variant.number),
        output_path=args.output,
        concurrency=args.concurrency,
        experiment=variant.as_dict(),
    ))


if __name__ == "__main__":
    main()
