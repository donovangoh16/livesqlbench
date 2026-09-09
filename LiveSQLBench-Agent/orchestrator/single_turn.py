"""LiveSQLBench — Single-turn text-to-SQL agent pipeline.

The orchestrator:
1. Initializes the DB environment service
2. Initializes an agent session on the system-agent service
3. Sends the user query once
4. Reads the final session state for metrics
"""

import logging
import time
import traceback
from typing import Any, Dict

import httpx

from shared.config import settings
from experiment.variants import VariantConfig, get_variant

logger = logging.getLogger(__name__)

SYSTEM_AGENT_URL = f"http://localhost:{settings.system_agent_port}"
DB_ENV_URL = f"http://localhost:{settings.db_env_port}"

MAX_STEPS = 30


async def _post(url: str, payload: dict, timeout: float = 120.0) -> dict:
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


async def init_task_on_services(task_id: str, task_data: dict):
    payload = {
        "task_id": task_id,
        "task_data": {**task_data, "_interact_mode": "single-turn"},
    }
    await _post(f"{DB_ENV_URL}/init_task", payload)
    logger.info("  [%s] DB environment initialized", task_id)


async def init_agent_session(task_id: str, task_data: dict, variant: VariantConfig):
    state = {
        "task_id": task_id,
        "db_name": task_data["selected_database"],
        "user_query": task_data["query"],
        "task_category": task_data.get("category"),
        "steps_remaining": MAX_STEPS,
        "max_steps": MAX_STEPS,
        "total_reward": 0.0,
        "tool_trajectory": [],
        "adk_events": [],
        "phase1_completed": False,
        "task_done": False,
        "experiment_variant": variant.number,
        "requested_agent_profile": variant.requested_agent_profile,
        "requested_harness_profile": variant.requested_harness_profile,
        "active_agent_profile": variant.active_agent_profile,
        "active_harness_profile": variant.active_harness_profile,
    }
    return await _post(
        f"{SYSTEM_AGENT_URL}/init_session",
        {"task_id": task_id, "state": state, "reset": True},
        timeout=30.0,
    )


async def run_agent_session(task_id: str, message: str):
    return await _post(
        f"{SYSTEM_AGENT_URL}/run_session",
        {"task_id": task_id, "message": message},
        timeout=1800.0,
    )


async def cleanup_task_service(task_id: str):
    try:
        await _post(f"{DB_ENV_URL}/cleanup_task", {"task_id": task_id}, timeout=30.0)
    except Exception as e:
        logger.warning("Cleanup failed for %s: %s", task_id, e)


async def run_single_task(
    task_data: dict, variant_number: int | None = None
) -> Dict[str, Any]:
    variant = get_variant(
        settings.experiment_variant if variant_number is None else variant_number
    )
    instance_id = task_data["instance_id"]
    db_name = task_data["selected_database"]
    logger.info("Starting task: %s (db: %s)", instance_id, db_name)
    start_time = time.time()

    await init_task_on_services(instance_id, task_data)

    try:
        await init_agent_session(instance_id, task_data, variant)

        initial_message = (
            f"Database: {db_name}\n"
            f"Task ID: {instance_id}\n\n"
            f"User Query:\n{task_data['query']}\n\n"
            f"You have {MAX_STEPS} steps. Each tool call costs 1 step.\n"
            f"You have ONE submission attempt — make it count."
        )

        run_result = await run_agent_session(instance_id, initial_message)
        state = run_result.get("state", {})
        from system_agent.harness import finalize_harness_metrics
        finalize_harness_metrics(state)
        elapsed = time.time() - start_time

        steps_used = MAX_STEPS - max(0, state.get("steps_remaining", MAX_STEPS))
        token_usage = state.get("token_usage", {})
        result = {
            "task_id": instance_id,
            "instance_id": instance_id,
            "database": db_name,
            "phase1_passed": state.get("phase1_completed", False),
            "total_reward": state.get("total_reward", 0.0),
            "elapsed_seconds": elapsed,
            "steps_used": steps_used,
            "steps_remaining": max(0, state.get("steps_remaining", MAX_STEPS)),
            "tool_trajectory": state.get("tool_trajectory", []),
            "adk_events": state.get("adk_events", []),
            "token_usage": token_usage,
            "input_tokens": token_usage.get("input_tokens", 0),
            "output_tokens": token_usage.get("output_tokens", 0),
            "total_tokens": token_usage.get("total_tokens", 0),
            "cached_input_tokens": token_usage.get("cached_input_tokens", 0),
            "final_response": run_result.get("response", ""),
            "experiment": variant.as_dict(),
        }
        if isinstance(state.get("harness_metrics"), dict):
            result["harness_metrics"] = state["harness_metrics"]
        logger.info(
            "Task %s done. Reward: %.2f, Steps used: %d, Time: %.1fs",
            instance_id,
            result["total_reward"],
            steps_used,
            elapsed,
        )
        return result
    finally:
        await cleanup_task_service(instance_id)


# i want to create a database-disjoint split to test whether the agent generalises to unseen databases.

# i should also add a output that is in pandas format to help in generating plots and analysis.
