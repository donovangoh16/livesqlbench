"""Create the matched 20-task cross-database regression subset."""

from __future__ import annotations

import json
from pathlib import Path


TASK_IDS = [
    "gaming_10", "gaming_M_2",
    "credit_3", "credit_7", "credit_9", "credit_M_2", "credit_M_3",
    "cybermarket_6", "cybermarket_8", "cybermarket_10",
    "cybermarket_M_1", "cybermarket_M_2",
    "museum_1", "museum_10",
    "solar_6", "solar_M_1", "solar_M_3", "solar_M_5",
    "gaming_M_1", "gaming_M_4",
]

SOURCE = Path("splits/lite_7db_train.jsonl")
OUTPUT = Path("splits/regression_20.jsonl")


def main() -> None:
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    by_id = {row["instance_id"]: row for row in rows}
    missing = [task_id for task_id in TASK_IDS if task_id not in by_id]
    if missing:
        raise ValueError(f"Missing task IDs: {missing}")
    selected = [by_id[task_id] for task_id in TASK_IDS]
    OUTPUT.write_text("".join(json.dumps(row) + "\n" for row in selected))
    if len(selected) != 20 or len({row["instance_id"] for row in selected}) != 20:
        raise ValueError("Regression subset must contain 20 unique tasks")
    print(f"Created {OUTPUT} with {len(selected)} tasks")


if __name__ == "__main__":
    main()
