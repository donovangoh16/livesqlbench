#!/usr/bin/env python3
"""Create the selected database-disjoint Lite train/test split."""

import json
from pathlib import Path


SOURCE = Path("livesqlbench-base-lite/livesqlbench_data_with_gt.jsonl")
OUTPUT_DIR = Path("splits")
TRAIN_DATABASES = {"credit", "cybermarket", "gaming", "solar", "museum"}
TEST_DATABASES = {"cross_db", "archeology"}


def load_tasks(path: Path) -> list[dict]:
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def write_tasks(path: Path, tasks: list[dict]) -> None:
    with path.open("w") as file:
        for task in tasks:
            file.write(json.dumps(task) + "\n")


def main() -> None:
    tasks = load_tasks(SOURCE)
    train_tasks = [task for task in tasks if task["selected_database"] in TRAIN_DATABASES]
    test_tasks = [task for task in tasks if task["selected_database"] in TEST_DATABASES]
    selected = TRAIN_DATABASES | TEST_DATABASES
    unexpected = {task["selected_database"] for task in tasks} - selected

    OUTPUT_DIR.mkdir(exist_ok=True)
    write_tasks(OUTPUT_DIR / "lite_7db_train.jsonl", train_tasks)
    write_tasks(OUTPUT_DIR / "lite_7db_test.jsonl", test_tasks)
    for database in TEST_DATABASES:
        write_tasks(
            OUTPUT_DIR / f"lite_{database}_test.jsonl",
            [task for task in test_tasks if task["selected_database"] == database],
        )
    (OUTPUT_DIR / "lite_7db_split.json").write_text(json.dumps({
        "source": str(SOURCE),
        "train_databases": sorted(TRAIN_DATABASES),
        "test_databases": sorted(TEST_DATABASES),
        "train_tasks": len(train_tasks),
        "test_tasks": len(test_tasks),
        "excluded_databases": sorted(unexpected),
    }, indent=2) + "\n")

    assert len(train_tasks) == 75, f"Expected 75 train tasks, got {len(train_tasks)}"
    assert len(test_tasks) == 30, f"Expected 30 test tasks, got {len(test_tasks)}"
    print("Created 75 train tasks and 30 test tasks in splits/.")


if __name__ == "__main__":
    main()
