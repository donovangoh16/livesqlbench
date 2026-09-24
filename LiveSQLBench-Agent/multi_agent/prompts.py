"""Phase prompts derived from the improved single-agent instruction."""

from system_agent.prompts import IMPROVED_INSTRUCTION


PHASE_HEADERS = (
    "PHASE 1 — PRE-PROCESSING",
    "PHASE 2 — QUERY PLANNING",
    "PHASE 3 — SQL GENERATION",
    "PHASE 4 — POST-PROCESSING",
)


def _phase_section(index: int) -> str:
    start = IMPROVED_INSTRUCTION.index(PHASE_HEADERS[index])
    if index + 1 < len(PHASE_HEADERS):
        end = IMPROVED_INSTRUCTION.index(PHASE_HEADERS[index + 1])
    else:
        end = len(IMPROVED_INSTRUCTION)
    return IMPROVED_INSTRUCTION[start:end].strip()


SHARED_INSTRUCTION = (
    "You are one specialist in a four-agent PostgreSQL workflow. The model, "
    "task budget, tools, validation rules, and evaluator are shared with the "
    "improved single-agent system. Work only on your assigned phase, use stored "
    "artifacts rather than reconstructing prior work, and return a compact "
    "handoff. Do not expose private chain-of-thought."
)

PREPROCESSING_INSTRUCTION = (
    SHARED_INSTRUCTION + "\n\n" + _phase_section(0) +
    "\n\nStop after PREPROCESSING_CONTEXT is valid; hand it to the planning agent. "
    "On recovery, preserve stored valid schema and KB evidence. Perform only the "
    "validator's required action, then finalize; do not repeat broad retrieval."
)

PLANNING_INSTRUCTION = (
    SHARED_INSTRUCTION + "\n\n" + _phase_section(1) +
    "\n\nUse the stored PREPROCESSING_CONTEXT. Stop after QUERY_PLAN is valid; "
    "hand it to the SQL-generation agent."
)

GENERATION_INSTRUCTION = (
    SHARED_INSTRUCTION + "\n\n" + _phase_section(2) +
    "\n\nIn the multi-agent workflow, stop after DRAFT_SQL passes "
    "validate_sql_to_plan. Execution and submission belong to post-processing."
)

POSTPROCESSING_INSTRUCTION = (
    SHARED_INSTRUCTION + "\n\n" + _phase_section(3) +
    "\n\nUse the stored validated QUERY_PLAN and DRAFT_SQL. Finish with the "
    "single permitted submission. If the incoming DRAFT_SQL is already validated "
    "and unchanged, execute or submit it directly. Re-run validate_sql_to_plan only "
    "after making a correction."
)

PHASE_INSTRUCTIONS = {
    "preprocessing": PREPROCESSING_INSTRUCTION,
    "planning": PLANNING_INSTRUCTION,
    "sql_generation": GENERATION_INSTRUCTION,
    "post_processing": POSTPROCESSING_INSTRUCTION,
}
