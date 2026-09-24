"""Phase-specific views over the existing improved tool implementations."""

from google.adk.tools import FunctionTool

from system_agent.tools import (
    diagnose_last_execution_error,
    execute_validated_sql,
    finalize_preprocessing_context,
    generate_and_validate_query_plan,
    inspect_database,
    prepare_knowledge_context,
    prepare_schema_context,
    submit_validated_sql,
    validate_sql_to_plan,
)


PHASE_TOOL_FUNCTIONS = {
    "preprocessing": (
        prepare_schema_context,
        prepare_knowledge_context,
        inspect_database,
        finalize_preprocessing_context,
    ),
    "planning": (generate_and_validate_query_plan,),
    "sql_generation": (validate_sql_to_plan,),
    "post_processing": (
        diagnose_last_execution_error,
        validate_sql_to_plan,
        execute_validated_sql,
        submit_validated_sql,
    ),
}


def get_phase_tool_functions(phase: str) -> tuple:
    try:
        return PHASE_TOOL_FUNCTIONS[phase]
    except KeyError as exc:
        raise ValueError(f"Unsupported multi-agent phase: {phase}") from exc


def get_phase_tools(phase: str) -> list[FunctionTool]:
    """Wrap the shared functions for ADK without copying their behavior."""
    return [FunctionTool(function) for function in get_phase_tool_functions(phase)]
