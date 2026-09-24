"""Definitions for the four specialist SQL agents."""

from dataclasses import dataclass
from typing import Any, Callable

from shared.config import settings
from shared.llm import build_adk_model
from multi_agent.prompts import PHASE_INSTRUCTIONS
from multi_agent.tools import get_phase_tool_functions, get_phase_tools

try:
    from google.adk import Agent
    from google.genai import types
    ADK_AVAILABLE = True
    ADK_IMPORT_ERROR = ""
except ImportError as exc:
    Agent = Any
    types = None
    ADK_AVAILABLE = False
    ADK_IMPORT_ERROR = str(exc)


@dataclass(frozen=True)
class PhaseAgentSpec:
    phase: str
    name: str
    description: str
    instruction: str
    tool_functions: tuple[Callable, ...]


def _spec(phase: str, description: str) -> PhaseAgentSpec:
    names = {
        "preprocessing": "sql_preprocessing_agent",
        "planning": "sql_planning_agent",
        "sql_generation": "sql_generation_agent",
        "post_processing": "sql_post_processing_agent",
    }
    return PhaseAgentSpec(
        phase=phase,
        name=names[phase],
        description=description,
        instruction=PHASE_INSTRUCTIONS[phase],
        tool_functions=get_phase_tool_functions(phase),
    )


AGENT_SPECS = {
    "preprocessing": _spec(
        "preprocessing", "Ground schema, join paths, values, and external knowledge.",
    ),
    "planning": _spec(
        "planning", "Produce and validate the structured Query or Management plan.",
    ),
    "sql_generation": _spec(
        "sql_generation", "Translate the validated plan into validated PostgreSQL.",
    ),
    "post_processing": _spec(
        "post_processing", "Execute, diagnose, minimally correct, and submit SQL.",
    ),
}


def build_phase_agent(phase: str) -> Agent:
    """Build one specialist with the same model and callbacks as Variant 1."""
    if not ADK_AVAILABLE:
        raise RuntimeError(f"google-adk runtime unavailable: {ADK_IMPORT_ERROR}")
    try:
        spec = AGENT_SPECS[phase]
    except KeyError as exc:
        raise ValueError(f"Unsupported multi-agent phase: {phase}") from exc

    from system_agent.callbacks import (
        after_tool_callback,
        before_model_callback,
        before_tool_callback,
    )

    return Agent(
        model=build_adk_model(settings.system_agent_model),
        name=spec.name,
        description=spec.description,
        instruction=spec.instruction,
        tools=get_phase_tools(phase),
        before_model_callback=before_model_callback,
        before_tool_callback=before_tool_callback,
        after_tool_callback=after_tool_callback,
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
    )


def build_agents() -> dict[str, Agent]:
    """Build all specialists. The coordinator will consume this in Milestone 3."""
    return {phase: build_phase_agent(phase) for phase in AGENT_SPECS}
