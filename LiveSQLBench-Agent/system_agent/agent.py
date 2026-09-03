"""LiveSQLBench ADK System Agent — Single-turn text-to-SQL."""

import logging
from typing import Any

from shared.config import settings

try:
    from google.adk import Agent
    from google.adk.tools import FunctionTool
    from google.genai import types
    ADK_AVAILABLE = True
    ADK_IMPORT_ERROR = ""
except ImportError as exc:
    Agent = Any
    FunctionTool = None
    types = None
    ADK_AVAILABLE = False
    ADK_IMPORT_ERROR = str(exc)

logger = logging.getLogger(__name__)

from shared.llm import build_adk_model as _build_model
from system_agent.prompts import BASELINE_INSTRUCTION, get_instruction

# Backwards-compatible name for code that imports the original prompt.
INSTRUCTION = BASELINE_INSTRUCTION


def build_agent(profile: str = "baseline") -> Agent:
    """Build the single-turn text-to-SQL agent."""
    if not ADK_AVAILABLE:
        raise RuntimeError(f"google-adk runtime unavailable: {ADK_IMPORT_ERROR}")

    from system_agent.tools import get_tools
    from system_agent.callbacks import (
        before_model_callback, before_tool_callback, after_tool_callback,
    )

    model = _build_model(settings.system_agent_model)
    return Agent(
        model=model,
        name="bird_interact_agent",
        description="Single-turn text-to-SQL agent.",
        instruction=get_instruction(profile),
        tools=get_tools(profile),
        before_model_callback=before_model_callback,
        before_tool_callback=before_tool_callback,
        after_tool_callback=after_tool_callback,
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
    )
