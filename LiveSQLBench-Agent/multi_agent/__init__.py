"""Four-phase multi-agent architecture for experiment variants 4 and 5.

Milestone 1 defines agents, prompts, tool scopes, and shared artifact names.
The runtime remains disconnected until the coordinator milestone.
"""

from multi_agent.agents import AGENT_SPECS, build_agents

__all__ = ["AGENT_SPECS", "build_agents"]
