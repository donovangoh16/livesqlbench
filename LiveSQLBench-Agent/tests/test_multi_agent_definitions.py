import unittest

from multi_agent.agents import AGENT_SPECS
from multi_agent.prompts import PHASE_HEADERS, PHASE_INSTRUCTIONS
from multi_agent.state import ARTIFACT_KEYS, PHASES, initial_multi_agent_state
from multi_agent.tools import PHASE_TOOL_FUNCTIONS, get_phase_tool_functions
from system_agent.tools import get_tools


class MultiAgentDefinitionTests(unittest.TestCase):
    def test_exactly_four_phase_agents_are_defined(self):
        self.assertEqual(tuple(AGENT_SPECS), PHASES)
        self.assertEqual(tuple(PHASE_INSTRUCTIONS), PHASES)
        self.assertEqual(tuple(PHASE_TOOL_FUNCTIONS), PHASES)

    def test_each_prompt_contains_its_original_phase_contract(self):
        for phase, header in zip(PHASES, PHASE_HEADERS):
            self.assertIn(header, PHASE_INSTRUCTIONS[phase])

    def test_phase_tools_are_the_existing_function_objects(self):
        improved_by_name = {
            tool.name: tool.func for tool in get_tools("improved")
        }
        for functions in PHASE_TOOL_FUNCTIONS.values():
            for function in functions:
                self.assertIs(improved_by_name[function.__name__], function)

    def test_combined_tool_capability_matches_improved_agent(self):
        combined = {
            function.__name__
            for functions in PHASE_TOOL_FUNCTIONS.values()
            for function in functions
        }
        improved = {tool.name for tool in get_tools("improved")}
        self.assertEqual(combined, improved)

    def test_tool_scopes(self):
        self.assertEqual(
            {function.__name__ for function in get_phase_tool_functions("planning")},
            {"generate_and_validate_query_plan"},
        )
        self.assertNotIn(
            "submit_validated_sql",
            {function.__name__ for function in get_phase_tool_functions("sql_generation")},
        )
        with self.assertRaises(ValueError):
            get_phase_tool_functions("unknown")

    def test_shared_state_has_metrics_but_does_not_duplicate_artifacts(self):
        state = initial_multi_agent_state()
        self.assertEqual(state["current_phase"], "preprocessing")
        self.assertEqual(tuple(state["agent_metrics"]), PHASES)
        self.assertEqual(tuple(ARTIFACT_KEYS), PHASES)
        self.assertNotIn("query_plan", state)


if __name__ == "__main__":
    unittest.main()
