import unittest

from experiment.variants import get_variant
from system_agent.prompts import BASELINE_INSTRUCTION, IMPROVED_INSTRUCTION, get_instruction
from system_agent.tools import get_tools


class VariantConfigurationTests(unittest.TestCase):
    def test_requested_factorial_profiles(self):
        expected = {
            0: ("baseline", "baseline"),
            1: ("improved", "baseline"),
            2: ("baseline", "improved"),
            3: ("improved", "improved"),
        }
        actual = {
            number: (
                get_variant(number).requested_agent_profile,
                get_variant(number).requested_harness_profile,
            )
            for number in range(4)
        }
        self.assertEqual(actual, expected)

    def test_prompt_change_is_only_active_for_variants_one_and_three(self):
        active_profiles = {
            number: get_variant(number).behavior_signature for number in range(4)
        }
        self.assertEqual(active_profiles, {
            0: ("baseline", "baseline"),
            1: ("improved", "baseline"),
            2: ("baseline", "baseline"),
            3: ("improved", "baseline"),
        })

    def test_invalid_variant_is_rejected(self):
        with self.assertRaises(ValueError):
            get_variant(4)

    def test_variants_resolve_to_the_expected_prompt(self):
        expected = {
            0: BASELINE_INSTRUCTION,
            1: IMPROVED_INSTRUCTION,
            2: BASELINE_INSTRUCTION,
            3: IMPROVED_INSTRUCTION,
        }
        for number, instruction in expected.items():
            profile = get_variant(number).active_agent_profile
            self.assertEqual(get_instruction(profile), instruction)

    def test_improved_prompt_scopes_execute_sql_across_three_phases(self):
        self.assertIn("one combined read-only inspection", IMPROVED_INSTRUCTION)
        self.assertIn("execute_validated_sql", IMPROVED_INSTRUCTION)
        self.assertIn("diagnose_last_execution_error", IMPROVED_INSTRUCTION)
        self.assertNotIn("optional data inspection in Phase 1", BASELINE_INSTRUCTION)

    def test_alien_few_shot_is_improved_only(self):
        self.assertIn("Few-shot", IMPROVED_INSTRUCTION)
        self.assertIn("COUNT(*) FILTER", IMPROVED_INSTRUCTION)
        self.assertNotIn("Few-shot", BASELINE_INSTRUCTION)

    def test_categorical_value_inspection_is_prompt_only_for_improved(self):
        self.assertIn("Inspection proves validity, not synonymy", IMPROVED_INSTRUCTION)
        self.assertIn("For every categorical predicate", IMPROVED_INSTRUCTION)
        self.assertIn("value_verified=true", IMPROVED_INSTRUCTION)
        self.assertNotIn("value_verified=true", BASELINE_INSTRUCTION)

    def test_low_token_semantic_rules_are_improved_only(self):
        markers = (
            "formula_dependencies",
            "exact output contract",
            "shortest declared foreign-key path",
            "unexpected empty result",
            "implausible bounded score",
            "/30.0, never /30",
            "same validation diagnosis more than twice",
            "semantic_contract.summary",
        )
        for marker in markers:
            self.assertIn(marker, IMPROVED_INSTRUCTION)
            self.assertNotIn(marker, BASELINE_INSTRUCTION)

    def test_generation_strategy_instruction_is_improved_only(self):
        marker = "Follow generation_strategy"
        self.assertIn(marker, IMPROVED_INSTRUCTION)
        self.assertNotIn(marker, BASELINE_INSTRUCTION)

    def test_improved_tools_are_only_used_by_variants_one_and_three(self):
        baseline_names = tuple(tool.name for tool in get_tools("baseline"))
        improved_names = tuple(tool.name for tool in get_tools("improved"))
        self.assertEqual(len(baseline_names), 8)
        self.assertEqual(len(improved_names), 9)
        self.assertIn("prepare_schema_context", improved_names)
        self.assertIn("prepare_knowledge_context", improved_names)
        self.assertNotIn("get_schema", improved_names)
        self.assertIn("finalize_preprocessing_context", improved_names)
        self.assertIn("generate_and_validate_query_plan", improved_names)
        self.assertIn("validate_sql_to_plan", improved_names)
        self.assertIn("execute_validated_sql", improved_names)
        self.assertIn("inspect_database", improved_names)
        self.assertIn("diagnose_last_execution_error", improved_names)
        self.assertIn("submit_validated_sql", improved_names)
        for number in (0, 2):
            self.assertEqual(
                tuple(tool.name for tool in get_tools(get_variant(number).active_agent_profile)),
                baseline_names,
            )
        for number in (1, 3):
            self.assertEqual(
                tuple(tool.name for tool in get_tools(get_variant(number).active_agent_profile)),
                improved_names,
            )


if __name__ == "__main__":
    unittest.main()
