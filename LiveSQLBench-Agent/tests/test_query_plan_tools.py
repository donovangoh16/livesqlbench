import json
import unittest
from types import SimpleNamespace

from system_agent.tools import (
    diagnose_execution_error,
    execute_validated_sql,
    generate_and_validate_query_plan,
    generate_query_plan,
    validate_query_plan,
    validate_sql_to_plan,
)


def valid_state():
    return {
        "preprocessing_completed": True,
        "preprocessing_context": {
            "selected_tables": ["vendors", "markets"],
            "selected_columns": [
                {"table": "vendors", "column": "vendregistry"},
                {"table": "vendors", "column": "mktref"},
                {"table": "markets", "column": "mktregistry"},
            ],
            "selected_join_edges": [{
                "left": "vendors.mktref", "right": "markets.mktregistry",
            }],
            "selected_knowledge": [],
            "unresolved_items": [],
        },
    }


class QueryPlanToolTests(unittest.TestCase):
    def _validated_query_context(self):
        context = SimpleNamespace(state=valid_state())
        generate_query_plan("Query", {
            "operation": "SELECT",
            "result_grain": "one row per vendor",
            "output_columns": ["vendors.vendregistry"],
            "source_tables": ["vendors", "markets"],
            "joins": [{
                "left": "vendors.mktref", "right": "markets.mktregistry",
                "type": "LEFT",
            }],
            "calculations": [], "conditions": [], "steps": ["join and project"],
            "requires_cte": False, "requires_subquery": False,
            "requires_set_operation": False,
        }, context)
        self.assertTrue(json.loads(validate_query_plan(context))["valid"])
        return context

    def test_valid_plan_is_promoted(self):
        context = SimpleNamespace(state=valid_state())
        generated = json.loads(generate_query_plan("Query", {
            "operation": "SELECT",
            "result_grain": "one row per vendor",
            "output_columns": ["vendors.vendregistry"],
            "source_tables": ["vendors", "markets"],
            "joins": [{
                "left": "vendors.mktref", "right": "markets.mktregistry",
                "type": "LEFT",
            }],
            "calculations": [], "conditions": [], "grouping": [],
            "aggregations": [], "ordering": [], "steps": ["join and project"],
            "requires_cte": False, "requires_subquery": False,
            "requires_set_operation": False,
        }, context))
        self.assertEqual(generated["difficulty"], "non_nested_complex")
        result = json.loads(validate_query_plan(context))
        self.assertTrue(result["valid"])
        self.assertTrue(context.state["query_plan_completed"])
        self.assertIn("query_plan", context.state)

    def test_common_plan_shapes_are_normalized_before_validation(self):
        context = SimpleNamespace(state=valid_state())
        generated = json.loads(generate_query_plan("Query", {
            "operation": "Aggregate vendors",
            "result_grain": "one row per vendor",
            "outputs": [{"expression": "v.vendregistry", "name": "vendor"}],
            "sources": [
                {"table": "vendors", "alias": "v"},
                "markets AS m",
            ],
            "joins": [{
                "join_type": "LEFT",
                "left_column": "v.mktref",
                "right_column": "m.mktregistry",
            }],
            "filters": [{"predicate": "v.vendregistry IS NOT NULL", "stage": "WHERE"}],
            "calculations": [],
            "steps": ["join and project"],
        }, context))
        self.assertTrue(generated["normalizations"])
        plan = context.state["query_plan_candidate"]["plan"]
        self.assertEqual(plan["operation"], "SELECT")
        self.assertEqual(plan["source_tables"], ["vendors", "markets"])
        self.assertEqual(plan["joins"][0]["left"], "vendors.mktref")
        self.assertEqual(plan["joins"][0]["right"], "markets.mktregistry")
        self.assertTrue(json.loads(validate_query_plan(context))["valid"])

    def test_combined_plan_tool_generates_and_validates(self):
        context = SimpleNamespace(state=valid_state())
        result = json.loads(generate_and_validate_query_plan("Query", {
            "operation": "SELECT",
            "result_grain": "one row per vendor",
            "output_columns": ["vendors.vendregistry"],
            "source_tables": ["vendors"],
            "joins": [], "calculations": [], "conditions": [],
            "steps": ["project vendor identifier"],
        }, context))
        self.assertTrue(result["valid"])
        self.assertTrue(context.state["query_plan_validated"])

    def test_state_backed_execution_requires_validated_sql(self):
        context = SimpleNamespace(state={})
        result = json.loads(execute_validated_sql(context))
        self.assertFalse(result["success"])
        self.assertIn("validate_sql_to_plan", result["next_action"])

    def test_invalid_plan_recommends_regeneration(self):
        context = SimpleNamespace(state=valid_state())
        generate_query_plan("Query", {
            "operation": "SELECT", "source_tables": ["unknown"], "steps": [],
        }, context)
        result = json.loads(validate_query_plan(context))
        self.assertFalse(result["valid"])
        self.assertIn("generate_query_plan again", result["recommendation"])
        self.assertFalse(context.state["query_plan_completed"])
        self.assertNotIn("query_plan", context.state)

    def test_regenerated_plan_creates_new_version(self):
        context = SimpleNamespace(state=valid_state())
        generate_query_plan("Query", {"operation": "SELECT"}, context)
        second = json.loads(generate_query_plan("Management", {
            "operation": "UPDATE", "source_tables": ["vendors"],
            "target_objects": ["vendors.vendregistry"],
            "mutations": [{"target": "vendors.vendregistry", "value": "x"}],
            "steps": ["update vendors"],
        }, context))
        self.assertEqual(second["version"], 2)
        self.assertEqual(len(context.state["query_plan_history"]), 2)

    def test_sql_matching_plan_is_recorded_as_draft(self):
        context = self._validated_query_context()
        result = json.loads(validate_sql_to_plan(
            "SELECT v.vendregistry FROM vendors v "
            "LEFT JOIN markets m ON v.mktref = m.mktregistry",
            1,
            context,
        ))
        self.assertTrue(result["valid"])
        self.assertTrue(context.state["sql_generation_completed"])
        self.assertIn("draft_sql", context.state)

    def test_sql_plan_mismatch_returns_natural_language_diagnosis(self):
        context = self._validated_query_context()
        result = json.loads(validate_sql_to_plan(
            "SELECT v.vendregistry FROM vendors v "
            "JOIN markets m ON v.mktref = m.mktregistry",
            1,
            context,
        ))
        self.assertFalse(result["valid"])
        self.assertEqual(result["diagnoses"][0]["type"], "join_type_mismatch")
        self.assertIn("Planned join types", result["diagnoses"][0]["message"])
        self.assertFalse(context.state["sql_generation_completed"])

    def test_execution_error_diagnosis_requires_plan_preserving_retry(self):
        context = self._validated_query_context()
        result = json.loads(diagnose_execution_error(
            "SELECT missing FROM vendors",
            'column "missing" does not exist',
            1,
            context,
        ))
        self.assertEqual(result["error_type"], "missing_column")
        self.assertEqual(result["return_to_phase"], "sql_generation")
        self.assertTrue(result["retry_allowed"])
        self.assertIn("execution_error_diagnoses", context.state)

    def test_execution_error_diagnosis_rejects_wrong_plan_version(self):
        context = self._validated_query_context()
        result = json.loads(diagnose_execution_error(
            "SELECT 1", "syntax error near SELECT", 99, context,
        ))
        self.assertFalse(result["retry_allowed"])
        self.assertIn("does not match", result["diagnostic_issues"][0])



if __name__ == "__main__":
    unittest.main()
