import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from system_agent.tools import (
    _generation_strategy,
    diagnose_execution_error,
    execute_validated_sql,
    generate_and_validate_query_plan,
    generate_query_plan,
    inspect_database,
    prepare_knowledge_context,
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

    def test_plan_normalizes_missing_stage_and_steps_without_retry(self):
        context = SimpleNamespace(state=valid_state())
        result = json.loads(generate_and_validate_query_plan("Query", {
            "operation": "SELECT", "result_grain": "one row per vendor",
            "source_tables": ["vendors"], "joins": [],
            "output_columns": ["vendors.vendregistry"],
            "conditions": [{"predicate": "vendors.vendregistry > 0"}],
        }, context))
        self.assertTrue(result["valid"])
        plan = context.state["query_plan"]["plan"]
        self.assertEqual(plan["conditions"][0]["stage"], "WHERE")
        self.assertTrue(plan["steps"])

    def test_plan_normalizes_string_rounding_contract(self):
        context = SimpleNamespace(state=valid_state())
        result = json.loads(generate_and_validate_query_plan("Query", {
            "operation": "SELECT", "result_grain": "one row per vendor",
            "source_tables": ["vendors"], "joins": [],
            "output_columns": [{
                "name": "vendor", "expression": "vendors.vendregistry",
                "rounding": "2_decimal_places",
            }],
        }, context))
        self.assertTrue(result["valid"])
        self.assertEqual(
            context.state["query_plan"]["plan"]["output_columns"][0]["rounding"], 2,
        )

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
        self.assertEqual(result["contract"]["output_count"], 1)
        self.assertIn("join_path_id", result["contract"])
        self.assertEqual(result["generation_strategy"]["shape"], "single_select")
        self.assertTrue(result["semantic_contract"]["semantic_valid"])
        self.assertIn("outputs=vendregistry", result["semantic_contract"]["summary"])

    def test_semantic_contract_requires_requested_ordering(self):
        state = valid_state()
        state["task_question"] = "Show vendors sorted from highest to lowest."
        context = SimpleNamespace(state=state)
        result = json.loads(generate_and_validate_query_plan("Query", {
            "operation": "SELECT", "result_grain": "one row per vendor",
            "source_tables": ["vendors"], "joins": [],
            "output_columns": ["vendors.vendregistry"],
        }, context))
        self.assertFalse(result["valid"])
        self.assertFalse(result["semantic_contract"]["checks"]["ordering_covered"])
        self.assertEqual(result["semantic_contract"]["next"], "regenerate_plan")

    def test_generation_strategy_covers_all_category_difficulty_pairs(self):
        expected = {
            ("Query", "easy"): "single_select",
            ("Query", "non_nested_complex"): "joined_select",
            ("Query", "nested_complex"): "staged_cte_pipeline",
            ("Management", "easy"): "single_statement",
            ("Management", "non_nested_complex"): "source_then_mutation",
            ("Management", "nested_complex"): "ordered_statement_pipeline",
        }
        for pair, shape in expected.items():
            self.assertEqual(_generation_strategy(*pair)["shape"], shape)

    @patch("system_agent.tools.execute_sql")
    def test_inspection_returns_exact_value_json(self, execute):
        execute.return_value = "devscope\n--------\nController\nGamepad"
        context = SimpleNamespace(state={})
        result = json.loads(inspect_database("SELECT DISTINCT devscope", context))
        self.assertEqual(result["values"], ["Controller", "Gamepad"])
        self.assertEqual(result["selection_rule"], "exact_match_only_no_inferred_synonyms")

    @patch("system_agent.tools.execute_sql")
    def test_execution_returns_compact_structured_summary(self, execute):
        execute.return_value = "score | label\n------+------\n8.5 | good\n7.0 | ok\n6.0 | low\n5.0 | low"
        context = SimpleNamespace(state={
            "draft_sql": "SELECT score, label FROM results",
            "sql_generation_completed": True,
            "query_plan": {"plan": {"output_columns": []}},
        })
        result = json.loads(execute_validated_sql(context))
        self.assertEqual(result["sample_row_count"], 4)
        self.assertEqual(len(result["sample_rows"]), 3)
        self.assertEqual(result["numeric_ranges"]["score"], [5.0, 8.5])
        self.assertEqual(result["next"], "submit")

    @patch("system_agent.tools._column_metadata")
    @patch("system_agent.tools._post_json")
    @patch("system_agent.tools._knowledge_names")
    def test_knowledge_dependencies_are_normalized_and_expanded(self, names, post, columns):
        names.return_value = [
            "Wireless Performance Efficiency (WPE)",
            "Wireless Performance Rating (WPR)",
            "Battery Efficiency Ratio (BER)",
        ]
        definitions = {
            "Wireless Performance Efficiency (WPE)": {
                "id": 35, "knowledge": "Wireless Performance Efficiency (WPE)",
                "definition": "WPE = WPR * SQRT(BER / 5.0)",
            },
            "Wireless Performance Rating (WPR)": {
                "id": 8, "knowledge": "Wireless Performance Rating (WPR)",
                "definition": "WPR = WlRangeM / 10.0",
            },
            "Battery Efficiency Ratio (BER)": {
                "id": 1, "knowledge": "Battery Efficiency Ratio (BER)",
                "definition": "BER = life * capacity / power",
            },
        }
        post.side_effect = lambda path, payload: {
            "knowledge": json.dumps(definitions[payload["knowledge_name"]])
        }
        columns.return_value = {
            "gaming|deviceidentity|wlrangem": "range",
            "gaming|testsessions|battlifeh": "battery life",
        }
        context = SimpleNamespace(state={"task_id": "gaming_6"})
        result = json.loads(prepare_knowledge_context(
            "Calculate WPE", ["WPE"], [], context,
        ))
        self.assertEqual({item["id"] for item in result["knowledge"]}, {1, 8, 35})
        self.assertEqual(result["knowledge"][0]["depends_on"], [
            "Wireless Performance Rating (WPR)", "Battery Efficiency Ratio (BER)",
        ])
        self.assertEqual(result["required_schema"]["deviceidentity"], ["wlrangem"])

    def test_state_backed_execution_requires_validated_sql(self):
        context = SimpleNamespace(state={})
        result = json.loads(execute_validated_sql(context))
        self.assertFalse(result["success"])
        self.assertIn("validate_sql_to_plan", result["next_action"])

    def test_population_constraint_must_appear_in_where(self):
        context = SimpleNamespace(state=valid_state())
        result = json.loads(generate_and_validate_query_plan("Query", {
            "operation": "SELECT", "result_grain": "one row per vendor",
            "source_tables": ["vendors"], "joins": [], "calculations": [],
            "conditions": [], "output_columns": ["vendors.vendregistry"],
            "population_constraints": [{
                "phrase": "positive vendor ids",
                "predicate": "vendors.vendregistry > 0",
            }],
            "steps": ["filter and project"],
        }, context))
        self.assertTrue(result["valid"])
        checked = json.loads(validate_sql_to_plan(
            "SELECT vendregistry FROM vendors", 1, context,
        ))
        self.assertTrue(any(
            item["type"] == "population_constraint_missing"
            for item in checked["diagnoses"]
        ))

    def test_categorical_constraint_requires_observed_verified_value(self):
        state = valid_state()
        state["last_database_inspection"] = {"values": ["Controller", "Gamepad"]}
        context = SimpleNamespace(state=state)
        result = json.loads(generate_and_validate_query_plan("Query", {
            "operation": "SELECT", "result_grain": "one row per vendor",
            "source_tables": ["vendors"], "joins": [],
            "output_columns": ["vendors.vendregistry"],
            "population_constraints": [{
                "phrase": "controllers", "predicate": "kind = 'Gamepad'",
                "value_verified": True,
            }],
        }, context))
        self.assertFalse(result["valid"])
        self.assertTrue(any("broader" in error for error in result["errors"]))

    def test_output_filter_requires_local_sql_filter(self):
        state = valid_state()
        state["preprocessing_context"]["selected_knowledge"] = [{"id": 50, "name": "Eligible"}]
        context = SimpleNamespace(state=state)
        result = json.loads(generate_and_validate_query_plan("Query", {
            "operation": "SELECT", "result_grain": "one row per vendor",
            "source_tables": ["vendors"], "joins": [], "calculations": [],
            "conditions": [], "steps": ["count eligible vendors"],
            "output_columns": [{"name": "eligible_count", "aggregate": "COUNT",
                                "expression": "COUNT(*)", "filter": "vendregistry > 0",
                                "knowledge_ids": [50]}],
        }, context))
        self.assertTrue(result["valid"])
        rejected = json.loads(validate_sql_to_plan(
            "SELECT COUNT(*) FROM vendors WHERE vendregistry > 0", 1, context,
        ))
        self.assertTrue(any(d["type"] == "output_filter_scope_mismatch" for d in rejected["diagnoses"]))

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

    def test_sql_rejects_extra_exposed_outputs(self):
        context = SimpleNamespace(state=valid_state())
        result = json.loads(generate_and_validate_query_plan("Query", {
            "operation": "SELECT", "result_grain": "one row per vendor",
            "source_tables": ["vendors"], "joins": [], "conditions": [],
            "output_columns": [{"name": "vendor", "expression": "vendors.vendregistry"}],
        }, context))
        self.assertTrue(result["valid"])
        checked = json.loads(validate_sql_to_plan(
            "SELECT vendregistry, mktref FROM vendors", 1, context,
        ))
        self.assertTrue(any(d["code"] == "OUTPUT_COUNT_MISMATCH" for d in checked["diagnoses"]))

    def test_transitive_kb_dependencies_count_as_used(self):
        state = valid_state()
        state["preprocessing_context"]["selected_knowledge"] = [
            {"id": 38, "name": "PAR"}, {"id": 30, "name": "CGPI"},
            {"id": 5, "name": "SPR"}, {"id": 31, "name": "RAI"},
        ]
        state["prepared_knowledge_context"] = {"knowledge": [
            {"id": 38, "name": "PAR", "depends_on": ["CGPI"]},
            {"id": 30, "name": "CGPI", "depends_on": ["SPR", "RAI"]},
            {"id": 5, "name": "SPR", "depends_on": []},
            {"id": 31, "name": "RAI", "depends_on": []},
        ]}
        context = SimpleNamespace(state=state)
        result = json.loads(generate_and_validate_query_plan("Query", {
            "operation": "SELECT", "result_grain": "one row per vendor",
            "source_tables": ["vendors"], "joins": [],
            "output_columns": [{"name": "par", "expression": "par", "knowledge_id": 38}],
            "calculations": [{"name": "par", "expression": "cgpi", "knowledge_id": 38}],
            "conditions": [],
        }, context))
        self.assertTrue(result["valid"])

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
