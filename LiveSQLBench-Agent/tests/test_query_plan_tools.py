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
    prepare_schema_context,
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

    @patch("system_agent.tools.rank_relevant_columns")
    @patch("system_agent.tools.get_selected_schema")
    @patch("system_agent.tools.rank_relevant_tables")
    def test_repeated_schema_preparation_preserves_original_question(
        self, tables, schema, columns,
    ):
        state = valid_state()
        state["task_question"] = "Show vendors ordered by score descending."
        context = SimpleNamespace(state=state)
        tables.return_value = json.dumps({"ranked_tables": []})
        schema.return_value = json.dumps({"tables": []})
        columns.return_value = json.dumps({"ranked_columns": []})
        prepare_schema_context("Find dependency columns", 3, 5, context)
        self.assertEqual(
            context.state["task_question"],
            "Show vendors ordered by score descending.",
        )

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
    def test_inspection_evidence_accumulates_and_handles_mice(self, execute):
        execute.side_effect = [
            "kind\n----\nKeyboard\nMouse",
            "name\n----\nRetail\nOnline",
        ]
        state = valid_state()
        state["preprocessing_context"]["selected_columns"].append(
            {"table": "vendors", "column": "kind"}
        )
        context = SimpleNamespace(state=state)
        inspect_database("SELECT DISTINCT v.kind FROM vendors v", context)
        inspect_database("SELECT DISTINCT m.name FROM markets m", context)
        self.assertEqual(context.state["database_inspections"]["vendors.kind"], [
            "Keyboard", "Mouse",
        ])
        result = json.loads(generate_and_validate_query_plan("Query", {
            "operation": "SELECT", "result_grain": "one row per vendor",
            "source_tables": ["vendors"], "joins": [],
            "output_columns": ["vendors.vendregistry"],
            "population_constraints": [{
                "phrase": "keyboards and mice",
                "predicate": "vendors.kind IN ('Keyboard', 'Mouse')",
                "value_verified": True,
            }],
        }, context))
        self.assertTrue(result["valid"])

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

    @patch("system_agent.tools._column_metadata", return_value={})
    @patch("system_agent.tools._post_json")
    @patch("system_agent.tools._knowledge_names")
    def test_later_kb_lookup_adds_evidence_without_replacing_requirements(
        self, names, post, _columns,
    ):
        names.return_value = ["Primary Score", "Exploratory Score"]
        definitions = {
            "Primary Score": {"id": 6, "knowledge": "Primary Score", "definition": "x / 2.0"},
            "Exploratory Score": {"id": 21, "knowledge": "Exploratory Score", "definition": "y / 3.0"},
        }
        post.side_effect = lambda path, payload: {
            "knowledge": json.dumps(definitions[payload["knowledge_name"]])
        }
        context = SimpleNamespace(state={"task_id": "credit_M_4"})
        first = json.loads(prepare_knowledge_context(
            "Use Primary Score", ["Primary Score"], [], context,
        ))
        second = json.loads(prepare_knowledge_context(
            "Check Exploratory Score", ["Exploratory Score"], [], context,
        ))
        self.assertEqual(first["required_knowledge_ids"], [6])
        self.assertEqual(second["required_knowledge_ids"], [6])
        self.assertEqual({item["id"] for item in second["knowledge"]}, {6, 21})

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

    def test_json_path_keys_are_not_treated_as_categorical_values(self):
        state = valid_state()
        state["preprocessing_context"]["selected_tables"].append("bank_and_transactions")
        state["preprocessing_context"]["selected_columns"].append({
            "table": "bank_and_transactions", "column": "chaninvdatablock",
        })
        state["last_database_inspection"] = {"values": ["High", "Yes"]}
        context = SimpleNamespace(state=state)
        result = json.loads(generate_and_validate_query_plan("Query", {
            "operation": "SELECT", "result_grain": "one row per transaction",
            "source_tables": ["bank_and_transactions"], "joins": [],
            "output_columns": ["bank_and_transactions.bankexpref"],
            "population_constraints": [{
                "phrase": "Digital First Customer",
                "predicate": (
                    "bank_and_transactions.chaninvdatablock->>'onlineuse' = 'High' "
                    "AND bank_and_transactions.chaninvdatablock->>'autopay' = 'Yes'"
                ),
                "value_verified": True,
            }],
        }, context))
        self.assertTrue(result["valid"], result.get("errors"))

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

    def test_parameter_only_management_function_needs_no_source_table(self):
        context = SimpleNamespace(state=valid_state())
        result = json.loads(generate_and_validate_query_plan("Management", {
            "operation": "CREATE FUNCTION",
            "target_objects": [{"type": "function", "name": "calculate_score"}],
            "parameters": [{"name": "x", "type": "numeric"}],
            "return_contract": {"type": "numeric"},
            "language": "sql",
            "body_requirements": [{"expression": "x * 2.0"}],
        }, context))
        self.assertTrue(result["valid"], result.get("errors"))
        checked = json.loads(validate_sql_to_plan(
            "CREATE FUNCTION calculate_score(x numeric) RETURNS numeric "
            "LANGUAGE SQL AS $$ SELECT x * 2.0 $$;", 1, context,
        ))
        self.assertTrue(checked["valid"], checked.get("diagnoses"))

    def test_management_target_dictionary_uses_its_name(self):
        context = SimpleNamespace(state=valid_state())
        result = json.loads(generate_and_validate_query_plan("Management", {
            "operation": "ALTER TABLE",
            "target_objects": [{"type": "table", "name": "vendors"}],
            "schema_changes": [{
                "action": "ADD COLUMN", "name": "active", "type": "boolean",
            }],
        }, context))
        self.assertTrue(result["valid"], result.get("errors"))
        checked = json.loads(validate_sql_to_plan(
            "ALTER TABLE vendors ADD COLUMN active boolean;", 1, context,
        ))
        self.assertTrue(checked["valid"], checked.get("diagnoses"))

    def test_management_scalar_condition_is_normalized(self):
        context = SimpleNamespace(state=valid_state())
        result = json.loads(generate_and_validate_query_plan("Management", {
            "operation": "DELETE",
            "target_objects": ["vendors"],
            "affected_row_conditions": "vendors.vendregistry > 10",
        }, context))
        self.assertTrue(result["valid"], result.get("errors"))
        condition = context.state["query_plan"]["plan"]["affected_row_conditions"][0]
        self.assertEqual(condition["expression"], "vendors.vendregistry > 10")

    def test_management_condition_separates_phrase_from_sql_expression(self):
        context = SimpleNamespace(state=valid_state())
        result = json.loads(generate_and_validate_query_plan("Management", {
            "operation": "DELETE", "target_objects": ["vendors"],
            "affected_row_conditions": (
                "Delete vendors meeting the policy: vendors.vendregistry > 10 AND vendors.mktref > 0"
            ),
        }, context))
        self.assertTrue(result["valid"], result.get("errors"))
        condition = context.state["query_plan"]["plan"]["affected_row_conditions"][0]
        self.assertEqual(condition["phrase"], "Delete vendors meeting the policy")
        self.assertEqual(
            condition["expression"],
            "vendors.vendregistry > 10 AND vendors.mktref > 0",
        )

    def test_management_sql_template_statement_is_semantically_normalized(self):
        context = SimpleNamespace(state=valid_state())
        result = json.loads(generate_and_validate_query_plan("Management", {
            "operation": "CREATE FUNCTION",
            "target_objects": [{"type": "function", "name": "calculate_score"}],
            "body_requirements": [{"expression": "x * 2.0"}],
            "required_statements": [
                "CREATE OR REPLACE FUNCTION calculate_score(...) RETURNS numeric LANGUAGE SQL AS $$ ... $$;"
            ],
        }, context))
        self.assertTrue(result["valid"], result.get("errors"))
        statement = context.state["query_plan"]["plan"]["required_statements"][0]
        self.assertEqual(statement["operation"], "CREATE_FUNCTION")
        self.assertEqual(statement["target"], "calculate_score")
        checked = json.loads(validate_sql_to_plan(
            "CREATE FUNCTION calculate_score(x numeric) RETURNS numeric "
            "LANGUAGE SQL AS $$ SELECT x * 2.0 $$;", 1, context,
        ))
        self.assertTrue(checked["valid"], checked.get("diagnoses"))

    def test_management_statement_alias_fields_are_normalized(self):
        context = SimpleNamespace(state=valid_state())
        result = json.loads(generate_and_validate_query_plan("Management", {
            "operation": "CREATE INDEX",
            "target_objects": [{"type": "index", "name": "idx_vendor_market"}],
            "required_statements": [{
                "statement_type": "CREATE_INDEX",
                "object_name": "idx_vendor_market",
            }],
        }, context))
        self.assertTrue(result["valid"], result.get("errors"))
        statement = context.state["query_plan"]["plan"]["required_statements"][0]
        self.assertEqual(statement["operation"], "CREATE_INDEX")
        self.assertEqual(statement["target"], "idx_vendor_market")

    def test_management_sql_kind_and_compound_operations_are_normalized(self):
        context = SimpleNamespace(state=valid_state())
        result = json.loads(generate_and_validate_query_plan("Management", {
            "operation": "ALTER TABLE",
            "target_objects": [{"type": "table", "name": "vendors"}],
            "schema_changes": [{"action": "ADD COLUMN", "name": "active", "type": "boolean"}],
            "affected_row_conditions": "all rows in vendors",
            "required_statements": [
                {"statement_type": "alter_table_add_column", "target": "vendors"},
                {"sql_kind": "UPDATE_BACKFILL", "target": "vendors"},
            ],
        }, context))
        self.assertTrue(result["valid"], result.get("errors"))
        plan = context.state["query_plan"]["plan"]
        self.assertEqual(plan["affected_row_conditions"], [])
        self.assertEqual(
            [item["operation"] for item in plan["required_statements"]],
            ["ALTER_TABLE", "UPDATE"],
        )

    def test_nested_details_sql_identifies_do_block(self):
        context = SimpleNamespace(state=valid_state())
        result = json.loads(generate_and_validate_query_plan("Management", {
            "operation": "ALTER TABLE",
            "target_objects": [{"type": "table", "name": "vendors"}],
            "schema_changes": [{"action": "ADD COLUMN", "name": "active", "type": "boolean"}],
            "required_statements": [{
                "statement_type": "sql", "target": "vendors",
                "details": {"sql": "DO $$ BEGIN UPDATE vendors SET active = true; END $$;"},
            }],
        }, context))
        self.assertTrue(result["valid"], result.get("errors"))
        statement = context.state["query_plan"]["plan"]["required_statements"][0]
        self.assertEqual(statement["operation"], "DO_BLOCK")

    def test_arbitrary_nested_sql_intent_is_detected(self):
        context = SimpleNamespace(state=valid_state())
        result = json.loads(generate_and_validate_query_plan("Management", {
            "operation": "ALTER TABLE",
            "target_objects": [{"type": "table", "name": "vendors"}],
            "schema_changes": [{"action": "ADD COLUMN", "name": "active", "type": "boolean"}],
            "required_statements": [{
                "metadata": {"sql_intent": "ALTER TABLE vendors ADD COLUMN active boolean"},
            }],
        }, context))
        self.assertTrue(result["valid"], result.get("errors"))
        statement = context.state["query_plan"]["plan"]["required_statements"][0]
        self.assertEqual(statement["operation"], "ALTER_TABLE")
        self.assertEqual(statement["target"], "vendors")

    def test_statement_template_beats_longer_sql_like_prose(self):
        context = SimpleNamespace(state=valid_state())
        result = json.loads(generate_and_validate_query_plan("Management", {
            "operation": "CREATE_INDEX",
            "target_objects": [{"type": "index", "name": "idx_vendor_market"}],
            "required_statements": [{
                "statement_purpose": (
                    "Create a highly optimized index for repeated filtering and reporting"
                ),
                "statement_template": "CREATE INDEX idx_vendor_market ON vendors (mktref)",
            }],
        }, context))
        self.assertTrue(result["valid"], result.get("errors"))
        statement = context.state["query_plan"]["plan"]["required_statements"][0]
        self.assertEqual(statement["operation"], "CREATE_INDEX")
        self.assertEqual(statement["target"], "idx_vendor_market")

    def test_plan_rejects_clamping_not_present_in_kb_definition(self):
        state = valid_state()
        state["preprocessing_context"]["selected_knowledge"] = [{"id": 6}]
        state["prepared_knowledge_context"] = {
            "required_knowledge_ids": [6],
            "knowledge": [{
                "id": 6, "name": "FSI",
                "definition": "FSI = 0.3 * (1 - debt_ratio) + 0.7 * liquid_ratio",
                "depends_on": [],
            }],
        }
        context = SimpleNamespace(state=state)
        result = json.loads(generate_and_validate_query_plan("Management", {
            "operation": "UPDATE",
            "target_objects": [{"type": "table", "name": "vendors"}],
            "mutations": [{
                "target": "score",
                "expression": "LEAST(GREATEST(0.3 * (1 - debt_ratio), 0), 1)",
                "knowledge_id": 6,
            }],
        }, context))
        self.assertFalse(result["valid"])
        self.assertTrue(any("unrequested clamping" in error for error in result["errors"]))

    def test_calculated_column_plan_requires_update_backfill(self):
        state = valid_state()
        state["preprocessing_context"]["selected_knowledge"] = [{"id": 10}]
        state["prepared_knowledge_context"] = {
            "required_knowledge_ids": [10],
            "knowledge": [{"id": 10, "name": "Prime", "definition": "prime = score > 720"}],
        }
        context = SimpleNamespace(state=state)
        result = json.loads(generate_and_validate_query_plan("Management", {
            "operation": "ALTER_TABLE_ADD_COLUMN",
            "target_objects": [{"type": "column", "name": "vendors.prime"}],
            "calculations": [{"name": "prime", "expression": "score > 720", "knowledge_id": 10}],
            "required_statements": [{"statement": "ALTER TABLE vendors ADD COLUMN prime boolean"}],
        }, context))
        self.assertFalse(result["valid"])
        self.assertTrue(any("UPDATE backfill" in error for error in result["errors"]))

    def test_sql_comments_cannot_satisfy_kb_expression(self):
        state = valid_state()
        state["preprocessing_context"]["selected_knowledge"] = [{"id": 5}]
        state["prepared_knowledge_context"] = {
            "required_knowledge_ids": [5],
            "knowledge": [{"id": 5, "name": "score", "definition": "score = x * 2.0"}],
        }
        context = SimpleNamespace(state=state)
        planned = json.loads(generate_and_validate_query_plan("Management", {
            "operation": "CREATE_FUNCTION",
            "target_objects": [{"type": "function", "name": "calculate_score"}],
            "body_requirements": [{"expression": "x * 2.0", "knowledge_id": 5}],
        }, context))
        self.assertTrue(planned["valid"], planned.get("errors"))
        checked = json.loads(validate_sql_to_plan(
            "CREATE FUNCTION calculate_score(x numeric) RETURNS numeric "
            "LANGUAGE SQL AS $$ SELECT x /* x * 2.0 */ $$;", 1, context,
        ))
        self.assertFalse(checked["valid"])
        self.assertTrue(any(
            item["type"] == "kb_expression_mismatch" for item in checked["diagnoses"]
        ))

    def test_not_null_after_add_requires_backfill(self):
        context = SimpleNamespace(state=valid_state())
        planned = json.loads(generate_and_validate_query_plan("Management", {
            "operation": "ALTER_TABLE",
            "target_objects": [{"type": "table", "name": "vendors"}],
            "columns": [{"name": "prime", "type": "boolean"}],
        }, context))
        self.assertTrue(planned["valid"], planned.get("errors"))
        checked = json.loads(validate_sql_to_plan(
            "ALTER TABLE vendors ADD COLUMN prime boolean; "
            "ALTER TABLE vendors ALTER COLUMN prime SET NOT NULL;", 1, context,
        ))
        self.assertFalse(checked["valid"])
        self.assertTrue(any(
            item["type"] == "unsafe_not_null_transition" for item in checked["diagnoses"]
        ))

    def test_repeated_statement_operations_match_sequentially(self):
        context = SimpleNamespace(state=valid_state())
        result = json.loads(generate_and_validate_query_plan("Management", {
            "operation": "ALTER TABLE",
            "target_objects": [{"type": "table", "name": "vendors"}],
            "schema_changes": [{"action": "ADD COLUMN", "name": "active", "type": "boolean"}],
            "required_statements": [
                {"statement": "ALTER TABLE vendors ADD COLUMN active boolean"},
                {"sql_kind": "UPDATE_BACKFILL", "target": "vendors"},
                {"statement": "ALTER TABLE vendors ALTER COLUMN active SET NOT NULL"},
            ],
        }, context))
        self.assertTrue(result["valid"], result.get("errors"))
        checked = json.loads(validate_sql_to_plan(
            "ALTER TABLE vendors ADD COLUMN active boolean; "
            "UPDATE vendors SET active = false; "
            "ALTER TABLE vendors ALTER COLUMN active SET NOT NULL;",
            1, context,
        ))
        self.assertTrue(checked["valid"], checked.get("diagnoses"))

    def test_symbolic_formula_helper_name_is_not_required_literal_sql(self):
        state = valid_state()
        state["preprocessing_context"]["selected_knowledge"] = [{"id": 5}]
        state["prepared_knowledge_context"] = {
            "required_knowledge_ids": [5],
            "knowledge": [{"id": 5, "name": "score", "depends_on": []}],
        }
        context = SimpleNamespace(state=state)
        result = json.loads(generate_and_validate_query_plan("Management", {
            "operation": "CREATE FUNCTION",
            "target_objects": [{"type": "function", "name": "calculate_score"}],
            "body_requirements": [{"expression": "bound_value(x)", "knowledge_id": 5}],
        }, context))
        self.assertTrue(result["valid"], result.get("errors"))
        checked = json.loads(validate_sql_to_plan(
            "CREATE FUNCTION calculate_score(x numeric) RETURNS numeric "
            "LANGUAGE SQL AS $$ SELECT LEAST(GREATEST(x, 0), 1) $$;",
            1, context,
        ))
        self.assertTrue(checked["valid"], checked.get("diagnoses"))

    def test_management_statement_sql_field_is_parsed_as_fallback(self):
        context = SimpleNamespace(state=valid_state())
        result = json.loads(generate_and_validate_query_plan("Management", {
            "operation": "ALTER TABLE",
            "target_objects": [{"type": "table", "name": "vendors"}],
            "schema_changes": [{"action": "ADD COLUMN", "name": "active", "type": "boolean"}],
            "required_statements": [{
                "order": 1,
                "statement": "ALTER TABLE vendors ADD COLUMN active boolean",
            }],
        }, context))
        self.assertTrue(result["valid"], result.get("errors"))
        statement = context.state["query_plan"]["plan"]["required_statements"][0]
        self.assertEqual(statement["operation"], "ALTER_TABLE")
        self.assertEqual(statement["target"], "vendors")

    def test_only_required_not_all_retrieved_kb_blocks_management_plan(self):
        state = valid_state()
        state["preprocessing_context"]["selected_knowledge"] = [
            {"id": 0}, {"id": 1}, {"id": 5},
        ]
        state["prepared_knowledge_context"] = {
            "required_knowledge_ids": [5],
            "knowledge": [{"id": 5, "name": "CHS", "depends_on": []}],
        }
        context = SimpleNamespace(state=state)
        result = json.loads(generate_and_validate_query_plan("Management", {
            "operation": "CREATE FUNCTION",
            "target_objects": [{"type": "function", "name": "calculate_chs"}],
            "body_requirements": [{"expression": "x * 2.0", "knowledge_id": 5}],
        }, context))
        self.assertTrue(result["valid"], result.get("errors"))

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

    def test_repeated_sql_diagnosis_becomes_nonretryable(self):
        context = self._validated_query_context()
        sql = (
            "SELECT v.vendregistry FROM vendors v "
            "JOIN markets m ON v.mktref = m.mktregistry"
        )
        first = json.loads(validate_sql_to_plan(sql, 1, context))
        second = json.loads(validate_sql_to_plan(sql, 1, context))
        self.assertFalse(first["valid"])
        self.assertTrue(first["retryable"])
        self.assertEqual(second["same_diagnosis_count"], 2)
        self.assertFalse(second["retryable"])
        history_length = len(context.state["sql_validation_history"])
        third = json.loads(validate_sql_to_plan(sql, 1, context))
        self.assertFalse(third["retryable"])
        self.assertEqual(third["next"], "revise_plan_once_or_stop")
        self.assertEqual(len(context.state["sql_validation_history"]), history_length)

    def test_validated_sql_can_be_revalidated_after_execution_correction(self):
        context = self._validated_query_context()
        first = json.loads(validate_sql_to_plan(
            "SELECT v.vendregistry FROM vendors v "
            "LEFT JOIN markets m ON v.mktref = m.mktregistry",
            1, context,
        ))
        self.assertTrue(first["valid"])
        corrected = json.loads(validate_sql_to_plan(
            "SELECT v.vendregistry FROM vendors AS v "
            "LEFT JOIN markets AS m ON v.mktref = m.mktregistry",
            1, context,
        ))
        self.assertTrue(corrected["valid"], corrected.get("diagnoses"))

    @patch("system_agent.tools.execute_sql")
    def test_aggregate_inspection_flattens_values_and_maps_source(self, execute):
        execute.return_value = (
            "devscope_values | mechanical_columns\n"
            "------------------------------------\n"
            "['Controller', 'Keyboard', 'Mouse'] | ['ergorate', 'palmangle', 'wristflag']"
        )
        context = SimpleNamespace(state={"task_question": "Show gaming controllers"})
        sql = (
            "SELECT "
            "(SELECT array_agg(DISTINCT devscope::text) FROM testsessions) AS devscope_values, "
            "(SELECT array_agg(DISTINCT column_name) FROM information_schema.columns "
            "WHERE table_name='mechanical') AS mechanical_columns"
        )
        result = json.loads(inspect_database(sql, context))
        self.assertEqual(result["values_by_column"]["devscope_values"], [
            "Controller", "Keyboard", "Mouse",
        ])
        self.assertEqual(context.state["database_inspections"]["testsessions.devscope"], [
            "Controller", "Keyboard", "Mouse",
        ])
        self.assertEqual(result["required_population"], [{
            "phrase": "Controller", "column": "testsessions.devscope",
            "value": "Controller", "value_verified": True,
        }])

    def test_request_named_inspected_value_is_required_in_plan(self):
        state = valid_state()
        state["task_question"] = "Show active vendors"
        state["preprocessing_context"]["selected_columns"].append(
            {"table": "vendors", "column": "status"}
        )
        state["required_population"] = [{
            "phrase": "Active", "column": "vendors.status",
            "value": "Active", "value_verified": True,
        }]
        context = SimpleNamespace(state=state)
        result = json.loads(generate_and_validate_query_plan("Query", {
            "operation": "SELECT", "result_grain": "one row per vendor",
            "source_tables": ["vendors"], "joins": [],
            "output_columns": ["vendors.vendregistry"],
            "population_constraints": [],
        }, context))
        self.assertFalse(result["valid"])
        self.assertTrue(any("vendors.status" in error for error in result["errors"]))

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

    def test_percentile_rank_defaults_to_zero_to_one_scale(self):
        state = valid_state()
        state["task_question"] = "Show the percentile ranking within each market."
        context = SimpleNamespace(state=state)
        result = json.loads(generate_and_validate_query_plan("Query", {
            "operation": "SELECT", "result_grain": "one row per vendor",
            "source_tables": ["vendors"], "joins": [],
            "output_columns": [{
                "name": "percentile_rank",
                "expression": "PERCENT_RANK() OVER (ORDER BY vendors.vendregistry)",
                "scale": "0_to_100",
            }],
        }, context))
        self.assertTrue(result["valid"])
        self.assertEqual(
            context.state["query_plan"]["plan"]["output_columns"][0]["scale"], "0_to_1",
        )
        checked = json.loads(validate_sql_to_plan(
            "SELECT PERCENT_RANK() OVER (ORDER BY vendregistry) * 100 FROM vendors",
            1, context,
        ))
        self.assertTrue(any(d["code"] == "OUTPUT_SCALE_MISMATCH" for d in checked["diagnoses"]))

    def test_population_validator_accepts_calculated_alias_in_where(self):
        context = SimpleNamespace(state=valid_state())
        result = json.loads(generate_and_validate_query_plan("Query", {
            "operation": "SELECT", "result_grain": "one row per vendor",
            "source_tables": ["vendors"], "joins": [],
            "output_columns": [{"name": "par", "expression": "par"}],
            "calculations": [{
                "name": "par", "expression": "vendors.vendregistry * 2.0",
            }],
            "population_constraints": [{
                "phrase": "PAR exceeding 8.5",
                "predicate": "vendors.vendregistry * 2.0 > 8.5",
            }],
            "requires_cte": True,
        }, context))
        self.assertTrue(result["valid"])
        checked = json.loads(validate_sql_to_plan(
            "WITH calc AS (SELECT vendregistry * 2.0 AS par FROM vendors) "
            "SELECT par FROM calc WHERE par > 8.5", 1, context,
        ))
        self.assertTrue(checked["valid"], checked.get("diagnoses"))

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
