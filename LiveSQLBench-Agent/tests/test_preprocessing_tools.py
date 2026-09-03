import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from system_agent.tools import (
    finalize_preprocessing_context,
    get_schema_summary,
    get_selected_schema,
    rank_relevant_columns,
    rank_relevant_tables,
)


SCHEMA = '''CREATE TABLE "vendors" (
    "id" integer PRIMARY KEY,
    "market_id" integer,
    "name" text,
    FOREIGN KEY ("market_id") REFERENCES "markets" ("id")
);
CREATE TABLE "markets" (
    "id" integer PRIMARY KEY,
    "region" text
);'''


class CompactSchemaToolTests(unittest.TestCase):
    @patch("system_agent.tools._schema_text", return_value=SCHEMA)
    def test_schema_summary_omits_non_key_columns(self, _schema_text):
        result = json.loads(get_schema_summary(SimpleNamespace(state={})))
        self.assertEqual(result["table_count"], 2)
        vendors = next(x for x in result["tables"] if x["table"] == "vendors")
        self.assertEqual(vendors["column_count"], 3)
        self.assertNotIn("columns", vendors)
        self.assertEqual(vendors["foreign_keys"][0]["references"], "markets.id")

    @patch("system_agent.tools._column_metadata")
    @patch("system_agent.tools._schema_text", return_value=SCHEMA)
    def test_selected_schema_returns_only_requested_tables(self, _schema_text, metadata):
        metadata.return_value = {
            "db|vendors|id": "Vendor identifier",
            "db|vendors|name": "Vendor display name",
            "db|markets|region": "Market region",
        }
        context = SimpleNamespace(state={})
        result = json.loads(get_selected_schema(["vendors"], context))
        self.assertEqual([x["table"] for x in result["tables"]], ["vendors"])
        self.assertEqual(result["missing_tables"], [])
        self.assertEqual(len(result["tables"][0]["columns"]), 3)
        self.assertIsInstance(result["tables"][0]["columns"][0], str)
        self.assertIn("selected_schema_details", context.state)
        stored_column = context.state["selected_schema_details"]["tables"][0]["columns"][0]
        self.assertIn("definition", stored_column)

    @patch("system_agent.tools._column_metadata")
    def test_table_ranking_has_hard_result_cap(self, metadata):
        metadata.return_value = {
            f"db|table_{index}|column": "value" for index in range(12)
        }
        result = json.loads(rank_relevant_tables(
            "value", 100, SimpleNamespace(state={})
        ))
        self.assertEqual(len(result["ranked_tables"]), 5)

    @patch("system_agent.tools._column_metadata")
    def test_column_ranking_returns_hint_and_stores_full_meaning(self, metadata):
        metadata.return_value = {
            "db|vendors|name": "x" * 150,
        }
        context = SimpleNamespace(state={})
        result = json.loads(rank_relevant_columns(
            "vendor name", ["vendors"], 5, context,
        ))
        returned = result["ranked_columns"][0]
        stored = context.state["ranked_column_candidates"][0]
        self.assertNotIn("meaning", returned)
        self.assertLessEqual(len(returned["meaning_hint"]), 96)
        self.assertGreater(len(stored["meaning"]), len(returned["meaning_hint"]))


class FinalizePreprocessingContextTests(unittest.TestCase):
    @patch("system_agent.tools._knowledge_names", return_value=["Vendor Network Centrality"])
    @patch("system_agent.tools._post_json")
    @patch("system_agent.tools._column_metadata")
    def test_valid_context_is_stored(self, metadata, post_json, _knowledge_names):
        metadata.return_value = {
            "db|vendors|vendregistry": "Primary key",
            "db|vendors|mktref": "Foreign key",
            "db|markets|mktregistry": "Primary key",
        }
        post_json.return_value = {
            "schema": (
                'CREATE TABLE "vendors" (\n'
                'mktref varchar,\n'
                'FOREIGN KEY (mktref) REFERENCES markets(mktregistry)\n);'
            )
        }
        context = SimpleNamespace(state={})
        result = json.loads(finalize_preprocessing_context(
            selected_tables=["vendors", "markets"],
            selected_columns=[
                {"table": "vendors", "column": "vendregistry", "role": "output"},
                {"table": "vendors", "column": "mktref", "role": "join"},
                {"table": "markets", "column": "mktregistry", "role": "join"},
            ],
            selected_join_edges=[{
                "left": "vendors.mktref", "right": "markets.mktregistry",
            }],
            required_knowledge_phrases=["Vendor Network Centrality"],
            selected_knowledge=[{"id": 12, "name": "Vendor Network Centrality"}],
            unresolved_items=[],
            tool_context=context,
        ))
        self.assertTrue(result["valid"])
        self.assertTrue(context.state["preprocessing_completed"])
        self.assertIn("preprocessing_context", context.state)

    @patch("system_agent.tools._knowledge_names", return_value=[])
    @patch("system_agent.tools._post_json")
    @patch("system_agent.tools._column_metadata")
    def test_split_join_edge_fields_are_normalized(self, metadata, post_json, _knowledge_names):
        metadata.return_value = {
            "db|vendors|mktref": "Foreign key",
            "db|markets|mktregistry": "Primary key",
        }
        post_json.return_value = {
            "schema": (
                'CREATE TABLE "vendors" (\n'
                'mktref varchar,\n'
                'FOREIGN KEY (mktref) REFERENCES markets(mktregistry)\n);'
            )
        }
        context = SimpleNamespace(state={})
        result = json.loads(finalize_preprocessing_context(
            selected_tables=["vendors", "markets"],
            selected_columns=[
                {"table": "vendors", "column": "mktref", "role": "join"},
                {"table": "markets", "column": "mktregistry", "role": "join"},
            ],
            selected_join_edges=[{
                "left_table": "vendors", "left_column": "mktref",
                "right_table": "markets", "right_column": "mktregistry",
            }],
            required_knowledge_phrases=[], selected_knowledge=[],
            unresolved_items=[], tool_context=context,
        ))
        self.assertTrue(result["valid"])
        self.assertEqual(context.state["preprocessing_context"]["selected_join_edges"], [{
            "left": "vendors.mktref", "right": "markets.mktregistry",
        }])

    @patch("system_agent.tools._knowledge_names", return_value=[])
    @patch("system_agent.tools._post_json", return_value={"schema": ""})
    @patch("system_agent.tools._column_metadata", return_value={})
    def test_invalid_context_is_not_completed(self, *_mocks):
        context = SimpleNamespace(state={})
        result = json.loads(finalize_preprocessing_context(
            selected_tables=["missing"], selected_columns=[], selected_join_edges=[],
            required_knowledge_phrases=[], selected_knowledge=[],
            unresolved_items=["unknown metric"], tool_context=context,
        ))
        self.assertFalse(result["valid"])
        self.assertFalse(context.state["preprocessing_completed"])
        self.assertNotIn("preprocessing_context", context.state)


if __name__ == "__main__":
    unittest.main()
