import unittest
from unittest.mock import patch

from orchestrator.evaluation_metrics import (
    aggregate_evaluation_metrics, calculate_task_evaluation_metrics,
)


class EvaluationMetricsTests(unittest.TestCase):
    def task(self):
        return {
            "selected_database": "example",
            "category": "Query",
            "sol_sql": ["SELECT u.id FROM users u"],
            "external_knowledge": [7],
        }

    def result(self, variant=0):
        return {
            "phase1_passed": True,
            "steps_used": 2,
            "total_tokens": 100,
            "elapsed_seconds": 3.0,
            "experiment": {"number": variant},
            "tool_trajectory": [
                {"tool": "get_knowledge_definition", "args": {"knowledge_name": "7"},
                 "result": '{"id":7}'},
                {"tool": "submit_sql", "args": {"sql": "SELECT u.id FROM users u"},
                 "result": "ok"},
            ],
        }

    @patch("orchestrator.evaluation_metrics._kb_catalog", return_value={"7"})
    @patch("orchestrator.evaluation_metrics._schema_catalog", return_value=({"users"}, {"users.id"}))
    def test_common_metrics_and_variant_applicability(self, _schema, _kb):
        metrics = calculate_task_evaluation_metrics(self.task(), self.result(0))
        self.assertTrue(metrics["common"]["task_success"])
        self.assertEqual(metrics["common"]["table"]["f1"], 1.0)
        self.assertEqual(
            metrics["common"]["invalid_references"]["overall_invalid_reference_rate"],
            0.0,
        )
        self.assertFalse(metrics["phase_specific"]["applicable"])
        self.assertFalse(metrics["harness_specific"]["applicable"])

    @patch("orchestrator.evaluation_metrics._kb_catalog", return_value={"7"})
    @patch("orchestrator.evaluation_metrics._schema_catalog", return_value=({"users"}, {"users.id"}))
    def test_aggregate_common_metrics(self, _schema, _kb):
        first = self.result(0)
        first["evaluation_metrics"] = calculate_task_evaluation_metrics(self.task(), first)
        second = self.result(0)
        second["phase1_passed"] = False
        second["evaluation_metrics"] = calculate_task_evaluation_metrics(self.task(), second)
        aggregate = aggregate_evaluation_metrics([first, second])
        self.assertEqual(aggregate["task_success_rate"], 0.5)
        self.assertEqual(aggregate["submission_rate"], 1.0)
        self.assertEqual(aggregate["table_f1"], 1.0)
        self.assertEqual(aggregate["average_steps_used"], 2.0)

    @patch("orchestrator.evaluation_metrics._kb_catalog", return_value=set())
    @patch("orchestrator.evaluation_metrics._schema_catalog", return_value=(set(), set()))
    def test_invalid_schema_and_kb_are_separate(self, _schema, _kb):
        metrics = calculate_task_evaluation_metrics(self.task(), self.result(0))["common"]
        invalid = metrics["invalid_references"]
        self.assertEqual(invalid["invalid_table_count"], 1)
        self.assertEqual(invalid["invalid_column_count"], 1)
        self.assertEqual(invalid["invalid_kb_count"], 1)
        self.assertTrue(invalid["task_has_invalid_reference"])


if __name__ == "__main__":
    unittest.main()
