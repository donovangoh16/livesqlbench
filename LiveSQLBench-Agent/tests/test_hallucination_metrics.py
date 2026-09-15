import unittest

from orchestrator.hallucination_metrics import calculate_hallucination_metrics


class HallucinationMetricsTests(unittest.TestCase):
    def test_phase_references_and_explicit_errors_are_reported(self):
        task = {
            "sol_sql": ["SELECT u.id FROM users AS u"],
            "external_knowledge": [7],
        }
        trajectory = [
            {
                "tool": "finalize_preprocessing_context",
                "args": {
                    "selected_tables": ["users", "invented"],
                    "selected_columns": [
                        {"table": "users", "columns": [{"name": "id"}]}
                    ],
                    "selected_knowledge": [{"id": 7}],
                },
                "result": '{"valid":false,"errors":["Selected table does not exist"]}',
            },
            {
                "tool": "generate_and_validate_query_plan",
                "args": {
                    "plan": {
                        "source_tables": ["users", "invented"],
                        "output_columns": [
                            {"source_expression": "users.id"},
                            {"source_expression": "invented.value"},
                        ],
                        "calculations": [{"knowledge_id": 7}],
                    }
                },
                "result": '{"valid":true}',
            },
            {
                "tool": "validate_sql_to_plan",
                "args": {"sql": "SELECT u.id FROM users AS u"},
                "result": '{"valid":true}',
            },
            {"tool": "submit_validated_sql", "args": {}, "result": "ok"},
        ]

        metrics = calculate_hallucination_metrics(task, trajectory)

        self.assertEqual(
            metrics["phases"]["preprocessing"]["tables"]["hallucination_rate"],
            0.5,
        )
        self.assertEqual(
            metrics["phases"]["query_planning"]["columns"]["hallucinated_count"],
            1,
        )
        self.assertEqual(
            metrics["phases"]["sql_generation"]["tables"]["hallucination_rate"],
            0.0,
        )
        self.assertTrue(metrics["explicit_unsupported_reference_detected"])


if __name__ == "__main__":
    unittest.main()
