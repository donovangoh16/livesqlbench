import unittest

from system_agent.harness import (
    authorize_tool_call, finalize_harness_metrics, observe_tool_call,
)


class HarnessObservationTests(unittest.TestCase):
    class AdkLikeState:
        """Minimal ADK State interface: deliberately does not implement pop."""

        def __init__(self, values):
            self.values = dict(values)

        def get(self, key, default=None):
            return self.values.get(key, default)

        def setdefault(self, key, default=None):
            return self.values.setdefault(key, default)

        def __getitem__(self, key):
            return self.values[key]

        def __setitem__(self, key, value):
            self.values[key] = value

    def test_baseline_harness_does_not_create_metrics(self):
        state = {
            "active_agent_profile": "baseline",
            "active_harness_profile": "baseline",
        }
        observe_tool_call(state, "get_schema", {}, "CREATE TABLE vendors (...)")
        self.assertNotIn("harness_metrics", state)

    def test_variant_two_uses_baseline_adapter_without_blocking(self):
        state = {
            "active_agent_profile": "baseline",
            "active_harness_profile": "improved",
        }
        self.assertIsNone(observe_tool_call(
            state, "get_schema", {}, "CREATE TABLE vendors (...)"
        ))
        self.assertIsNone(observe_tool_call(
            state, "execute_sql", {"sql": "SELECT 1"}, {"success": True}
        ))
        metrics = state["harness_metrics"]
        self.assertEqual(metrics["mode"], "workflow_enforcement")
        self.assertEqual(metrics["adapter"], "baseline")
        self.assertEqual(metrics["event_counts"], {
            "schema_read": 1, "sql_execution": 1,
        })
        self.assertIn("preprocessing->generation", metrics["phase_transitions"])

    def test_variant_three_observes_phases_duplicates_and_no_progress(self):
        state = {
            "active_agent_profile": "improved",
            "active_harness_profile": "improved",
        }
        observe_tool_call(
            state, "finalize_preprocessing_context", {"selected_tables": ["vendors"]},
            {"valid": True},
        )
        repeated_args = {"category": "Query", "plan": {"source_tables": ["vendors"]}}
        repeated_error = {"valid": False, "errors": ["output missing"]}
        observe_tool_call(
            state, "generate_and_validate_query_plan", repeated_args, repeated_error,
        )
        observe_tool_call(
            state, "generate_and_validate_query_plan", repeated_args, repeated_error,
        )
        metrics = state["harness_metrics"]
        self.assertEqual(metrics["adapter"], "improved")
        self.assertEqual(metrics["duplicate_calls_detected"], 1)
        self.assertEqual(metrics["no_progress_events_detected"], 1)
        self.assertEqual(metrics["max_no_progress_streak"], 1)
        self.assertEqual(metrics["retry_counts"]["plan_attempt"], 1)
        self.assertIn("preprocessing->planning", metrics["phase_transitions"])

    def test_variant_two_requires_schema_and_successful_exact_query(self):
        state = {
            "active_agent_profile": "baseline",
            "active_harness_profile": "improved",
        }
        blocked = authorize_tool_call(state, "execute_sql", {"sql": "SELECT 1"})
        self.assertEqual(blocked["reason"], "schema_required_before_execution")
        observe_tool_call(state, "get_schema", {}, "CREATE TABLE vendors (...)")
        self.assertIsNone(authorize_tool_call(state, "execute_sql", {"sql": "SELECT 1"}))
        observe_tool_call(state, "execute_sql", {"sql": "SELECT 1"}, "1\n-\n1")
        self.assertIsNone(authorize_tool_call(
            state, "submit_sql", {"sql": " SELECT  1; "},
        ))
        rejected = authorize_tool_call(state, "submit_sql", {"sql": "SELECT 2"})
        self.assertEqual(rejected["reason"], "submitted_sql_not_successfully_executed")

    def test_variant_two_uses_ast_equivalence_for_submission(self):
        state = {
            "active_agent_profile": "baseline",
            "active_harness_profile": "improved",
        }
        observe_tool_call(state, "get_schema", {}, "CREATE TABLE metrics(value numeric)")
        executed = "SELECT NULLIF(value,0) AS score FROM metrics WHERE value > 1"
        submitted = (
            "-- formatting changed before submission\n"
            "select nullif(value, 0) as score\n"
            "from metrics where value>1;"
        )
        observe_tool_call(state, "execute_sql", {"sql": executed}, "score\n-----\n2")
        self.assertIsNone(authorize_tool_call(
            state, "submit_sql", {"sql": submitted},
        ))
        changed = authorize_tool_call(
            state, "submit_sql",
            {"sql": "SELECT NULLIF(value, 0) AS score FROM metrics WHERE value > 2"},
        )
        self.assertEqual(changed["reason"], "submitted_sql_not_successfully_executed")

    def test_variant_three_enforces_artifact_gates_and_query_execution(self):
        state = {
            "active_agent_profile": "improved",
            "active_harness_profile": "improved",
        }
        blocked = authorize_tool_call(
            state, "generate_and_validate_query_plan", {"category": "Query"},
        )
        self.assertEqual(blocked["reason"], "preprocessing_not_finalized")
        state.update({
            "preprocessing_completed": True,
            "query_plan_validated": True,
            "query_plan": {"category": "Query"},
            "sql_generation_completed": True,
            "draft_sql": "SELECT 1",
        })
        blocked = authorize_tool_call(state, "submit_validated_sql", {})
        self.assertEqual(blocked["reason"], "current_query_sql_not_successfully_executed")
        state.update({
            "last_execution_succeeded": True,
            "last_execution_sql": "SELECT 1",
        })
        self.assertIsNone(authorize_tool_call(state, "submit_validated_sql", {}))

    def test_management_submission_does_not_require_execution(self):
        state = {
            "active_agent_profile": "improved",
            "active_harness_profile": "improved",
            "query_plan": {"category": "Management"},
            "sql_generation_completed": True,
            "draft_sql": "CREATE TABLE example(id integer)",
        }
        self.assertIsNone(authorize_tool_call(state, "submit_validated_sql", {}))

    def test_baseline_cte_update_is_treated_as_management(self):
        state = {
            "active_agent_profile": "baseline",
            "active_harness_profile": "improved",
            "_harness_schema_grounded": True,
        }
        sql = "WITH candidate_rows AS (SELECT 1 AS id) UPDATE vendors SET active=true"
        self.assertIsNone(authorize_tool_call(state, "submit_sql", {"sql": sql}))

    def test_terminal_reason_is_filled(self):
        state = {
            "active_agent_profile": "baseline",
            "active_harness_profile": "improved",
            "steps_remaining": 12,
        }
        result = finalize_harness_metrics(state)
        self.assertEqual(result["termination_reason"], "agent_stopped")

    def test_successful_execution_supports_adk_state_without_pop(self):
        state = self.AdkLikeState({
            "active_agent_profile": "baseline",
            "active_harness_profile": "improved",
            "_harness_failed_sql_hash": "old",
        })
        observe_tool_call(state, "execute_sql", {"sql": "SELECT 1"}, "1\n-\n1")
        self.assertIsNone(state.get("_harness_failed_sql_hash"))
        self.assertTrue(state.get("_harness_successful_sql_hash"))


if __name__ == "__main__":
    unittest.main()
