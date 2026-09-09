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
        self.assertIsNone(authorize_tool_call(state, "get_schema", {}))

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

    def test_exact_duplicate_is_blocked_until_meaningful_state_changes(self):
        state = {
            "active_agent_profile": "baseline",
            "active_harness_profile": "improved",
        }
        self.assertIsNone(authorize_tool_call(state, "get_schema", {}))
        observe_tool_call(state, "get_schema", {}, "CREATE TABLE vendors (...)")
        blocked = authorize_tool_call(state, "get_schema", {})
        self.assertEqual(blocked["reason"], "duplicate_call_without_state_change")
        self.assertEqual(state["harness_metrics"]["duplicate_calls_blocked"], 1)

    def test_event_no_progress_survives_intervening_calls_and_blocks_retry(self):
        state = {
            "active_agent_profile": "improved",
            "active_harness_profile": "improved",
            "preprocessing_completed": True,
        }
        failure = {"valid": False, "errors": [{"code": "missing_filter"}]}
        observe_tool_call(
            state, "generate_and_validate_query_plan", {"plan": {"version": 1}}, failure,
        )
        observe_tool_call(
            state, "inspect_database", {"sql": "SELECT DISTINCT status FROM t"},
            {"success": True, "rows": ["active"]},
        )
        observe_tool_call(
            state, "generate_and_validate_query_plan", {"plan": {"version": 1}}, failure,
        )
        blocked = authorize_tool_call(
            state, "generate_and_validate_query_plan", {"plan": {"version": 3}},
        )
        self.assertEqual(blocked["reason"], "no_progress_without_state_change")
        self.assertEqual(state["harness_metrics"]["no_progress_calls_blocked"], 1)

    def test_meaningful_recovery_reopens_no_progress_event(self):
        state = {
            "active_agent_profile": "improved",
            "active_harness_profile": "improved",
            "preprocessing_completed": True,
        }
        failure = {"valid": False, "errors": ["missing output"]}
        observe_tool_call(state, "generate_and_validate_query_plan", {"v": 1}, failure)
        observe_tool_call(state, "generate_and_validate_query_plan", {"v": 1}, failure)
        blocked = authorize_tool_call(
            state, "generate_and_validate_query_plan", {"v": 3},
        )
        self.assertEqual(blocked["reason"], "no_progress_without_state_change")
        observe_tool_call(
            state, "finalize_preprocessing_context", {"selected_tables": ["vendors"]},
            {"valid": True},
        )
        self.assertIsNone(authorize_tool_call(
            state, "generate_and_validate_query_plan", {"v": 3},
        ))

    def test_query_event_budget_blocks_only_after_calibrated_limit(self):
        state = {
            "active_agent_profile": "improved",
            "active_harness_profile": "improved",
            "preprocessing_completed": True,
        }
        for version in range(1, 6):
            args = {"plan": {"version": version}}
            self.assertIsNone(authorize_tool_call(
                state, "generate_and_validate_query_plan", args,
            ))
            observe_tool_call(
                state, "generate_and_validate_query_plan", args,
                {"valid": False, "errors": [{"code": f"issue_{version}"}]},
            )
        blocked = authorize_tool_call(
            state, "generate_and_validate_query_plan", {"plan": {"version": 6}},
        )
        self.assertEqual(blocked["reason"], "event_budget_exhausted")
        self.assertEqual(state["harness_metrics"]["budget_blocks"], 1)

    def test_management_budget_preserves_observed_ten_plan_success_path(self):
        state = {
            "active_agent_profile": "improved",
            "active_harness_profile": "improved",
            "preprocessing_completed": True,
            "task_category": "Management",
        }
        for version in range(1, 11):
            args = {"category": "Management", "plan": {"version": version}}
            self.assertIsNone(authorize_tool_call(
                state, "generate_and_validate_query_plan", args,
            ))
            observe_tool_call(
                state, "generate_and_validate_query_plan", args,
                {"valid": False, "errors": [{"code": f"issue_{version}"}]},
            )
        self.assertEqual(
            state["harness_metrics"]["budget_limits"]["plan_attempt"], 12,
        )

    def test_execution_error_requires_diagnosis_and_material_correction(self):
        state = {
            "active_agent_profile": "improved",
            "active_harness_profile": "improved",
            "query_plan_validated": True,
            "query_plan": {"category": "Query", "version": 1},
            "sql_generation_completed": True,
            "draft_sql": "SELECT bad_column FROM vendors",
            "draft_sql_plan_version": 1,
        }
        observe_tool_call(
            state, "execute_validated_sql", {},
            {"success": False, "error": "column bad_column does not exist"},
        )
        state["last_execution_succeeded"] = False
        blocked = authorize_tool_call(state, "execute_validated_sql", {})
        self.assertEqual(blocked["reason"], "execution_diagnosis_required_before_retry")
        blocked = authorize_tool_call(
            state, "validate_sql_to_plan",
            {"sql": "SELECT bad_column FROM vendors", "plan_version": 1},
        )
        self.assertEqual(
            blocked["reason"], "execution_diagnosis_required_before_correction",
        )
        observe_tool_call(
            state, "diagnose_last_execution_error", {},
            {"error_type": "missing_column", "retry_allowed": True},
        )
        blocked = authorize_tool_call(
            state, "validate_sql_to_plan",
            {"sql": "SELECT bad_column FROM vendors", "plan_version": 1},
        )
        self.assertEqual(blocked["reason"], "correction_required_after_diagnosis")
        self.assertIsNone(authorize_tool_call(
            state, "validate_sql_to_plan",
            {"sql": "SELECT vendor_id FROM vendors", "plan_version": 1},
        ))
        state.update({
            "draft_sql": "SELECT vendor_id FROM vendors",
            "draft_sql_plan_version": 1,
        })
        self.assertIsNone(authorize_tool_call(state, "execute_validated_sql", {}))

    def test_semantic_warning_requires_revision_and_clean_execution(self):
        state = {
            "active_agent_profile": "improved",
            "active_harness_profile": "improved",
            "query_plan_validated": True,
            "query_plan": {"category": "Query", "version": 1},
            "sql_generation_completed": True,
            "draft_sql": "SELECT vendor_id FROM vendors WHERE active = false",
            "draft_sql_plan_version": 1,
        }
        observe_tool_call(
            state, "execute_validated_sql", {},
            {
                "success": True,
                "empty": False,
                "has_rows": True,
                "columns": ["vendor_id"],
                "alerts": [{"code": "VALUE_OUTSIDE_EXPECTED_RANGE"}],
            },
        )
        blocked = authorize_tool_call(state, "submit_validated_sql", {})
        self.assertEqual(blocked["reason"], "semantic_review_required_before_submission")
        blocked = authorize_tool_call(
            state, "validate_sql_to_plan",
            {"sql": state["draft_sql"], "plan_version": 1},
        )
        self.assertEqual(blocked["reason"], "semantic_revision_required")
        revised_sql = "SELECT vendor_id FROM vendors WHERE active = true"
        self.assertIsNone(authorize_tool_call(
            state, "validate_sql_to_plan", {"sql": revised_sql, "plan_version": 1},
        ))
        state["draft_sql"] = revised_sql
        state["last_execution_sql"] = revised_sql
        state["last_execution_succeeded"] = True
        observe_tool_call(
            state, "execute_validated_sql", {},
            {"success": True, "empty": False, "has_rows": True, "columns": ["vendor_id"]},
        )
        self.assertIsNone(authorize_tool_call(state, "submit_validated_sql", {}))
        metrics = state["harness_metrics"]
        self.assertEqual(metrics["semantic_reviews_required"], 1)
        self.assertEqual(metrics["semantic_reviews_completed"], 1)

    def test_empty_result_is_advisory_and_does_not_block_submission(self):
        sql = "SELECT vendor_id FROM vendors WHERE active = false"
        state = {
            "active_agent_profile": "improved",
            "active_harness_profile": "improved",
            "query_plan_validated": True,
            "query_plan": {"category": "Query", "version": 1},
            "sql_generation_completed": True,
            "draft_sql": sql,
            "draft_sql_plan_version": 1,
            "last_execution_sql": sql,
            "last_execution_succeeded": True,
        }
        observe_tool_call(
            state, "execute_validated_sql", {},
            {"success": True, "empty": True, "has_rows": False, "columns": ["vendor_id"]},
        )
        self.assertIsNone(authorize_tool_call(state, "submit_validated_sql", {}))
        metrics = state["harness_metrics"]
        self.assertEqual(metrics["empty_result_advisories"], 1)
        self.assertEqual(metrics["semantic_reviews_required"], 0)

    def test_changed_preprocessing_evidence_resets_no_progress(self):
        state = {
            "active_agent_profile": "improved",
            "active_harness_profile": "improved",
        }
        failure = {"valid": False, "errors": ["knowledge phrase not covered"]}
        first = {
            "selected_tables": ["devices"],
            "selected_columns": [],
            "selected_join_edges": [],
            "required_knowledge_phrases": ["keyboard"],
            "selected_knowledge": [],
            "unresolved_items": [],
        }
        second = {**first, "selected_knowledge": [{"knowledge_id": 11}]}
        observe_tool_call(state, "finalize_preprocessing_context", first, failure)
        observe_tool_call(state, "finalize_preprocessing_context", second, failure)
        third = {**second, "selected_knowledge": [{"knowledge_id": 11}, {"knowledge_id": 2}]}
        self.assertIsNone(authorize_tool_call(
            state, "finalize_preprocessing_context", third,
        ))
        self.assertEqual(state["harness_metrics"]["no_progress_events_detected"], 0)

    def test_changed_management_plan_is_progress_and_no_progress_is_advisory(self):
        state = {
            "active_agent_profile": "improved",
            "active_harness_profile": "improved",
            "preprocessing_completed": True,
            "task_category": "Management",
        }
        failure = {"valid": False, "errors": ["formula mismatch"]}
        first = {"category": "Management", "plan": {"language": "plpgsql"}}
        second = {"category": "Management", "plan": {"language": "sql"}}
        observe_tool_call(state, "generate_and_validate_query_plan", first, failure)
        observe_tool_call(state, "generate_and_validate_query_plan", second, failure)
        self.assertEqual(state["harness_metrics"]["no_progress_events_detected"], 0)
        self.assertIsNone(authorize_tool_call(
            state, "generate_and_validate_query_plan",
            {"category": "Management", "plan": {"language": "sql", "revision": 2}},
        ))

        # Even a repeated Management outcome is advisory rather than a hard
        # no-progress gate; exact duplicate-call protection remains separate.
        observe_tool_call(state, "generate_and_validate_query_plan", second, failure)
        self.assertEqual(
            state["harness_metrics"]["management_no_progress_advisories"], 1,
        )
        self.assertIsNone(authorize_tool_call(
            state, "generate_and_validate_query_plan",
            {"category": "Management", "plan": {"language": "sql", "revision": 3}},
        ))

    def test_variant_two_uses_changed_sql_without_unavailable_diagnosis_tool(self):
        state = {
            "active_agent_profile": "baseline",
            "active_harness_profile": "improved",
            "_harness_schema_grounded": True,
        }
        observe_tool_call(
            state, "execute_sql", {"sql": "SELECT bad FROM vendors"},
            "SQL Error: column bad does not exist",
        )
        self.assertIsNone(authorize_tool_call(
            state, "execute_sql", {"sql": "SELECT vendor_id FROM vendors"},
        ))


if __name__ == "__main__":
    unittest.main()
