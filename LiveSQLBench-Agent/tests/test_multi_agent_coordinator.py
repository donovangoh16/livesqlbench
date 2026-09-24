import unittest

from multi_agent.coordinator import (
    PHASE_AGENT_NAMES, MultiAgentCoordinator, invalidate_for_return,
    phase_message, recovery_directive,
)


class FakeRuntime:
    def __init__(self, stop_at=None):
        self.stop_at = stop_at
        self.sessions = {}
        self.calls = []

    async def init_session(self, task_id, state, reset=True):
        self.sessions[task_id] = dict(state)
        return {"task_id": task_id}

    async def run_turn(self, task_id, message):
        state = self.sessions[task_id]
        phase = state["active_phase_agent"]
        self.calls.append((phase, message, state["steps_remaining"]))
        state["model_turns"] = int(state.get("model_turns", 0)) + 1
        trajectory = list(state.get("tool_trajectory", []))
        trajectory.append({"tool": f"{phase}_tool"})
        state["tool_trajectory"] = trajectory
        previous_usage = state.get("token_usage", {}) or {}
        state["token_usage"] = {
            "input_tokens": int(previous_usage.get("input_tokens", 0)) + 100,
            "cached_input_tokens": int(previous_usage.get("cached_input_tokens", 0)) + 60,
            "uncached_input_tokens": int(previous_usage.get("uncached_input_tokens", 0)) + 40,
            "output_tokens": int(previous_usage.get("output_tokens", 0)) + 20,
            "total_tokens": int(previous_usage.get("total_tokens", 0)) + 120,
        }
        if phase != self.stop_at:
            if phase == "preprocessing":
                state["preprocessing_completed"] = True
                state["preprocessing_context"] = {"selected_tables": ["items"]}
            elif phase == "planning":
                state["query_plan_validated"] = True
                state["query_plan"] = {"version": 1, "category": "Query"}
            elif phase == "sql_generation":
                state["sql_generation_completed"] = True
                state["draft_sql"] = "SELECT 1"
            elif phase == "post_processing":
                state["task_done"] = True
        return {"state": state, "response": phase}


class MultiAgentCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    def test_every_phase_has_a_display_agent_name(self):
        self.assertEqual(
            set(PHASE_AGENT_NAMES),
            {"preprocessing", "planning", "sql_generation", "post_processing"},
        )

    def initial_state(self):
        return {
            "task_id": "t1",
            "user_query": "Return one row",
            "steps_remaining": 30,
            "active_agent_profile": "improved",
            "active_harness_profile": "baseline",
            "active_orchestration_profile": "multi",
        }

    async def test_forward_only_runs_all_four_agents_in_order(self):
        runtime = FakeRuntime()
        coordinator = MultiAgentCoordinator(runtime)
        await coordinator.init_task("t1", self.initial_state())
        result = await coordinator.run("t1", "initial")
        self.assertEqual(
            [phase for phase, _, _ in runtime.calls],
            ["preprocessing", "planning", "sql_generation", "post_processing"],
        )
        self.assertEqual(result["state"]["steps_remaining"], 30)
        self.assertEqual(len(result["state"]["phase_transitions"]), 3)
        self.assertTrue(result["state"]["task_done"])
        for phase in ("preprocessing", "planning", "sql_generation", "post_processing"):
            metrics = result["state"]["agent_metrics"][phase]
            self.assertEqual(metrics["activations"], 1)
            self.assertEqual(metrics["model_calls"], 1)
            self.assertEqual(metrics["tool_calls"], 1)
            self.assertEqual(metrics["total_tokens"], 120)
        self.assertTrue(result["state"]["submission_attempted"])
        self.assertTrue(all(result["state"]["phase_completion"].values()))

    async def test_incomplete_phase_stops_without_running_downstream_agents(self):
        runtime = FakeRuntime(stop_at="planning")
        coordinator = MultiAgentCoordinator(runtime)
        await coordinator.init_task("t1", self.initial_state())
        result = await coordinator.run("t1", "initial")
        self.assertEqual([call[0] for call in runtime.calls], ["preprocessing", "planning"])
        self.assertEqual(result["state"]["multi_agent_stop_reason"], "planning_incomplete")

    async def test_validator_can_route_planning_back_to_preprocessing(self):
        class RoutingRuntime(FakeRuntime):
            def __init__(self):
                super().__init__()
                self.planning_calls = 0

            async def run_turn(self, task_id, message):
                state = self.sessions[task_id]
                phase = state["active_phase_agent"]
                if phase == "planning":
                    self.planning_calls += 1
                    if self.planning_calls == 1:
                        self.calls.append((phase, message, state["steps_remaining"]))
                        state["model_turns"] = int(state.get("model_turns", 0)) + 1
                        state["query_plan_validated"] = False
                        state["query_plan_validation"] = {
                            "return_to_phase": "preprocessing",
                            "required_action": "ground_missing_join",
                            "retryable": False,
                        }
                        return {"state": state, "response": "planning failed"}
                return await super().run_turn(task_id, message)

        runtime = RoutingRuntime()
        coordinator = MultiAgentCoordinator(runtime)
        await coordinator.init_task("t1", self.initial_state())
        result = await coordinator.run("t1", "initial")
        self.assertEqual(
            [call[0] for call in runtime.calls],
            [
                "preprocessing", "planning", "preprocessing", "planning",
                "sql_generation", "post_processing",
            ],
        )
        self.assertEqual(result["state"]["backward_transitions"], 1)
        self.assertIn("planning->preprocessing", result["state"]["phase_transitions"])
        recovery_message = runtime.calls[2][1]
        self.assertIn("ground_missing_join", recovery_message)
        self.assertEqual(result["state"]["routing_events"][0], {
            "from": "planning",
            "to": "preprocessing",
            "required_action": "ground_missing_join",
            "retryable": False,
            "evidence_before": result["state"]["routing_events"][0]["evidence_before"],
            "evidence_after": result["state"]["routing_events"][0]["evidence_after"],
            "new_evidence_acquired": False,
        })

    def test_recovery_directive_uses_existing_validator_fields(self):
        directive = recovery_directive({
            "sql_validation_history": [{
                "return_to_phase": "query_planning",
                "required_action": "repair_output_contract",
                "retryable": True,
            }],
        }, "sql_generation")
        self.assertEqual(directive["return_to_phase"], "planning")
        self.assertEqual(directive["required_action"], "repair_output_contract")

    def test_return_to_planning_invalidates_plan_and_sql_but_keeps_preprocessing(self):
        state = {
            "preprocessing_context": {"selected_tables": ["items"]},
            "preprocessing_completed": True,
            "query_plan": {"version": 1},
            "query_plan_validated": True,
            "draft_sql": "SELECT 1",
            "sql_generation_completed": True,
            "task_done": True,
        }
        invalidate_for_return(state, "planning")
        self.assertIn("preprocessing_context", state)
        self.assertTrue(state["preprocessing_completed"])
        self.assertNotIn("query_plan", state)
        self.assertNotIn("draft_sql", state)
        self.assertFalse(state["task_done"])

    def test_handoffs_contain_artifacts_not_prior_conversation(self):
        state = self.initial_state()
        state["preprocessing_context"] = {"selected_tables": ["items"]}
        message = phase_message("planning", state)
        self.assertIn("PREPROCESSING_CONTEXT", message)
        self.assertIn("selected_tables", message)
        self.assertNotIn("tool_trajectory", message)


if __name__ == "__main__":
    unittest.main()
